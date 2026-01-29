"""
Audio Processing Module
Handles audio file loading, preprocessing, and speech-to-text conversion
Uses OpenAI Whisper or faster-whisper for transcription
"""

import torch
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Optional, Dict, Tuple
from loguru import logger
import soundfile as sf
import librosa
import numpy as np

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.warning("faster-whisper not available, falling back to standard whisper")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("whisper not available")


class AudioProcessor:
    """
    Processes audio files and converts speech to text using Whisper models
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the AudioProcessor with configuration
        
        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config
        self.model = None
        self.model_type = config.get("model_type", "faster-whisper")
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = config.get("sample_rate", 16000)
        
        logger.info(f"Initializing AudioProcessor with {self.model_type} on {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load the speech-to-text model"""
        try:
            model_name = self.config.get("model_name", "base")
            
            if self.model_type == "faster-whisper" and FASTER_WHISPER_AVAILABLE:
                # Load faster-whisper model (more efficient)
                compute_type = self.config.get("compute_type", "float16")
                self.model = WhisperModel(
                    model_name,
                    device=self.device,
                    compute_type=compute_type if self.device == "cuda" else "int8"
                )
                logger.info(f"✓ Loaded faster-whisper model: {model_name}")
                
            elif WHISPER_AVAILABLE:
                # Load standard whisper model
                self.model = whisper.load_model(model_name, device=self.device)
                logger.info(f"✓ Loaded whisper model: {model_name}")
                
            else:
                raise ImportError("Neither whisper nor faster-whisper is available")
                
        except Exception as e:
            logger.error(f"Failed to load speech-to-text model: {e}")
            raise
    
    def load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return audio data with sample rate
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            logger.info(f"Loading audio file: {audio_path}")
            
            # Load audio file using librosa (handles multiple formats)
            audio_data, sr = librosa.load(
                str(audio_path),
                sr=self.sample_rate,
                mono=True
            )
            
            logger.info(f"✓ Loaded audio: {len(audio_data)/sr:.2f}s, {sr}Hz")
            return audio_data, sr
            
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise
    
    def preprocess_audio(self, audio_data: np.ndarray, 
                        normalize: bool = True) -> np.ndarray:
        """
        Preprocess audio data (normalize, remove silence, etc.)
        
        Args:
            audio_data: Audio data as numpy array
            normalize: Whether to normalize audio amplitude
            
        Returns:
            Preprocessed audio data
        """
        try:
            # Normalize audio
            if normalize:
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    audio_data = audio_data / max_val
            
            # Remove leading/trailing silence
            audio_data, _ = librosa.effects.trim(
                audio_data,
                top_db=20,
                frame_length=2048,
                hop_length=512
            )
            
            logger.info("✓ Audio preprocessing completed")
            return audio_data
            
        except Exception as e:
            logger.warning(f"Audio preprocessing failed, using original: {e}")
            return audio_data
    
    def transcribe(self, audio_path: Path, 
                   language: Optional[str] = None) -> Dict[str, any]:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es') or None for auto-detect
            
        Returns:
            Dictionary containing transcript and metadata
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        try:
            logger.info(f"Starting transcription: {audio_path.name}")
            
            # Set language
            lang = language or self.config.get("language", None)
            
            if self.model_type == "faster-whisper" and FASTER_WHISPER_AVAILABLE:
                # Use faster-whisper
                segments, info = self.model.transcribe(
                    str(audio_path),
                    language=lang,
                    task=self.config.get("task", "transcribe"),
                    beam_size=5,
                    vad_filter=True,  # Voice activity detection
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                
                # Collect segments
                transcript_segments = []
                full_text = []
                
                for segment in segments:
                    transcript_segments.append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                        "confidence": segment.avg_logprob
                    })
                    full_text.append(segment.text.strip())
                
                result = {
                    "text": " ".join(full_text),
                    "segments": transcript_segments,
                    "language": info.language,
                    "language_probability": info.language_probability,
                    "duration": info.duration,
                    "model_type": "faster-whisper",
                }
                
            else:
                # Use standard whisper
                result_whisper = self.model.transcribe(
                    str(audio_path),
                    language=lang,
                    task=self.config.get("task", "transcribe"),
                    fp16=(self.device == "cuda")
                )
                
                result = {
                    "text": result_whisper["text"].strip(),
                    "segments": [
                        {
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": seg["text"].strip(),
                            "confidence": seg.get("avg_logprob", 0.0)
                        }
                        for seg in result_whisper.get("segments", [])
                    ],
                    "language": result_whisper.get("language", lang or "en"),
                    "model_type": "whisper",
                }
            
            logger.info(f"✓ Transcription completed: {len(result['text'])} characters")
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def process_audio_file(self, audio_path: Path, 
                          preprocess: bool = True,
                          language: Optional[str] = None) -> Dict[str, any]:
        """
        Complete audio processing pipeline: load, preprocess, and transcribe
        
        Args:
            audio_path: Path to audio file
            preprocess: Whether to preprocess audio
            language: Language code or None for auto-detect
            
        Returns:
            Dictionary containing transcript and metadata
        """
        try:
            # Validate audio file
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Load audio (for validation and metadata)
            audio_data, sr = self.load_audio(audio_path)
            duration = len(audio_data) / sr
            
            logger.info(f"Processing audio: {audio_path.name} ({duration:.2f}s)")
            
            # Transcribe
            result = self.transcribe(audio_path, language=language)
            
            # Add metadata
            result["audio_file"] = str(audio_path.name)
            result["audio_duration"] = duration
            result["sample_rate"] = sr
            
            return result
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise
    
    def batch_transcribe(self, audio_paths: list, 
                        language: Optional[str] = None) -> list:
        """
        Transcribe multiple audio files
        
        Args:
            audio_paths: List of audio file paths
            language: Language code or None for auto-detect
            
        Returns:
            List of transcription results
        """
        results = []
        total = len(audio_paths)
        
        for idx, audio_path in enumerate(audio_paths, 1):
            try:
                logger.info(f"Processing {idx}/{total}: {audio_path.name}")
                result = self.process_audio_file(audio_path, language=language)
                results.append({
                    "success": True,
                    "file": str(audio_path),
                    "result": result
                })
            except Exception as e:
                logger.error(f"Failed to process {audio_path.name}: {e}")
                results.append({
                    "success": False,
                    "file": str(audio_path),
                    "error": str(e)
                })
        
        return results
    
    def get_audio_info(self, audio_path: Path) -> Dict[str, any]:
        """
        Get metadata about an audio file without transcribing
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio metadata
        """
        try:
            audio_data, sr = self.load_audio(audio_path)
            duration = len(audio_data) / sr
            
            return {
                "filename": audio_path.name,
                "duration_seconds": round(duration, 2),
                "sample_rate": sr,
                "channels": 1,  # Mono
                "file_size_mb": round(audio_path.stat().st_size / (1024 * 1024), 2)
            }
        except Exception as e:
            logger.error(f"Failed to get audio info: {e}")
            return {"error": str(e)}


# Test function
if __name__ == "__main__":
    print("AudioProcessor module loaded successfully!")
    print(f"Faster-whisper available: {FASTER_WHISPER_AVAILABLE}")
    print(f"Whisper available: {WHISPER_AVAILABLE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
