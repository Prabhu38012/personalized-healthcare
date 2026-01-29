# Sample Medical Consultation Audio Files

This directory is where you should place your medical consultation audio files for processing.

## Supported Audio Formats

- WAV (`.wav`)
- MP3 (`.mp3`)
- M4A (`.m4a`)
- FLAC (`.flac`)
- OGG (`.ogg`)

## Audio Quality Guidelines

For best transcription results:

1. **Sample Rate**: 16kHz or higher
2. **Bit Depth**: 16-bit or higher
3. **Channels**: Mono or Stereo (will be converted to mono)
4. **Duration**: Up to 1 hour (longer files may take more time)
5. **Background Noise**: Minimize as much as possible
6. **Volume**: Clear and audible speech

## Example File Naming

Use descriptive names for easy identification:

```
consultation_2026-01-26_patient001.wav
dr_smith_followup_20260126.mp3
initial_checkup_john_doe.wav
```

## Getting Sample Audio

If you don't have medical consultation audio files, you can:

1. **Record a sample conversation** using your phone or computer
2. **Use text-to-speech** to create synthetic medical dialogues
3. **Find public domain medical recordings** (ensure proper licensing)

## Privacy & Security

⚠️ **IMPORTANT**: 
- Ensure you have proper consent to record and process consultations
- Follow HIPAA and local privacy regulations
- Implement proper access controls and data encryption
- Remove or anonymize patient identifying information

## Processing Your Audio

### Method 1: Web Interface
1. Run: `streamlit run app.py`
2. Upload your audio file through the web interface
3. View results in real-time

### Method 2: Python Script
```python
from pathlib import Path
from modules import AudioProcessor
from config import SPEECH_TO_TEXT_CONFIG

processor = AudioProcessor(SPEECH_TO_TEXT_CONFIG)
result = processor.process_audio_file(Path("data/audio_files/your_file.wav"))
print(result["text"])
```

## Processed Files

After processing, results will be saved in:
- Transcripts: `data/transcripts/`
- Summaries: `data/summaries/`
- Prescriptions: `data/prescriptions/`

## Need Help?

See the main README.md for:
- Installation instructions
- Troubleshooting guide
- API documentation
- More examples
