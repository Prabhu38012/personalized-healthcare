"""
Data Handler Module
Handles saving and loading of transcripts, summaries, and prescriptions
Supports JSON and CSV formats
"""

import json
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from loguru import logger


class DataHandler:
    """
    Handles data storage and retrieval for medical consultation data
    """
    
    def __init__(self, config: Dict):
        """
        Initialize DataHandler with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.format = config.get("format", "json")
        self.pretty_print = config.get("pretty_print", True)
        self.include_timestamp = config.get("include_timestamp", True)
        self.include_metadata = config.get("include_metadata", True)
        
        # Get directories from config
        self.transcripts_dir = config.get("transcripts_dir", Path("data/transcripts"))
        self.summaries_dir = config.get("summaries_dir", Path("data/summaries"))
        self.prescriptions_dir = config.get("prescriptions_dir", Path("data/prescriptions"))
        
        # Ensure directories exist
        self._ensure_directories()
        
        logger.info("DataHandler initialized")
    
    def _ensure_directories(self):
        """Create directories if they don't exist"""
        for directory in [self.transcripts_dir, self.summaries_dir, self.prescriptions_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _add_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add metadata to data dictionary
        
        Args:
            data: Original data dictionary
            
        Returns:
            Data with added metadata
        """
        if not self.include_metadata:
            return data
        
        metadata = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0"
        }
        
        if self.include_timestamp:
            metadata["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        data["metadata"] = metadata
        return data
    
    def _generate_filename(self, base_name: str, extension: str) -> str:
        """
        Generate filename with optional timestamp
        
        Args:
            base_name: Base name for the file
            extension: File extension (e.g., 'json', 'csv')
            
        Returns:
            Generated filename
        """
        if self.include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{base_name}_{timestamp}.{extension}"
        return f"{base_name}.{extension}"
    
    def save_transcript(self, transcript_data: Dict[str, Any], 
                       filename: Optional[str] = None) -> Path:
        """
        Save transcript data
        
        Args:
            transcript_data: Dictionary containing transcript data
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        try:
            # Add metadata
            data = self._add_metadata(transcript_data.copy())
            
            # Generate filename
            if not filename:
                audio_file = transcript_data.get("audio_file", "transcript")
                base_name = Path(audio_file).stem
                filename = self._generate_filename(base_name, "json")
            
            # Save path
            save_path = self.transcripts_dir / filename
            
            # Save as JSON
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2 if self.pretty_print else None, ensure_ascii=False)
            
            logger.info(f"✓ Transcript saved: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to save transcript: {e}")
            raise
    
    def save_summary(self, summary_data: Dict[str, Any], 
                    filename: Optional[str] = None) -> Path:
        """
        Save summary data
        
        Args:
            summary_data: Dictionary containing summary data
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        try:
            # Add metadata
            data = self._add_metadata(summary_data.copy())
            
            # Generate filename
            if not filename:
                filename = self._generate_filename("summary", "json")
            
            # Save path
            save_path = self.summaries_dir / filename
            
            # Save based on format
            if self.format == "json" or self.format == "both":
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2 if self.pretty_print else None, ensure_ascii=False)
                logger.info(f"✓ Summary saved (JSON): {save_path}")
            
            if self.format == "csv" or self.format == "both":
                csv_path = save_path.with_suffix('.csv')
                # Flatten data for CSV
                flat_data = self._flatten_dict(data)
                df = pd.DataFrame([flat_data])
                df.to_csv(csv_path, index=False)
                logger.info(f"✓ Summary saved (CSV): {csv_path}")
            
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")
            raise
    
    def save_prescriptions(self, prescriptions: List[Dict[str, Any]], 
                          filename: Optional[str] = None) -> Path:
        """
        Save prescription data
        
        Args:
            prescriptions: List of prescription dictionaries
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        try:
            # Prepare data
            data = {
                "prescriptions": prescriptions,
                "count": len(prescriptions)
            }
            data = self._add_metadata(data)
            
            # Generate filename
            if not filename:
                filename = self._generate_filename("prescriptions", "json")
            
            # Save path
            save_path = self.prescriptions_dir / filename
            
            # Save based on format
            if self.format == "json" or self.format == "both":
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2 if self.pretty_print else None, ensure_ascii=False)
                logger.info(f"✓ Prescriptions saved (JSON): {save_path}")
            
            if self.format == "csv" or self.format == "both":
                csv_path = save_path.with_suffix('.csv')
                df = pd.DataFrame(prescriptions)
                # Handle list columns
                for col in df.columns:
                    if df[col].apply(lambda x: isinstance(x, list)).any():
                        df[col] = df[col].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
                df.to_csv(csv_path, index=False)
                logger.info(f"✓ Prescriptions saved (CSV): {csv_path}")
            
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to save prescriptions: {e}")
            raise
    
    def save_complete_report(self, audio_file: str, transcript: Dict[str, Any],
                           summary: Dict[str, Any], prescriptions: List[Dict[str, Any]]) -> Path:
        """
        Save a complete medical consultation report
        
        Args:
            audio_file: Name of the audio file
            transcript: Transcript data
            summary: Summary data
            prescriptions: List of prescriptions
            
        Returns:
            Path to saved report
        """
        try:
            # Prepare complete report
            report = {
                "audio_file": audio_file,
                "transcript": transcript,
                "summary": summary,
                "prescriptions": prescriptions,
                "prescription_count": len(prescriptions)
            }
            
            report = self._add_metadata(report)
            
            # Generate filename
            base_name = Path(audio_file).stem
            filename = self._generate_filename(f"report_{base_name}", "json")
            
            # Save path
            save_path = self.summaries_dir / filename
            
            # Save report
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2 if self.pretty_print else None, ensure_ascii=False)
            
            logger.info(f"✓ Complete report saved: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to save complete report: {e}")
            raise
    
    def load_transcript(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Load transcript from file
        
        Args:
            filename: Name of the transcript file
            
        Returns:
            Transcript data or None if failed
        """
        try:
            file_path = self.transcripts_dir / filename
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"✓ Transcript loaded: {filename}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load transcript: {e}")
            return None
    
    def load_summary(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Load summary from file
        
        Args:
            filename: Name of the summary file
            
        Returns:
            Summary data or None if failed
        """
        try:
            file_path = self.summaries_dir / filename
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"✓ Summary loaded: {filename}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load summary: {e}")
            return None
    
    def load_prescriptions(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Load prescriptions from file
        
        Args:
            filename: Name of the prescriptions file
            
        Returns:
            Prescriptions data or None if failed
        """
        try:
            file_path = self.prescriptions_dir / filename
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"✓ Prescriptions loaded: {filename}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load prescriptions: {e}")
            return None
    
    def list_files(self, directory: str = "transcripts") -> List[str]:
        """
        List all files in a directory
        
        Args:
            directory: Directory name ('transcripts', 'summaries', 'prescriptions')
            
        Returns:
            List of filenames
        """
        dir_map = {
            "transcripts": self.transcripts_dir,
            "summaries": self.summaries_dir,
            "prescriptions": self.prescriptions_dir
        }
        
        target_dir = dir_map.get(directory, self.transcripts_dir)
        
        try:
            files = [f.name for f in Path(target_dir).glob("*.json")]
            logger.info(f"Found {len(files)} files in {directory}")
            return sorted(files, reverse=True)  # Most recent first
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', 
                     sep: str = '_') -> Dict[str, Any]:
        """
        Flatten nested dictionary for CSV export
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for recursion
            sep: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                items.append((new_key, '; '.join(map(str, v))))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def export_to_csv(self, data_list: List[Dict[str, Any]], 
                     output_path: Path) -> bool:
        """
        Export list of dictionaries to CSV
        
        Args:
            data_list: List of data dictionaries
            output_path: Path to save CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Flatten all dictionaries
            flattened = [self._flatten_dict(d) for d in data_list]
            
            # Create DataFrame and save
            df = pd.DataFrame(flattened)
            df.to_csv(output_path, index=False)
            
            logger.info(f"✓ Exported {len(data_list)} records to CSV: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export to CSV: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about stored data
        
        Returns:
            Dictionary with file counts
        """
        stats = {
            "transcripts": len(self.list_files("transcripts")),
            "summaries": len(self.list_files("summaries")),
            "prescriptions": len(self.list_files("prescriptions"))
        }
        
        logger.info(f"Statistics: {stats}")
        return stats


# Test function
if __name__ == "__main__":
    print("Testing DataHandler...")
    
    # Sample configuration
    config = {
        "format": "json",
        "pretty_print": True,
        "include_timestamp": True,
        "include_metadata": True,
        "transcripts_dir": Path("test_data/transcripts"),
        "summaries_dir": Path("test_data/summaries"),
        "prescriptions_dir": Path("test_data/prescriptions")
    }
    
    handler = DataHandler(config)
    
    # Test data
    transcript = {
        "text": "Patient complains of fever and cough.",
        "audio_file": "consultation_001.wav",
        "duration": 120.5
    }
    
    summary = {
        "general_summary": "Patient has flu symptoms.",
        "diagnosis": "Common cold"
    }
    
    prescriptions = [
        {
            "medicine": "Paracetamol",
            "dosage": "500mg",
            "frequency": "three times daily"
        }
    ]
    
    print("\n Testing save functions...")
    try:
        # Test save transcript
        path = handler.save_transcript(transcript)
        print(f"✓ Transcript saved to: {path}")
        
        # Test save summary
        path = handler.save_summary(summary)
        print(f"✓ Summary saved to: {path}")
        
        # Test save prescriptions
        path = handler.save_prescriptions(prescriptions)
        print(f"✓ Prescriptions saved to: {path}")
        
        # Get statistics
        stats = handler.get_statistics()
        print(f"\n✓ Statistics: {stats}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
    
    print("\n✅ DataHandler test completed!")
