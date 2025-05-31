#!/usr/bin/env python3
"""
Voice Harvester - Extract voice samples from audio files
Perfect for creating TTS reference voices from audiobooks, podcasts, etc.
"""

import argparse
import logging
from pathlib import Path
import torchaudio as ta
import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class VoiceHarvester:
    def __init__(self):
        logging.info("🎤 Voice Harvester initialized")
    
    def extract_clean_segments(self, audio_file: Path, output_dir: Path, 
                             segment_length: int = 10, min_silence: float = 0.5):
        """Extract clean voice segments from audio file"""
        
        logging.info(f"📂 Processing: {audio_file}")
        
        try:
            # Load audio
            waveform, sample_rate = ta.load(str(audio_file))
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            duration = waveform.shape[1] / sample_rate
            logging.info(f"📊 Audio: {duration:.1f}s, {sample_rate}Hz")
            
            # Detect voice activity (simple energy-based)
            voice_segments = self._detect_voice_segments(waveform, sample_rate, min_silence)
            
            logging.info(f"🔍 Found {len(voice_segments)} voice segments")
            
            # Extract segments
            output_dir.mkdir(parents=True, exist_ok=True)
            extracted_count = 0
            
            for i, (start, end) in enumerate(voice_segments):
                segment_duration = end - start
                
                # Skip very short segments
                if segment_duration < 3.0:
                    continue
                
                # Trim to desired length
                if segment_duration > segment_length:
                    # Take from the middle for best quality
                    mid_point = (start + end) / 2
                    start = mid_point - segment_length / 2
                    end = mid_point + segment_length / 2
                
                # Extract audio segment
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                segment = waveform[:, start_sample:end_sample]
                
                # Check audio quality
                if self._is_good_quality(segment, sample_rate):
                    output_file = output_dir / f"voice_segment_{extracted_count:03d}.wav"
                    ta.save(str(output_file), segment, sample_rate)
                    
                    logging.info(f"✅ Extracted: {output_file.name} ({segment_duration:.1f}s)")
                    extracted_count += 1
                    
                    # Limit number of extractions
                    if extracted_count >= 20:
                        break
            
            logging.info(f"🎉 Extracted {extracted_count} voice segments")
            return extracted_count
            
        except Exception as e:
            logging.error(f"❌ Error processing {audio_file}: {e}")
            return 0
    
    def _detect_voice_segments(self, waveform: torch.Tensor, sample_rate: int, 
                              min_silence: float = 0.5):
        """Simple voice activity detection"""
        
        # Calculate energy in windows
        window_size = int(0.1 * sample_rate)  # 100ms windows
        hop_size = window_size // 2
        
        energies = []
        for i in range(0, waveform.shape[1] - window_size, hop_size):
            window = waveform[:, i:i + window_size]
            energy = torch.mean(window ** 2).item()
            energies.append(energy)
        
        energies = np.array(energies)
        
        # Threshold for voice detection (adaptive)
        threshold = np.percentile(energies, 25)  # Bottom 25% is likely silence
        
        # Find voice segments
        voice_mask = energies > threshold
        segments = []
        
        in_voice = False
        start_time = 0
        
        for i, is_voice in enumerate(voice_mask):
            time = i * hop_size / sample_rate
            
            if is_voice and not in_voice:
                # Start of voice segment
                start_time = time
                in_voice = True
            elif not is_voice and in_voice:
                # End of voice segment
                if time - start_time > min_silence:  # Minimum segment length
                    segments.append((start_time, time))
                in_voice = False
        
        # Handle case where voice continues to end
        if in_voice:
            segments.append((start_time, len(energies) * hop_size / sample_rate))
        
        return segments
    
    def _is_good_quality(self, segment: torch.Tensor, sample_rate: int) -> bool:
        """Check if audio segment is good quality for TTS"""
        
        # Check for minimum length
        duration = segment.shape[1] / sample_rate
        if duration < 3.0:
            return False
        
        # Check for reasonable volume
        rms = torch.sqrt(torch.mean(segment ** 2))
        if rms < 0.01 or rms > 0.8:  # Too quiet or too loud
            return False
        
        # Check for clipping
        max_val = torch.max(torch.abs(segment))
        if max_val > 0.95:  # Likely clipped
            return False
        
        return True
    
    def create_voice_library(self, source_dir: Path, output_dir: Path):
        """Process multiple audio files to create voice library"""
        
        logging.info(f"📚 Creating voice library from: {source_dir}")
        
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.m4a', '*.flac', '*.ogg']:
            audio_files.extend(source_dir.glob(ext))
        
        if not audio_files:
            logging.error("❌ No audio files found!")
            return
        
        logging.info(f"📁 Found {len(audio_files)} audio files")
        
        total_extracted = 0
        
        for audio_file in audio_files:
            # Create subdirectory for each source
            voice_output_dir = output_dir / audio_file.stem
            
            extracted = self.extract_clean_segments(
                audio_file, 
                voice_output_dir,
                segment_length=10  # 10-second segments
            )
            
            total_extracted += extracted
        
        logging.info(f"🎉 Voice library complete! Total segments: {total_extracted}")
        
        # Create usage guide
        guide_file = output_dir / "USAGE_GUIDE.txt"
        with open(guide_file, 'w') as f:
            f.write("""Voice Library Usage Guide
=========================

This directory contains extracted voice segments suitable for TTS cloning.

Usage:
1. Listen to segments to find voices you like
2. Use the best segments as reference voices:
   python audiobook_tts.py book.txt --voice voices/speaker_001/voice_segment_003.wav

Tips:
- Choose segments with clear, natural speech
- Avoid segments with background noise or music
- 5-15 second segments work best for TTS cloning
- Test different segments to find your preferred voice style

Legal Note:
Only use voices from content you have permission to use!
""")
        
        logging.info(f"📝 Created usage guide: {guide_file}")

def download_librivox_sample():
    """Example of how to download a LibriVox sample"""
    
    print("""
🎙️ LibriVox Voice Samples

To get free voice samples from LibriVox:

1. Visit https://librivox.org/
2. Browse by reader or genre
3. Download MP3 files
4. Use this script to extract clean segments

Popular readers with clear voices:
- LibriVox volunteers often have excellent diction
- Look for "solo" readings (single narrator)
- Fiction works often have more expressive narration

Example downloads:
wget https://archive.org/download/[book_identifier]/[chapter].mp3

Then process with:
python voice_harvester.py --extract downloaded_audiobook.mp3
""")

def main():
    parser = argparse.ArgumentParser(
        description="Extract voice samples from audio files for TTS cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python voice_harvester.py --extract audiobook.mp3
  python voice_harvester.py --library audiobooks/ --output voice_library/
  python voice_harvester.py --librivox-info
        """
    )
    
    parser.add_argument(
        "--extract",
        help="Extract voice segments from single audio file"
    )
    
    parser.add_argument(
        "--library",
        help="Create voice library from directory of audio files"
    )
    
    parser.add_argument(
        "--output",
        default="extracted_voices",
        help="Output directory for extracted voices (default: extracted_voices)"
    )
    
    parser.add_argument(
        "--segment-length",
        type=int,
        default=10,
        help="Target segment length in seconds (default: 10)"
    )
    
    parser.add_argument(
        "--librivox-info",
        action="store_true",
        help="Show info about downloading LibriVox samples"
    )
    
    args = parser.parse_args()
    
    if args.librivox_info:
        download_librivox_sample()
        return
    
    harvester = VoiceHarvester()
    output_dir = Path(args.output)
    
    if args.extract:
        input_file = Path(args.extract)
        if not input_file.exists():
            logging.error(f"Input file not found: {input_file}")
            return
        
        harvester.extract_clean_segments(
            input_file, 
            output_dir, 
            segment_length=args.segment_length
        )
    
    elif args.library:
        source_dir = Path(args.library)
        if not source_dir.exists():
            logging.error(f"Source directory not found: {source_dir}")
            return
        
        harvester.create_voice_library(source_dir, output_dir)
    
    else:
        logging.error("Please specify --extract, --library, or --librivox-info")

if __name__ == "__main__":
    main()