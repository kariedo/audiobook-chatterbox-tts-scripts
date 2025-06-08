#!/usr/bin/env python3
"""
Audio Converter and Splitter Module
Handles audio format conversion and intelligent file splitting using FFmpeg
"""

import logging
import subprocess
import json
import math
from pathlib import Path
from typing import List


class AudioConverter:
    """Handles audio format conversion using FFmpeg"""
    
    @staticmethod
    def check_ffmpeg_available() -> bool:
        """Check if FFmpeg is available on the system"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False
    
    @staticmethod
    def convert_to_mp3(wav_file: Path, mp3_file: Path, bitrate: str = "128k", 
                       remove_wav: bool = False, metadata: dict = None) -> bool:
        """Convert WAV file to MP3 using FFmpeg with optional metadata"""
        
        if not AudioConverter.check_ffmpeg_available():
            logging.error("❌ FFmpeg not found! Please install FFmpeg for MP3 conversion")
            logging.error("   Install guide: https://ffmpeg.org/download.html")
            return False
        
        try:
            # FFmpeg command for high-quality audiobook conversion
            cmd = [
                'ffmpeg',
                '-i', str(wav_file),          # Input file
                '-codec:a', 'libmp3lame',     # MP3 encoder
                '-b:a', bitrate,              # Bitrate
                '-ar', '22050',               # Sample rate (good for speech)
                '-ac', '1',                   # Mono (smaller file, fine for audiobooks)
                '-compression_level', '2',    # Good compression vs speed balance
            ]
            
            # Add metadata if provided
            if metadata:
                if metadata.get('artist'):
                    cmd.extend(['-metadata', f"artist={metadata['artist']}"])
                if metadata.get('album'):
                    cmd.extend(['-metadata', f"album={metadata['album']}"])
                if metadata.get('title'):
                    cmd.extend(['-metadata', f"title={metadata['title']}"])
                if metadata.get('track'):
                    cmd.extend(['-metadata', f"track={metadata['track']}"])
                if metadata.get('genre'):
                    cmd.extend(['-metadata', f"genre={metadata['genre']}"])
            
            cmd.extend([
                '-y',                         # Overwrite output file
                str(mp3_file)
            ])
            
            logging.info(f"🔄 Converting {wav_file.name} to MP3 (bitrate: {bitrate})...")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                # Check if output file was created and has reasonable size
                if mp3_file.exists() and mp3_file.stat().st_size > 1024:  # At least 1KB
                    
                    # Calculate compression ratio
                    wav_size = wav_file.stat().st_size
                    mp3_size = mp3_file.stat().st_size
                    ratio = (1 - mp3_size / wav_size) * 100
                    
                    logging.info(f"✅ MP3 conversion successful!")
                    logging.info(f"   Size reduction: {ratio:.1f}% ({wav_size//1024//1024}MB → {mp3_size//1024//1024}MB)")
                    
                    # Remove WAV file if requested
                    if remove_wav:
                        try:
                            wav_file.unlink()
                            logging.info(f"🗑️ Removed original WAV file: {wav_file.name}")
                        except Exception as e:
                            logging.warning(f"Could not remove WAV file: {e}")
                    
                    return True
                else:
                    logging.error(f"❌ MP3 conversion failed: output file missing or too small")
                    return False
            else:
                logging.error(f"❌ FFmpeg conversion failed:")
                logging.error(f"   stdout: {result.stdout}")
                logging.error(f"   stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logging.error(f"❌ MP3 conversion timed out for {wav_file.name}")
            return False
        except Exception as e:
            logging.error(f"❌ MP3 conversion error: {e}")
            return False
    
    @staticmethod
    def get_audio_info(file_path: Path) -> dict:
        """Get audio file information using FFprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(file_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                info = json.loads(result.stdout)
                
                # Extract relevant information
                format_info = info.get('format', {})
                stream_info = info.get('streams', [{}])[0]
                
                return {
                    'duration': float(format_info.get('duration', 0)),
                    'size': int(format_info.get('size', 0)),
                    'bitrate': int(format_info.get('bit_rate', 0)),
                    'sample_rate': int(stream_info.get('sample_rate', 0)),
                    'channels': int(stream_info.get('channels', 0)),
                    'codec': stream_info.get('codec_name', 'unknown')
                }
            else:
                return {}
                
        except Exception as e:
            logging.warning(f"Could not get audio info: {e}")
            return {}


class AudioSplitter:
    """Handles splitting audio files into time-limited segments with sentence boundary awareness"""
    
    @staticmethod
    def find_silence_breaks(input_file: Path, min_silence_duration: float = 0.3, 
                           silence_threshold: float = -40) -> List[float]:
        """Find silence breaks in audio that likely correspond to sentence boundaries"""
        
        if not AudioConverter.check_ffmpeg_available():
            return []
        
        try:
            # Use FFmpeg to detect silence
            cmd = [
                'ffmpeg',
                '-i', str(input_file),
                '-af', f'silencedetect=noise={silence_threshold}dB:d={min_silence_duration}',
                '-f', 'null',
                '-'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Parse silence detection output
            silence_breaks = []
            
            for line in result.stderr.split('\n'):
                if 'silence_end:' in line:
                    # Extract the end time of silence (good split point)
                    try:
                        # Format: [silencedetect @ 0x...] silence_end: 12.345 | silence_duration: 0.678
                        parts = line.split('silence_end:')[1].split('|')[0].strip()
                        silence_end_time = float(parts)
                        silence_breaks.append(silence_end_time)
                    except (IndexError, ValueError):
                        continue
            
            logging.info(f"🔍 Found {len(silence_breaks)} potential sentence breaks in audio")
            return sorted(silence_breaks)
            
        except Exception as e:
            logging.warning(f"Could not detect silence breaks: {e}")
            return []
    
    @staticmethod
    def find_optimal_split_points(total_duration: float, target_duration: float, 
                                 silence_breaks: List[float], tolerance: float = 0.15) -> List[float]:
        """Find optimal split points that respect sentence boundaries while staying close to target duration"""
        
        if not silence_breaks:
            # Fallback to fixed intervals if no silence detection
            num_segments = math.ceil(total_duration / target_duration)
            return [i * target_duration for i in range(1, num_segments)]
        
        split_points = []
        current_target = target_duration
        max_tolerance = target_duration * tolerance  # Allow 15% deviation by default
        
        while current_target < total_duration:
            # Find the silence break closest to our target time
            best_break = None
            best_distance = float('inf')
            
            for break_time in silence_breaks:
                # Only consider breaks that haven't been used and are within tolerance
                if break_time <= current_target + max_tolerance and break_time >= current_target - max_tolerance:
                    distance = abs(break_time - current_target)
                    if distance < best_distance:
                        best_distance = distance
                        best_break = break_time
            
            if best_break is not None:
                split_points.append(best_break)
                # Next target is based on actual split point, not ideal interval
                current_target = best_break + target_duration
            else:
                # No good break found, use the target time (fallback)
                split_points.append(current_target)
                current_target += target_duration
        
        logging.info(f"📍 Selected {len(split_points)} optimal split points with sentence boundaries")
        return split_points
    
    @staticmethod
    def split_audio_by_time(input_file: Path, output_base_name: str, max_minutes: int = 5, 
                           mp3_enabled: bool = False, mp3_bitrate: str = "128k", 
                           remove_wav: bool = False, metadata: dict = None, 
                           smart_split: bool = True) -> List[Path]:
        """Split audio file into time-limited segments with optional sentence boundary awareness"""
        
        if not AudioConverter.check_ffmpeg_available():
            logging.error("❌ FFmpeg not found! Cannot split audio files")
            return []
        
        try:
            # Get audio duration first
            audio_info = AudioConverter.get_audio_info(input_file)
            total_duration = audio_info.get('duration', 0)
            
            if total_duration == 0:
                logging.error("❌ Could not determine audio duration")
                return []
            
            max_seconds = max_minutes * 60
            
            if smart_split:
                # Find silence breaks that correspond to sentence boundaries
                logging.info("🔍 Analyzing audio for sentence boundaries...")
                silence_breaks = AudioSplitter.find_silence_breaks(
                    input_file, 
                    min_silence_duration=0.3,  # 300ms minimum silence
                    silence_threshold=-35      # Adjust based on audio quality
                )
                
                # Find optimal split points that respect sentence boundaries
                split_points = AudioSplitter.find_optimal_split_points(
                    total_duration, 
                    max_seconds, 
                    silence_breaks,
                    tolerance=0.20  # Allow 20% deviation to find good breaks
                )
                
                if not split_points:
                    logging.warning("⚠️ No optimal split points found, using fixed intervals")
                    num_segments = math.ceil(total_duration / max_seconds)
                    split_points = [i * max_seconds for i in range(1, num_segments)]
                else:
                    logging.info(f"🎯 Using sentence-aware splitting with {len(split_points)} break points")
            else:
                # Use fixed time intervals (legacy behavior)
                logging.info("⏰ Using fixed time intervals (sentence boundaries ignored)")
                num_segments = math.ceil(total_duration / max_seconds)
                split_points = [i * max_seconds for i in range(1, num_segments)]
                silence_breaks = []  # No sentence boundary info
            
            logging.info(f"🔄 Splitting audio into {len(split_points) + 1} segments with sentence boundaries...")
            
            output_files = []
            prev_split = 0.0
            
            for i, split_point in enumerate(split_points + [total_duration]):
                segment_num = f"{i+1:03d}"
                segment_start = prev_split
                segment_duration = split_point - prev_split
                
                # Skip very short segments (less than 30 seconds)
                if segment_duration < 30 and i < len(split_points):
                    continue
                
                # Determine output format and filename
                if mp3_enabled:
                    output_file = input_file.parent / f"{output_base_name}_{segment_num}.mp3"
                    codec_args = ['-codec:a', 'libmp3lame', '-b:a', mp3_bitrate]
                else:
                    output_file = input_file.parent / f"{output_base_name}_{segment_num}.wav"
                    codec_args = ['-codec:a', 'pcm_s16le']
                
                # FFmpeg command to extract segment
                cmd = [
                    'ffmpeg',
                    '-i', str(input_file),
                    '-ss', str(segment_start),      # Start time (sentence boundary)
                    '-t', str(segment_duration),    # Duration (to next sentence boundary)
                    '-ar', '22050',                 # Sample rate
                    '-ac', '1',                     # Mono
                    *codec_args,                    # Codec specific args
                ]
                
                # Add metadata if MP3 and metadata provided
                if mp3_enabled and metadata:
                    if metadata.get('artist'):
                        cmd.extend(['-metadata', f"artist={metadata['artist']}"])
                    if metadata.get('album'):
                        cmd.extend(['-metadata', f"album={metadata['album']}"])
                    if metadata.get('title'):
                        # For segments, create title with track info
                        track_title = f"{metadata.get('title', 'Audiobook')} - Part {i+1}"
                        cmd.extend(['-metadata', f"title={track_title}"])
                    else:
                        cmd.extend(['-metadata', f"title=Part {i+1}"])
                    if metadata.get('track'):
                        cmd.extend(['-metadata', f"track={i+1}"])
                    else:
                        cmd.extend(['-metadata', f"track={i+1}"])
                    if metadata.get('genre'):
                        cmd.extend(['-metadata', f"genre={metadata['genre']}"])
                
                cmd.extend([
                    '-y',                        # Overwrite
                    str(output_file)
                ])
                
                # Show the actual time range for this segment
                start_min, start_sec = divmod(int(segment_start), 60)
                end_time = segment_start + segment_duration
                end_min, end_sec = divmod(int(end_time), 60)
                
                logging.info(f"📁 Creating segment {segment_num}: {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d} ({segment_duration:.1f}s)")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0 and output_file.exists():
                    # Verify the file has content
                    if output_file.stat().st_size > 1024:  # At least 1KB
                        output_files.append(output_file)
                        
                        # Get segment info
                        segment_info = AudioConverter.get_audio_info(output_file)
                        actual_duration = segment_info.get('duration', 0)
                        
                        # Check if this split was at a sentence boundary
                        is_sentence_boundary = split_point in silence_breaks if i < len(split_points) else True
                        boundary_indicator = "🎯" if is_sentence_boundary else "⏰"
                        
                        logging.info(f"✅ Segment {segment_num} created: {actual_duration:.1f}s, {output_file.stat().st_size//1024//1024}MB {boundary_indicator}")
                    else:
                        logging.warning(f"⚠️ Segment {segment_num} is too small, skipping")
                        if output_file.exists():
                            output_file.unlink()
                else:
                    logging.error(f"❌ Failed to create segment {segment_num}")
                    logging.error(f"   stderr: {result.stderr}")
                
                prev_split = split_point
            
            # Remove original file if requested and segments were created successfully
            if remove_wav and output_files and input_file.suffix.lower() == '.wav':
                try:
                    input_file.unlink()
                    logging.info(f"🗑️ Removed original file: {input_file.name}")
                except Exception as e:
                    logging.warning(f"Could not remove original file: {e}")
            
            logging.info(f"🎵 Audio splitting complete: {len(output_files)} segments created")
            
            return output_files
            
        except Exception as e:
            logging.error(f"❌ Audio splitting error: {e}")
            return []