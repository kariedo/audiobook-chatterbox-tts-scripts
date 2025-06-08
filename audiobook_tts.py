#!/usr/bin/env python3
"""
Audiobook TTS Generator
Converts text files and ebooks to audiobooks using Chatterbox TTS
Supports chunking, parallel processing, resume capability, voice cloning, MP3 conversion, and time-limited file splitting
"""

import argparse
import logging
import os
import sys
import time
import json
import re
import subprocess
import shutil
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple
import hashlib
import math

# Suppress known warnings from dependencies
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*LoRACompatibleLinear.*")
warnings.filterwarnings("ignore", message=".*torch.backends.cuda.sdp_kernel.*")
warnings.filterwarnings("ignore", message=".*LlamaSdpaAttention.*")
warnings.filterwarnings("ignore", message=".*past_key_values.*")
warnings.filterwarnings("ignore", message=".*attn_implementation.*")
warnings.filterwarnings("ignore", message=".*scaled_dot_product_attention.*")

# Set environment variables to suppress additional warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Setup logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def setup_mac_compatibility():
    """Setup Mac M4 compatibility for Chatterbox"""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    map_location = torch.device(device)
    
    torch_load_original = torch.load
    def patched_torch_load(*args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = map_location
        return torch_load_original(*args, **kwargs)
    
    torch.load = patched_torch_load
    return device

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

class TextProcessor:
    """Handles different text file formats"""
    
    @staticmethod
    def read_file(file_path: Path) -> str:
        """Read text from various file formats"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.fb2':
            return TextProcessor._read_fb2(file_path)
        elif suffix in ['.txt', '.md']:
            return TextProcessor._read_text(file_path)
        elif suffix == '.epub':
            return TextProcessor._read_epub(file_path)
        else:
            # Try as plain text
            return TextProcessor._read_text(file_path)
    
    @staticmethod
    def _read_text(file_path: Path) -> str:
        """Read plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['cp1251', 'latin1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            
            # Last resort: read as bytes and decode with errors='ignore'
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    
    @staticmethod
    def _read_fb2(file_path: Path) -> str:
        """Read FictionBook2 format"""
        try:
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # FB2 uses namespaces
            ns = {'fb': 'http://www.gribuser.ru/xml/fictionbook/2.0'}
            
            # Extract text from body
            body = root.find('.//fb:body', ns)
            if body is None:
                # Try without namespace
                body = root.find('.//body')
            
            if body is None:
                logging.warning("Could not find body in FB2 file, trying to extract all text")
                return ET.tostring(root, encoding='unicode', method='text')
            
            # Extract text content
            text_parts = []
            for elem in body.iter():
                if elem.text:
                    text_parts.append(elem.text.strip())
                if elem.tail:
                    text_parts.append(elem.tail.strip())
            
            return '\n'.join(filter(None, text_parts))
            
        except Exception as e:
            logging.warning(f"Error parsing FB2 file: {e}, trying as plain text")
            return TextProcessor._read_text(file_path)
    
    @staticmethod
    def _read_epub(file_path: Path) -> str:
        """Read EPUB format with advanced text cleaning"""
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
            import re
            
            book = epub.read_epub(str(file_path))
            text_parts = []
            
            # Get book metadata for filtering
            title = book.get_metadata('DC', 'title')
            title_text = title[0][0] if title else ""
            
            logging.info(f"📖 Processing EPUB: {title_text}")
            
            chapter_count = 0
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    # Skip navigation and TOC files
                    item_name = item.get_name().lower()
                    if any(skip in item_name for skip in ['nav', 'toc', 'contents', 'index', 'cover']):
                        logging.info(f"⏭️ Skipping navigation file: {item_name}")
                        continue
                    
                    try:
                        content = item.get_content()
                        soup = BeautifulSoup(content, 'html.parser')
                        
                        # Remove script and style elements
                        for script in soup(["script", "style", "meta", "link"]):
                            script.decompose()
                        
                        # Remove navigation elements
                        for nav in soup.find_all(['nav', 'aside'], class_=re.compile(r'nav|toc|sidebar', re.I)):
                            nav.decompose()
                        
                        # Remove elements by common navigation IDs/classes
                        for element in soup.find_all(attrs={'class': re.compile(r'nav|toc|menu|header|footer|sidebar', re.I)}):
                            element.decompose()
                        
                        for element in soup.find_all(attrs={'id': re.compile(r'nav|toc|menu|header|footer', re.I)}):
                            element.decompose()
                        
                        # Extract clean text
                        text = soup.get_text(separator=' ', strip=True)
                        
                        if text and len(text.strip()) > 50:  # Only include substantial content
                            # Clean the extracted text
                            cleaned_text = TextProcessor._clean_epub_text(text)
                            
                            if cleaned_text and len(cleaned_text.strip()) > 50:
                                text_parts.append(cleaned_text)
                                chapter_count += 1
                                logging.info(f"✅ Extracted chapter {chapter_count}: {len(cleaned_text)} chars")
                    
                    except Exception as e:
                        logging.warning(f"Error processing EPUB item {item_name}: {e}")
                        continue
            
            if not text_parts:
                logging.warning("No content extracted from EPUB, trying alternative method")
                return TextProcessor._read_epub_fallback(file_path)
            
            final_text = '\n\n'.join(text_parts)
            logging.info(f"📚 EPUB processing complete: {len(final_text)} characters from {chapter_count} chapters")
            
            return final_text
            
        except ImportError:
            logging.warning("ebooklib not installed, install with: pip install ebooklib beautifulsoup4")
            return TextProcessor._read_text(file_path)
        except Exception as e:
            logging.warning(f"Error parsing EPUB file: {e}, trying fallback method")
            return TextProcessor._read_epub_fallback(file_path)
    
    @staticmethod
    def _clean_epub_text(text: str) -> str:
        """Clean text extracted from EPUB"""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common ebook artifacts
        text = re.sub(r'\*\*\*+', '', text)  # Star separators
        text = re.sub(r'---+', '', text)     # Dash separators
        text = re.sub(r'___+', '', text)     # Underscore separators
        
        # Remove page numbers and chapter markers that might have leaked through
        text = re.sub(r'\bPage \d+\b', '', text, flags=re.I)
        text = re.sub(r'\bChapter \d+\b(?!\w)', '', text, flags=re.I)
        
        # Remove common navigation text
        navigation_patterns = [
            r'\bTable of Contents\b',
            r'\bNext Chapter\b',
            r'\bPrevious Chapter\b',
            r'\bBack to Top\b',
            r'\bReturn to\b.*?(?=\.|$)',
            r'\bCopyright.*?(?=\.|$)',
            r'\bISBN.*?(?=\.|$)',
            r'\bPublished by.*?(?=\.|$)',
        ]
        
        for pattern in navigation_patterns:
            text = re.sub(pattern, '', text, flags=re.I)
        
        # Remove standalone numbers (likely page numbers)
        text = re.sub(r'\b\d+\b(?=\s|$)', '', text)
        
        # Clean up punctuation spacing
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Remove empty lines and excessive spacing
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text.strip()
    
    @staticmethod
    def _read_epub_fallback(file_path: Path) -> str:
        """Fallback EPUB reader using zipfile"""
        try:
            import zipfile
            import xml.etree.ElementTree as ET
            from bs4 import BeautifulSoup
            
            text_parts = []
            
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                # Look for content files
                for file_name in zip_file.namelist():
                    if file_name.endswith(('.xhtml', '.html', '.htm')) and 'nav' not in file_name.lower():
                        try:
                            content = zip_file.read(file_name).decode('utf-8', errors='ignore')
                            soup = BeautifulSoup(content, 'html.parser')
                            
                            # Remove unwanted elements
                            for unwanted in soup(['script', 'style', 'nav', 'header', 'footer']):
                                unwanted.decompose()
                            
                            text = soup.get_text(separator=' ', strip=True)
                            if text and len(text.strip()) > 100:
                                cleaned = TextProcessor._clean_epub_text(text)
                                if cleaned:
                                    text_parts.append(cleaned)
                        
                        except Exception as e:
                            logging.warning(f"Error reading {file_name}: {e}")
                            continue
            
            return '\n\n'.join(text_parts) if text_parts else ""
            
        except Exception as e:
            logging.error(f"Fallback EPUB reading failed: {e}")
            return TextProcessor._read_text(file_path)

class TextChunker:
    """Intelligently chunks text for optimal TTS processing with sentence boundary respect"""
    
    def __init__(self, max_chars: int = 200, min_chars: int = 50):
        self.max_chars = max_chars
        self.min_chars = min_chars
        
        # Common abbreviations that don't end sentences
        self.abbreviations = {
            'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'vs', 'etc', 'inc', 'ltd', 'corp',
            'co', 'st', 'ave', 'blvd', 'rd', 'apt', 'no', 'vol', 'ch', 'sec', 'fig', 'pg',
            'pp', 'ed', 'eds', 'rev', 'repr', 'trans', 'cf', 'e.g', 'i.e', 'viz', 'al',
            'govt', 'dept', 'univ', 'assn', 'bros', 'ph.d', 'm.d', 'b.a', 'm.a', 'j.d',
            'u.s', 'u.k', 'u.s.a', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep',
            'oct', 'nov', 'dec', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'
        }
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into optimal chunks respecting sentence boundaries"""
        
        # Clean up text
        text = self._clean_text(text)
        
        # Split into paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Get sentences from this paragraph
            sentences = self._smart_sentence_split(paragraph)
            
            if not sentences:
                continue
            
            # Group sentences into chunks
            paragraph_chunks = self._group_sentences_into_chunks(sentences)
            chunks.extend(paragraph_chunks)
        
        # Validate and clean up chunks
        final_chunks = self._validate_chunks(chunks)
        
        return final_chunks
    
    def _smart_sentence_split(self, text: str) -> List[str]:
        """Advanced sentence splitting that handles abbreviations and edge cases"""
        
        # First pass: identify potential sentence boundaries
        potential_breaks = []
        i = 0
        
        while i < len(text):
            char = text[i]
            
            # Look for sentence-ending punctuation
            if char in '.!?':
                # Check if this is really a sentence end
                if self._is_sentence_boundary(text, i):
                    potential_breaks.append(i)
            
            i += 1
        
        # Split text at confirmed sentence boundaries
        sentences = []
        start = 0
        
        for break_pos in potential_breaks:
            # Include the punctuation in the sentence
            sentence = text[start:break_pos + 1].strip()
            if sentence:
                sentences.append(sentence)
            start = break_pos + 1
        
        # Don't forget the last sentence if it doesn't end with punctuation
        if start < len(text):
            last_sentence = text[start:].strip()
            if last_sentence:
                # Add period if missing sentence-ending punctuation
                if not last_sentence[-1] in '.!?':
                    last_sentence += '.'
                sentences.append(last_sentence)
        
        return sentences
    
    def _is_sentence_boundary(self, text: str, pos: int) -> bool:
        """Determine if punctuation at position marks a real sentence boundary"""
        
        if pos >= len(text) - 1:
            return True  # End of text
        
        char = text[pos]
        
        # Handle multiple punctuation (e.g., "...")
        if pos < len(text) - 1 and text[pos + 1] in '.!?':
            return False  # Part of multi-character punctuation
        
        # Look at the word before the punctuation
        word_start = pos - 1
        while word_start >= 0 and text[word_start].isalpha():
            word_start -= 1
        word_start += 1
        
        if word_start < pos:
            word_before = text[word_start:pos].lower()
            
            # Check if it's a known abbreviation
            if word_before in self.abbreviations:
                return False
            
            # Check for single letter abbreviations (e.g., "A. Smith")
            if len(word_before) == 1 and char == '.':
                # Look ahead to see if next word is capitalized (likely a name)
                next_word_start = pos + 1
                while next_word_start < len(text) and text[next_word_start].isspace():
                    next_word_start += 1
                
                if next_word_start < len(text) and text[next_word_start].isupper():
                    return False  # Likely an initial
        
        # Look at what comes after the punctuation
        next_char_pos = pos + 1
        while next_char_pos < len(text) and text[next_char_pos].isspace():
            next_char_pos += 1
        
        if next_char_pos >= len(text):
            return True  # End of text
        
        next_char = text[next_char_pos]
        
        # Sentence boundary if next character is uppercase or quote
        if next_char.isupper() or next_char in '"\'':
            return True
        
        # Check for numbers (e.g., "version 2.0 was released")
        if char == '.' and next_char.isdigit():
            return False
        
        # Default to not a sentence boundary for lowercase continuation
        return False
    
    def _group_sentences_into_chunks(self, sentences: List[str]) -> List[str]:
        """Group sentences into chunks respecting size limits"""
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Calculate the size if we add this sentence
            if current_chunk:
                test_chunk = current_chunk + " " + sentence
            else:
                test_chunk = sentence
            
            # If adding this sentence would exceed max_chars
            if len(test_chunk) > self.max_chars:
                # Save current chunk if it has content
                if current_chunk and len(current_chunk.strip()) >= self.min_chars:
                    chunks.append(current_chunk.strip())
                
                # Handle very long sentences that exceed max_chars by themselves
                if len(sentence) > self.max_chars:
                    # Split long sentence at clause boundaries
                    clause_chunks = self._split_long_sentence(sentence)
                    chunks.extend(clause_chunks)
                    current_chunk = ""
                else:
                    current_chunk = sentence
            else:
                current_chunk = test_chunk
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk.strip()) >= self.min_chars:
            chunks.append(current_chunk.strip())
        elif current_chunk.strip():
            # Very short final chunk - try to merge with previous
            if chunks:
                last_chunk = chunks[-1] + " " + current_chunk.strip()
                if len(last_chunk) <= self.max_chars * 1.1:  # Allow 10% overflow for merging
                    chunks[-1] = last_chunk
                else:
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split an overly long sentence at natural break points"""
        
        # Try to split at clause boundaries (commas, semicolons, conjunctions)
        clause_patterns = [
            r',\s+(?=and\s+)',     # ", and "
            r',\s+(?=but\s+)',     # ", but "
            r',\s+(?=or\s+)',      # ", or "
            r',\s+(?=so\s+)',      # ", so "
            r',\s+(?=yet\s+)',     # ", yet "
            r';\s+',               # "; "
            r',\s+(?=which\s+)',   # ", which "
            r',\s+(?=that\s+)',    # ", that "
            r',\s+(?=when\s+)',    # ", when "
            r',\s+(?=where\s+)',   # ", where "
            r',\s+(?=while\s+)',   # ", while "
            r',\s+'                # Any other comma (last resort)
        ]
        
        chunks = []
        remaining = sentence
        
        while len(remaining) > self.max_chars and remaining:
            best_split = -1
            best_pattern = None
            
            # Find the best split point within our character limit
            for pattern in clause_patterns:
                matches = list(re.finditer(pattern, remaining))
                for match in matches:
                    split_pos = match.end()
                    if split_pos <= self.max_chars:
                        best_split = split_pos
                        best_pattern = pattern
                    else:
                        break  # No more valid splits with this pattern
                
                if best_split > 0:
                    break
            
            if best_split > 0:
                # Split at the best position
                chunk = remaining[:best_split].strip()
                remaining = remaining[best_split:].strip()
                
                if chunk:
                    chunks.append(chunk)
            else:
                # No good split found, force split at word boundary
                words = remaining.split()
                chunk_words = []
                chunk_length = 0
                
                for word in words:
                    test_length = chunk_length + len(word) + (1 if chunk_words else 0)
                    if test_length <= self.max_chars:
                        chunk_words.append(word)
                        chunk_length = test_length
                    else:
                        break
                
                if chunk_words:
                    chunk = ' '.join(chunk_words)
                    chunks.append(chunk)
                    remaining = ' '.join(words[len(chunk_words):])
                else:
                    # Single word longer than max_chars, just take it
                    chunks.append(remaining)
                    remaining = ""
        
        # Add remaining text
        if remaining.strip():
            chunks.append(remaining.strip())
        
        return chunks
    
    def _validate_chunks(self, chunks: List[str]) -> List[str]:
        """Validate and clean up final chunks"""
        
        validated_chunks = []
        
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            
            # Ensure chunk doesn't start with punctuation or lowercase (except quotes)
            if chunk and chunk[0].islower() and chunk[0] not in '"\'':
                # Try to fix by capitalizing
                chunk = chunk[0].upper() + chunk[1:]
            
            # Ensure chunk ends with sentence-ending punctuation
            if chunk and chunk[-1] not in '.!?':
                chunk += '.'
            
            # Remove chunks that are too short or just punctuation
            if len(chunk.strip('.,!?;"\'')) >= 3:  # At least 3 non-punctuation characters
                validated_chunks.append(chunk)
        
        return validated_chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Normalize quotes first
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Clean up common ebook artifacts
        text = re.sub(r'\*\*\*+', '', text)  # Stars
        text = re.sub(r'---+', '', text)     # Dashes
        
        # Remove excessive whitespace but preserve single spaces
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double newlines
        text = re.sub(r'[\r\f\v]+', ' ', text)  # Other whitespace to space
        
        # Fix spacing around punctuation (but preserve abbreviations)
        text = re.sub(r'\s+([!?,:;])', r'\1', text)  # Remove space before punctuation (but not periods)
        # Only add space after sentence end if it's followed by a letter (avoid abbreviations like U.S.A.)
        text = re.sub(r'([.!?])([A-Z][a-z])', r'\1 \2', text)  # Ensure space after sentence end only for new sentences
        
        return text.strip()

class ProgressTracker:
    """Tracks processing progress and allows resuming"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.progress_file = output_dir / "progress.json"
        self.progress = self._load_progress()
    
    def _load_progress(self) -> dict:
        """Load existing progress"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Could not load progress file: {e}")
        
        return {
            "total_chunks": 0,
            "completed_chunks": [],
            "failed_chunks": [],
            "start_time": None,
            "last_update": None,
            "session_start_time": None,
            "chunk_times": []  # Store completion times for rate calculation
        }
    
    def save_progress(self):
        """Save current progress"""
        self.progress["last_update"] = datetime.now().isoformat()
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            logging.error(f"Could not save progress: {e}")
    
    def mark_chunk_completed(self, chunk_index: int):
        """Mark a chunk as completed"""
        if chunk_index not in self.progress["completed_chunks"]:
            self.progress["completed_chunks"].append(chunk_index)
            
            # Store completion time for rate calculation
            completion_time = datetime.now().isoformat()
            if "chunk_times" not in self.progress:
                self.progress["chunk_times"] = []
            self.progress["chunk_times"].append({
                "chunk": chunk_index,
                "time": completion_time
            })
            
            # Keep only last 50 completion times for rate calculation
            if len(self.progress["chunk_times"]) > 50:
                self.progress["chunk_times"] = self.progress["chunk_times"][-50:]
        
        # Remove from failed if it was there
        if chunk_index in self.progress["failed_chunks"]:
            self.progress["failed_chunks"].remove(chunk_index)
        
        self.save_progress()
    
    def mark_chunk_failed(self, chunk_index: int):
        """Mark a chunk as failed"""
        if chunk_index not in self.progress["failed_chunks"]:
            self.progress["failed_chunks"].append(chunk_index)
        self.save_progress()
    
    def is_chunk_completed(self, chunk_index: int) -> bool:
        """Check if chunk is already completed"""
        return chunk_index in self.progress["completed_chunks"]
    
    def get_completion_stats(self) -> Tuple[int, int, int]:
        """Get completion statistics"""
        total = self.progress["total_chunks"]
        completed = len(self.progress["completed_chunks"])
        failed = len(self.progress["failed_chunks"])
        return completed, failed, total
    
    def calculate_eta(self) -> Tuple[str, float]:
        """Calculate estimated time to completion"""
        completed = len(self.progress["completed_chunks"])
        total = self.progress["total_chunks"]
        
        if completed == 0:
            return "Calculating...", 0.0
        
        # Use the last few chunks to calculate rate
        if "chunk_times" not in self.progress or len(self.progress["chunk_times"]) < 3:
            return "Calculating...", 0.0
        
        # Get the last 5 chunks to get a more stable rate
        recent_chunks = self.progress["chunk_times"][-5:]
        
        if len(recent_chunks) < 3:
            return "Calculating...", 0.0
        
        # Calculate average time between chunks, filtering out large gaps
        intervals = []
        for i in range(1, len(recent_chunks)):
            prev_time = datetime.fromisoformat(recent_chunks[i-1]["time"])
            curr_time = datetime.fromisoformat(recent_chunks[i]["time"])
            interval = (curr_time - prev_time).total_seconds()
            
            # Only include intervals that seem reasonable (less than 2 minutes)
            # This filters out gaps from session restarts
            if interval < 120:  # 2 minutes max
                intervals.append(interval)
        
        if len(intervals) == 0:
            return "Calculating...", 0.0
        
        # Calculate average time per chunk from valid intervals
        avg_time_per_chunk = sum(intervals) / len(intervals)
        chunks_per_second = 1.0 / avg_time_per_chunk if avg_time_per_chunk > 0 else 0.0
        
        if chunks_per_second <= 0:
            return "Calculating...", 0.0
        
        # Calculate remaining time
        remaining_chunks = total - completed
        remaining_seconds = remaining_chunks * avg_time_per_chunk
        
        # Format time
        if remaining_seconds < 60:
            eta_str = f"{remaining_seconds:.0f}s"
        elif remaining_seconds < 3600:
            minutes = int(remaining_seconds // 60)
            seconds = int(remaining_seconds % 60)
            eta_str = f"{minutes}m {seconds}s"
        else:
            hours = int(remaining_seconds // 3600)
            minutes = int((remaining_seconds % 3600) // 60)
            eta_str = f"{hours}h {minutes}m"
        
        return eta_str, chunks_per_second
    
    def set_session_start(self):
        """Mark the start of the current processing session"""
        if not self.progress.get("session_start_time"):
            self.progress["session_start_time"] = datetime.now().isoformat()
            self.save_progress()

class AudioValidator:
    """Validates generated audio chunks for quality issues"""
    
    @staticmethod
    def validate_audio_chunk(audio_file: Path, chunk_text: str, expected_duration_range: Tuple[float, float] = (0.5, 30.0)) -> dict:
        """
        Validate an audio chunk for common issues
        
        Returns dict with validation results:
        - is_valid: bool
        - issues: list of detected problems
        - metrics: dict with audio metrics
        """
        validation_result = {
            "is_valid": True,
            "issues": [],
            "metrics": {}
        }
        
        try:
            if not audio_file.exists():
                validation_result["is_valid"] = False
                validation_result["issues"].append("File does not exist")
                return validation_result
            
            # Check file size
            file_size = audio_file.stat().st_size
            if file_size < 1024:  # Less than 1KB
                validation_result["is_valid"] = False
                validation_result["issues"].append(f"File too small: {file_size} bytes")
                return validation_result
            
            # Load and analyze audio
            try:
                waveform, sample_rate = ta.load(str(audio_file))
                validation_result["metrics"]["sample_rate"] = sample_rate
                validation_result["metrics"]["channels"] = waveform.shape[0]
                validation_result["metrics"]["samples"] = waveform.shape[1]
                
                # Calculate duration
                duration = waveform.shape[1] / sample_rate
                validation_result["metrics"]["duration"] = duration
                
                # Check duration is reasonable
                if duration < expected_duration_range[0]:
                    validation_result["issues"].append(f"Audio too short: {duration:.2f}s (expected min {expected_duration_range[0]}s)")
                elif duration > expected_duration_range[1]:
                    validation_result["issues"].append(f"Audio too long: {duration:.2f}s (expected max {expected_duration_range[1]}s)")
                
                # Check for silence (all samples near zero)
                audio_rms = torch.sqrt(torch.mean(waveform ** 2))
                validation_result["metrics"]["rms_level"] = float(audio_rms)
                
                if audio_rms < 0.001:  # Very quiet audio
                    validation_result["issues"].append(f"Audio appears silent: RMS {audio_rms:.6f}")
                
                # Check for clipping (samples at max/min values)
                max_amplitude = torch.max(torch.abs(waveform))
                validation_result["metrics"]["max_amplitude"] = float(max_amplitude)
                
                if max_amplitude > 0.95:  # Near clipping
                    validation_result["issues"].append(f"Audio may be clipped: max amplitude {max_amplitude:.3f}")
                
                # Check for DC offset
                dc_offset = torch.mean(waveform)
                validation_result["metrics"]["dc_offset"] = float(dc_offset)
                
                if abs(dc_offset) > 0.1:
                    validation_result["issues"].append(f"Significant DC offset: {dc_offset:.3f}")
                
                # Check for excessive silence at start/end
                silence_threshold = 0.01
                start_silence, end_silence = AudioValidator._detect_silence_boundaries(waveform, silence_threshold)
                validation_result["metrics"]["start_silence"] = start_silence / sample_rate
                validation_result["metrics"]["end_silence"] = end_silence / sample_rate
                
                # Warn if more than 1 second of silence at start/end
                if start_silence / sample_rate > 1.0:
                    validation_result["issues"].append(f"Excessive silence at start: {start_silence/sample_rate:.2f}s")
                if end_silence / sample_rate > 1.0:
                    validation_result["issues"].append(f"Excessive silence at end: {end_silence/sample_rate:.2f}s")
                
                # Check for reasonable correlation with text length
                text_length = len(chunk_text.strip())
                words = len(chunk_text.split())
                chars_per_second = text_length / duration if duration > 0 else 0
                words_per_minute = (words * 60) / duration if duration > 0 else 0
                
                validation_result["metrics"]["text_length"] = text_length
                validation_result["metrics"]["word_count"] = words
                validation_result["metrics"]["chars_per_second"] = chars_per_second
                validation_result["metrics"]["words_per_minute"] = words_per_minute
                
                # Typical reading speeds: 150-250 WPM, 8-15 chars/second
                if words_per_minute > 300:
                    validation_result["issues"].append(f"Speech too fast: {words_per_minute:.0f} WPM (expected < 300)")
                elif words_per_minute < 50 and words > 5:  # Only flag if substantial text
                    validation_result["issues"].append(f"Speech too slow: {words_per_minute:.0f} WPM (expected > 50)")
                
                # Check for audio artifacts (sudden volume spikes)
                if len(waveform[0]) > sample_rate:  # Only for audio longer than 1 second
                    volume_variance = AudioValidator._detect_volume_spikes(waveform)
                    validation_result["metrics"]["volume_variance"] = volume_variance
                    
                    if volume_variance > 10.0:  # High variance indicates possible artifacts
                        validation_result["issues"].append(f"High volume variance detected: {volume_variance:.2f} (possible artifacts)")
                
            except Exception as e:
                validation_result["is_valid"] = False
                validation_result["issues"].append(f"Audio loading error: {str(e)}")
                return validation_result
            
            # Set overall validity
            if validation_result["issues"]:
                validation_result["is_valid"] = False
                
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["issues"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    @staticmethod
    def _detect_silence_boundaries(waveform: torch.Tensor, threshold: float = 0.01) -> Tuple[int, int]:
        """Detect silence at the start and end of audio"""
        audio = waveform[0] if waveform.dim() > 1 else waveform
        abs_audio = torch.abs(audio)
        
        # Find start of audio (first non-silent sample)
        start_silence = 0
        for i, sample in enumerate(abs_audio):
            if sample > threshold:
                start_silence = i
                break
        else:
            start_silence = len(abs_audio)  # All silence
        
        # Find end of audio (last non-silent sample)
        end_silence = 0
        for i, sample in enumerate(reversed(abs_audio)):
            if sample > threshold:
                end_silence = i
                break
        else:
            end_silence = len(abs_audio)  # All silence
        
        return start_silence, end_silence
    
    @staticmethod
    def _detect_volume_spikes(waveform: torch.Tensor, window_size: int = 2048) -> float:
        """Detect sudden volume changes that might indicate artifacts"""
        audio = waveform[0] if waveform.dim() > 1 else waveform
        
        # Calculate RMS for overlapping windows
        rms_values = []
        for i in range(0, len(audio) - window_size, window_size // 2):
            window = audio[i:i + window_size]
            rms = torch.sqrt(torch.mean(window ** 2))
            rms_values.append(float(rms))
        
        if len(rms_values) < 2:
            return 0.0
        
        # Calculate variance in RMS levels (higher = more spiky)
        rms_tensor = torch.tensor(rms_values)
        variance = torch.var(rms_tensor)
        
        return float(variance)


class EndingDetector:
    """Detects problematic TTS endings where RMS energy rises and stays elevated"""
    
    def __init__(self):
        self.sample_rate = 24000
        
    def analyze_ending_pattern(self, audio_file: Path, analysis_duration: float = 5.0) -> dict:
        """Analyze the ending pattern of an audio file"""
        try:
            audio, sr = librosa.load(str(audio_file), sr=self.sample_rate)
            duration = len(audio) / sr
            
            if duration < analysis_duration:
                return {'is_problematic': False, 'reason': f'Audio too short ({duration:.2f}s)'}
                
            # Extract the last N seconds
            start_sample = len(audio) - int(analysis_duration * sr)
            ending_segment = audio[start_sample:]
            
            # Calculate RMS energy in small windows
            window_size = int(sr * 0.1)  # 100ms windows
            hop_size = int(sr * 0.05)    # 50ms hop
            
            rms_values = []
            for i in range(0, len(ending_segment) - window_size, hop_size):
                window = ending_segment[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                rms_values.append(rms)
            
            rms_values = np.array(rms_values)
            if len(rms_values) < 10:
                return {'is_problematic': False, 'reason': 'Insufficient data for analysis'}
                
            # Smooth the RMS values
            rms_smooth = gaussian_filter1d(rms_values, sigma=2)
            
            # Find the minimum RMS value and its position
            min_idx = np.argmin(rms_smooth)
            min_rms = rms_smooth[min_idx]
            
            # Analyze what happens after the minimum
            remaining_frames = len(rms_smooth) - min_idx
            if remaining_frames < 8:
                return {'is_problematic': False, 'reason': 'Not enough data after minimum'}
                
            post_min_values = rms_smooth[min_idx:]
            
            # Focus on final 25% for true ending trend
            final_25_pct = int(len(post_min_values) * 0.75)
            if final_25_pct < len(post_min_values) and len(post_min_values) - final_25_pct >= 3:
                final_values = post_min_values[final_25_pct:]
                final_indices = np.arange(len(final_values))
                final_slope = np.polyfit(final_indices, final_values, 1)[0]
            else:
                final_slope = 0
                
            # Check if the very end (last 10%) is higher than mid-point
            last_10_pct_idx = int(len(post_min_values) * 0.9)
            if last_10_pct_idx < len(post_min_values):
                mid_point_rms = np.mean(post_min_values[len(post_min_values)//2:int(len(post_min_values)*0.8)])
                end_10_pct_rms = np.mean(post_min_values[last_10_pct_idx:])
                end_vs_mid_ratio = end_10_pct_rms / (mid_point_rms + 1e-10)
            else:
                end_vs_mid_ratio = 1.0
                
            # Compare first quarter vs last quarter after minimum
            quarter_len = len(post_min_values) // 4
            if quarter_len > 1:
                first_quarter = np.mean(post_min_values[:quarter_len])
                last_quarter = np.mean(post_min_values[-quarter_len:])
                quarter_ratio = last_quarter / (first_quarter + 1e-10)
            else:
                quarter_ratio = 1.0
                
            # Look at the final value compared to minimum
            end_rms = rms_smooth[-1]
            end_recovery_ratio = end_rms / (min_rms + 1e-10)
            
            # DETECTION LOGIC: Final portion has upward trend AND end stays elevated
            is_problematic = False
            reason = 'normal_fade'
            
            if (final_slope > min_rms * 0.1 and end_vs_mid_ratio > 2.0):
                if (end_recovery_ratio > 5.0 and quarter_ratio > 3.0):
                    is_problematic = True
                    reason = f'Rising trend detected: final_slope={final_slope:.6f}, end/mid_ratio={end_vs_mid_ratio:.2f}, recovery={end_recovery_ratio:.2f}x'
            
            return {
                'is_problematic': is_problematic,
                'reason': reason,
                'final_slope': final_slope,
                'end_vs_mid_ratio': end_vs_mid_ratio,
                'end_recovery_ratio': end_recovery_ratio,
                'quarter_ratio': quarter_ratio
            }
            
        except Exception as e:
            return {'is_problematic': False, 'reason': f'Analysis error: {e}'}


class AudiobookTTS:
    """Main TTS audiobook generator"""
    
    def __init__(self, voice_file: Optional[str] = None, max_workers: int = 2, 
                 exaggeration: float = 0.8, cfg_weight: float = 0.8, pitch_shift: float = 0.0,
                 mp3_bitrate: str = "128k", mp3_enabled: bool = False, remove_wav: bool = False,
                 split_minutes: int = 5, memory_cleanup_interval: int = 5, debug_memory: bool = False,
                 metadata: dict = None, smart_split: bool = True):
        self.device = setup_mac_compatibility()
        logging.info(f"Initializing Chatterbox TTS on {self.device}...")
        
        # Set MPS memory allocation strategy for better memory management
        if self.device == "mps":
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            # Force immediate cleanup
            torch.mps.empty_cache()
        
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.voice_file = voice_file
        # Reduce default workers for MPS to prevent OOM
        self.max_workers = 1 if self.device == "mps" else max_workers
        
        # Voice characteristics settings
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.pitch_shift = pitch_shift  # Semitones to shift pitch
        
        # MP3 conversion settings
        self.mp3_enabled = mp3_enabled
        self.mp3_bitrate = mp3_bitrate
        self.remove_wav = remove_wav
        
        # File splitting settings
        self.split_minutes = split_minutes
        self.smart_split = smart_split
        
        # MP3 metadata settings
        self.metadata = metadata or {}
        
        # Check FFmpeg availability if MP3 is enabled or splitting is requested
        if self.mp3_enabled or self.split_minutes > 0:
            if not AudioConverter.check_ffmpeg_available():
                if self.mp3_enabled:
                    logging.warning("⚠️ FFmpeg not found - MP3 conversion will be disabled")
                    self.mp3_enabled = False
                if self.split_minutes > 0:
                    logging.warning("⚠️ FFmpeg not found - File splitting will be disabled")
                    self.split_minutes = 0
            else:
                if self.mp3_enabled:
                    logging.info(f"🎵 MP3 conversion enabled (bitrate: {mp3_bitrate})")
                if self.split_minutes > 0:
                    logging.info(f"✂️ File splitting enabled (max {split_minutes} minutes per file)")
        
        # Initialize memory tracking and performance monitoring
        self.chunks_since_cleanup = 0
        self.cleanup_interval = memory_cleanup_interval  # Clear cache every N chunks
        self.performance_history = []  # Track processing times to detect slowdown
        self.adaptive_cleanup = True  # Enable adaptive cleanup based on performance
        self.severe_degradation_count = 0  # Count severe performance issues
        self.model_reinit_threshold = 2  # Reinitialize model after N severe degradations (lowered for faster recovery)
        self.debug_memory = debug_memory  # Enable detailed memory debugging
        
        # Warmup
        with torch.no_grad():
            _ = self.model.generate("Warmup", exaggeration=self.exaggeration, cfg_weight=self.cfg_weight)
        
        # Initial cleanup after warmup
        self._force_memory_cleanup()
        
        # Initialize ending detector for quality control
        self.ending_detector = EndingDetector()
        self.regenerated_chunks_count = 0
        
        logging.info(f"✅ TTS model ready for audiobook generation")
        logging.info(f"   Voice settings: exag={self.exaggeration}, cfg={self.cfg_weight}")
        if self.pitch_shift != 0:
            logging.info(f"   Pitch shift: {self.pitch_shift:+.1f} semitones")
        logging.info(f"   Memory cleanup interval: every {self.cleanup_interval} chunks")
        logging.info(f"   Ending detection enabled for quality control")
    
    def apply_pitch_shift(self, wav: torch.Tensor) -> torch.Tensor:
        """Apply pitch shifting to generated audio"""
        if self.pitch_shift == 0:
            return wav
        
        try:
            # Calculate pitch ratio
            pitch_ratio = 2 ** (self.pitch_shift / 12.0)
            original_sr = self.model.sr
            new_sr = int(original_sr * pitch_ratio)
            
            # Apply pitch shift via resampling
            resampler = ta.transforms.Resample(original_sr, new_sr)
            shifted_wav = resampler(wav)
            
            # Resample back to maintain playback speed
            resampler_back = ta.transforms.Resample(new_sr, original_sr)
            final_wav = resampler_back(shifted_wav)
            
            return final_wav
            
        except Exception as e:
            logging.warning(f"Pitch shift failed: {e}, using original audio")
            return wav
    
    def _get_memory_usage(self) -> dict:
        """Get current memory usage statistics"""
        stats = {}
        
        try:
            # Get basic system memory info
            import psutil
            system_memory = psutil.virtual_memory()
            stats['system_used_mb'] = system_memory.used / 1024 / 1024
            stats['system_available_mb'] = system_memory.available / 1024 / 1024
            stats['system_percent'] = system_memory.percent
            
            # Get current process memory
            process = psutil.Process()
            memory_info = process.memory_info()
            stats['rss_mb'] = memory_info.rss / 1024 / 1024
            stats['vms_mb'] = memory_info.vms / 1024 / 1024
            
        except Exception as e:
            # Fallback to basic resource tracking
            import resource
            try:
                usage = resource.getrusage(resource.RUSAGE_SELF)
                stats['rss_mb'] = usage.ru_maxrss / 1024  # macOS reports in bytes
                stats['system_percent'] = 0.0  # Can't determine without psutil
            except:
                stats['rss_mb'] = 0
                stats['system_percent'] = 0.0
        
        try:
            # Add GPU memory if available (this is most important)
            if self.device == "mps":
                # MPS memory tracking
                if hasattr(torch.mps, 'current_allocated_memory'):
                    stats['gpu_allocated_mb'] = torch.mps.current_allocated_memory() / 1024 / 1024
                else:
                    stats['gpu_allocated_mb'] = 0
                    
                if hasattr(torch.mps, 'max_memory_allocated'):
                    stats['gpu_max_allocated_mb'] = torch.mps.max_memory_allocated() / 1024 / 1024
                else:
                    stats['gpu_max_allocated_mb'] = 0
                    
            elif self.device == "cuda":
                stats['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
                stats['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
                stats['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
                stats['gpu_max_reserved_mb'] = torch.cuda.max_memory_reserved() / 1024 / 1024
            else:
                stats['gpu_allocated_mb'] = 0
                
        except Exception as e:
            stats['gpu_allocated_mb'] = 0
            stats['gpu_max_allocated_mb'] = 0
        
        try:
            # Add torch object counts for debugging
            import gc
            torch_objects = [obj for obj in gc.get_objects() if torch.is_tensor(obj)]
            stats['torch_tensor_count'] = len(torch_objects)
            
            # Calculate total tensor memory (simplified)
            tensor_memory_mb = 0
            for obj in torch_objects[:100]:  # Only check first 100 to avoid performance issues
                try:
                    if hasattr(obj, 'numel') and hasattr(obj, 'element_size'):
                        tensor_memory_mb += obj.numel() * obj.element_size() / 1024 / 1024
                except:
                    pass
            stats['torch_tensor_memory_mb'] = tensor_memory_mb
            
        except Exception as e:
            stats['torch_tensor_count'] = 0
            stats['torch_tensor_memory_mb'] = 0
            
        return stats
    
    def _calculate_performance_metrics(self, generation_time: float, text_length: int, word_count: int, text_content: str = "") -> dict:
        """Calculate performance metrics considering text complexity"""
        if generation_time <= 0 or text_length <= 0:
            return {"chars_per_sec": 0, "words_per_min": 0, "expected_time": 0, "performance_ratio": 1.0}
        
        # Calculate actual performance metrics
        chars_per_sec = text_length / generation_time
        words_per_min = (word_count * 60) / generation_time
        
        # Calculate expected generation time based on text complexity
        # Base rate: ~10-15 chars/sec for typical TTS (varies by complexity)
        base_chars_per_sec = 12.0  # Conservative baseline
        
        # Complexity adjustments
        complexity_factor = 1.0
        
        # Longer texts may have slightly lower throughput due to memory usage
        if text_length > 150:
            complexity_factor *= 1.1
        
        # Texts with many punctuation marks are more complex (use text_content if provided, otherwise estimate)
        if text_content:
            punct_count = sum(1 for c in text_content if c in '.,!?;:()[]"\'')
        else:
            # Estimate based on typical punctuation density
            punct_count = int(text_length * 0.08)  # ~8% punctuation in typical text
        
        punct_density = punct_count / max(text_length, 1)
        if punct_density > 0.15:  # High punctuation density
            complexity_factor *= 1.2
        
        expected_chars_per_sec = base_chars_per_sec / complexity_factor
        expected_time = text_length / expected_chars_per_sec
        
        # Performance ratio: how we're doing vs expected (1.0 = exactly as expected, <1.0 = better, >1.0 = worse)
        performance_ratio = generation_time / expected_time if expected_time > 0 else 1.0
        
        return {
            "chars_per_sec": chars_per_sec,
            "words_per_min": words_per_min,
            "expected_time": expected_time,
            "performance_ratio": performance_ratio
        }
    
    def _detect_performance_degradation(self, current_time: float, text_length: int = 100, word_count: int = 20, text_content: str = "") -> bool:
        """Detect if performance is degrading based on text-relative metrics"""
        
        # Calculate performance metrics for this chunk
        perf_metrics = self._calculate_performance_metrics(current_time, text_length, word_count, text_content)
        performance_ratio = perf_metrics["performance_ratio"]
        
        # Store performance ratios instead of absolute times
        if not hasattr(self, 'performance_ratios'):
            self.performance_ratios = []
        
        self.performance_ratios.append(performance_ratio)
        
        # Keep only last 15 measurements for faster detection
        if len(self.performance_ratios) > 15:
            self.performance_ratios = self.performance_ratios[-15:]
        
        # Still track absolute times for compatibility
        self.performance_history.append(current_time)
        if len(self.performance_history) > 15:
            self.performance_history = self.performance_history[-15:]
        
        # Need at least 6 measurements to detect trend
        if len(self.performance_ratios) < 6:
            return False
        
        # Check for immediate severe slowdown (>3x expected time)
        if performance_ratio > 3.0:
            return True
        
        # Compare recent average to earlier average (using performance ratios)
        recent_avg_ratio = sum(self.performance_ratios[-3:]) / 3  # Last 3 chunks
        earlier_avg_ratio = sum(self.performance_ratios[:3]) / 3   # First 3 chunks
        
        # If recent chunks are taking 50% longer relative to text complexity
        degradation_threshold = 1.5
        is_degraded = recent_avg_ratio > earlier_avg_ratio * degradation_threshold
        
        # Also check if we have consistent poor performance (>2x expected)
        recent_slow_count = sum(1 for ratio in self.performance_ratios[-5:] if ratio > 2.0)
        if recent_slow_count >= 3:  # 3 out of last 5 chunks are slow relative to text
            return True
            
        return is_degraded
    
    def _clear_model_caches(self):
        """Clear internal model caches that cause performance degradation"""
        try:
            # Clear T3 model KV-cache if it exists
            if hasattr(self.model, 't3') and hasattr(self.model.t3, 'patched_model'):
                # Clear any cached past_key_values or internal state
                if hasattr(self.model.t3.patched_model, 'clear_cache'):
                    self.model.t3.patched_model.clear_cache()
                
                # Force clear any transformer internal caches
                if hasattr(self.model.t3.patched_model, 'tfmr'):
                    tfmr = self.model.t3.patched_model.tfmr
                    # Clear position embeddings cache if it exists
                    if hasattr(tfmr, '_position_embeddings_cache'):
                        tfmr._position_embeddings_cache.clear()
                    # Clear any attention caches
                    for layer in getattr(tfmr, 'layers', []):
                        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'past_key_value'):
                            layer.self_attn.past_key_value = None
            
            # Clear S3Gen flow cache if it exists
            if hasattr(self.model, 's3gen'):
                s3gen = self.model.s3gen
                # Clear conditional flow matching caches
                if hasattr(s3gen, 'cond_cfm') and hasattr(s3gen.cond_cfm, 'flow_cache'):
                    # Reset flow cache to empty state
                    if hasattr(s3gen.cond_cfm, 'reset_cache'):
                        s3gen.cond_cfm.reset_cache()
                    elif hasattr(s3gen.cond_cfm, 'flow_cache'):
                        # Manually clear the flow cache
                        s3gen.cond_cfm.flow_cache = torch.zeros_like(s3gen.cond_cfm.flow_cache[:, :, :0])
                
                # Clear LRU resampler cache occasionally
                if hasattr(s3gen, 'get_resampler') and hasattr(s3gen.get_resampler, 'cache_clear'):
                    s3gen.get_resampler.cache_clear()
            
            # Clear any other potential caches
            if hasattr(self.model, 'clear_caches'):
                self.model.clear_caches()
                
        except Exception as e:
            logging.warning(f"Could not clear model caches: {e}")
    
    def _force_memory_cleanup(self, aggressive: bool = False):
        """Force aggressive memory cleanup"""
        # Log memory before cleanup
        memory_before = self._get_memory_usage()
        
        # Clear model-specific caches first (most important)
        if aggressive:
            self._clear_model_caches()
        
        if self.device == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()
            if aggressive:
                # Multiple rounds of cleanup for severe cases
                for _ in range(3):
                    torch.mps.empty_cache()
                    torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if aggressive:
                for _ in range(3):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if aggressive:
            # Multiple rounds of garbage collection
            for _ in range(3):
                gc.collect()
        
        # Model warmup after aggressive cleanup to reset internal state
        if aggressive:
            try:
                logging.info("🔄 Performing model warmup after cleanup...")
                with torch.no_grad():
                    # Brief generation to reset model internal state
                    _ = self.model.generate("Reset", exaggeration=self.exaggeration, cfg_weight=self.cfg_weight)
                
                # Clean up warmup artifacts
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logging.warning(f"Model warmup after cleanup failed: {e}")
        
        # Log memory after cleanup
        memory_after = self._get_memory_usage()
        rss_freed = memory_before.get('rss_mb', 0) - memory_after.get('rss_mb', 0)
        gpu_freed = memory_before.get('gpu_allocated_mb', 0) - memory_after.get('gpu_allocated_mb', 0)
        
        cleanup_type = "aggressive" if aggressive else "standard"
        if rss_freed > 10 or gpu_freed > 10:  # Only log if significant cleanup occurred
            logging.info(f"💾 {cleanup_type.title()} memory cleanup freed: {rss_freed:.1f}MB RAM, {gpu_freed:.1f}MB GPU")
    
    def _reinitialize_model(self):
        """Reinitialize the TTS model as a last resort for severe performance issues"""
        logging.warning("🔄 Reinitializing TTS model due to severe performance degradation...")
        
        # Clear all memory first
        self._force_memory_cleanup(aggressive=True)
        
        try:
            # Reinitialize the model
            del self.model
            if self.device == "mps":
                torch.mps.empty_cache()
                torch.mps.synchronize()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Recreate model
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            
            # Quick warmup
            with torch.no_grad():
                _ = self.model.generate("Reinit test", exaggeration=self.exaggeration, cfg_weight=self.cfg_weight)
            
            # Reset all counters
            self.chunks_since_cleanup = 0
            self.performance_history = []
            self.severe_degradation_count = 0
            
            logging.info("✅ Model reinitialization completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"❌ Model reinitialization failed: {e}")
            return False
    
    def generate_chunk(self, chunk_text: str, output_file: Path, max_retries: int = 2) -> bool:
        """Generate audio for a single chunk with validation and retry logic"""
        import time
        chunk_start_time = time.time()
        
        for attempt in range(max_retries + 1):
            is_retry = attempt > 0
            if is_retry:
                logging.warning(f"🔄 Retry attempt {attempt}/{max_retries} for chunk generation")
                # Aggressive cleanup before retry
                self._force_memory_cleanup(aggressive=True)
                self.chunks_since_cleanup = 0
                time.sleep(1.0)  # Brief pause before retry
                
            try:
                # More frequent light cleanup every 3 chunks
                if not is_retry:  # Skip on retries since we do aggressive cleanup above
                    self.chunks_since_cleanup += 1
                    
                    # Always do light cleanup before generation
                    if self.device == "mps":
                        torch.mps.empty_cache()
                    elif self.device == "cuda":
                        torch.cuda.empty_cache()
                    
                    # Periodic deep cleanup
                    if self.chunks_since_cleanup >= self.cleanup_interval:
                        logging.info(f"🧹 Performing periodic memory cleanup (every {self.cleanup_interval} chunks)")
                        self._force_memory_cleanup()
                        self.chunks_since_cleanup = 0
                    elif self.chunks_since_cleanup % 2 == 0:
                        # Light cleanup every 2 chunks including model cache clearing (more frequent)
                        self._clear_model_caches()
                        import gc
                        gc.collect()
                
                # Track memory before TTS generation
                memory_before_tts = self._get_memory_usage()
                tts_start_time = time.time()
                
                with torch.no_grad():
                    if self.voice_file and os.path.exists(self.voice_file):
                        wav = self.model.generate(
                            chunk_text,
                            audio_prompt_path=self.voice_file,
                            exaggeration=self.exaggeration,
                            cfg_weight=self.cfg_weight
                        )
                    else:
                        wav = self.model.generate(
                            chunk_text,
                            exaggeration=self.exaggeration,
                            cfg_weight=self.cfg_weight
                        )
                
                tts_duration = time.time() - tts_start_time
                memory_after_tts = self._get_memory_usage()
                
                # Calculate performance metrics and log appropriately
                text_len = len(chunk_text)
                word_count = len(chunk_text.split())
                punct_count = sum(1 for c in chunk_text if c in '.,!?;:')
                complexity_score = (punct_count / max(word_count, 1)) * 100  # Punctuation density as complexity proxy
                
                # Get performance metrics based on text complexity
                perf_metrics = self._calculate_performance_metrics(tts_duration, text_len, word_count, chunk_text)
                performance_ratio = perf_metrics["performance_ratio"]
                chars_per_sec = perf_metrics["chars_per_sec"]
                expected_time = perf_metrics["expected_time"]
                
                # Log based on performance ratio rather than absolute time
                should_log_detailed = (performance_ratio > 1.8 or self.debug_memory)  # 80% slower than expected
                
                if should_log_detailed:
                    gpu_growth = memory_after_tts.get('gpu_allocated_mb', 0) - memory_before_tts.get('gpu_allocated_mb', 0)
                    tensor_growth = memory_after_tts.get('torch_tensor_count', 0) - memory_before_tts.get('torch_tensor_count', 0)
                    
                    if performance_ratio > 2.5:  # >2.5x slower than expected
                        logging.warning(f"🔍 TTS generation slow: {tts_duration:.1f}s (expected {expected_time:.1f}s, {performance_ratio:.1f}x), {chars_per_sec:.1f} chars/s, GPU growth: {gpu_growth:.1f}MB, Text: {text_len}chars/{word_count}words/complexity:{complexity_score:.1f}%")
                    elif performance_ratio > 1.8:  # >1.8x slower than expected
                        logging.info(f"🔍 TTS timing: {tts_duration:.1f}s (expected {expected_time:.1f}s, {performance_ratio:.1f}x), {chars_per_sec:.1f} chars/s, GPU growth: {gpu_growth:.1f}MB, Text: {text_len}chars/{word_count}words")
                    elif self.debug_memory:
                        logging.info(f"🔍 TTS timing: {tts_duration:.1f}s (expected {expected_time:.1f}s, {performance_ratio:.1f}x), {chars_per_sec:.1f} chars/s, GPU growth: {gpu_growth:.1f}MB, Text: {text_len}chars/{word_count}words")
                
                # Apply pitch shift if specified
                wav = self.apply_pitch_shift(wav)
                
                # Save audio
                ta.save(str(output_file), wav, self.model.sr)
                
                # Validate generated audio
                validation_result = AudioValidator.validate_audio_chunk(output_file, chunk_text)
                
                if not validation_result["is_valid"]:
                    logging.warning(f"🔍 Audio validation failed (attempt {attempt + 1}):")
                    for issue in validation_result["issues"]:
                        logging.warning(f"   - {issue}")
                    
                    # Log metrics for debugging
                    metrics = validation_result["metrics"]
                    if metrics:
                        duration = metrics.get("duration", 0)
                        rms = metrics.get("rms_level", 0)
                        words_per_min = metrics.get("words_per_minute", 0)
                        max_amp = metrics.get("max_amplitude", 0)
                        logging.warning(f"   Metrics: {duration:.2f}s, RMS:{rms:.4f}, {words_per_min:.0f}WPM, MaxAmp:{max_amp:.3f}")
                    
                    # Clean up failed audio and try again (unless last attempt)
                    if output_file.exists():
                        output_file.unlink()
                    del wav
                    
                    if attempt < max_retries:
                        logging.info(f"   Retrying with aggressive cleanup...")
                        continue
                    else:
                        logging.error(f"❌ Audio validation failed after {max_retries + 1} attempts")
                        return False
                else:
                    # Validation passed
                    if is_retry:
                        logging.info(f"✅ Audio validation passed on retry attempt {attempt}")
                    elif validation_result["issues"]:
                        # Has minor issues but still valid
                        logging.info(f"⚠️ Audio generated with minor issues:")
                        for issue in validation_result["issues"]:
                            logging.info(f"   - {issue}")
                    
                    # Log detailed metrics if debug enabled or if there were issues
                    if self.debug_memory or validation_result["issues"]:
                        metrics = validation_result["metrics"]
                        duration = metrics.get("duration", 0)
                        rms = metrics.get("rms_level", 0)
                        words_per_min = metrics.get("words_per_minute", 0)
                        max_amp = metrics.get("max_amplitude", 0)
                        volume_var = metrics.get("volume_variance", 0)
                        logging.info(f"🔍 Audio metrics: {duration:.2f}s, RMS:{rms:.4f}, {words_per_min:.0f}WPM, MaxAmp:{max_amp:.3f}, VolVar:{volume_var:.2f}")
                
                # Immediate cleanup
                del wav
                
                # Light cleanup after generation (deep cleanup handled periodically)
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
                
                # Performance monitoring and adaptive cleanup (only on successful generation)
                chunk_duration = time.time() - chunk_start_time
                
                # Use text-aware performance detection for final chunk timing
                chunk_perf_metrics = self._calculate_performance_metrics(chunk_duration, text_len, word_count, chunk_text)
                chunk_performance_ratio = chunk_perf_metrics["performance_ratio"]
                
                # Log detailed timing breakdown for debugging (only if significantly slow relative to text)
                if chunk_performance_ratio > 2.0:  # 2x slower than expected for this text length
                    memory_after = self._get_memory_usage()
                    expected_chunk_time = chunk_perf_metrics["expected_time"]
                    logging.warning(f"🐌 Slow chunk generation: {chunk_duration:.1f}s (expected {expected_chunk_time:.1f}s, {chunk_performance_ratio:.1f}x slower) (GPU: {memory_after.get('gpu_allocated_mb', 0):.0f}MB, Tensors: {memory_after.get('torch_tensor_count', 0)})")
                
                if self.adaptive_cleanup and self._detect_performance_degradation(chunk_duration, text_len, word_count, chunk_text):
                    self.severe_degradation_count += 1
                    logging.warning(f"⚠️ Performance degradation detected ({self.severe_degradation_count}/{self.model_reinit_threshold})")
                    
                    if self.severe_degradation_count >= self.model_reinit_threshold:
                        # Try model reinitialization as last resort (lowered threshold for faster recovery)
                        logging.info(f"🔄 Triggering model reinitialization after {self.severe_degradation_count} severe degradations")
                        if self._reinitialize_model():
                            logging.info("✅ Model reinitialized successfully - performance should improve")
                        else:
                            logging.error("❌ Model reinitialization failed, continuing with aggressive cleanup")
                            self._force_memory_cleanup(aggressive=True)
                            self.chunks_since_cleanup = 0
                            self.performance_history = []
                            self.severe_degradation_count = 0
                    else:
                        # Try aggressive cleanup first
                        logging.info("🧹 Triggering aggressive cleanup")
                        self._force_memory_cleanup(aggressive=True)
                        self.chunks_since_cleanup = 0
                        self.performance_history = []
                
                return True
                
            except torch.mps.OutOfMemoryError as e:
                logging.error(f"MPS out of memory generating chunk (attempt {attempt + 1}): {e}")
                # Force aggressive cleanup on OOM and reset counter
                self._force_memory_cleanup(aggressive=True)
                self.chunks_since_cleanup = 0
                self.performance_history = []  # Reset history
                
                if attempt < max_retries:
                    logging.info("Retrying after OOM cleanup...")
                    continue
                else:
                    return False
                    
            except Exception as e:
                logging.error(f"Error generating chunk (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    logging.info("Retrying after error...")
                    continue
                else:
                    return False
        
        # Should not reach here
        return False
    
    def process_audiobook(self, input_file: Path, time_limit: Optional[int] = None) -> Path:
        """Process entire audiobook with chunking and parallel processing"""
        
        # Create output directory
        output_dir = Path(input_file.stem)
        output_dir.mkdir(exist_ok=True)
        
        logging.info(f"📚 Processing audiobook: {input_file}")
        logging.info(f"📁 Output directory: {output_dir}")
        
        # Read and chunk text
        logging.info("📖 Reading and chunking text...")
        text_processor = TextProcessor()
        text = text_processor.read_file(input_file)
        
        chunker = TextChunker(max_chars=200)  # Optimal chunk size for speed
        chunks = chunker.chunk_text(text)
        
        logging.info(f"📄 Created {len(chunks)} chunks from {len(text)} characters")
        
        # Setup progress tracking
        progress = ProgressTracker(output_dir)
        progress.progress["total_chunks"] = len(chunks)
        if not progress.progress["start_time"]:
            progress.progress["start_time"] = datetime.now().isoformat()
        progress.save_progress()
        
        # Check for existing chunks
        chunks_to_process = []
        for i, chunk in enumerate(chunks):
            chunk_file = output_dir / f"chunk_{i:04d}.wav"
            
            if progress.is_chunk_completed(i) and chunk_file.exists():
                logging.info(f"⏭️ Skipping chunk {i:04d} (already completed)")
                continue
            
            chunks_to_process.append((i, chunk, chunk_file))
        
        if not chunks_to_process:
            logging.info("🎉 All chunks already completed!")
            return self._combine_chunks(output_dir, len(chunks), input_file.stem)
        
        logging.info(f"🚀 Processing {len(chunks_to_process)} remaining chunks...")
        
        # Set session start time for ETA calculation
        progress.set_session_start()
        
        # Time limit setup
        start_time = datetime.now()
        time_limit_reached = False
        
        # Process chunks with parallel execution
        completed_count = 0
        failed_count = 0
        
        def process_single_chunk(chunk_data):
            """Process a single chunk (for parallel execution)"""
            chunk_index, chunk_text, chunk_file = chunk_data
            
            logging.info(f"🎵 Generating chunk {chunk_index:04d}: {chunk_text[:50]}...")
            
            # Add delay between chunks for memory recovery
            import time
            time.sleep(0.8)  # Increased delay for better memory recovery
            
            # Record processing start time
            chunk_start_time = time.time()
            
            success = self.generate_chunk(chunk_text, chunk_file)
            chunk_duration = time.time() - chunk_start_time
            
            # Get memory stats after chunk generation
            memory_stats = self._get_memory_usage()
            
            if success:
                # Check for problematic endings and re-generate if needed
                detection_result = self.ending_detector.analyze_ending_pattern(chunk_file)
                if detection_result['is_problematic']:
                    logging.warning(f"🔍 Chunk {chunk_index:04d} has problematic ending: {detection_result['reason']}")
                    logging.info(f"🔄 Re-generating chunk {chunk_index:04d} to fix ending issue...")
                    
                    # Remove the problematic chunk
                    if chunk_file.exists():
                        chunk_file.unlink()
                    
                    # Re-generate the chunk (with one retry)
                    regeneration_success = self.generate_chunk(chunk_text, chunk_file, max_retries=1)
                    if regeneration_success:
                        # Check the re-generated chunk
                        recheck_result = self.ending_detector.analyze_ending_pattern(chunk_file)
                        if recheck_result['is_problematic']:
                            logging.warning(f"⚠️ Chunk {chunk_index:04d} still problematic after regeneration, keeping anyway")
                        else:
                            logging.info(f"✅ Chunk {chunk_index:04d} regeneration successful - ending issue resolved")
                        self.regenerated_chunks_count += 1
                        logging.info(f"📊 Total chunks regenerated: {self.regenerated_chunks_count}")
                    else:
                        logging.error(f"❌ Failed to regenerate chunk {chunk_index:04d}, keeping original")
                        # Restore original chunk by regenerating without ending check
                        self.generate_chunk(chunk_text, chunk_file, max_retries=0)
                
                progress.mark_chunk_completed(chunk_index)
                try:
                    audio_length = ta.info(str(chunk_file)).num_frames / ta.info(str(chunk_file)).sample_rate
                    logging.info(f"✅ Completed chunk {chunk_index:04d} ({chunk_duration:.1f}s)")
                except Exception:
                    logging.info(f"✅ Completed chunk {chunk_index:04d} ({chunk_duration:.1f}s)")
                
                # Log detailed memory usage for every chunk if debug mode enabled
                if self.debug_memory:
                    ram_mb = memory_stats.get('rss_mb', 0)
                    gpu_mb = memory_stats.get('gpu_allocated_mb', 0)
                    gpu_max_mb = memory_stats.get('gpu_max_allocated_mb', 0)
                    sys_percent = memory_stats.get('system_percent', 0)
                    tensor_count = memory_stats.get('torch_tensor_count', 0)
                    tensor_memory_mb = memory_stats.get('torch_tensor_memory_mb', 0)
                    
                    logging.info(f"💾 Chunk {chunk_index:04d} Memory: RAM={ram_mb:.0f}MB, GPU={gpu_mb:.0f}MB (max={gpu_max_mb:.0f}MB), SysRAM={sys_percent:.1f}%, Tensors={tensor_count} ({tensor_memory_mb:.0f}MB)")
                else:
                    # Just log basic memory stats
                    ram_mb = memory_stats.get('rss_mb', 0)
                    gpu_mb = memory_stats.get('gpu_allocated_mb', 0)
                    tensor_count = memory_stats.get('torch_tensor_count', 0)
                    logging.info(f"💾 Chunk {chunk_index:04d}: {ram_mb:.0f}MB RAM, {gpu_mb:.0f}MB GPU, {tensor_count} tensors")
            else:
                progress.mark_chunk_failed(chunk_index)
                logging.error(f"❌ Failed chunk {chunk_index:04d}")
            
            return success
        
        # Reduce workers to prevent memory exhaustion
        safe_workers = min(self.max_workers, 1 if self.device == "mps" else 2)
        logging.info(f"🔧 Using {safe_workers} workers for memory safety")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=safe_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(process_single_chunk, chunk_data): chunk_data[0] 
                for chunk_data in chunks_to_process
            }
            
            # Process completed chunks
            for future in as_completed(future_to_chunk):
                # Check time limit
                if time_limit:
                    elapsed = (datetime.now() - start_time).total_seconds() / 60
                    if elapsed >= time_limit:
                        logging.warning(f"⏰ Time limit reached ({time_limit} minutes)")
                        time_limit_reached = True
                        
                        # Cancel remaining futures
                        for f in future_to_chunk:
                            if not f.done():
                                f.cancel()
                        break
                
                try:
                    success = future.result()
                    if success:
                        completed_count += 1
                    else:
                        failed_count += 1
                    
                    # Progress update with ETA
                    total_completed, total_failed, total_chunks = progress.get_completion_stats()
                    eta_str, chunks_per_sec = progress.calculate_eta()
                    
                    # Calculate percentage
                    percentage = (total_completed / total_chunks * 100) if total_chunks > 0 else 0
                    
                    logging.info(f"📊 Progress: {total_completed}/{total_chunks} ({percentage:.1f}%) completed, {total_failed} failed")
                    logging.info(f"⏱️ ETA: {eta_str} (Rate: {chunks_per_sec:.1f} chunks/sec)")
                    
                    # Log memory usage every 50 chunks to monitor trends
                    if total_completed % 50 == 0 and total_completed > 0:
                        memory_stats = self._get_memory_usage()
                        ram_usage = memory_stats.get('rss_mb', 0)
                        gpu_usage = memory_stats.get('gpu_allocated_mb', 0)
                        logging.info(f"💾 Memory usage: {ram_usage:.0f}MB RAM, {gpu_usage:.0f}MB GPU")
                    
                except Exception as e:
                    logging.error(f"Chunk processing error: {e}")
                    failed_count += 1
        
        # Final statistics
        total_completed, total_failed, total_chunks = progress.get_completion_stats()
        
        logging.info(f"\n🎯 Generation Complete!")
        logging.info(f"   Completed: {total_completed}/{total_chunks}")
        logging.info(f"   Failed: {total_failed}")
        
        if time_limit_reached:
            logging.info(f"   Stopped due to time limit: {time_limit} minutes")
        
        # Combine completed chunks
        if total_completed > 0:
            return self._combine_chunks(output_dir, len(chunks), input_file.stem)
        else:
            logging.error("No chunks were successfully generated!")
            return output_dir
    
    def _combine_chunks(self, output_dir: Path, total_chunks: int, base_filename: str) -> Path:
        """Combine individual chunk files into final audiobook"""
        
        logging.info("🔗 Combining chunks into final audiobook...")
        
        audio_segments = []
        sample_rate = None
        
        for i in range(total_chunks):
            chunk_file = output_dir / f"chunk_{i:04d}.wav"
            
            if chunk_file.exists():
                try:
                    wav, sr = ta.load(str(chunk_file))
                    audio_segments.append(wav)
                    
                    if sample_rate is None:
                        sample_rate = sr
                    
                except Exception as e:
                    logging.warning(f"Could not load chunk {i:04d}: {e}")
        
        if not audio_segments:
            logging.error("No audio segments to combine!")
            return output_dir
        
        # Add small pause between chunks (0.5 seconds)
        pause_samples = int(0.5 * sample_rate)
        pause = torch.zeros(1, pause_samples)
        
        # Combine all segments
        combined_audio = audio_segments[0]
        for segment in audio_segments[1:]:
            combined_audio = torch.cat([combined_audio, pause, segment], dim=1)
        
        # Save final audiobook as WAV
        final_wav_file = output_dir / "audiobook.wav"
        ta.save(str(final_wav_file), combined_audio, sample_rate)
        
        # Calculate statistics
        total_duration = combined_audio.shape[1] / sample_rate
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = int(total_duration % 60)
        
        logging.info(f"🎉 WAV audiobook complete!")
        logging.info(f"   File: {final_wav_file}")
        logging.info(f"   Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
        logging.info(f"   Chunks used: {len(audio_segments)}/{total_chunks}")
        
        # Handle file splitting and MP3 conversion
        final_output_files = []
        
        if self.split_minutes > 0:
            # Split the combined audiobook into time-limited files
            logging.info(f"✂️ Splitting audiobook into {self.split_minutes}-minute segments...")
            
            split_files = AudioSplitter.split_audio_by_time(
                input_file=final_wav_file,
                output_base_name=base_filename,
                max_minutes=self.split_minutes,
                mp3_enabled=self.mp3_enabled,
                mp3_bitrate=self.mp3_bitrate,
                remove_wav=self.remove_wav,
                metadata=self.metadata,
                smart_split=self.smart_split
            )
            
            if split_files:
                final_output_files = split_files
                logging.info(f"📱 Split audiobook into {len(split_files)} files")
                
                # Remove the combined file if split was successful and we're keeping MP3 only
                if self.remove_wav and self.mp3_enabled:
                    try:
                        final_wav_file.unlink()
                        logging.info(f"🗑️ Removed combined WAV file: {final_wav_file.name}")
                    except Exception as e:
                        logging.warning(f"Could not remove combined WAV file: {e}")
            else:
                logging.warning("⚠️ File splitting failed, keeping combined file")
                final_output_files = [final_wav_file]
        
        elif self.mp3_enabled:
            # Convert to MP3 without splitting
            final_mp3_file = output_dir / f"{base_filename}.mp3"
            logging.info("🔄 Converting final audiobook to MP3...")
            
            conversion_success = AudioConverter.convert_to_mp3(
                wav_file=final_wav_file,
                mp3_file=final_mp3_file,
                bitrate=self.mp3_bitrate,
                remove_wav=self.remove_wav,
                metadata=self.metadata
            )
            
            if conversion_success:
                final_output_files = [final_mp3_file]
                
                # Get audio info for the MP3
                mp3_info = AudioConverter.get_audio_info(final_mp3_file)
                if mp3_info:
                    logging.info(f"📱 MP3 audiobook details:")
                    logging.info(f"   File: {final_mp3_file}")
                    logging.info(f"   Size: {mp3_info.get('size', 0) // 1024 // 1024}MB")
                    logging.info(f"   Bitrate: {mp3_info.get('bitrate', 0) // 1000}kbps")
                    logging.info(f"   Format: {mp3_info.get('codec', 'mp3')}")
            else:
                final_output_files = [final_wav_file]
        else:
            # Keep WAV file as is
            final_output_files = [final_wav_file]
        
        # Clean up individual chunk files if requested
        if self.remove_wav and (self.mp3_enabled or self.split_minutes > 0):
            self._cleanup_chunk_files(output_dir, total_chunks)
        
        # Log final output files
        if final_output_files:
            logging.info("📋 Final output files:")
            for output_file in final_output_files:
                file_info = AudioConverter.get_audio_info(output_file)
                duration = file_info.get('duration', 0)
                size_mb = output_file.stat().st_size // 1024 // 1024
                logging.info(f"   📄 {output_file.name} - {duration//60:.0f}:{duration%60:02.0f} ({size_mb}MB)")
        
        return output_dir
    
    def _cleanup_chunk_files(self, output_dir: Path, total_chunks: int):
        """Remove individual chunk WAV files after successful conversion"""
        removed_count = 0
        
        logging.info("🧹 Cleaning up individual chunk files...")
        
        for i in range(total_chunks):
            chunk_file = output_dir / f"chunk_{i:04d}.wav"
            
            if chunk_file.exists():
                try:
                    chunk_file.unlink()
                    removed_count += 1
                except Exception as e:
                    logging.warning(f"Could not remove chunk file {chunk_file.name}: {e}")
        
        if removed_count > 0:
            logging.info(f"🗑️ Removed {removed_count} chunk files")

def main():
    parser = argparse.ArgumentParser(
        description="Convert text files and ebooks to audiobooks using Chatterbox TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python audiobook_tts.py book.txt
  python audiobook_tts.py novel.fb2 --voice voices/narrator.wav
  python audiobook_tts.py story.epub --limit-minutes 30 --mp3
  python audiobook_tts.py document.txt --voice voices/reader.wav --mp3 --mp3-bitrate 192k
  python audiobook_tts.py book.txt --mp3 --remove-wav --mp3-bitrate 256k --split-minutes 10
  python audiobook_tts.py novel.txt --split-minutes 3 --mp3 --tag "Author Name - Book Title"
  python audiobook_tts.py book.txt --mp3 --tag "Arthur Conan Doyle - The Adventures of Sherlock Holmes"
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Input text file (supports .txt, .fb2, .epub, .md)"
    )
    
    parser.add_argument(
        "--voice",
        help="Voice reference file for cloning (e.g., voices/myvoice.wav)"
    )
    
    parser.add_argument(
        "--limit-minutes",
        type=int,
        help="Maximum processing time in minutes"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)"
    )
    
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.8,
        help="Voice exaggeration level (0.2-1.2, lower=higher pitch, default: 0.8)"
    )
    
    parser.add_argument(
        "--cfg-weight", 
        type=float,
        default=0.8,
        help="CFG weight (0.2-1.0, lower=lighter voice, default: 0.8)"
    )
    
    parser.add_argument(
        "--pitch-shift",
        type=float,
        default=0.0,
        help="Pitch shift in semitones (+/-4, positive=higher, negative=lower)"
    )
    
    # MP3 conversion arguments
    parser.add_argument(
        "--mp3",
        action="store_true",
        help="Convert final audiobook to MP3 format"
    )
    
    parser.add_argument(
        "--mp3-bitrate",
        type=str,
        default="128k",
        choices=["64k", "96k", "128k", "160k", "192k", "256k", "320k"],
        help="MP3 bitrate for conversion (default: 128k)"
    )
    
    parser.add_argument(
        "--remove-wav",
        action="store_true",
        help="Remove WAV files after MP3 conversion (saves disk space)"
    )
    
    # File splitting arguments
    parser.add_argument(
        "--split-minutes",
        type=int,
        default=5,
        help="Split output into files of maximum X minutes each with sentence boundary awareness (default: 5, set to 0 to disable)"
    )
    
    parser.add_argument(
        "--disable-smart-split",
        action="store_true",
        help="Disable sentence-aware splitting and use fixed time intervals (may cut mid-sentence)"
    )
    
    parser.add_argument(
        "--memory-cleanup-interval",
        type=int,
        default=5,
        help="Perform deep memory cleanup every N chunks to prevent slowdown (default: 5, lower=more frequent)"
    )
    
    parser.add_argument(
        "--debug-memory",
        action="store_true",
        help="Enable detailed memory usage logging for every chunk to debug performance issues"
    )
    
    # MP3 metadata arguments
    parser.add_argument(
        "--tag",
        type=str,
        help="MP3 metadata in format 'Author - Book Title' (e.g., 'Arthur Conan Doyle - The Adventures of Sherlock Holmes')"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_file = Path(args.input_file)
    if not input_file.exists():
        logging.error(f"Input file not found: {input_file}")
        sys.exit(1)
    
    # Validate voice file if provided
    if args.voice and not os.path.exists(args.voice):
        logging.error(f"Voice file not found: {args.voice}")
        sys.exit(1)
    
    # Validate MP3 settings
    if args.remove_wav and not args.mp3 and args.split_minutes == 0:
        logging.warning("--remove-wav specified without --mp3 or --split-minutes, ignoring...")
        args.remove_wav = False
    
    # Validate split settings
    if args.split_minutes < 0:
        logging.error("--split-minutes must be 0 or positive")
        sys.exit(1)
    
    # Parse metadata from tag if provided
    metadata = {}
    if args.tag:
        # Parse format: "Author - Book Title"
        if ' - ' in args.tag:
            author, title = args.tag.split(' - ', 1)
            metadata['artist'] = author.strip()
            metadata['album'] = title.strip()
            metadata['title'] = title.strip()
            metadata['genre'] = 'Audiobook'
        else:
            # If no separator, treat entire string as title
            metadata['album'] = args.tag.strip()
            metadata['title'] = args.tag.strip()
            metadata['genre'] = 'Audiobook'
    
    # Log configuration
    logging.info("🎙️ Audiobook TTS Generator Starting...")
    logging.info(f"   Input: {input_file}")
    logging.info(f"   Voice: {args.voice or 'Default model voice'}")
    logging.info(f"   Voice settings: exag={args.exaggeration}, cfg={args.cfg_weight}")
    logging.info(f"   Pitch shift: {args.pitch_shift:+.1f} semitones")
    logging.info(f"   Time limit: {args.limit_minutes or 'None'} minutes")
    logging.info(f"   Workers: {args.workers}")
    
    if args.split_minutes > 0:
        split_mode = "fixed intervals" if args.disable_smart_split else "sentence-aware"
        logging.info(f"   File splitting: enabled (max {args.split_minutes} minutes per file, {split_mode})")
        output_format = "MP3" if args.mp3 else "WAV"
        logging.info(f"   Output format: {output_format}")
        if args.mp3:
            logging.info(f"   MP3 bitrate: {args.mp3_bitrate}")
    elif args.mp3:
        logging.info(f"   MP3 conversion: enabled (bitrate: {args.mp3_bitrate})")
        logging.info(f"   File splitting: disabled")
    else:
        logging.info("   MP3 conversion: disabled")
        logging.info("   File splitting: disabled")
    
    if args.remove_wav:
        logging.info(f"   Remove WAV files: {args.remove_wav}")
    
    if metadata:
        logging.info("🏷️ MP3 Metadata:")
        if metadata.get('artist'):
            logging.info(f"   Author: {metadata['artist']}")
        if metadata.get('album'):
            logging.info(f"   Book: {metadata['album']}")
    
    try:
        # Initialize TTS generator
        tts_generator = AudiobookTTS(
            voice_file=args.voice,
            max_workers=args.workers,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            pitch_shift=args.pitch_shift,
            mp3_enabled=args.mp3,
            mp3_bitrate=args.mp3_bitrate,
            remove_wav=args.remove_wav,
            split_minutes=args.split_minutes,
            memory_cleanup_interval=args.memory_cleanup_interval,
            debug_memory=args.debug_memory,
            metadata=metadata,
            smart_split=not args.disable_smart_split
        )
        
        # Process audiobook
        output_dir = tts_generator.process_audiobook(
            input_file=input_file,
            time_limit=args.limit_minutes
        )
        
        logging.info(f"🎉 Audiobook generation complete! Check: {output_dir}")
        
        # Show regenerated chunks statistics
        if tts_generator.regenerated_chunks_count > 0:
            logging.info(f"🔄 Quality control: {tts_generator.regenerated_chunks_count} chunks were regenerated due to ending issues")
        else:
            logging.info(f"✅ Quality control: All chunks passed ending detection")
        
        # List final output files
        final_files = []
        
        # Look for split files or single files
        if args.split_minutes > 0:
            # Look for split files with the new naming convention
            base_name = input_file.stem
            extension = ".mp3" if args.mp3 else ".wav"
            
            split_files = sorted(output_dir.glob(f"{base_name}_*.{extension.lstrip('.')}"))
            for split_file in split_files:
                final_files.append(f"🎵 {split_file.name}")
        else:
            # Look for single combined files
            wav_file = output_dir / "audiobook.wav"
            mp3_file = output_dir / f"{input_file.stem}.mp3"
            
            if mp3_file.exists():
                final_files.append(f"🎵 {mp3_file.name}")
            elif wav_file.exists():
                final_files.append(f"📄 {wav_file.name}")
        
        if final_files:
            logging.info("📋 Final output files:")
            for file_info in final_files:
                logging.info(f"   {file_info}")
        
    except KeyboardInterrupt:
        logging.info("⏹️ Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Error during generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()