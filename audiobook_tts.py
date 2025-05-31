#!/usr/bin/env python3
"""
Audiobook TTS Generator
Converts text files and ebooks to audiobooks using Chatterbox TTS
Supports chunking, parallel processing, resume capability, and voice cloning
"""

import argparse
import logging
import os
import sys
import time
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple
import hashlib

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

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
    """Intelligently chunks text for optimal TTS processing"""
    
    def __init__(self, max_chars: int = 200):
        self.max_chars = max_chars
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into optimal chunks for TTS"""
        
        # Clean up text
        text = self._clean_text(text)
        
        # Split into paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            if len(paragraph) <= self.max_chars:
                chunks.append(paragraph)
            else:
                # Split long paragraphs at sentence boundaries
                sentences = self._split_sentences(paragraph)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + " " + sentence) <= self.max_chars and current_chunk:
                        current_chunk += " " + sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Clean up common ebook artifacts
        text = re.sub(r'\*\*\*+', '', text)  # Stars
        text = re.sub(r'---+', '', text)     # Dashes
        
        return text.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Add back punctuation to sentences (except the last one)
        for i in range(len(sentences) - 1):
            sentences[i] += "."
        
        return sentences

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
            "last_update": None
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

class AudiobookTTS:
    """Main TTS audiobook generator"""
    
    def __init__(self, voice_file: Optional[str] = None, max_workers: int = 2, 
                 exaggeration: float = 0.8, cfg_weight: float = 0.8, pitch_shift: float = 0.0):
        self.device = setup_mac_compatibility()
        logging.info(f"Initializing Chatterbox TTS on {self.device}...")
        
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.voice_file = voice_file
        self.max_workers = max_workers
        
        # Voice characteristics settings
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.pitch_shift = pitch_shift  # Semitones to shift pitch
        
        # Warmup
        with torch.no_grad():
            _ = self.model.generate("Warmup", exaggeration=self.exaggeration, cfg_weight=self.cfg_weight)
        
        logging.info(f"✅ TTS model ready for audiobook generation")
        logging.info(f"   Voice settings: exag={self.exaggeration}, cfg={self.cfg_weight}")
        if self.pitch_shift != 0:
            logging.info(f"   Pitch shift: {self.pitch_shift:+.1f} semitones")
    
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
    
    def generate_chunk(self, chunk_text: str, output_file: Path) -> bool:
        """Generate audio for a single chunk"""
        try:
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
            
            # Apply pitch shift if specified
            wav = self.apply_pitch_shift(wav)
            
            # Save audio
            ta.save(str(output_file), wav, self.model.sr)
            
            # Clear GPU memory
            if self.device in ["mps", "cuda"]:
                if self.device == "mps":
                    torch.mps.empty_cache()
                else:
                    torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            logging.error(f"Error generating chunk: {e}")
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
            return self._combine_chunks(output_dir, len(chunks))
        
        logging.info(f"🚀 Processing {len(chunks_to_process)} remaining chunks...")
        
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
            
            success = self.generate_chunk(chunk_text, chunk_file)
            
            if success:
                progress.mark_chunk_completed(chunk_index)
                audio_length = ta.info(str(chunk_file)).num_frames / ta.info(str(chunk_file)).sample_rate
                logging.info(f"✅ Completed chunk {chunk_index:04d} ({audio_length:.1f}s)")
            else:
                progress.mark_chunk_failed(chunk_index)
                logging.error(f"❌ Failed chunk {chunk_index:04d}")
            
            return success
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
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
                    
                    # Progress update
                    total_completed, total_failed, total_chunks = progress.get_completion_stats()
                    logging.info(f"📊 Progress: {total_completed}/{total_chunks} completed, {total_failed} failed")
                    
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
            return self._combine_chunks(output_dir, len(chunks))
        else:
            logging.error("No chunks were successfully generated!")
            return output_dir
    
    def _combine_chunks(self, output_dir: Path, total_chunks: int) -> Path:
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
        
        # Save final audiobook
        final_file = output_dir / "audiobook.wav"
        ta.save(str(final_file), combined_audio, sample_rate)
        
        # Calculate statistics
        total_duration = combined_audio.shape[1] / sample_rate
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = int(total_duration % 60)
        
        logging.info(f"🎉 Audiobook complete!")
        logging.info(f"   File: {final_file}")
        logging.info(f"   Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
        logging.info(f"   Chunks used: {len(audio_segments)}/{total_chunks}")
        
        return output_dir

def main():
    parser = argparse.ArgumentParser(
        description="Convert text files and ebooks to audiobooks using Chatterbox TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python audiobook_tts.py book.txt
  python audiobook_tts.py novel.fb2 --voice voices/narrator.wav
  python audiobook_tts.py story.epub --limit-minutes 30
  python audiobook_tts.py document.txt --voice voices/reader.wav --limit-minutes 60
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
    
    # Log configuration
    logging.info("🎙️ Audiobook TTS Generator Starting...")
    logging.info(f"   Input: {input_file}")
    logging.info(f"   Voice: {args.voice or 'Default model voice'}")
    logging.info(f"   Voice settings: exag={args.exaggeration}, cfg={args.cfg_weight}")
    logging.info(f"   Pitch shift: {args.pitch_shift:+.1f} semitones")
    logging.info(f"   Time limit: {args.limit_minutes or 'None'} minutes")
    logging.info(f"   Workers: {args.workers}")
    
    try:
        # Initialize TTS generator
        tts_generator = AudiobookTTS(
            voice_file=args.voice,
            max_workers=args.workers,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            pitch_shift=args.pitch_shift
        )
        
        # Process audiobook
        output_dir = tts_generator.process_audiobook(
            input_file=input_file,
            time_limit=args.limit_minutes
        )
        
        logging.info(f"🎉 Audiobook generation complete! Check: {output_dir}")
        
    except KeyboardInterrupt:
        logging.info("⏹️ Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Error during generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()