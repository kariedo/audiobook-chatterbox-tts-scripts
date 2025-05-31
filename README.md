# Audiobook TTS Generator Suite

A comprehensive collection of Python scripts for converting text files and ebooks into high-quality audiobooks using Chatterbox TTS. Optimized for Apple Silicon with voice cloning, pitch control, and parallel processing.

## 🎯 Quick Start

```bash
# Basic audiobook generation
python audiobook_tts.py your_book.txt

# With custom voice and pitch adjustment
python audiobook_tts.py book.epub --voice voices/narrator.wav --exaggeration 0.3 --pitch-shift +1
```

## 📚 Scripts Overview

### 📖 `audiobook_tts.py` - Main Audiobook Generator
Converts text files and ebooks into complete audiobooks with professional features.

**Features:**
- Supports `.txt`, `.epub`, `.fb2`, `.md` files
- Smart text chunking for optimal processing
- Parallel processing for faster generation
- Resume capability (picks up where you left off)
- Voice cloning support
- Time limits and progress tracking
- Pitch and voice characteristic control

**Usage:**
```bash
# Basic usage
python audiobook_tts.py book.txt

# Advanced usage
python audiobook_tts.py novel.epub \
  --voice voices/narrator.wav \
  --limit-minutes 60 \
  --exaggeration 0.5 \
  --cfg-weight 0.7 \
  --pitch-shift -1.5 \
  --workers 3
```

**Output:** Creates folder `book/` with individual chunks and final `audiobook.wav`

---

### 🔍 `epub_preview.py` - EPUB Debug Tool
Analyzes EPUB files to debug text extraction issues and preview content before conversion.

**Features:**
- Shows EPUB internal structure and metadata
- Previews extracted text before TTS conversion
- Identifies navigation vs content files
- Saves clean extracted text for manual review

**Usage:**
```bash
# Show EPUB structure
python epub_preview.py book.epub --structure

# Preview extracted text
python epub_preview.py book.epub --extract --preview 1000

# Save clean text for review
python epub_preview.py book.epub --save clean_book.txt
```

**When to use:** When EPUB conversion produces gibberish or poor results.

---

### 📄 `simple_epub_reader.py` - Alternative EPUB Processor
Fallback EPUB reader using basic zipfile extraction when `epub_preview.py` fails.

**Features:**
- Works with problematic EPUB files
- Uses zipfile instead of ebooklib
- Lists all files in EPUB for debugging
- Extracts and cleans text directly

**Usage:**
```bash
python simple_epub_reader.py problematic_book.epub
```

**Output:** Creates `book_extracted.txt` with clean text, then use:
```bash
python audiobook_tts.py book_extracted.txt
```

---

### 🎛️ `voice_pitch_tuner.py` - Voice Customization Tool
Interactive tool for finding optimal voice pitch and characteristics before generating full audiobooks.

**Features:**
- Test different voice parameter combinations
- Real-time pitch adjustment with audio post-processing
- Interactive tuning mode
- Generate comparison samples
- Save optimal settings for audiobook generation

**Usage:**
```bash
# Interactive tuning mode
python voice_pitch_tuner.py --interactive

# Test parameter combinations
python voice_pitch_tuner.py --test-params

# Generate pitch variations
python voice_pitch_tuner.py --pitch-variations --voice voices/myvoice.wav
```

**Voice Settings Guide:**
- **Higher Pitch:** `--exaggeration 0.2 --cfg-weight 0.3 --pitch-shift +2`
- **Lower Pitch:** `--exaggeration 1.0 --cfg-weight 0.8 --pitch-shift -2`
- **Neutral:** `--exaggeration 0.5 --cfg-weight 0.5`

---

### 🎤 `voice_harvester.py` - Voice Sample Extractor
Extracts clean voice segments from audiobooks, podcasts, or recordings for TTS voice cloning.

**Features:**
- Automatic voice activity detection
- Extracts optimal-length segments (5-15 seconds)
- Quality filtering (removes clipped or noisy audio)
- Batch processing for multiple files
- Creates organized voice library

**Usage:**
```bash
# Extract from single audiobook
python voice_harvester.py --extract audiobook.mp3

# Create voice library from multiple files
python voice_harvester.py --library audiobooks/ --output voice_library/

# Get LibriVox download guide
python voice_harvester.py --librivox-info
```

**Legal Voice Sources:**
- **LibriVox.org** - Free public domain audiobooks
- **Archive.org** - Historical recordings and speeches
- **Your own recordings** - Always legal!

## 🛠️ Installation

### Prerequisites
```bash
# Install Python dependencies
pip install torch torchaudio chatterbox-tts

# Optional dependencies for enhanced features
pip install ebooklib beautifulsoup4  # Better EPUB support
pip install sounddevice              # For voice recording
```

### Apple Silicon (M4 Mac) Setup
The scripts are optimized for M4 Macs with automatic MPS (Metal Performance Shaders) acceleration.

## 🔄 Typical Workflow

1. **Test EPUB extraction** (if using EPUB files):
   ```bash
   python epub_preview.py book.epub --extract
   ```

2. **Find optimal voice settings**:
   ```bash
   python voice_pitch_tuner.py --interactive
   ```

3. **Generate audiobook**:
   ```bash
   python audiobook_tts.py book.epub --exaggeration 0.3 --cfg-weight 0.4
   ```

4. **If interrupted, resume**:
   ```bash
   python audiobook_tts.py book.epub  # Automatically resumes
   ```

## 📁 Directory Structure

After running scripts, you'll have:
```
your_project/
├── book/                    # Audiobook output
│   ├── chunk_0001.wav      # Individual segments
│   ├── chunk_0002.wav
│   ├── progress.json       # Resume data
│   └── audiobook.wav       # Final combined book
├── pitch_tests/            # Voice tuning samples
├── voice_library/          # Extracted voice samples
└── extracted_voices/       # Harvested voice segments
```

## ⚡ Performance Tips

- **M4 Mac:** Uses MPS acceleration automatically
- **Parallel Processing:** Use `--workers 2-4` for faster generation
- **Optimal Chunk Size:** Default 200 characters works best
- **Memory Management:** Scripts automatically clear GPU cache
- **Resume Feature:** Large books can be processed in sessions with `--limit-minutes`

## 🎯 Voice Quality Guidelines

**For Best Results:**
- Use 5-15 second voice samples
- Choose clear, single-speaker audio
- Avoid background music or noise
- Test voice settings with short samples first

**Parameter Optimization:**
- Start with interactive tuner to find your preferred sound
- Lower `exaggeration` values = higher, lighter voices
- Lower `cfg_weight` values = more expressive, variable voices
- Use `pitch_shift` for fine-tuning after generation

## 🆘 Troubleshooting

**EPUB Issues:**
- Try `epub_preview.py` first to debug extraction
- Use `simple_epub_reader.py` for problematic files
- Convert to `.txt` manually if needed

**Voice Quality:**
- Use `voice_pitch_tuner.py` to find optimal settings
- Test with short samples before full books
- Check voice sample quality with `voice_harvester.py`

**Performance:**
- Ensure you're using MPS acceleration (shows in logs)
- Reduce `--workers` if experiencing memory issues
- Use `--limit-minutes` for very large books

## 📋 Examples

```bash
# Complete workflow with custom voice
python voice_harvester.py --extract narrator_audiobook.mp3
python voice_pitch_tuner.py --voice extracted_voices/voice_segment_003.wav --interactive
python audiobook_tts.py novel.epub --voice extracted_voices/voice_segment_003.wav --exaggeration 0.4

# Quick audiobook with pitch adjustment
python audiobook_tts.py book.txt --pitch-shift -1 --limit-minutes 30

# Debug problematic EPUB
python epub_preview.py problematic.epub --save clean_text.txt
python audiobook_tts.py clean_text.txt
```

## ⚖️ Legal Notice

Only use voice samples from:
- ✅ Public domain content (LibriVox)
- ✅ Your own recordings
- ✅ Content you have permission to use
- ❌ Avoid copyrighted material without permission

---

*Optimized for Apple Silicon Macs with Chatterbox TTS*