# Audiobook TTS Generator Suite

A comprehensive collection of Python scripts for converting text files and ebooks into high-quality audiobooks using Chatterbox TTS. Optimized for Apple Silicon with voice cloning, pitch control, parallel processing, automatic MP3 conversion, and time-limited file splitting.

## 🎯 Quick Start

```bash
# Basic audiobook generation (creates 5-minute MP3 files by default)
python audiobook_tts.py your_book.txt --mp3

# With custom voice, pitch adjustment, and 3-minute file segments
python audiobook_tts.py book.epub --voice voices/narrator.wav --exaggeration 0.3 --pitch-shift +1 --mp3 --split-minutes 3
```

## 📚 Scripts Overview

### 📖 `audiobook_tts.py` - Main Audiobook Generator
Converts text files and ebooks into complete audiobooks with professional features, automatic MP3 conversion, and time-limited file splitting.

**Features:**
- Supports `.txt`, `.epub`, `.fb2`, `.md` files
- **🎯 Smart sentence-aware text chunking** - Ensures complete sentences in each WAV file
- **🔍 Advanced audio validation** - Automatically detects and retries problematic audio chunks
- **✂️ Intelligent sentence-aware MP3 splitting** - Prevents mid-sentence cuts using silence detection
- **📊 Text-length-relative performance monitoring** - Accurate detection of processing issues
- Parallel processing for faster generation
- Resume capability (picks up where you left off)
- Voice cloning support
- Time limits and progress tracking
- Pitch and voice characteristic control
- **🎵 Automatic MP3 conversion with FFmpeg**
- **📱 Configurable bitrates for quality vs file size**
- **⏰ Time-limited file splitting (5 minutes default)**
- **📝 Proper file naming convention: `<filename>_001.mp3`, `<filename>_002.mp3`**
- **🧹 Optional WAV file cleanup to save disk space**
- **🏷️ MP3 metadata tagging with author and book title**
- **⏱️ Real-time ETA calculation and progress percentage**
- **🧠 Optimized memory management for Apple Silicon**
- **🔕 Clean output with suppressed dependency warnings**

**Usage:**
```bash
# Basic usage (creates 5-minute MP3 segments)
python audiobook_tts.py book.txt --mp3

# Custom segment length
python audiobook_tts.py book.txt --mp3 --split-minutes 3

# Single file output (disable splitting)
python audiobook_tts.py book.txt --mp3 --split-minutes 0

# Advanced usage with custom settings and smart splitting
python audiobook_tts.py novel.epub \
  --voice voices/narrator.wav \
  --limit-minutes 60 \
  --exaggeration 0.5 \
  --cfg-weight 0.7 \
  --pitch-shift -1.5 \
  --workers 3 \
  --mp3 \
  --mp3-bitrate 192k \
  --split-minutes 7 \
  --remove-wav

# Disable smart splitting for compatibility with older workflows
python audiobook_tts.py book.txt --mp3 --disable-smart-split

# High-quality MP3 with 10-minute segments and metadata
python audiobook_tts.py book.txt --mp3 --mp3-bitrate 320k --split-minutes 10 --tag "Author Name - Book Title"

# Space-saving mobile version with 3-minute segments and metadata
python audiobook_tts.py book.txt --mp3 --mp3-bitrate 96k --split-minutes 3 --remove-wav --tag "Arthur Conan Doyle - The Adventures of Sherlock Holmes"
```

**File Splitting Options:**
- `--split-minutes X`: Split into X-minute files (default: 5, set to 0 to disable)
- **🎯 Smart sentence-aware splitting**: Automatically detects sentence boundaries to prevent mid-sentence cuts
- `--disable-smart-split`: Use traditional time-based splitting for compatibility
- Creates files with naming convention: `<filename>_001.mp3`, `<filename>_002.mp3`, etc.
- File extension is removed from base filename automatically
- Works with both MP3 and WAV output formats

**MP3 Conversion Options:**
- `--mp3`: Enable MP3 conversion using FFmpeg
- `--mp3-bitrate`: Choose quality (`64k`, `96k`, `128k`, `160k`, `192k`, `256k`, `320k`)
- `--remove-wav`: Delete WAV files after MP3 conversion to save space
- `--tag "Author - Title"`: Add MP3 metadata with author and book title

**Output Examples:**

*With splitting enabled (default):*
```
book/
├── book_001.mp3    # First 5 minutes
├── book_002.mp3    # Next 5 minutes
├── book_003.mp3    # Next 5 minutes
└── ...
```

*With splitting disabled (`--split-minutes 0`):*
```
book/
├── chunk_0001.wav  # Individual processing chunks
├── chunk_0002.wav
├── audiobook.wav   # Full combined audiobook
└── book.mp3        # Final MP3 (if --mp3 used)
```

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
python audiobook_tts.py book_extracted.txt --mp3 --split-minutes 5
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

### 🔍 `bad_ending_detector.py` - TTS Ending Quality Analyzer
Standalone tool for detecting problematic TTS audio endings with artifacts, distorted sound effects, or unnatural audio patterns.

**Features:**
- Analyzes the final 5 seconds of audio files for problematic patterns
- Detects rising energy trends that indicate TTS artifacts
- Provides confidence scoring and detailed metrics
- Batch processing for analyzing multiple files
- Visualization output with energy trend graphs
- Comprehensive pattern classification

**Usage:**
```bash
# Analyze a single audio file
python bad_ending_detector.py chunk_0001.wav

# Analyze with visualization output
python bad_ending_detector.py chunk_0001.wav --visualize

# Batch analyze all WAV files in a directory
python bad_ending_detector.py --batch ./audiobook_output/

# Analyze last 3 seconds instead of default 5
python bad_ending_detector.py chunk_0001.wav --duration 3.0
```

**Detection Patterns:**
- **Problematic Rising Trend**: Energy increases and stays elevated at the end
- **Sustained Elevation**: Audio doesn't fade naturally after speech
- **Normal Fade**: Proper audio ending with natural energy decay

**Integration with Main System:**
The detection algorithm from this tool is automatically integrated into `audiobook_tts.py` for real-time quality control during generation. Use this standalone tool for manual analysis or debugging specific audio files.

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

---

### 🚀 `bulk_audiobook.sh` - Bulk URL Processing Script
Processes multiple URLs from a text file and generates audiobooks with organized output.

**Features:**
- Batch process multiple URLs from a single input file  
- Extracts content using `html_extractor.py`
- Generates audiobooks with MP3 conversion and splitting
- Organizes all output files into named directory structure
- Automatic cleanup between processing cycles
- Progress tracking and error reporting

**Usage:**
```bash
# Make script executable (first time only)
chmod +x bulk_audiobook.sh

# Process URLs from file
./bulk_audiobook.sh urls.txt
```

**Input File Format (urls.txt):**
```
https://example.com/article1.html
https://example.com/article2.html  
# This is a comment - will be skipped
https://news.site.com/story.html
https://blog.example.com/essay.html
```

**Output Structure:**
```
urls/                           # Directory named after input file
├── article1_001.mp3           # First URL's audiobook files
├── article1_002.mp3
├── article2_001.mp3           # Second URL's audiobook files  
├── article2_002.mp3
├── story_001.mp3              # Third URL's audiobook files
└── essay_001.mp3              # Fourth URL's audiobook files
```

**Processing Steps Per URL:**
1. Extract web content to clean text file (`domain_filename.txt`)
2. Generate audiobook with MP3 conversion and 10-minute splitting
3. Copy all MP3 files to organized output directory  
4. Clean up temporary files between URLs
5. Report success/failure for each URL

**Configuration:**
- **Split Duration:** 10 minutes per MP3 file (modify `--split-minutes 10` in script)
- **Output Directory:** Named after input file without extension
- **MP3 Quality:** Uses default 128k bitrate (modify `--mp3-bitrate` in script)
- **Error Handling:** Continues processing remaining URLs if one fails

**Integration Example:**
```bash
# Create URL list
cat > tech_articles.txt << EOF
https://example.com/article1.html
https://blog.site.com/tutorial.html  
https://news.example.com/feature.html
EOF

# Process all articles
./bulk_audiobook.sh tech_articles.txt

# Result: tech_articles/ directory with all audiobook files organized
```

**Customization:**
Edit the script to modify default settings:
- Change `--split-minutes 10` for different segment lengths
- Add `--mp3-bitrate 192k` for higher quality
- Add `--remove-wav` for automatic cleanup
- Add `--tag "Author - Collection"` for metadata

---

### 🌐 `html_extractor.py` - Web Content to Text Converter
Extracts clean, readable text from web pages and local HTML files for audiobook generation.

**Features:**
- Fetch content from remote URLs or local HTML files
- Intelligent text extraction with content filtering
- Removes navigation, ads, and formatting elements
- Smart text cleaning and formatting
- Automatic filename generation
- Robust error handling and logging
- **🔧 Compatible with cron jobs and automation**
- **📱 Mobile-friendly User-Agent headers**
- **🧹 Removes scripts, styles, and non-content elements**

**Usage:**
```bash
# Extract from web page
python html_extractor.py https://example.com/article.html

# Extract from local HTML file
python html_extractor.py local_page.html

# Custom output filename
python html_extractor.py https://news.site.com/story.html -o story.txt

# Verbose logging for debugging
python html_extractor.py https://example.com/page.html --verbose
```

**Integration with Audiobook Generation:**
```bash
# Complete workflow: Web page to audiobook
python html_extractor.py https://example.com/long-article.html -o article.txt
python audiobook_tts.py article.txt --mp3 --split-minutes 5

# Batch process multiple web pages
for url in $(cat urls.txt); do
    python html_extractor.py "$url"
    python audiobook_tts.py "$(basename "$url").txt" --mp3 --split-minutes 3
done
```

**Supported Content Sources:**
- **News articles** - Clean extraction from news websites
- **Blog posts** - Removes sidebars and comments
- **Documentation** - Technical content and guides
- **Wikipedia articles** - Clean text without navigation
- **Local HTML files** - Offline content processing

**Output Quality Features:**
- Preserves paragraph structure and formatting
- Removes excessive whitespace and empty lines
- Maintains readability for TTS conversion
- Handles various character encodings
- Smart cleanup of HTML artifacts

## 🛠️ Installation

### Quick Setup (Recommended)

**Important:** Chatterbox TTS requires Python 3.11 or earlier. Python 3.12+ is not currently supported.

```bash
# 1. Clone the repository
git clone <repository-url>
cd audiobook-chatterbox-tts-scripts

# 2. Create and activate virtual environment with Python 3.11
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Verify Python version (should be 3.11.x)
python --version

# 4. Install all dependencies
pip install -r requirements.txt

# 5. Install FFmpeg for MP3 conversion and file splitting
# See FFmpeg installation guide below

# 6. Test the installation
python audiobook_tts.py --help
```

### Manual Installation

**Prerequisites:** Python 3.11 or earlier (Python 3.12+ not supported by Chatterbox TTS)

```bash
# Verify Python version first
python --version  # Should be 3.11.x or earlier

# Core dependencies (required)
pip install torch>=2.0.0 torchaudio>=2.0.0 chatterbox-tts

# Audio processing
pip install librosa>=0.10.0 sounddevice>=0.4.0 scipy>=1.10.0 numpy>=1.24.0

# Text and file format support
pip install beautifulsoup4>=4.12.0 requests>=2.28.0 ebooklib>=0.18

# System utilities
pip install psutil>=5.9.0

# Or install everything at once:
pip install -r requirements.txt
```

### Python Version Compatibility

- **✅ Supported:** Python 3.8, 3.9, 3.10, 3.11
- **❌ Not Supported:** Python 3.12 and newer
- **Recommended:** Python 3.11 (tested and verified working)

If you have Python 3.12+ installed, install Python 3.11 alongside it:

```bash
# macOS (using Homebrew)
brew install python@3.11

# Ubuntu/Debian
sudo apt install python3.11 python3.11-venv

# Windows
# Download Python 3.11 from python.org and install
```

### FFmpeg Installation for MP3 Conversion and File Splitting

**macOS (using Homebrew):**
```bash
brew install ffmpeg
```

**Windows:**
```bash
# Download from https://ffmpeg.org/download.html
# Or using chocolatey:
choco install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Verification:**
```bash
ffmpeg -version  # Should show FFmpeg version info
```

### Apple Silicon (M4 Mac) Setup
The scripts are optimized for M4 Macs with automatic MPS (Metal Performance Shaders) acceleration, aggressive memory management, and intelligent worker allocation to prevent out-of-memory crashes.

## 🔄 Typical Workflow

1. **Extract or download source content**:
   ```bash
   # From web page
   python html_extractor.py https://example.com/article.html -o article.txt
   
   # Test EPUB extraction (if using EPUB files)
   python epub_preview.py book.epub --extract
   ```

2. **Find optimal voice settings**:
   ```bash
   python voice_pitch_tuner.py --interactive
   ```

3. **Generate audiobook with MP3 conversion and splitting**:
   ```bash
   python audiobook_tts.py book.epub --exaggeration 0.3 --cfg-weight 0.4 --mp3 --mp3-bitrate 160k --split-minutes 5
   ```

4. **If interrupted, resume**:
   ```bash
   python audiobook_tts.py book.epub --mp3 --split-minutes 5  # Automatically resumes and applies splitting
   ```

## 📁 Directory Structure

After running scripts with file splitting enabled, you'll have:
```
your_project/
├── book/                      # Audiobook output directory
│   ├── chunk_0001.wav        # Individual processing segments
│   ├── chunk_0002.wav
│   ├── progress.json         # Resume data
│   ├── audiobook.wav         # Temporary combined file (removed if --remove-wav)
│   ├── book_001.mp3          # Final split files (5 min each)
│   ├── book_002.mp3          # Second segment
│   ├── book_003.mp3          # Third segment
│   └── book_004.mp3          # Final segment
├── pitch_tests/              # Voice tuning samples
├── voice_library/            # Extracted voice samples
└── extracted_voices/         # Harvested voice segments
```

## ⚡ Performance Tips

- **M4 Mac:** Uses MPS acceleration automatically with optimized memory management
- **Parallel Processing:** Uses 1 worker for MPS devices, 2-4 workers for other systems
- **Smart Chunking:** Intelligent sentence boundary detection for optimal processing
- **Audio Quality:** Automatic validation and retry system ensures reliable output
- **Performance Monitoring:** Text-length-aware performance detection prevents false slow warnings
- **Memory Management:** Aggressive GPU cache clearing and memory synchronization
- **Resume Feature:** Large books can be processed in sessions with `--limit-minutes`
- **File Splitting:** Smart sentence-aware splitting happens automatically after audio generation
- **MP3 Conversion:** Optimized for speech content with minimal processing overhead
- **Progress Tracking:** Real-time ETA and completion percentage display
- **Clean Output:** Dependency warnings automatically suppressed for cleaner logs

## 🏷️ MP3 Metadata and Tagging

**Automatic MP3 Metadata Support:**
- Add author and book information to generated MP3 files
- Proper track numbering for split files
- Compatible with media players and podcast apps
- Genre automatically set to "Audiobook"

**Usage:**
```bash
# Basic metadata with author and title
python audiobook_tts.py book.txt --mp3 --tag "Arthur Conan Doyle - The Adventures of Sherlock Holmes"

# Split files get automatic track numbering and titles:
# - Track 1: "The Adventures of Sherlock Holmes - Part 1"
# - Track 2: "The Adventures of Sherlock Holmes - Part 2"
# - etc.

# If no author separator, entire string becomes title
python audiobook_tts.py book.txt --mp3 --tag "Book Title Only"
```

**Metadata Applied:**
- **Artist/Author:** From text before " - " separator
- **Album:** Book title (text after " - " separator or entire string)
- **Title:** Book title for single files, "Book Title - Part N" for split files
- **Track:** Sequential numbering for split files (1, 2, 3, etc.)
- **Genre:** Automatically set to "Audiobook"

**Media Player Benefits:**
- Organized library browsing by author and book
- Proper track progression in playlists
- Audiobook-specific categorization
- Seamless integration with podcast and audiobook apps

## 🎵 File Splitting and MP3 Quality Guidelines

**File Splitting Benefits:**
- **Easier navigation:** Jump to specific chapters or sections
- **Better compatibility:** Some devices have file size limits
- **Convenient sharing:** Share specific portions without splitting manually
- **Resume listening:** Easier to remember where you left off

**Recommended Split Durations:**
- **3-5 minutes:** Optimal for mobile listening and navigation
- **10-15 minutes:** Good for longer listening sessions
- **20+ minutes:** Chapter-like segments for books
- **0 (disabled):** Single file for simple playback

**Bitrate Recommendations:**
- **64k-96k:** Mobile devices, very small files, basic quality
- **128k:** Good quality, standard for audiobooks (default)
- **160k-192k:** High quality, good balance of size vs quality
- **256k-320k:** Excellent quality, larger files, archival purposes

**File Size Comparison (5-minute segment):**
- WAV (original): ~50MB per segment
- MP3 128k: ~5MB per segment (90% smaller)
- MP3 192k: ~7.5MB per segment (85% smaller)
- MP3 320k: ~12.5MB per segment (75% smaller)

**Audio Optimization for Speech:**
- Sample rate: 22kHz (perfect for speech)
- Channels: Mono (smaller files, adequate for audiobooks)
- Compression: Optimized for speech content

## 📊 Progress Tracking and ETA

The audiobook generator now provides detailed progress information during generation:

**Real-time Progress Display:**
```
📊 Progress: 966/2103 (45.9%) completed, 0 failed
⏱️ ETA: 1h 47m (Rate: 2.1 chunks/sec)
```

**Progress Features:**
- **Completion Percentage:** Shows exact progress through the book
- **ETA Calculation:** Estimates time to completion based on current speed
- **Processing Rate:** Displays chunks processed per second
- **Failed Chunk Tracking:** Monitors and reports any processing failures
- **Session Resumption:** Maintains progress across interrupted sessions

**ETA Accuracy:**
- Calculates from actual processing speed over the last 50 chunks
- Updates in real-time as processing speed changes
- Accounts for system performance variations
- More accurate for longer processing sessions

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

**Web Content Issues:**
- Use `html_extractor.py` for web pages and HTML files
- Supports both remote URLs and local HTML files
- Handles various website layouts and content structures
- Check output text quality before audiobook generation

**EPUB Issues:**
- Try `epub_preview.py` first to debug extraction
- Use `simple_epub_reader.py` for problematic files
- Convert to `.txt` manually if needed

**Voice Quality:**
- Use `voice_pitch_tuner.py` to find optimal settings
- Test with short samples before full books
- Check voice sample quality with `voice_harvester.py`
- Analyze problematic audio endings with `bad_ending_detector.py`

**MP3 Conversion and File Splitting Issues:**
- Ensure FFmpeg is installed: `ffmpeg -version`
- Check FFmpeg installation guide above
- Both MP3 conversion and file splitting automatically disabled if FFmpeg not found
- Use `--split-minutes 0` to disable splitting if only MP3 conversion is needed
- Use `--mp3-bitrate` to adjust quality vs file size

**File Splitting Issues:**
- Large files may take longer to split - this is normal
- If splitting fails, original combined file is preserved
- Use `--remove-wav` carefully - only removes files after successful conversion
- Check disk space before processing large books with splitting enabled

**Performance:**
- Ensure you're using MPS acceleration (shows in logs)
- Memory management is automatic for Apple Silicon Macs
- Worker count is automatically optimized (1 for MPS, 2-4 for others)
- Use `--limit-minutes` for very large books
- File splitting adds minimal processing time after audio generation
- ETA calculation provides accurate completion estimates
- Progress percentage shows real-time completion status

## 📋 Examples

### Complete Workflow Examples

```bash
# Web article to audiobook with custom voice and splitting
python html_extractor.py https://example.com/long-article.html -o article.txt
python voice_pitch_tuner.py --voice voices/narrator.wav --interactive
python audiobook_tts.py article.txt \
  --voice voices/narrator.wav \
  --exaggeration 0.4 \
  --mp3 \
  --mp3-bitrate 160k \
  --split-minutes 5 \
  --remove-wav \
  --tag "Author Name - Article Title"

# Complete workflow with custom voice, MP3, metadata, and 3-minute segments
python voice_harvester.py --extract narrator_audiobook.mp3
python voice_pitch_tuner.py --voice extracted_voices/voice_segment_003.wav --interactive
python audiobook_tts.py novel.epub \
  --voice extracted_voices/voice_segment_003.wav \
  --exaggeration 0.4 \
  --mp3 \
  --mp3-bitrate 160k \
  --split-minutes 3 \
  --remove-wav \
  --tag "Author Name - Novel Title"

# Quick audiobook with pitch adjustment and 5-minute MP3 segments
python audiobook_tts.py book.txt --pitch-shift -1 --limit-minutes 30 --mp3 --split-minutes 5

# High-quality archival version with 15-minute segments
python audiobook_tts.py book.txt --mp3 --mp3-bitrate 320k --split-minutes 15

# Mobile-optimized version with 3-minute segments and cleanup
python audiobook_tts.py book.txt --mp3 --mp3-bitrate 96k --split-minutes 3 --remove-wav

# Single file output (no splitting)
python audiobook_tts.py book.txt --mp3 --mp3-bitrate 192k --split-minutes 0

# Debug problematic EPUB with MP3 output and splitting
python epub_preview.py problematic.epub --save clean_text.txt
python audiobook_tts.py clean_text.txt --mp3 --mp3-bitrate 128k --split-minutes 5

# Web content extraction and processing
python html_extractor.py https://news.example.com/article.html --verbose
python audiobook_tts.py news_example_com_article.txt --mp3 --split-minutes 3
```

### File Splitting Scenarios

```bash
# Podcast-style short segments (3 minutes)
python audiobook_tts.py book.txt --mp3 --split-minutes 3

# Chapter-like segments (20 minutes)
python audiobook_tts.py book.txt --mp3 --split-minutes 20

# Traditional audiobook segments (10 minutes)
python audiobook_tts.py book.txt --mp3 --split-minutes 10

# Maximum compatibility (5 minutes, lower bitrate)
python audiobook_tts.py book.txt --mp3 --mp3-bitrate 96k --split-minutes 5
```

### Space-Saving Workflows

```bash
# Generate with immediate cleanup, splitting, and metadata
python audiobook_tts.py book.txt --mp3 --mp3-bitrate 128k --split-minutes 5 --remove-wav --tag "Author Name - Book Title"
# Result: Only numbered MP3 files remain (book_001.mp3, book_002.mp3, etc.) with proper metadata

# Compare file splitting approaches
python audiobook_tts.py book.txt --mp3 --split-minutes 3   # Many small files
python audiobook_tts.py book.txt --mp3 --split-minutes 15  # Fewer large files
python audiobook_tts.py book.txt --mp3 --split-minutes 0   # Single file
```

### Batch Processing with File Splitting

```bash
# Process multiple books with consistent 5-minute splitting and metadata
for book in *.epub; do
    # Extract base name for title
    title=$(basename "$book" .epub)
    python audiobook_tts.py "$book" --mp3 --mp3-bitrate 160k --split-minutes 5 --remove-wav --tag "Unknown Author - $title"
done

# Batch process web articles
while IFS= read -r url; do
    python html_extractor.py "$url"
    filename=$(python -c "from html_extractor import HTMLTextExtractor; print(HTMLTextExtractor().generate_output_filename('$url'))")
    python audiobook_tts.py "$filename" --mp3 --mp3-bitrate 128k --split-minutes 3
done < urls.txt

# Create different versions for different use cases with metadata
python audiobook_tts.py book.txt --mp3 --mp3-bitrate 96k --split-minutes 3 --tag "Author - Book Title"    # Mobile
python audiobook_tts.py book.txt --mp3 --mp3-bitrate 192k --split-minutes 10 --tag "Author - Book Title"  # Desktop
python audiobook_tts.py book.txt --mp3 --mp3-bitrate 320k --split-minutes 0 --tag "Author - Book Title"   # Archive
```

## 🎛️ Advanced File Management

**Naming Convention Benefits:**
- Files are automatically named: `<filename>_001.mp3`, `<filename>_002.mp3`
- File extension is removed from base name (e.g., `book.txt` becomes `book_001.mp3`)
- Sequential numbering ensures proper playlist order
- Compatible with most media players and podcast apps

**File Organization Tips:**
```bash
# Output structure for "My Novel.epub" with 5-minute segments:
My Novel/
├── My Novel_001.mp3  # Minutes 0-5
├── My Novel_002.mp3  # Minutes 5-10
├── My Novel_003.mp3  # Minutes 10-15
└── ...

# Use descriptive folder organization:
mkdir -p "Audiobooks/$(basename "$book" .epub)"
python audiobook_tts.py "$book" --mp3 --split-minutes 5
```

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

## ⚖️ Legal Notice

Only use voice samples from:
- ✅ Public domain content (LibriVox)
- ✅ Your own recordings
- ✅ Content you have permission to use
- ❌ Avoid copyrighted material without permission

**Web Content Usage:**
- ✅ Public domain articles and documentation
- ✅ Content you own or have permission to use
- ✅ Fair use educational content (check local laws)
- ❌ Copyrighted articles without permission
- ❌ Paywalled content circumvention

---

*Optimized for Apple Silicon Macs with Chatterbox TTS, FFmpeg MP3 conversion, intelligent file splitting*