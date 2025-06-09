#!/bin/bash

# Bulk Audiobook Generator
# Processes a list of URLs and generates audiobooks

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <url_list.txt> [audiobook_tts_options...]"
    echo "Example: $0 urls.txt"
    echo "Example: $0 urls.txt --exaggeration 0.6 --cfg-weight 0.8 --workers 2 --voice extracted_voices/voice_segment_012.wav"
    exit 1
fi

URL_FILE="$1"
shift  # Remove first argument, rest are TTS options
TTS_OPTIONS="$@"
BASE_NAME=$(basename "$URL_FILE" .txt)
OUTPUT_DIR="$BASE_NAME"

if [ ! -f "$URL_FILE" ]; then
    echo "Error: File '$URL_FILE' not found"
    exit 1
fi

echo "Starting bulk audiobook generation"
echo "Input file: $URL_FILE"
echo "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process each URL
while IFS= read -r url || [ -n "$url" ]; do
    # Skip empty lines and comments
    [[ -z "$url" || "$url" =~ ^[[:space:]]*# ]] && continue
    
    echo ""
    echo "Processing: $url"
    
    # Extract content and generate audiobook
    if python html_extractor.py "$url"; then
        # Find the generated text file (domain_filename.txt pattern)
        domain=$(echo "$url" | sed -E 's|https?://||; s|www\.||; s|/.*||')
        txt_file=$(ls "${domain}"*.txt 2>/dev/null | head -1)
        
        if [ -n "$txt_file" ] && [ -f "$txt_file" ]; then
            echo "Generating audiobook from: $txt_file"
            echo "TTS Options: $TTS_OPTIONS"
            
            # Get base filename without extension for matching generated files
            base_name=$(basename "$txt_file" .txt)
            
            python audiobook_tts.py "$txt_file" --mp3 --split-minutes 10 $TTS_OPTIONS
            
            # Copy MP3 files from the generated audiobook directory
            audiobook_dir="${base_name}"
            if [ -d "$audiobook_dir" ] && ls "$audiobook_dir"/*.mp3 1> /dev/null 2>&1; then
                echo "Copying MP3 files from $audiobook_dir/ to $OUTPUT_DIR/"
                cp "$audiobook_dir"/*.mp3 "$OUTPUT_DIR/"
                echo "Found $(ls -1 "$audiobook_dir"/*.mp3 | wc -l) MP3 files"
            else
                echo "No MP3 files found in $audiobook_dir/ directory"
            fi
            
            # Cleanup text file only (keep audiobook directory with WAV files)
            rm -f "$txt_file"
            
            echo "Completed: $url"
        else
            echo "Could not find extracted text file for: $url"
        fi
    else
        echo "Failed to extract content from: $url"
    fi
    
done < "$URL_FILE"

echo ""
echo "Bulk processing complete!"
echo "All audiobooks saved to: $OUTPUT_DIR/"
echo "Total files: $(ls -1 "$OUTPUT_DIR"/*.mp3 2>/dev/null | wc -l)"