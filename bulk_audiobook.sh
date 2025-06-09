#!/bin/bash

# Bulk Audiobook Generator
# Processes a list of URLs and generates audiobooks

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <url_list.txt>"
    echo "Example: $0 urls.txt"
    exit 1
fi

URL_FILE="$1"
BASE_NAME=$(basename "$URL_FILE" .txt)
OUTPUT_DIR="$BASE_NAME"

if [ ! -f "$URL_FILE" ]; then
    echo "Error: File '$URL_FILE' not found"
    exit 1
fi

echo "🚀 Starting bulk audiobook generation"
echo "📁 Input file: $URL_FILE"
echo "📂 Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Process each URL
while IFS= read -r url || [ -n "$url" ]; do
    # Skip empty lines and comments
    [[ -z "$url" || "$url" =~ ^[[:space:]]*# ]] && continue
    
    echo ""
    echo "🌐 Processing: $url"
    
    # Extract content and generate audiobook
    if python html_extractor.py "$url"; then
        # Find the generated text file (domain_filename.txt pattern)
        domain=$(echo "$url" | sed -E 's|https?://||; s|www\.||; s|/.*||')
        txt_file=$(ls "${domain}"*.txt 2>/dev/null | head -1)
        
        if [ -n "$txt_file" ] && [ -f "$txt_file" ]; then
            echo "📖 Generating audiobook from: $txt_file"
            python audiobook_tts.py "$txt_file" --mp3 --split-minutes 10
            
            # Copy all MP3 files to output directory
            if ls *.mp3 1> /dev/null 2>&1; then
                echo "📦 Copying MP3 files to $OUTPUT_DIR/"
                cp *.mp3 "$OUTPUT_DIR/"
                rm *.mp3
            fi
            
            # Cleanup
            rm -f "$txt_file"
            rm -rf "${txt_file%.*}"/
            
            echo "✅ Completed: $url"
        else
            echo "❌ Could not find extracted text file for: $url"
        fi
    else
        echo "❌ Failed to extract content from: $url"
    fi
    
done < "$URL_FILE"

echo ""
echo "🎉 Bulk processing complete!"
echo "📂 All audiobooks saved to: $OUTPUT_DIR/"
echo "📊 Total files: $(ls -1 "$OUTPUT_DIR"/*.mp3 2>/dev/null | wc -l)"