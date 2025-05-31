#!/usr/bin/env python3
"""
Simple EPUB Reader - Alternative approach using zipfile
Works even when ebooklib has issues
"""

import zipfile
import sys
from pathlib import Path
import re
from bs4 import BeautifulSoup

def read_epub_simple(epub_path):
    """Read EPUB using basic zipfile approach"""
    
    print(f"📚 Reading EPUB: {epub_path}")
    
    try:
        with zipfile.ZipFile(epub_path, 'r') as zip_file:
            print(f"📁 Files in EPUB:")
            
            # List all files
            file_list = zip_file.namelist()
            
            html_files = []
            for filename in file_list:
                print(f"   {filename}")
                
                # Look for content files
                if filename.endswith(('.xhtml', '.html', '.htm')):
                    # Skip navigation files
                    if not any(skip in filename.lower() for skip in ['nav', 'toc', 'contents', 'ncx']):
                        html_files.append(filename)
            
            print(f"\n📄 Found {len(html_files)} content files:")
            for f in html_files:
                print(f"   {f}")
            
            # Extract text from content files
            all_text = []
            
            for html_file in html_files:
                print(f"\n🔍 Processing: {html_file}")
                
                try:
                    # Read file content
                    content = zip_file.read(html_file)
                    
                    # Try to decode
                    try:
                        text_content = content.decode('utf-8')
                    except UnicodeDecodeError:
                        text_content = content.decode('utf-8', errors='ignore')
                    
                    print(f"   Raw content sample: {text_content[:200]}...")
                    
                    # Parse HTML
                    soup = BeautifulSoup(text_content, 'html.parser')
                    
                    # Remove scripts, styles, etc.
                    for tag in soup(['script', 'style', 'meta', 'link', 'head']):
                        tag.decompose()
                    
                    # Get text
                    text = soup.get_text(separator=' ', strip=True)
                    
                    # Clean text
                    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
                    text = re.sub(r'\*\*\*+', '', text)  # Remove star separators
                    text = re.sub(r'---+', '', text)   # Remove dash separators
                    
                    if text and len(text.strip()) > 100:
                        all_text.append(text.strip())
                        print(f"   ✅ Extracted {len(text)} characters")
                        print(f"   Preview: {text[:300]}...")
                    else:
                        print(f"   ❌ No substantial text found")
                
                except Exception as e:
                    print(f"   ❌ Error processing {html_file}: {e}")
            
            # Combine all text
            if all_text:
                final_text = '\n\n'.join(all_text)
                print(f"\n📊 Total extracted: {len(final_text):,} characters")
                
                # Save to file
                output_file = Path(epub_path).stem + "_extracted.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(final_text)
                
                print(f"💾 Saved to: {output_file}")
                
                # Show first part
                print(f"\n📖 First 1000 characters:")
                print(final_text[:1000])
                print("...")
                
                return final_text
            else:
                print("❌ No text could be extracted!")
                return ""
    
    except Exception as e:
        print(f"❌ Error reading EPUB: {e}")
        import traceback
        traceback.print_exc()
        return ""

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python simple_epub_reader.py book.epub")
        sys.exit(1)
    
    epub_file = sys.argv[1]
    if not Path(epub_file).exists():
        print(f"File not found: {epub_file}")
        sys.exit(1)
    
    read_epub_simple(epub_file)