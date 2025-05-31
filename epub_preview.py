#!/usr/bin/env python3
"""
EPUB Preview Tool
Debug and preview what text will be extracted from EPUB files
"""

import argparse
import logging
from pathlib import Path
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def preview_epub_structure(epub_path: Path):
    """Show the internal structure of an EPUB file"""
    try:
        import ebooklib
        from ebooklib import epub
        
        book = epub.read_epub(str(epub_path))
        
        # Show metadata - fix the unhashable dict error
        print("\n📖 EPUB Metadata:")
        try:
            # Try different ways to get metadata
            title = book.get_metadata('DC', 'title')
            if title:
                print(f"   Title: {title[0][0] if title and len(title[0]) > 0 else 'Unknown'}")
            
            creator = book.get_metadata('DC', 'creator')
            if creator:
                print(f"   Author: {creator[0][0] if creator and len(creator[0]) > 0 else 'Unknown'}")
            
            language = book.get_metadata('DC', 'language')
            if language:
                print(f"   Language: {language[0][0] if language and len(language[0]) > 0 else 'Unknown'}")
            
            # Try to get all metadata safely
            try:
                all_metadata = book.metadata
                if all_metadata:
                    for namespace, items in all_metadata.items():
                        if items:  # Only show if there are items
                            print(f"   Namespace {namespace}: {len(items)} items")
            except Exception as meta_error:
                print(f"   Metadata extraction error: {meta_error}")
        
        except Exception as e:
            print(f"   Metadata error: {e}")
            print("   Continuing without metadata...")
        
        print(f"\n📁 EPUB Contents:")
        
        chapter_count = 0
        nav_count = 0
        image_count = 0
        style_count = 0
        
        try:
            items = list(book.get_items())
            print(f"   Total items found: {len(items)}")
            
            for item in items:
                try:
                    item_type = item.get_type()
                    item_name = item.get_name()
                    
                    if item_type == ebooklib.ITEM_DOCUMENT:
                        # Check if it's navigation
                        if any(nav_word in item_name.lower() for nav_word in ['nav', 'toc', 'contents', 'index', 'cover']):
                            nav_count += 1
                            print(f"   📋 NAV: {item_name}")
                        else:
                            chapter_count += 1
                            # Get content size safely
                            try:
                                content = item.get_content()
                                content_size = len(content) if content else 0
                                print(f"   📄 CHAPTER: {item_name} ({content_size} bytes)")
                            except Exception as content_error:
                                print(f"   📄 CHAPTER: {item_name} (error: {content_error})")
                    
                    elif item_type == ebooklib.ITEM_IMAGE:
                        image_count += 1
                        print(f"   🖼️  IMAGE: {item_name}")
                    elif item_type == ebooklib.ITEM_STYLE:
                        style_count += 1
                        print(f"   🎨 STYLE: {item_name}")
                    else:
                        print(f"   📦 OTHER: {item_name} ({item_type})")
                
                except Exception as item_error:
                    print(f"   ❌ Error processing item: {item_error}")
        
        except Exception as items_error:
            print(f"   Error getting items: {items_error}")
        
        print(f"\n📊 Summary: {chapter_count} chapters, {nav_count} navigation files, {image_count} images, {style_count} styles")
        
        if chapter_count == 0:
            print("⚠️ WARNING: No chapters found! This EPUB might have an unusual structure.")
        
    except ImportError:
        print("❌ ebooklib not installed. Install with: pip install ebooklib beautifulsoup4")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading EPUB: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)

def extract_epub_text(epub_path: Path, preview_length: int = 500):
    """Extract and preview text from EPUB"""
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
        import re
        
        book = epub.read_epub(str(epub_path))
        
        print(f"\n🔍 Text Extraction Preview (first {preview_length} chars per chapter):")
        
        chapter_count = 0
        total_chars = 0
        
        try:
            items = list(book.get_items())
            document_items = [item for item in items if item.get_type() == ebooklib.ITEM_DOCUMENT]
            print(f"Found {len(document_items)} document items to process")
            
            for item in document_items:
                try:
                    item_name = item.get_name()
                    
                    # Skip navigation files
                    if any(skip in item_name.lower() for skip in ['nav', 'toc', 'contents', 'index', 'cover']):
                        print(f"\n⏭️ Skipping: {item_name}")
                        continue
                    
                    print(f"\n📝 Processing: {item_name}")
                    
                    try:
                        content = item.get_content()
                        if not content:
                            print("❌ No content found")
                            continue
                        
                        # Try to decode content
                        if isinstance(content, bytes):
                            try:
                                content_str = content.decode('utf-8')
                            except UnicodeDecodeError:
                                content_str = content.decode('utf-8', errors='ignore')
                        else:
                            content_str = str(content)
                        
                        soup = BeautifulSoup(content_str, 'html.parser')
                        
                        # Show raw HTML sample
                        print(f"Raw HTML sample: {content_str[:200]}...")
                        
                        # Remove unwanted elements
                        for script in soup(["script", "style", "meta", "link"]):
                            script.decompose()
                        
                        # Remove navigation elements
                        for nav in soup.find_all(['nav', 'aside']):
                            nav.decompose()
                        
                        # Extract text
                        text = soup.get_text(separator=' ', strip=True)
                        
                        if text and len(text.strip()) > 50:
                            # Clean text
                            cleaned_text = clean_epub_text(text)
                            
                            if cleaned_text and len(cleaned_text.strip()) > 50:
                                chapter_count += 1
                                total_chars += len(cleaned_text)
                                
                                print(f"📖 Cleaned text ({len(cleaned_text)} chars):")
                                print(f"   {cleaned_text[:preview_length]}...")
                                
                                if len(cleaned_text) > preview_length:
                                    print(f"   ... (and {len(cleaned_text) - preview_length} more characters)")
                            else:
                                print("❌ No usable text after cleaning")
                        else:
                            print("❌ No substantial text found")
                    
                    except Exception as content_error:
                        print(f"❌ Content processing error: {content_error}")
                        import traceback
                        traceback.print_exc()
                
                except Exception as item_error:
                    print(f"❌ Item processing error: {item_error}")
        
        except Exception as items_error:
            print(f"❌ Error getting items: {items_error}")
            import traceback
            traceback.print_exc()
        
        print(f"\n📊 Extraction Summary:")
        print(f"   Chapters processed: {chapter_count}")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Average per chapter: {total_chars // chapter_count if chapter_count > 0 else 0:,}")
        
        if total_chars < 1000:
            print("\n⚠️ WARNING: Very little text extracted. This might indicate:")
            print("   - EPUB has unusual structure")
            print("   - Text is in images (scanned book)")
            print("   - Heavy DRM protection")
            print("   - Malformed EPUB file")
            print("\n💡 Try using calibre to convert EPUB to TXT first:")
            print("   ebook-convert book.epub book.txt")
        
    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        import traceback
        traceback.print_exc()

def clean_epub_text(text: str) -> str:
    """Clean text extracted from EPUB (same as main script)"""
    import re
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common ebook artifacts
    text = re.sub(r'\*\*\*+', '', text)
    text = re.sub(r'---+', '', text)
    text = re.sub(r'___+', '', text)
    
    # Remove page numbers and chapter markers
    text = re.sub(r'\bPage \d+\b', '', text, flags=re.I)
    text = re.sub(r'\bChapter \d+\b(?!\w)', '', text, flags=re.I)
    
    # Remove navigation text
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
    
    # Remove standalone numbers
    text = re.sub(r'\b\d+\b(?=\s|$)', '', text)
    
    # Clean punctuation spacing
    text = re.sub(r'\s+([.!?,:;])', r'\1', text)
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    
    # Remove empty lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)
    
    return text.strip()

def save_extracted_text(epub_path: Path, output_path: Path):
    """Extract and save clean text to file"""
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
        import re
        
        book = epub.read_epub(str(epub_path))
        text_parts = []
        
        print(f"🔄 Extracting text from {epub_path}...")
        
        chapter_count = 0
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                item_name = item.get_name()
                
                if any(skip in item_name.lower() for skip in ['nav', 'toc', 'contents', 'index', 'cover']):
                    continue
                
                try:
                    content = item.get_content()
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Remove unwanted elements
                    for script in soup(["script", "style", "meta", "link"]):
                        script.decompose()
                    
                    for nav in soup.find_all(['nav', 'aside'], class_=re.compile(r'nav|toc|sidebar', re.I)):
                        nav.decompose()
                    
                    text = soup.get_text(separator=' ', strip=True)
                    
                    if text and len(text.strip()) > 50:
                        cleaned_text = clean_epub_text(text)
                        
                        if cleaned_text and len(cleaned_text.strip()) > 50:
                            text_parts.append(f"# Chapter {chapter_count + 1}\n\n{cleaned_text}")
                            chapter_count += 1
                            print(f"✅ Extracted chapter {chapter_count}")
                
                except Exception as e:
                    print(f"⚠️ Error processing {item_name}: {e}")
        
        if text_parts:
            final_text = '\n\n\n'.join(text_parts)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_text)
            
            print(f"💾 Saved clean text to: {output_path}")
            print(f"📊 {len(final_text):,} characters from {chapter_count} chapters")
        else:
            print("❌ No text could be extracted!")
    
    except Exception as e:
        print(f"❌ Error saving text: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Preview and debug EPUB text extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python epub_preview.py book.epub --structure
  python epub_preview.py book.epub --extract --preview 1000
  python epub_preview.py book.epub --save book_clean.txt
        """
    )
    
    parser.add_argument("epub_file", help="EPUB file to analyze")
    
    parser.add_argument(
        "--structure",
        action="store_true",
        help="Show EPUB internal structure"
    )
    
    parser.add_argument(
        "--extract",
        action="store_true", 
        help="Extract and preview text"
    )
    
    parser.add_argument(
        "--preview",
        type=int,
        default=500,
        help="Number of characters to preview per chapter (default: 500)"
    )
    
    parser.add_argument(
        "--save",
        help="Save extracted clean text to file"
    )
    
    args = parser.parse_args()
    
    epub_path = Path(args.epub_file)
    if not epub_path.exists():
        print(f"❌ EPUB file not found: {epub_path}")
        sys.exit(1)
    
    print(f"🔍 Analyzing EPUB: {epub_path}")
    
    if args.structure or (not args.extract and not args.save):
        preview_epub_structure(epub_path)
    
    if args.extract:
        extract_epub_text(epub_path, args.preview)
    
    if args.save:
        save_extracted_text(epub_path, Path(args.save))

if __name__ == "__main__":
    main()