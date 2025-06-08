#!/usr/bin/env python3
"""
Text Processing Module
Contains TextProcessor and TextChunker classes for handling different text file formats
and intelligently chunking text for optimal TTS processing
"""

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List


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