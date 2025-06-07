#!/usr/bin/env python3
"""
HTML to Text Extractor
Extracts clean text content from remote URLs or local HTML files
and saves it as formatted .txt files.
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

try:
    from bs4 import BeautifulSoup
    import requests
except ImportError:
    print("Required packages not installed. Install with:")
    print("pip install beautifulsoup4 requests")
    sys.exit(1)

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class HTMLTextExtractor:
    """Extract clean text from HTML content."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def fetch_content(self, source):
        """
        Fetch HTML content from URL or local file.
        
        Args:
            source (str): URL or local file path
            
        Returns:
            str: HTML content
        """
        if self.is_url(source):
            return self.fetch_from_url(source)
        else:
            return self.fetch_from_file(source)
    
    def is_url(self, source):
        """Check if source is a URL."""
        try:
            result = urlparse(source)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def fetch_from_url(self, url):
        """Fetch HTML content from remote URL."""
        logger.info(f"Fetching content from URL: {url}")
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            logger.info(f"Successfully fetched {len(response.content)} bytes from {url}")
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch URL {url}: {e}")
            raise
    
    def fetch_from_file(self, file_path):
        """Read HTML content from local file."""
        logger.info(f"Reading content from local file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            logger.info(f"Successfully read {len(content)} characters from {file_path}")
            return content
        except IOError as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            raise
    
    def extract_text(self, html_content):
        """
        Extract clean text from HTML content.
        
        Args:
            html_content (str): Raw HTML content
            
        Returns:
            str: Clean, formatted text
        """
        logger.info("Parsing HTML and extracting text content")
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up the text
        text = self.clean_text(text)
        
        logger.info(f"Extracted {len(text)} characters of clean text")
        return text
    
    def clean_text(self, text):
        """
        Clean and format extracted text.
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Cleaned and formatted text
        """
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Reduce multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Replace multiple spaces/tabs with single space
        text = re.sub(r' *\n *', '\n', text)  # Remove spaces around newlines
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        
        # Remove empty lines at the beginning and end
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        
        # Join lines back together
        text = '\n'.join(lines)
        
        return text
    
    def generate_output_filename(self, source):
        """
        Generate output filename based on source.
        
        Args:
            source (str): URL or file path
            
        Returns:
            str: Output filename
        """
        if self.is_url(source):
            # Extract domain and path for URL
            parsed = urlparse(source)
            domain = parsed.netloc.replace('www.', '')
            path = parsed.path.strip('/')
            
            if path:
                # Use last part of path
                filename = path.split('/')[-1]
                if '.' in filename:
                    filename = filename.rsplit('.', 1)[0]
            else:
                filename = domain
            
            # Clean filename
            filename = re.sub(r'[^\w\-_.]', '_', filename)
            return f"{domain}_{filename}.txt"
        else:
            # Use local file name
            path = Path(source)
            return f"{path.stem}.txt"
    
    def save_text(self, text, output_file):
        """
        Save text content to file.
        
        Args:
            text (str): Text content to save
            output_file (str): Output file path
        """
        logger.info(f"Saving text content to: {output_file}")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Successfully saved {len(text)} characters to {output_file}")
        except IOError as e:
            logger.error(f"Failed to save file {output_file}: {e}")
            raise
    
    def process(self, source, output_file=None):
        """
        Main processing function.
        
        Args:
            source (str): URL or local file path
            output_file (str, optional): Output file path
            
        Returns:
            str: Path to output file
        """
        try:
            # Fetch HTML content
            html_content = self.fetch_content(source)
            
            # Extract text
            text = self.extract_text(html_content)
            
            # Generate output filename if not provided
            if not output_file:
                output_file = self.generate_output_filename(source)
            
            # Save text to file
            self.save_text(text, output_file)
            
            return output_file
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Extract clean text content from HTML sources',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://example.com/page.html
  %(prog)s local_file.html
  %(prog)s https://example.com/page.html -o output.txt
  %(prog)s local_file.html --output custom_output.txt
        """
    )
    
    parser.add_argument(
        'source',
        help='URL or local HTML file path'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output text file path (auto-generated if not specified)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize extractor
    extractor = HTMLTextExtractor()
    
    try:
        # Process the source
        output_file = extractor.process(args.source, args.output)
        
        print(f"Success! Text extracted and saved to: {output_file}")
        logger.info("Processing completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Process failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()