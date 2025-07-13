# scripts/convert_pdf_to_md.py

import pdfplumber
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def convert_pdf_to_markdown(pdf_path: str, output_path: str):
    logger.info(f"Starting conversion of {pdf_path}")
    markdown_output = ""
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            logger.info(f"PDF opened successfully. Total pages: {len(pdf.pages)}")
            
            for page_num, page in enumerate(pdf.pages, 1):
                logger.debug(f"Processing page {page_num}")
                text = page.extract_text()
                
                if text:
                    logger.debug(f"Page {page_num}: Extracted {len(text)} characters")
                    logger.debug(f"Page {page_num} text preview: {text[:100]}...")
                    markdown_output += text + "\n\n"
                else:
                    logger.warning(f"Page {page_num}: No text extracted")
    
    except Exception as e:
        logger.error(f"Error opening PDF {pdf_path}: {e}")
        return
    
    if markdown_output.strip():
        try:
            # Ensure UTF-8 encoding when writing the file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_output)
            logger.info(f"Successfully converted {pdf_path} to {output_path}")
            logger.info(f"Total characters written: {len(markdown_output)}")
        except Exception as e:
            logger.error(f"Error writing to {output_path}: {e}")
    else:
        logger.warning(f"No content extracted from {pdf_path}")

if __name__ == "__main__":
    logger.info("Starting PDF to Markdown conversion process")
    
    input_dir = Path("data")  # Relative to project root
    output_dir = Path("Outputs/markdown")  # Relative to project root

    logger.info(f"Input directory: {input_dir.absolute()}")
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    # Check if input directory exists
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir.absolute()}")
        # Let's check what's in the current directory
        current_dir = Path(".")
        logger.info(f"Current directory: {current_dir.absolute()}")
        logger.info(f"Current directory contents: {list(current_dir.iterdir())}")
        exit(1)
    
    # Create output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory created/verified: {output_dir.absolute()}")
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        exit(1)

    # Find PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {input_dir}")
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    if not pdf_files:
        logger.warning("No PDF files found in input directory")
        exit(0)
    
    for pdf_file in pdf_files:
        logger.info(f"Processing: {pdf_file.name}")
        output_file = output_dir / (pdf_file.stem + ".md")
        convert_pdf_to_markdown(str(pdf_file), str(output_file))
    
    logger.info("PDF conversion process completed")
