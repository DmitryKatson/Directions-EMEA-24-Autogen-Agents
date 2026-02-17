import re
import PyPDF2
import os
# from tqdm import tqdm # No longer using tqdm directly here

def extract_text_from_pdf(file_path, progress_callback=None):
    """Extracts text from PDF, calling progress_callback if provided."""
    # Try to import PyMuPDF for rendering page images (optional)
    try:
        import pymupdf
        pymupdf_available = True
        # Open PDF with PyMuPDF for rendering (keep open for all pages)
        pdf_doc = pymupdf.open(file_path)
    except ImportError:
        pymupdf_available = False
        pdf_doc = None
    
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        num_pages = len(reader.pages)
        if progress_callback:
            progress_callback(0, num_pages, f"Reading {os.path.basename(file_path)}...")
        
        # Replace tqdm loop with manual loop and callback calls
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            # Clean page headers/footers
            page_text = clean_page_headers_and_footers(page_text)
            text += page_text + "\n"  # Add new lines between pages to avoid collisions
            
            # Render page to image if PyMuPDF is available
            page_image_bytes = None
            if progress_callback and pymupdf_available and pdf_doc:
                try:
                    # Render current page to image
                    pymupdf_page = pdf_doc[i]
                    # Use moderate DPI (150) to keep size reasonable while remaining readable
                    pix = pymupdf_page.get_pixmap(dpi=150)
                    page_image_bytes = pix.tobytes("png")
                except Exception as e:
                    print(f"Warning: Could not render page {i+1} to image: {e}")
                    page_image_bytes = None
            
            if progress_callback:
                progress_callback(i + 1, num_pages, f"Extracting text from PDF page {i+1}/{num_pages}", 
                                page_image_bytes=page_image_bytes, page_num=i+1)
        
        print("-" * 50)
    
    # Close PyMuPDF document if it was opened
    if pdf_doc:
        pdf_doc.close()
    
    return text

def clean_page_headers_and_footers(text):
    """
    Removes repetitive headers/footers that appear on each page, like 'PART', 'CHAPTER', and page numbers.
    """
    # Remove common header/footer patterns like "PART", "PAGE", and unwanted whitespace
    text = re.sub(r'PART \d+:\s+[A-Za-z ]+|PAGE \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s{2,}', ' ', text)  # Remove excessive spaces
    return text

def chunk_by_chapters(text, progress_callback=None):
    """Chunks text by chapters, calling progress_callback if provided."""
    # Regex pattern to match "Chapter X: Title"
    chapter_pattern = re.compile(r'(CHAPTER\s\d+:\s+)', re.IGNORECASE)

    # Find all chapter headings and their positions
    chapters = list(chapter_pattern.finditer(text))

    chapter_chunks = {}
    num_chapters = len(chapters)

    # If there are chapters, process them
    if chapters:
        if progress_callback:
             progress_callback(0, num_chapters, "Starting chapter chunking...")
             
        # Replace tqdm loop with manual loop and callback calls
        for i in range(num_chapters):
            match = chapters[i]
            chapter_title = match.group(0).strip().rstrip(':')  # Get the chapter title (e.g., "CHAPTER 1")
            start_pos = match.end()  # Position where the chapter content starts

            # Determine where this chapter ends:
            if i + 1 < num_chapters:
                end_pos = chapters[i + 1].start()  # The next chapter's start position
            else:
                end_pos = len(text)  # If this is the last chapter, go to the end of the document

            # Extract the content between the current chapter and the next chapter
            chapter_content = text[start_pos:end_pos].strip()

            # Include the chapter title in the content
            full_content = chapter_title + "\n" + chapter_content

            # Save the chapter title and content in the dictionary
            chapter_chunks[chapter_title] = full_content
            if progress_callback:
                progress_callback(i + 1, num_chapters, f"Chunking chapter {i+1}/{num_chapters}")

    print("-" * 50)
    return chapter_chunks

def process_pdf(file_path, progress_callback=None):
    """Processes PDF, passing progress_callback to sub-functions."""
    file_name = os.path.basename(file_path)
    print(f"Reading {file_name} ... 📚")    
    print("-" * 50)

    # Pass callback to extract_text_from_pdf
    full_text = extract_text_from_pdf(file_path, progress_callback=progress_callback)
    
    print("Okay, now I'll prepare to remember this! 🧠💪")
    print("Chunking into chapters...")
    print("-" * 50)

    # Pass callback to chunk_by_chapters
    return chunk_by_chapters(full_text, progress_callback=progress_callback)

if __name__ == "__main__":
    # Example progress callback for standalone execution
    def print_progress(step, total, description):
        percent = (step / total) * 100 if total > 0 else 0
        print(f"PROGRESS: {description} - {step}/{total} ({percent:.2f}%)")
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_file_path = os.path.join(script_dir, "Business_Central_API_v2.0_Reference_2nd_Edition.pdf")
    
    chapter_chunks = process_pdf(pdf_file_path, progress_callback=print_progress)

    # Display the chapters and their content
    for title, content in chapter_chunks.items():
        print(f"Title: {title}")
