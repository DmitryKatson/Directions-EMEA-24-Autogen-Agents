"""
Pre-generate text extraction and page images for the default Business Central API PDF.
Run this once (e.g. before a demo) so that indexing uses the cache and skips slow parsing/rendering.

Usage (from repo root or demo4/embedding):
  python -m demo4.embedding.prepare_default_cache
  # or
  cd demo4/embedding && python prepare_default_cache.py
"""
import os
import pickle
import sys

# Ensure we can import from embedding dir
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from chunking import extract_text_from_pdf, chunk_by_chapters

DEFAULT_PDF_BASENAME = "Business_Central_API_v2.0_Reference_2nd_Edition.pdf"
CACHE_SUBDIR = "cache"
CACHE_KEY = "Business_Central_API_v2.0_Reference_2nd_Edition"  # PDF name without extension


def get_cache_dir():
    return os.path.join(SCRIPT_DIR, CACHE_SUBDIR, CACHE_KEY)


def prepare_cache(pdf_path=None):
    if pdf_path is None:
        pdf_path = os.path.join(SCRIPT_DIR, DEFAULT_PDF_BASENAME)
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"Default PDF not found: {pdf_path}")

    cache_dir = get_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)

    # Collect page images during extraction
    page_images = []  # list of (page_num, image_bytes)

    def collect_images(step, total, description, **kwargs):
        page_image_bytes = kwargs.get("page_image_bytes")
        page_num = kwargs.get("page_num")
        if page_image_bytes and page_num:
            page_images.append((page_num, page_image_bytes))
        if total and step % 50 == 0:
            print(f"  {description}")

    print("Extracting text and rendering page images...")
    full_text = extract_text_from_pdf(pdf_path, progress_callback=collect_images)
    print(f"Extracted {len(full_text)} chars, {len(page_images)} page images.")

    print("Chunking by chapters...")
    chapter_chunks = chunk_by_chapters(full_text)
    print(f"Got {len(chapter_chunks)} chapters.")

    # Save chapter chunks (dict: chapter_title -> content)
    chunks_path = os.path.join(cache_dir, "chapter_chunks.pkl")
    with open(chunks_path, "wb") as f:
        pickle.dump(chapter_chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved chapter chunks to {chunks_path}")

    # Save page images (list of (page_num, bytes))
    images_path = os.path.join(cache_dir, "page_images.pkl")
    with open(images_path, "wb") as f:
        pickle.dump(page_images, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(page_images)} page images to {images_path}")

    return cache_dir


if __name__ == "__main__":
    try:
        cache_dir = prepare_cache()
        print(f"Cache ready at: {cache_dir}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
