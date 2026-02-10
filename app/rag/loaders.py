"""
Document Loaders

This module demonstrates LangChain's document loading capabilities.
Document loaders extract text and metadata from various file formats.

Key concepts:
- PyPDFLoader: Load PDF documents
- TextLoader: Load plain text files
- WebBaseLoader: Load web pages
- DirectoryLoader: Batch load multiple files
- Metadata extraction and enrichment
"""

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    UnstructuredHTMLLoader,
)
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
from pathlib import Path
import mimetypes


# ============================================================================
# PDF Document Loader
# ============================================================================

def load_pdf(file_path: str, extract_images: bool = False) -> List[Document]:
    """
    Load PDF document with metadata.
    
    PyPDFLoader extracts text page by page, preserving page numbers.
    
    Args:
        file_path: Path to PDF file
        extract_images: Whether to extract images (not implemented in basic loader)
        
    Returns:
        List of Document objects (one per page)
        
    Example:
        docs = load_pdf("manual.pdf")
        for doc in docs:
            print(f"Page {doc.metadata['page']}: {doc.page_content[:100]}")
    
    Metadata included:
        - source: File path
        - page: Page number
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Enrich metadata
    for doc in documents:
        doc.metadata["file_type"] = "pdf"
        doc.metadata["source"] = file_path
    
    return documents


# ============================================================================
# Text File Loader
# ============================================================================

def load_text_file(file_path: str, encoding: str = "utf-8") -> List[Document]:
    """
    Load plain text file.
    
    Args:
        file_path: Path to text file
        encoding: File encoding (default: utf-8)
        
    Returns:
        List with single Document object
        
    Example:
        docs = load_text_file("notes.txt")
        print(docs[0].page_content)
    """
    loader = TextLoader(file_path, encoding=encoding)
    documents = loader.load()
    
    # Enrich metadata
    for doc in documents:
        doc.metadata["file_type"] = "txt"
        doc.metadata["source"] = file_path
    
    return documents


# ============================================================================
# HTML/Web Loader
# ============================================================================

def load_html_file(file_path: str) -> List[Document]:
    """
    Load HTML file.
    
    Extracts text content from HTML, removing tags.
    
    Args:
        file_path: Path to HTML file
        
    Returns:
        List with single Document object
        
    Example:
        docs = load_html_file("page.html")
    """
    loader = UnstructuredHTMLLoader(file_path)
    documents = loader.load()
    
    # Enrich metadata
    for doc in documents:
        doc.metadata["file_type"] = "html"
        doc.metadata["source"] = file_path
    
    return documents


# ============================================================================
# Directory Loader (Batch Loading)
# ============================================================================

def load_directory(
    directory_path: str,
    glob_pattern: str = "**/*",
    show_progress: bool = True,
    use_multithreading: bool = True,
) -> List[Document]:
    """
    Load all documents from a directory.
    
    Automatically detects file types and uses appropriate loaders.
    
    Args:
        directory_path: Path to directory
        glob_pattern: Pattern to match files (default: all files)
        show_progress: Show progress bar
        use_multithreading: Use multiple threads for faster loading
        
    Returns:
        List of all loaded documents
        
    Example:
        # Load all PDFs
        docs = load_directory("./documents", glob_pattern="**/*.pdf")
        
        # Load all text files
        docs = load_directory("./documents", glob_pattern="**/*.txt")
        
        # Load everything
        docs = load_directory("./documents")
    """
    # For PDFs
    if "*.pdf" in glob_pattern or glob_pattern == "**/*":
        loader = DirectoryLoader(
            directory_path,
            glob=glob_pattern if "*.pdf" in glob_pattern else "**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=show_progress,
            use_multithreading=use_multithreading,
        )
        return loader.load()
    
    # For text files
    elif "*.txt" in glob_pattern:
        loader = DirectoryLoader(
            directory_path,
            glob=glob_pattern,
            loader_cls=TextLoader,
            show_progress=show_progress,
            use_multithreading=use_multithreading,
        )
        return loader.load()
    
    # Default: try to load all supported files
    else:
        all_docs = []
        
        # Load PDFs
        try:
            pdf_loader = DirectoryLoader(
                directory_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=show_progress,
                use_multithreading=use_multithreading,
            )
            all_docs.extend(pdf_loader.load())
        except Exception as e:
            print(f"Error loading PDFs: {e}")
        
        # Load text files
        try:
            txt_loader = DirectoryLoader(
                directory_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=show_progress,
                use_multithreading=use_multithreading,
            )
            all_docs.extend(txt_loader.load())
        except Exception as e:
            print(f"Error loading text files: {e}")
        
        return all_docs


# ============================================================================
# Smart Document Loader (Auto-detect type)
# ============================================================================

def load_document(file_path: str) -> List[Document]:
    """
    Load document with automatic type detection.
    
    Detects file type and uses appropriate loader.
    
    Args:
        file_path: Path to document
        
    Returns:
        List of Document objects
        
    Example:
        docs = load_document("unknown_file.pdf")  # Auto-detects PDF
        docs = load_document("data.txt")  # Auto-detects text
    """
    # Get file extension
    file_path_obj = Path(file_path)
    extension = file_path_obj.suffix.lower()
    
    # Detect MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    
    # Route to appropriate loader
    if extension == ".pdf" or mime_type == "application/pdf":
        return load_pdf(file_path)
    elif extension == ".txt" or mime_type == "text/plain":
        return load_text_file(file_path)
    elif extension in [".html", ".htm"] or mime_type == "text/html":
        return load_html_file(file_path)
    else:
        # Try text loader as fallback
        try:
            return load_text_file(file_path)
        except Exception as e:
            raise ValueError(f"Unsupported file type: {extension}. Error: {e}")


# ============================================================================
# Metadata Enrichment
# ============================================================================

def enrich_document_metadata(
    documents: List[Document],
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Add additional metadata to documents.
    
    Useful for adding context like:
    - Upload timestamp
    - User ID
    - Document category
    - Custom tags
    
    Args:
        documents: List of documents
        additional_metadata: Dict of metadata to add
        
    Returns:
        Documents with enriched metadata
        
    Example:
        docs = load_pdf("manual.pdf")
        enriched = enrich_document_metadata(
            docs,
            {
                "uploaded_by": "user123",
                "category": "technical",
                "tags": ["manual", "reference"]
            }
        )
    """
    if not additional_metadata:
        return documents
    
    for doc in documents:
        doc.metadata.update(additional_metadata)
    
    return documents


# ============================================================================
# Educational Note: Document Loading Best Practices
# ============================================================================
"""
**Document Loading Concepts:**

1. **Document Structure**:
   - page_content: The actual text content
   - metadata: Dictionary of metadata (source, page, etc.)
   
2. **Metadata Importance**:
   - Enables source citation
   - Allows filtering during retrieval
   - Provides context for chunks
   - Essential for production systems

3. **Batch Loading**:
   - Use DirectoryLoader for multiple files
   - Enable multithreading for speed
   - Show progress for user feedback
   - Handle errors gracefully

4. **File Type Detection**:
   - Use mimetypes for reliable detection
   - Fall back to extension checking
   - Provide clear error messages
   - Support common formats

**Best Practices:**

1. **Always preserve metadata**:
   - Source file path
   - Page numbers (for PDFs)
   - Upload timestamp
   - User/category info

2. **Handle errors**:
   - Some files may be corrupted
   - Encoding issues are common
   - Don't let one file break the whole batch

3. **Enrich metadata early**:
   - Add metadata at load time
   - Easier than adding later
   - Enables better retrieval

4. **Consider file size**:
   - Large PDFs may need streaming
   - Split very large documents
   - Monitor memory usage

**Common Issues:**

1. **Encoding errors**:
   - Try different encodings (utf-8, latin-1, cp1252)
   - Use error handling in TextLoader

2. **PDF extraction quality**:
   - Some PDFs have poor text extraction
   - Scanned PDFs need OCR (not included here)
   - Tables may not extract well

3. **HTML cleaning**:
   - HTML may have lots of boilerplate
   - Consider using BeautifulSoup for better control
   - Remove scripts, styles, navigation

**Performance Tips:**

1. **Use multithreading**:
   - Significantly faster for multiple files
   - Safe for I/O-bound operations

2. **Cache loaded documents**:
   - Don't reload unchanged files
   - Store in database or file system

3. **Lazy loading**:
   - Load documents on-demand
   - Don't load all at startup
"""
