#!/usr/bin/env python3
"""
Example Usage Script

This script demonstrates how to use the Intelligent Q&A System components.
Run this to test the RAG chains without starting the full API.
"""

import asyncio
from pathlib import Path

# Import components
from app.rag.loaders import load_pdf, load_text_file
from app.rag.splitters import split_documents_with_metadata
from app.rag.vector_store import create_vector_store, save_vector_store, load_vector_store, vector_store_exists
from app.rag.basic_rag import create_basic_rag_chain
from app.rag.conversational_rag import ConversationalRAGWithMemory
from app.rag.adaptive_rag import create_adaptive_rag_chain
from app.models.llm_config import get_chat_model_with_fallback


async def main():
    """
    Demonstrate the Q&A system components.
    """
    print("🚀 Intelligent Q&A System - Example Usage\n")
    
    # ========================================================================
    # Step 1: Load Documents
    # ========================================================================
    print("📄 Step 1: Loading documents...")
    
    # Create a sample document
    sample_doc_path = Path("sample_document.txt")
    if not sample_doc_path.exists():
        sample_content = """
        Company Refund Policy
        
        Our company offers a 30-day money-back guarantee on all products.
        
        To request a refund:
        1. Contact customer support within 30 days of purchase
        2. Provide your order number
        3. Explain the reason for the refund
        
        International orders may take 5-7 business days for refund processing.
        Domestic orders are processed within 2-3 business days.
        
        For technical support, email support@example.com or call 1-800-SUPPORT.
        """
        sample_doc_path.write_text(sample_content)
        print(f"✅ Created sample document: {sample_doc_path}")
    
    # Load the document
    docs = load_text_file(str(sample_doc_path))
    print(f"✅ Loaded {len(docs)} document(s)")
    
    # ========================================================================
    # Step 2: Split into Chunks
    # ========================================================================
    print("\n✂️  Step 2: Splitting into chunks...")
    
    chunks = split_documents_with_metadata(docs, chunk_size=200, chunk_overlap=50)
    print(f"✅ Created {len(chunks)} chunks")
    
    # ========================================================================
    # Step 3: Create Vector Store
    # ========================================================================
    print("\n🗄️  Step 3: Creating vector store...")
    
    if not vector_store_exists():
        vector_store = create_vector_store(chunks)
        save_vector_store(vector_store)
        print("✅ Vector store created and saved")
    else:
        vector_store = load_vector_store()
        print("✅ Vector store loaded from disk")
    
    # ========================================================================
    # Step 4: Test Basic RAG
    # ========================================================================
    print("\n🤖 Step 4: Testing Basic RAG...")
    
    basic_chain = create_basic_rag_chain(vector_store)
    
    question1 = "What is the refund policy?"
    print(f"\nQ: {question1}")
    answer1 = basic_chain.invoke({"question": question1})
    print(f"A: {answer1}\n")
    
    # ========================================================================
    # Step 5: Test Conversational RAG
    # ========================================================================
    print("💬 Step 5: Testing Conversational RAG...")
    
    conv_rag = ConversationalRAGWithMemory(vector_store)
    
    # First question
    question2 = "How long does it take to get a refund?"
    print(f"\nQ: {question2}")
    answer2 = conv_rag.invoke(question2)
    print(f"A: {answer2}\n")
    
    # Follow-up question (uses history)
    question3 = "What about for international orders?"
    print(f"Q: {question3}")
    answer3 = conv_rag.invoke(question3)
    print(f"A: {answer3}\n")
    
    # ========================================================================
    # Step 6: Test Adaptive RAG
    # ========================================================================
    print("🎯 Step 6: Testing Adaptive RAG...")
    
    adaptive_chain = create_adaptive_rag_chain(vector_store)
    
    # General knowledge question (should use direct answer)
    question4 = "What is Python?"
    print(f"\nQ: {question4}")
    answer4 = adaptive_chain.invoke({"question": question4})
    print(f"A: {answer4}\n")
    
    # Company-specific question (should use RAG)
    question5 = "How do I contact support?"
    print(f"Q: {question5}")
    answer5 = adaptive_chain.invoke({"question": question5})
    print(f"A: {answer5}\n")
    
    # ========================================================================
    # Done
    # ========================================================================
    print("✅ All examples completed successfully!")
    print("\n📚 Next steps:")
    print("1. Start the API: python app/main.py")
    print("2. Visit docs: http://localhost:8000/docs")
    print("3. Upload your own documents via /documents/upload")
    print("4. Ask questions via /qa/ask")


if __name__ == "__main__":
    asyncio.run(main())
