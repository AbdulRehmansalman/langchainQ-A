# Intelligent Q&A System - Quick Start Guide

## 🚀 Quick Setup (5 minutes)

### Prerequisites
- Python 3.8+
- Node.js 16+
- OpenAI API key

### One-Command Setup
```bash
./setup.sh
```

This will:
- Create Python virtual environment
- Install all dependencies (backend + frontend)
- Create `.env` file
- Set up directories

### Manual Setup

#### Backend
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

#### Frontend
```bash
cd frontend
npm install
```

---

## 🎯 Running the Application

### Start Backend
```bash
source venv/bin/activate
python app/main.py
```
Backend runs on: http://localhost:8000

### Start Frontend (in new terminal)
```bash
cd frontend
npm run dev
```
Frontend runs on: http://localhost:3000

---

## 📚 Quick Tour

### 1. Chat Interface
- Ask questions about your documents
- View conversation history
- See source citations
- Create new conversations

### 2. Document Upload
- Drag and drop PDF, TXT, or MD files
- Automatic indexing into vector store
- View uploaded documents
- Delete documents

### 3. Evaluation Dashboard
- View system performance metrics
- Understand retrieval quality (Recall, Precision, MRR)
- Monitor generation quality (Faithfulness, Relevance, Utilization)
- Learn about evaluation best practices

---

## 🔑 API Endpoints

### Q&A
- `POST /qa/ask` - Ask a question
- `GET /qa/conversations` - List conversations
- `GET /qa/conversations/{id}` - Get conversation history
- `DELETE /qa/conversations/{id}` - Delete conversation

### Documents
- `POST /documents/upload` - Upload document
- `GET /documents/` - List documents
- `DELETE /documents/{id}` - Delete document

### Evaluation
- `POST /evaluation/retrieval` - Evaluate retrieval
- `POST /evaluation/generation` - Evaluate generation
- `POST /evaluation/rag` - Full RAG evaluation
- `GET /evaluation/metrics` - Get system metrics

**Full API docs:** http://localhost:8000/docs

---

## 📖 Documentation

- **[README.md](README.md)** - Full project documentation
- **[Evaluation Guide](../brain/e791c240-dc0b-45ca-9cab-32c062502f0b/evaluation_guide.md)** - Comprehensive metrics guide
- **[Walkthrough](../brain/e791c240-dc0b-45ca-9cab-32c062502f0b/walkthrough.md)** - Complete feature walkthrough
- **[Example Usage](example_usage.py)** - Standalone examples

---

## 🎓 Learning Resources

### LangChain Concepts Demonstrated
- ✅ Prompts (ChatPromptTemplate, MessagesPlaceholder)
- ✅ Models (OpenAI, Ollama, fallbacks)
- ✅ Output Parsers (Pydantic, structured output)
- ✅ Document Loaders (PDF, text, HTML)
- ✅ Text Splitters (recursive, token-based)
- ✅ Embeddings (OpenAI with caching)
- ✅ Vector Stores (FAISS)
- ✅ Retrievers (vector, multi-query, BM25, ensemble)
- ✅ RAG Patterns (basic, conversational, adaptive)
- ✅ Tools (custom tools, retriever-as-tool)
- ✅ LCEL (pipe operator, RunnableParallel, RunnableBranch)

### RAG Patterns
- **Basic RAG** - Simple retrieval + generation
- **Conversational RAG** - With chat history
- **Adaptive RAG** - Intelligent routing (RAG vs direct)

### Evaluation Metrics
- **Retrieval:** Recall@K, Precision@K, MRR
- **Generation:** Faithfulness, Answer Relevance, Context Utilization
- **Human:** 5-point scale evaluation

---

## 🛠️ Troubleshooting

### Backend won't start
- Check Python version: `python3 --version` (need 3.8+)
- Activate venv: `source venv/bin/activate`
- Check `.env` has `OPENAI_API_KEY`

### Frontend won't start
- Check Node version: `node --version` (need 16+)
- Run `npm install` in frontend directory
- Check backend is running on port 8000

### No documents retrieved
- Upload documents first via Documents tab
- Wait for indexing to complete
- Check vector store path in `.env`

### Evaluation metrics not working
- Requires OpenAI API key
- Check API key has sufficient credits
- Metrics use GPT for evaluation

---

## 💡 Tips

1. **Start Simple**: Upload a small document and ask basic questions
2. **Try Different RAG Patterns**: Compare basic vs adaptive RAG
3. **Monitor Metrics**: Check evaluation dashboard regularly
4. **Experiment**: Try different chunk sizes, K values, etc.
5. **Read the Code**: Heavily commented for learning

---

## 🎉 You're Ready!

The system is fully functional and ready to use. Start by:
1. Uploading a document
2. Asking questions in the chat
3. Viewing metrics in the evaluation dashboard

**Happy learning!** 🚀
