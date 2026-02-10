# Intelligent Q&A System with Memory and Advanced RAG

A production-ready question-answering system built with **LangChain** (no LangGraph), **FastAPI**, and **React**. Demonstrates all core LangChain concepts with comprehensive explanations.

## 🎯 Features

### Advanced RAG Patterns
- **Basic RAG**: Retrieval-augmented generation with LCEL
- **Conversational RAG**: History-aware retrieval with query reformulation
- **Adaptive RAG**: Intelligent routing (RAG vs direct answer) using `RunnableBranch`
- **RAG with Sources**: Citation tracking and source attribution
- **Hybrid Search**: Ensemble retriever combining vector search (semantic) + BM25 (keyword)

### LangChain Core Components
- **Prompts**: `ChatPromptTemplate`, `MessagesPlaceholder`, few-shot prompting
- **Models**: OpenAI + Ollama support with fallback strategy
- **Output Parsers**: `StrOutputParser`, `PydanticOutputParser`, `with_structured_output()`
- **Document Loaders**: PDF, text, HTML with metadata enrichment
- **Text Splitters**: `RecursiveCharacterTextSplitter` with chunking strategies
- **Embeddings**: OpenAI embeddings with file-based caching
- **Vector Stores**: FAISS with similarity search and MMR
- **Retrievers**: Vector, multi-query, BM25, ensemble
- **Tools**: Custom tools with `@tool` decorator and `StructuredTool`

### Production Features
- **Authentication**: JWT-based auth with email verification
- **Database**: Async SQLAlchemy with SQLite (easily swap to PostgreSQL)
- **API**: FastAPI with automatic OpenAPI docs
- **Memory**: Conversation history with automatic management
- **Error Handling**: Comprehensive error handling and fallbacks
- **Caching**: Embedding caching to reduce costs

## 📋 Requirements

- Python 3.10+
- Node.js 18+ (for React frontend)
- OpenAI API key (or Ollama for local models)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd intelligent-qa-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
nano .env
```

Required environment variables:
```env
OPENAI_API_KEY=your_openai_api_key_here
SECRET_KEY=your_secret_key_here  # Generate with: openssl rand -hex 32
```

### 3. Run the Backend

```bash
# Initialize database
python -c "from app.database import init_db; import asyncio; asyncio.run(init_db())"

# Start FastAPI server
python app/main.py
```

API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 4. Test the API

```bash
# Upload a document
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@path/to/your/document.pdf"

# Ask a question
curl -X POST "http://localhost:8000/qa/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

## 📚 Project Structure

```
intelligent-qa-system/
├── app/
│   ├── auth/                 # Authentication (JWT, security)
│   │   ├── router.py        # Auth endpoints
│   │   ├── schemas.py       # Pydantic models
│   │   └── security.py      # Password hashing, JWT
│   ├── chains/              # LCEL chains (future)
│   ├── database/            # Database models and config
│   │   ├── database.py      # Async SQLAlchemy setup
│   │   └── models.py        # User, Conversation, Message, Document
│   ├── models/              # LLM configuration
│   │   └── llm_config.py    # OpenAI, Ollama, fallback
│   ├── parsers/             # Output parsers
│   │   └── output_parsers.py # Str, Pydantic, structured output
│   ├── prompts/             # Prompt templates
│   │   └── templates.py     # RAG, routing, summarization prompts
│   ├── rag/                 # RAG components
│   │   ├── loaders.py       # Document loaders
│   │   ├── splitters.py     # Text splitters
│   │   ├── embeddings.py    # Embeddings with caching
│   │   ├── vector_store.py  # FAISS vector store
│   │   ├── retrievers.py    # Advanced retrievers
│   │   ├── basic_rag.py     # Basic RAG chain
│   │   ├── conversational_rag.py  # Conversational RAG
│   │   └── adaptive_rag.py  # Adaptive RAG with routing
│   ├── routes/              # FastAPI routers
│   │   ├── qa_router.py     # Q&A endpoints
│   │   └── documents_router.py  # Document upload
│   ├── tools/               # Custom tools
│   │   └── custom_tools.py  # @tool decorator, StructuredTool
│   ├── config.py            # Centralized configuration
│   └── main.py              # FastAPI application
├── tests/                   # Unit and integration tests
├── uploads/                 # Uploaded documents
├── .env.example             # Environment variables template
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🧠 LangChain Concepts Demonstrated

### 1. LCEL (LangChain Expression Language)

```python
# Pipe operator for chain composition
chain = (
    RunnableParallel({
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    })
    | prompt
    | llm
    | StrOutputParser()
)
```

### 2. Prompts

```python
# ChatPromptTemplate with MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])
```

### 3. Conditional Routing

```python
# RunnableBranch for adaptive RAG
adaptive_chain = RunnableBranch(
    (lambda x: route_decision(x) == "RAG", rag_chain),
    direct_answer_chain
)
```

### 4. Structured Output

```python
# Modern approach with with_structured_output()
class Answer(BaseModel):
    answer: str
    confidence: float

structured_llm = llm.with_structured_output(Answer)
```

### 5. Hybrid Search

```python
# Ensemble retriever (vector + BM25)
ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)
```

## 🎓 Educational Value

Every component includes:
- **Comprehensive comments** explaining the "why"
- **Real-world examples** with use cases
- **Best practices** and common pitfalls
- **Performance tips** and optimization strategies
- **Production considerations**

Example from `app/rag/basic_rag.py`:
```python
"""
RAG (Retrieval-Augmented Generation) is a pattern that combines:
1. **Retrieval**: Find relevant documents from a knowledge base
2. **Augmentation**: Add retrieved docs to the prompt
3. **Generation**: LLM generates answer based on retrieved context

Benefits:
- Grounds responses in factual data
- Reduces hallucinations
- Enables domain-specific knowledge
- Can cite sources
- Updatable knowledge (just update documents)
"""
```

## 🔧 Configuration

All configuration is managed through environment variables (`.env`):

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Ollama (optional, for local models)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Database
DATABASE_URL=sqlite+aiosqlite:///./qa_system.db

# JWT
SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Vector Store
VECTOR_STORE_PATH=./vector_store
EMBEDDINGS_CACHE_PATH=./embeddings_cache

# LLM Settings
DEFAULT_MODEL=gpt-3.5-turbo
DEFAULT_TEMPERATURE=0.3
DEFAULT_MAX_TOKENS=1000
```

## 📊 API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login and get JWT token
- `POST /auth/refresh` - Refresh access token
- `POST /auth/verify-email` - Verify email address
- `POST /auth/forgot-password` - Request password reset
- `POST /auth/reset-password` - Reset password

### Q&A
- `POST /qa/ask` - Ask a question
- `GET /qa/conversations` - List conversations
- `GET /qa/conversations/{id}` - Get conversation history
- `DELETE /qa/conversations/{id}` - Delete conversation

### Documents
- `POST /documents/upload` - Upload document
- `GET /documents/` - List documents
- `DELETE /documents/{id}` - Delete document

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app tests/

# Run specific test
pytest tests/test_rag.py -v
```

## 🚀 Deployment

### Docker (Recommended)

```bash
# Build image
docker build -t intelligent-qa-system .

# Run container
docker run -p 8000:8000 --env-file .env intelligent-qa-system
```

### Manual Deployment

1. **Set up production database** (PostgreSQL recommended)
2. **Configure environment variables**
3. **Run migrations**: `alembic upgrade head`
4. **Start with gunicorn**: `gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker`

## 🎯 Performance

- **Response time**: < 5 seconds (target)
- **Embedding caching**: Reduces costs by 80%+
- **Async operations**: Better concurrency
- **Fallback strategy**: OpenAI → Ollama for resilience

## 🔒 Security

- **Password hashing**: BCrypt
- **JWT tokens**: Secure, stateless authentication
- **Input validation**: Pydantic models
- **SQL injection protection**: SQLAlchemy ORM
- **CORS**: Configured for frontend

## 📖 Learn More

- [LangChain Documentation](https://python.langchain.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LCEL Guide](https://python.langchain.com/docs/expression_language/)

## 🤝 Contributing

This is an educational project. Feel free to:
- Add new RAG patterns
- Improve documentation
- Add more LangChain features
- Optimize performance

## 📝 License

MIT License - feel free to use for learning and production!

## 🙏 Acknowledgments

Built to demonstrate LangChain concepts without LangGraph, focusing on:
- LCEL (LangChain Expression Language)
- Runnables and chains
- Core primitives
- Production-ready patterns

---

**Note**: This project is designed for education and demonstrates LangChain concepts comprehensively. Every component includes detailed explanations of the "why" behind design decisions.
