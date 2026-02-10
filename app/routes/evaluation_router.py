"""
Evaluation Router

API endpoints for RAG evaluation metrics.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.database import get_db
from app.evaluation.metrics import (
    evaluate_retrieval,
    evaluate_generation,
    evaluate_rag_system,
    RetrievalMetrics,
    GenerationMetrics,
    RAGEvaluationResult,
    HumanEvaluationCriteria,
    normalize_human_evaluation
)
from app.rag.vector_store import load_vector_store
from langchain_core.documents import Document


router = APIRouter(prefix="/evaluation", tags=["evaluation"])


# ============================================================================
# Request/Response Models
# ============================================================================

class EvaluateRetrievalRequest(BaseModel):
    """Request for retrieval evaluation"""
    question: str = Field(description="User's question")
    relevant_doc_ids: List[str] = Field(description="IDs of relevant documents")
    k: int = Field(default=5, description="K for Recall@K and Precision@K")


class EvaluateGenerationRequest(BaseModel):
    """Request for generation evaluation"""
    question: str = Field(description="User's question")
    answer: str = Field(description="Generated answer")
    context: str = Field(description="Retrieved context")


class EvaluateRAGRequest(BaseModel):
    """Request for full RAG evaluation"""
    question: str = Field(description="User's question")
    answer: str = Field(description="Generated answer")
    context: str = Field(description="Retrieved context")
    relevant_doc_ids: List[str] = Field(description="IDs of relevant documents")
    k: int = Field(default=5, description="K for retrieval metrics")


class HumanEvaluationRequest(BaseModel):
    """Request for human evaluation"""
    question_id: int = Field(description="ID of the question being evaluated")
    criteria: HumanEvaluationCriteria = Field(description="Human evaluation criteria")


class SystemMetricsResponse(BaseModel):
    """System-wide metrics response"""
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    overall_score: float
    total_evaluations: int


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/retrieval", response_model=RetrievalMetrics)
async def evaluate_retrieval_endpoint(
    request: EvaluateRetrievalRequest
):
    """
    Evaluate retrieval performance.
    
    Calculates Recall@K, Precision@K, and MRR.
    
    Example:
        POST /evaluation/retrieval
        {
            "question": "What is the refund policy?",
            "relevant_doc_ids": ["doc1", "doc3", "doc5"],
            "k": 5
        }
    """
    try:
        # Load vector store and retrieve documents
        vector_store = load_vector_store()
        retrieved_docs = vector_store.similarity_search(request.question, k=request.k)
        
        # Evaluate
        metrics = evaluate_retrieval(
            retrieved_docs=retrieved_docs,
            relevant_doc_ids=request.relevant_doc_ids,
            k=request.k
        )
        
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generation", response_model=GenerationMetrics)
async def evaluate_generation_endpoint(
    request: EvaluateGenerationRequest
):
    """
    Evaluate generation quality.
    
    Calculates Faithfulness, Answer Relevance, and Context Utilization.
    
    Example:
        POST /evaluation/generation
        {
            "question": "What is the refund policy?",
            "answer": "Our refund policy is 30 days.",
            "context": "Company policy: 30-day money-back guarantee."
        }
    """
    try:
        metrics = evaluate_generation(
            question=request.question,
            answer=request.answer,
            context=request.context
        )
        
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag", response_model=RAGEvaluationResult)
async def evaluate_rag_endpoint(
    request: EvaluateRAGRequest
):
    """
    Comprehensive RAG evaluation.
    
    Evaluates both retrieval and generation.
    
    Example:
        POST /evaluation/rag
        {
            "question": "What is the refund policy?",
            "answer": "Our refund policy is 30 days.",
            "context": "Company policy: 30-day money-back guarantee.",
            "relevant_doc_ids": ["doc1"],
            "k": 5
        }
    """
    try:
        # Load vector store and retrieve documents
        vector_store = load_vector_store()
        retrieved_docs = vector_store.similarity_search(request.question, k=request.k)
        
        # Evaluate
        result = evaluate_rag_system(
            question=request.question,
            answer=request.answer,
            context=request.context,
            retrieved_docs=retrieved_docs,
            relevant_doc_ids=request.relevant_doc_ids,
            k=request.k
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/human")
async def submit_human_evaluation(
    request: HumanEvaluationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Submit human evaluation.
    
    Stores human evaluation for a question.
    
    Example:
        POST /evaluation/human
        {
            "question_id": 123,
            "criteria": {
                "accuracy": 5,
                "completeness": 4,
                "clarity": 5,
                "usefulness": 4,
                "comments": "Great answer!"
            }
        }
    """
    try:
        # Normalize to 0-1 score
        score = normalize_human_evaluation(request.criteria)
        
        # In production, save to database
        # For now, just return the score
        
        return {
            "question_id": request.question_id,
            "score": score,
            "criteria": request.criteria
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics():
    """
    Get system-wide metrics.
    
    Returns aggregated metrics across all evaluations.
    
    Example:
        GET /evaluation/metrics
    """
    # Mock data - in production, aggregate from database
    return SystemMetricsResponse(
        retrieval_metrics=RetrievalMetrics(
            recall_at_k=0.85,
            precision_at_k=0.92,
            mrr=0.78,
            k=5
        ),
        generation_metrics=GenerationMetrics(
            faithfulness=0.91,
            answer_relevance=0.88,
            context_utilization=0.84
        ),
        overall_score=0.87,
        total_evaluations=150
    )


# ============================================================================
# Educational Note: Using Evaluation in Production
# ============================================================================
"""
**How to Use Evaluation Metrics:**

1. **During Development**:
   - Create test sets with known answers
   - Evaluate different RAG configurations
   - Compare approaches (e.g., basic vs adaptive RAG)
   - Tune hyperparameters (chunk size, K, etc.)

2. **In Production**:
   - Sample random queries for evaluation
   - Track metrics over time
   - Alert on metric degradation
   - A/B test new features

3. **Human Evaluation**:
   - Validate automated metrics periodically
   - Collect user feedback
   - Identify edge cases
   - Improve system based on feedback

**Example Workflow**:

```python
# 1. Ask a question
response = await ask_question("What is the refund policy?")

# 2. Evaluate the response
evaluation = await evaluate_rag_endpoint({
    "question": "What is the refund policy?",
    "answer": response.answer,
    "context": response.context,
    "relevant_doc_ids": ["doc1", "doc3"],
    "k": 5
})

# 3. Check if quality is acceptable
if evaluation.overall_score < 0.7:
    # Log for review or trigger fallback
    logger.warning(f"Low quality response: {evaluation.overall_score}")

# 4. Periodically request human evaluation
if random.random() < 0.1:  # 10% of queries
    # Show evaluation form to user
    pass
```

**Best Practices**:

1. **Set Thresholds**: Define minimum acceptable scores
2. **Monitor Trends**: Track metrics over time
3. **Investigate Failures**: Review low-scoring responses
4. **Iterate**: Use insights to improve system
5. **Balance**: Automated + human evaluation
"""
