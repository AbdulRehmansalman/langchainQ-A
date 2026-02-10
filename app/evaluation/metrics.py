"""
RAG Evaluation Metrics

This module implements comprehensive evaluation metrics for RAG systems.

Key concepts:
- Retrieval metrics (Recall@K, Precision@K, MRR)
- Generation metrics (Faithfulness, Answer Relevance, Context Utilization)
- End-to-end evaluation
- Human evaluation framework
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import numpy as np

from app.models.llm_config import get_openai_chat_model


# ============================================================================
# What are RAG Evaluation Metrics?
# ============================================================================
"""
RAG evaluation has two main components:

1. **Retrieval Metrics**: How well does the system retrieve relevant documents?
   - Recall@K: What % of relevant docs are in top K results?
   - Precision@K: What % of top K results are relevant?
   - MRR (Mean Reciprocal Rank): How high is the first relevant doc?

2. **Generation Metrics**: How good is the generated answer?
   - Faithfulness: Is the answer grounded in retrieved context?
   - Answer Relevance: Does the answer address the question?
   - Context Utilization: How well does it use the context?

Why evaluate?
- Measure system quality
- Compare different approaches
- Identify areas for improvement
- Track performance over time
- Justify design decisions
"""


# ============================================================================
# Evaluation Data Models
# ============================================================================

class RetrievalMetrics(BaseModel):
    """Retrieval evaluation metrics"""
    recall_at_k: float = Field(description="Recall@K score (0-1)")
    precision_at_k: float = Field(description="Precision@K score (0-1)")
    mrr: float = Field(description="Mean Reciprocal Rank (0-1)")
    k: int = Field(description="K value used")


class GenerationMetrics(BaseModel):
    """Generation evaluation metrics"""
    faithfulness: float = Field(description="Faithfulness score (0-1)")
    answer_relevance: float = Field(description="Answer relevance score (0-1)")
    context_utilization: float = Field(description="Context utilization score (0-1)")


class RAGEvaluationResult(BaseModel):
    """Complete RAG evaluation result"""
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    overall_score: float = Field(description="Combined score (0-1)")


# ============================================================================
# Retrieval Metrics
# ============================================================================

def calculate_recall_at_k(
    retrieved_docs: List[Document],
    relevant_doc_ids: List[str],
    k: int = 5
) -> float:
    """
    Calculate Recall@K.
    
    Recall@K = (# relevant docs in top K) / (# total relevant docs)
    
    Measures: How many of the relevant documents did we retrieve?
    
    Args:
        retrieved_docs: Retrieved documents (ordered by relevance)
        relevant_doc_ids: IDs of documents that are actually relevant
        k: Number of top results to consider
        
    Returns:
        Recall@K score (0-1)
        
    Example:
        # 10 relevant docs exist
        # We retrieved 7 of them in top 5
        recall = calculate_recall_at_k(retrieved, relevant_ids, k=5)
        # Returns: 0.7 (7/10)
        
    Interpretation:
        - 1.0 = Perfect (retrieved all relevant docs)
        - 0.5 = Retrieved half of relevant docs
        - 0.0 = Retrieved no relevant docs
    """
    if not relevant_doc_ids:
        return 1.0  # No relevant docs to find
    
    # Get top K retrieved docs
    top_k = retrieved_docs[:k]
    
    # Count how many relevant docs are in top K
    retrieved_ids = {doc.metadata.get("id", doc.metadata.get("source")) for doc in top_k}
    relevant_retrieved = len(set(relevant_doc_ids) & retrieved_ids)
    
    # Recall = relevant retrieved / total relevant
    recall = relevant_retrieved / len(relevant_doc_ids)
    
    return recall


def calculate_precision_at_k(
    retrieved_docs: List[Document],
    relevant_doc_ids: List[str],
    k: int = 5
) -> float:
    """
    Calculate Precision@K.
    
    Precision@K = (# relevant docs in top K) / K
    
    Measures: How many of the retrieved documents are actually relevant?
    
    Args:
        retrieved_docs: Retrieved documents (ordered by relevance)
        relevant_doc_ids: IDs of documents that are actually relevant
        k: Number of top results to consider
        
    Returns:
        Precision@K score (0-1)
        
    Example:
        # Retrieved 5 docs, 4 are relevant
        precision = calculate_precision_at_k(retrieved, relevant_ids, k=5)
        # Returns: 0.8 (4/5)
        
    Interpretation:
        - 1.0 = Perfect (all retrieved docs are relevant)
        - 0.5 = Half of retrieved docs are relevant
        - 0.0 = No retrieved docs are relevant
    """
    if k == 0:
        return 0.0
    
    # Get top K retrieved docs
    top_k = retrieved_docs[:k]
    
    # Count how many are relevant
    retrieved_ids = {doc.metadata.get("id", doc.metadata.get("source")) for doc in top_k}
    relevant_retrieved = len(set(relevant_doc_ids) & retrieved_ids)
    
    # Precision = relevant retrieved / K
    precision = relevant_retrieved / k
    
    return precision


def calculate_mrr(
    retrieved_docs: List[Document],
    relevant_doc_ids: List[str]
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    MRR = 1 / (rank of first relevant document)
    
    Measures: How quickly do we find a relevant document?
    
    Args:
        retrieved_docs: Retrieved documents (ordered by relevance)
        relevant_doc_ids: IDs of documents that are actually relevant
        
    Returns:
        MRR score (0-1)
        
    Example:
        # First relevant doc is at position 3
        mrr = calculate_mrr(retrieved, relevant_ids)
        # Returns: 0.333 (1/3)
        
    Interpretation:
        - 1.0 = First doc is relevant
        - 0.5 = Second doc is relevant
        - 0.333 = Third doc is relevant
        - 0.0 = No relevant docs found
    """
    for rank, doc in enumerate(retrieved_docs, start=1):
        doc_id = doc.metadata.get("id", doc.metadata.get("source"))
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    
    return 0.0  # No relevant docs found


def evaluate_retrieval(
    retrieved_docs: List[Document],
    relevant_doc_ids: List[str],
    k: int = 5
) -> RetrievalMetrics:
    """
    Evaluate retrieval performance.
    
    Calculates all retrieval metrics at once.
    
    Args:
        retrieved_docs: Retrieved documents
        relevant_doc_ids: IDs of relevant documents
        k: K value for Recall@K and Precision@K
        
    Returns:
        RetrievalMetrics with all scores
        
    Example:
        metrics = evaluate_retrieval(retrieved, relevant_ids, k=5)
        print(f"Recall@5: {metrics.recall_at_k}")
        print(f"Precision@5: {metrics.precision_at_k}")
        print(f"MRR: {metrics.mrr}")
    """
    return RetrievalMetrics(
        recall_at_k=calculate_recall_at_k(retrieved_docs, relevant_doc_ids, k),
        precision_at_k=calculate_precision_at_k(retrieved_docs, relevant_doc_ids, k),
        mrr=calculate_mrr(retrieved_docs, relevant_doc_ids),
        k=k
    )


# ============================================================================
# Generation Metrics (LLM-based)
# ============================================================================

def evaluate_faithfulness(
    answer: str,
    context: str,
    llm: Optional[BaseChatModel] = None
) -> float:
    """
    Evaluate faithfulness (groundedness) of answer.
    
    Faithfulness: Is the answer supported by the context?
    
    Uses LLM to check if answer claims are grounded in context.
    
    Args:
        answer: Generated answer
        context: Retrieved context
        llm: LLM for evaluation (default: OpenAI)
        
    Returns:
        Faithfulness score (0-1)
        
    Example:
        context = "Our refund policy is 30 days."
        answer = "You can get a refund within 30 days."
        score = evaluate_faithfulness(answer, context)
        # Returns: ~1.0 (fully grounded)
        
        answer = "You can get a refund within 60 days."
        score = evaluate_faithfulness(answer, context)
        # Returns: ~0.0 (not grounded, hallucination)
        
    Why it matters:
        - Prevents hallucinations
        - Ensures trustworthy answers
        - Critical for production systems
    """
    if llm is None:
        llm = get_openai_chat_model(temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an evaluator. Assess if the answer is faithful to the context.

Faithfulness means:
- All claims in the answer are supported by the context
- No information is made up or hallucinated
- No contradictions with the context

Rate faithfulness from 0 to 1:
- 1.0 = Fully faithful (all claims supported)
- 0.5 = Partially faithful (some claims supported)
- 0.0 = Not faithful (claims not supported or contradicted)

Respond with ONLY a number between 0 and 1."""),
        ("human", """Context:
{context}

Answer:
{answer}

Faithfulness score (0-1):""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({"context": context, "answer": answer})
        score = float(result.strip())
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    except:
        return 0.5  # Default if parsing fails


def evaluate_answer_relevance(
    question: str,
    answer: str,
    llm: Optional[BaseChatModel] = None
) -> float:
    """
    Evaluate answer relevance to question.
    
    Answer Relevance: Does the answer address the question?
    
    Uses LLM to check if answer is relevant to the question.
    
    Args:
        question: User's question
        answer: Generated answer
        llm: LLM for evaluation
        
    Returns:
        Relevance score (0-1)
        
    Example:
        question = "What is the refund policy?"
        answer = "Our refund policy is 30 days."
        score = evaluate_answer_relevance(question, answer)
        # Returns: ~1.0 (highly relevant)
        
        answer = "We have great customer service."
        score = evaluate_answer_relevance(question, answer)
        # Returns: ~0.3 (not very relevant)
        
    Why it matters:
        - Ensures answers are on-topic
        - Improves user satisfaction
        - Detects routing failures
    """
    if llm is None:
        llm = get_openai_chat_model(temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an evaluator. Assess if the answer is relevant to the question.

Relevance means:
- The answer directly addresses the question
- The answer provides useful information for the question
- The answer is on-topic

Rate relevance from 0 to 1:
- 1.0 = Highly relevant (directly answers question)
- 0.5 = Somewhat relevant (related but not direct)
- 0.0 = Not relevant (off-topic or unrelated)

Respond with ONLY a number between 0 and 1."""),
        ("human", """Question:
{question}

Answer:
{answer}

Relevance score (0-1):""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({"question": question, "answer": answer})
        score = float(result.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.5


def evaluate_context_utilization(
    answer: str,
    context: str,
    llm: Optional[BaseChatModel] = None
) -> float:
    """
    Evaluate how well the answer uses the context.
    
    Context Utilization: Does the answer make good use of the context?
    
    Checks if important information from context is included.
    
    Args:
        answer: Generated answer
        context: Retrieved context
        llm: LLM for evaluation
        
    Returns:
        Utilization score (0-1)
        
    Example:
        context = "Refund policy: 30 days. Contact: support@example.com"
        answer = "You can get a refund within 30 days by contacting support@example.com"
        score = evaluate_context_utilization(answer, context)
        # Returns: ~1.0 (uses all relevant info)
        
        answer = "You can get a refund."
        score = evaluate_context_utilization(answer, context)
        # Returns: ~0.5 (missing details)
        
    Why it matters:
        - Ensures comprehensive answers
        - Maximizes value of retrieval
        - Improves answer quality
    """
    if llm is None:
        llm = get_openai_chat_model(temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an evaluator. Assess how well the answer utilizes the context.

Context utilization means:
- Important information from context is included in answer
- Relevant details are not omitted
- Context is used effectively

Rate utilization from 0 to 1:
- 1.0 = Excellent (uses all relevant context)
- 0.5 = Moderate (uses some context, misses some)
- 0.0 = Poor (doesn't use context effectively)

Respond with ONLY a number between 0 and 1."""),
        ("human", """Context:
{context}

Answer:
{answer}

Context utilization score (0-1):""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({"context": context, "answer": answer})
        score = float(result.strip())
        return max(0.0, min(1.0, score))
    except:
        return 0.5


def evaluate_generation(
    question: str,
    answer: str,
    context: str,
    llm: Optional[BaseChatModel] = None
) -> GenerationMetrics:
    """
    Evaluate generation performance.
    
    Calculates all generation metrics at once.
    
    Args:
        question: User's question
        answer: Generated answer
        context: Retrieved context
        llm: LLM for evaluation
        
    Returns:
        GenerationMetrics with all scores
        
    Example:
        metrics = evaluate_generation(question, answer, context)
        print(f"Faithfulness: {metrics.faithfulness}")
        print(f"Relevance: {metrics.answer_relevance}")
        print(f"Utilization: {metrics.context_utilization}")
    """
    return GenerationMetrics(
        faithfulness=evaluate_faithfulness(answer, context, llm),
        answer_relevance=evaluate_answer_relevance(question, answer, llm),
        context_utilization=evaluate_context_utilization(answer, context, llm)
    )


# ============================================================================
# End-to-End RAG Evaluation
# ============================================================================

def evaluate_rag_system(
    question: str,
    answer: str,
    context: str,
    retrieved_docs: List[Document],
    relevant_doc_ids: List[str],
    k: int = 5,
    llm: Optional[BaseChatModel] = None
) -> RAGEvaluationResult:
    """
    Comprehensive RAG system evaluation.
    
    Evaluates both retrieval and generation.
    
    Args:
        question: User's question
        answer: Generated answer
        context: Retrieved context (formatted)
        retrieved_docs: Retrieved documents
        relevant_doc_ids: IDs of relevant documents
        k: K for retrieval metrics
        llm: LLM for generation metrics
        
    Returns:
        Complete evaluation result
        
    Example:
        result = evaluate_rag_system(
            question="What is the refund policy?",
            answer=generated_answer,
            context=formatted_context,
            retrieved_docs=docs,
            relevant_doc_ids=["doc1", "doc3"],
            k=5
        )
        
        print(f"Overall Score: {result.overall_score}")
        print(f"Recall@5: {result.retrieval_metrics.recall_at_k}")
        print(f"Faithfulness: {result.generation_metrics.faithfulness}")
    """
    # Evaluate retrieval
    retrieval_metrics = evaluate_retrieval(retrieved_docs, relevant_doc_ids, k)
    
    # Evaluate generation
    generation_metrics = evaluate_generation(question, answer, context, llm)
    
    # Calculate overall score (weighted average)
    overall_score = (
        0.3 * retrieval_metrics.recall_at_k +
        0.2 * retrieval_metrics.precision_at_k +
        0.2 * generation_metrics.faithfulness +
        0.2 * generation_metrics.answer_relevance +
        0.1 * generation_metrics.context_utilization
    )
    
    return RAGEvaluationResult(
        retrieval_metrics=retrieval_metrics,
        generation_metrics=generation_metrics,
        overall_score=overall_score
    )


# ============================================================================
# Human Evaluation Framework
# ============================================================================

class HumanEvaluationCriteria(BaseModel):
    """Criteria for human evaluation"""
    accuracy: int = Field(ge=1, le=5, description="Answer accuracy (1-5)")
    completeness: int = Field(ge=1, le=5, description="Answer completeness (1-5)")
    clarity: int = Field(ge=1, le=5, description="Answer clarity (1-5)")
    usefulness: int = Field(ge=1, le=5, description="Answer usefulness (1-5)")
    comments: Optional[str] = Field(None, description="Additional comments")


def normalize_human_evaluation(criteria: HumanEvaluationCriteria) -> float:
    """
    Normalize human evaluation to 0-1 score.
    
    Args:
        criteria: Human evaluation criteria
        
    Returns:
        Normalized score (0-1)
        
    Example:
        eval = HumanEvaluationCriteria(
            accuracy=5,
            completeness=4,
            clarity=5,
            usefulness=4
        )
        score = normalize_human_evaluation(eval)
        # Returns: 0.9 ((5+4+5+4)/(4*5))
    """
    total = criteria.accuracy + criteria.completeness + criteria.clarity + criteria.usefulness
    max_score = 4 * 5  # 4 criteria, max 5 each
    return total / max_score


# ============================================================================
# Educational Note: Evaluation Best Practices
# ============================================================================
"""
**Why Evaluation Matters:**

1. **Measure Quality**: Quantify system performance
2. **Compare Approaches**: A/B test different RAG strategies
3. **Track Progress**: Monitor improvements over time
4. **Identify Issues**: Find weak points
5. **Build Trust**: Demonstrate reliability

**Retrieval Metrics Explained:**

1. **Recall@K**:
   - Question: "Did we find the relevant docs?"
   - High recall = Found most relevant docs
   - Low recall = Missed relevant docs
   - Improve: Better embeddings, larger K, better chunking

2. **Precision@K**:
   - Question: "Are the retrieved docs relevant?"
   - High precision = Most retrieved docs are relevant
   - Low precision = Many irrelevant docs retrieved
   - Improve: Better embeddings, smaller K, better filtering

3. **MRR (Mean Reciprocal Rank)**:
   - Question: "How quickly do we find relevant docs?"
   - High MRR = Relevant docs at top
   - Low MRR = Relevant docs buried
   - Improve: Better ranking, re-ranking

**Generation Metrics Explained:**

1. **Faithfulness**:
   - Question: "Is the answer grounded in context?"
   - High = No hallucinations
   - Low = Making things up
   - Critical for trust

2. **Answer Relevance**:
   - Question: "Does answer address the question?"
   - High = On-topic, useful
   - Low = Off-topic, not helpful
   - Improves user satisfaction

3. **Context Utilization**:
   - Question: "Does answer use the context well?"
   - High = Comprehensive, detailed
   - Low = Vague, missing details
   - Maximizes retrieval value

**Evaluation Strategies:**

1. **Automated Metrics** (this module):
   - Pros: Fast, scalable, consistent
   - Cons: May not capture all quality aspects
   - Use for: Continuous monitoring, A/B testing

2. **Human Evaluation**:
   - Pros: Captures nuance, user perspective
   - Cons: Slow, expensive, subjective
   - Use for: Quality validation, edge cases

3. **Hybrid Approach** (recommended):
   - Automated for most queries
   - Human for sample validation
   - Best of both worlds

**Best Practices:**

1. **Create test sets**:
   - Questions with known answers
   - Labeled relevant documents
   - Diverse query types

2. **Evaluate regularly**:
   - After each change
   - Before deployment
   - Continuous monitoring

3. **Use multiple metrics**:
   - No single metric tells full story
   - Balance retrieval and generation
   - Consider user satisfaction

4. **Set thresholds**:
   - Minimum acceptable scores
   - Alert on degradation
   - Block poor quality

5. **Iterate based on results**:
   - Low recall → Improve retrieval
   - Low faithfulness → Better prompts
   - Low relevance → Better routing

**Common Issues:**

1. **Low Recall**:
   - Embeddings not capturing meaning
   - Chunks too large/small
   - K too small
   - Solution: Tune embeddings, chunking, K

2. **Low Precision**:
   - Too many irrelevant docs
   - K too large
   - Poor ranking
   - Solution: Better filtering, smaller K, re-ranking

3. **Low Faithfulness**:
   - LLM hallucinating
   - Context not clear
   - Prompt issues
   - Solution: Better prompts, temperature=0, citations

4. **Low Relevance**:
   - Routing failures
   - Wrong retrieval
   - Poor generation
   - Solution: Better routing, improve retrieval

**Advanced Techniques:**

1. **RAGAS Framework**: Automated RAG evaluation
2. **LangSmith**: Tracing and evaluation
3. **Human-in-the-loop**: Sample validation
4. **A/B Testing**: Compare approaches
5. **Regression Testing**: Prevent degradation
"""
