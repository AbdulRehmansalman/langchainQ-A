"""
LangChain Prompt Templates

This module demonstrates comprehensive prompt engineering with LangChain.
Prompts are the foundation of LLM applications - they guide model behavior.

Key concepts covered:
- ChatPromptTemplate: Structured prompts for chat models
- MessagesPlaceholder: Dynamic message history injection
- System/Human/AI roles: Proper message formatting
- Few-shot prompting: Examples for better responses
- Prompt composition: Combining multiple templates
- Partial prompts: Pre-filled context
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    FewShotChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ============================================================================
# Base System Prompt
# ============================================================================
# The system prompt sets the AI's personality, capabilities, and constraints
QA_SYSTEM_PROMPT = """You are an intelligent Q&A assistant with access to a knowledge base.

Your capabilities:
- Answer questions accurately and concisely
- Cite sources when using retrieved documents
- Admit when you don't know something
- Maintain conversation context
- Provide helpful, relevant information

Guidelines:
- Be concise but complete
- Use bullet points for lists
- Cite sources using [Source: filename]
- If information is not in the knowledge base, say so clearly
- Stay on topic and be helpful

Current date: {current_date}
"""


# ============================================================================
# RAG Prompt (with Retrieved Documents)
# ============================================================================
# This prompt is used when we have retrieved relevant documents
# It instructs the model to answer based on the provided context

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    ("system", """You have access to the following relevant documents:

{context}

Use these documents to answer the user's question. Always cite your sources.
If the documents don't contain relevant information, say so clearly."""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{question}"),
])


# ============================================================================
# Direct Answer Prompt (No RAG)
# ============================================================================
# This prompt is used for general knowledge questions that don't need RAG
# It's simpler and faster than RAG

DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    ("system", "Answer the following question using your general knowledge. Be concise and accurate."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{question}"),
])


# ============================================================================
# Routing Prompt (Decide RAG vs Direct)
# ============================================================================
# This prompt helps decide whether to use RAG or answer directly
# It's used in adaptive RAG to route queries intelligently

ROUTING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a routing assistant. Analyze the user's question and determine if it requires searching a knowledge base or can be answered with general knowledge.

Respond with ONLY "RAG" or "DIRECT":
- RAG: Question is about specific documents, company info, technical details, or requires factual lookup
- DIRECT: Question is about general knowledge, definitions, common facts, or conversational

Examples:
Question: "What is Python?" → DIRECT
Question: "What does our company policy say about vacation?" → RAG
Question: "Explain machine learning" → DIRECT
Question: "What are the features mentioned in the product documentation?" → RAG
Question: "How do I reset my password?" → RAG (if there's a user manual)
Question: "What's the capital of France?" → DIRECT
"""),
    ("human", "Question: {question}"),
])


# ============================================================================
# Conversation Summarization Prompt
# ============================================================================
# Used to create summaries of conversations for memory management
# Summaries are more efficient than storing full conversation history

SUMMARIZATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Summarize the following conversation concisely. Focus on:
- Main topics discussed
- Key questions asked
- Important information provided
- Any unresolved issues

Keep the summary under 200 words."""),
    ("human", "Conversation to summarize:\n\n{conversation}"),
])


# ============================================================================
# Self-RAG Evaluation Prompt
# ============================================================================
# Used in Self-RAG to evaluate if retrieved documents are relevant
# This helps filter out irrelevant documents before generating answers

SELF_RAG_EVALUATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a document relevance evaluator. 
    
Given a question and a document, determine if the document is relevant to answering the question.

Respond with ONLY "RELEVANT" or "NOT_RELEVANT":
- RELEVANT: Document contains information that helps answer the question
- NOT_RELEVANT: Document doesn't contain useful information for the question

Be strict - only mark as RELEVANT if the document actually helps answer the question."""),
    ("human", """Question: {question}

Document:
{document}

Is this document relevant?"""),
])


# ============================================================================
# Few-Shot Prompting Example
# ============================================================================
# Few-shot prompting provides examples to guide the model's responses
# This improves consistency and quality

# Define example Q&A pairs
qa_examples = [
    {
        "question": "What is the refund policy?",
        "answer": "According to our refund policy [Source: policy.pdf], customers can request a full refund within 30 days of purchase. The item must be unused and in original packaging."
    },
    {
        "question": "How do I contact support?",
        "answer": "Based on the support documentation [Source: support.md], you can contact support via:\n- Email: support@example.com\n- Phone: 1-800-SUPPORT\n- Live chat: Available 9 AM - 5 PM EST"
    },
]

# Create few-shot prompt template
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{question}"),
    ("ai", "{answer}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=qa_examples,
)

# Full prompt with few-shot examples
FEW_SHOT_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    ("system", "Here are some examples of good responses:\n"),
    few_shot_prompt,
    ("system", "\nNow answer the following question using the provided context:"),
    ("system", "Context:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{question}"),
])


# ============================================================================
# Partial Prompts (Pre-filled Context)
# ============================================================================
# Partial prompts allow pre-filling some variables
# Useful for context that doesn't change per request

from datetime import datetime

def get_rag_prompt_with_date():
    """
    Get RAG prompt with current date pre-filled.
    
    This demonstrates partial prompt application - some variables
    are filled at creation time, others at runtime.
    """
    return RAG_PROMPT_TEMPLATE.partial(
        current_date=datetime.now().strftime("%Y-%m-%d")
    )


# ============================================================================
# Prompt Composition
# ============================================================================
# Combine multiple prompts for complex scenarios

def create_contextual_rag_prompt(user_role: str = "user"):
    """
    Create a RAG prompt customized for user role.
    
    This demonstrates prompt composition - building complex prompts
    from simpler components based on runtime conditions.
    
    Args:
        user_role: User's role (user, admin, premium)
        
    Returns:
        Customized ChatPromptTemplate
    """
    role_context = {
        "user": "You are helping a standard user. Keep explanations simple.",
        "admin": "You are helping an administrator. You can use technical terms.",
        "premium": "You are helping a premium user. Provide detailed, comprehensive answers.",
    }
    
    return ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM_PROMPT),
        ("system", role_context.get(user_role, role_context["user"])),
        ("system", "Context:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{question}"),
    ])


# ============================================================================
# Educational Note: Prompt Engineering Best Practices
# ============================================================================
"""
Key prompt engineering concepts:

1. **ChatPromptTemplate vs PromptTemplate**:
   - ChatPromptTemplate: For chat models (GPT-3.5, GPT-4, Claude)
   - PromptTemplate: For completion models (older GPT-3)
   - Chat models understand roles (system, human, ai)

2. **Message Roles**:
   - System: Sets behavior and context (not visible to user)
   - Human: User's input
   - AI: Assistant's responses
   - Proper role usage improves response quality

3. **MessagesPlaceholder**:
   - Dynamically inject conversation history
   - Enables context-aware responses
   - Optional=True allows prompts to work without history

4. **Few-Shot Prompting**:
   - Provide examples of desired behavior
   - Improves consistency and quality
   - Especially useful for specific formats or styles

5. **Partial Prompts**:
   - Pre-fill variables that don't change
   - Reduces repetition
   - Improves performance

6. **Prompt Composition**:
   - Build complex prompts from simple components
   - Customize based on runtime conditions
   - Maintains consistency while allowing flexibility

Best practices:
- Be specific about desired output format
- Provide clear instructions
- Use examples for complex tasks
- Keep prompts concise but complete
- Test prompts with various inputs
- Version control your prompts

Example usage in chains:
    from app.prompts.templates import RAG_PROMPT_TEMPLATE
    
    chain = RAG_PROMPT_TEMPLATE | llm | StrOutputParser()
    response = chain.invoke({
        "question": "What is the refund policy?",
        "context": retrieved_docs,
        "chat_history": previous_messages,
        "current_date": "2024-01-15"
    })
"""
