import logging
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from agentic_rag.llm.provider import get_llm

logger = logging.getLogger(__name__)


# ── Pydantic schemas for structured output ─────────────────────────────────────

class ContextGrade(BaseModel):
    """Grade whether retrieved context is sufficient to answer the question."""
    sufficient: bool = Field(
        description="True if context contains enough information to answer the question"
    )
    reason: str = Field(
        description="One sentence explaining the grading decision"
    )


class AnswerGrade(BaseModel):
    """Grade whether the generated answer is faithful to the context."""
    faithful: bool = Field(
        description="True if every claim in the answer is supported by the context"
    )
    reason: str = Field(
        description="One sentence explaining the grading decision"
    )


# ── Prompts ────────────────────────────────────────────────────────────────────

CONTEXT_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a grader assessing whether retrieved context is sufficient to answer a question.

Return JSON with:
- sufficient: true if context contains the information needed to answer, false otherwise
- reason: one sentence explaining your decision

Be strict — if the context only partially covers the question, return false."""),
    ("human", """Question: {question}

Retrieved Context:
{context}

Is this context sufficient to answer the question?"""),
])

ANSWER_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a grader assessing whether a generated answer is faithful to the provided context.

Return JSON with:
- faithful: true if every claim in the answer is directly supported by the context, false if the answer contains information not in the context
- reason: one sentence explaining your decision

Be strict — any claim not explicitly supported by context should return false."""),
    ("human", """Question: {question}

Context:
{context}

Generated Answer:
{answer}

Is this answer faithful to the context?"""),
])


# ── Grader functions ───────────────────────────────────────────────────────────

def grade_context(
    question: str,
    context: str,
) -> ContextGrade:
    """
    Grade whether retrieved context is sufficient to answer the question.
    Runs BEFORE generation to avoid wasting an LLM call on bad context.

    Args:
        question: User question
        context: Formatted retrieved chunks

    Returns:
        ContextGrade with sufficient bool + reason
    """
    llm = get_llm()

    # Force structured output matching ContextGrade schema
    structured_llm = llm.with_structured_output(ContextGrade)
    chain = CONTEXT_GRADE_PROMPT | structured_llm

    try:
        result = chain.invoke({
            "question": question,
            "context": context,
        })
        logger.info(f"Context grade: sufficient={result.sufficient} | {result.reason}")
        return result

    except Exception as e:
        logger.warning(f"Context grader failed: {e} — defaulting to sufficient=True")
        return ContextGrade(sufficient=True, reason="Grader failed — defaulting to proceed")


def grade_answer(
    question: str,
    context: str,
    answer: str,
) -> AnswerGrade:
    """
    Grade whether the generated answer is faithful to the context.
    Runs AFTER generation to catch hallucinations.

    Args:
        question: User question
        context: Retrieved context used for generation
        answer: LLM-generated answer

    Returns:
        AnswerGrade with faithful bool + reason
    """
    llm = get_llm()

    structured_llm = llm.with_structured_output(AnswerGrade)
    chain = ANSWER_GRADE_PROMPT | structured_llm

    try:
        result = chain.invoke({
            "question": question,
            "context": context,
            "answer": answer,
        })
        logger.info(f"Answer grade: faithful={result.faithful} | {result.reason}")
        return result

    except Exception as e:
        logger.warning(f"Answer grader failed: {e} — defaulting to faithful=True")
        return AnswerGrade(faithful=True, reason="Grader failed — defaulting to proceed")