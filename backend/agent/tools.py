"""
Agent tool definitions with typed schemas.

Each tool is a function with type annotations that auto-generate
OpenAI-compatible tool schemas. The agent invokes tools via function calling
and we execute them in the tool_executor_node.

Implements the full closed-loop: LLM decides → tool called → result parsed → fed back.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from backend.config import settings
from backend.monitoring.logger import get_logger
from backend.monitoring.metrics import metrics

logger = get_logger(__name__)


# ── Tool Functions ────────────────────────────────────────────────────────────


async def search_slides(
    query: str,
    upload_id: str,
    top_k: int = 5,
    slide_range_start: int | None = None,
    slide_range_end: int | None = None,
) -> list[dict[str, Any]]:
    """
    Search the slide deck for content relevant to the query.

    Uses hybrid search (semantic + keyword) with diversity filtering.
    Returns relevant chunks with slide numbers and confidence scores.
    """
    from backend.rag.retriever import HybridRetriever
    from backend.rag.vectorstore import VectorStore

    vs = VectorStore()
    retriever = HybridRetriever(vs)

    slide_filter = slide_range_start if slide_range_start else None
    results = await retriever.retrieve(
        query=query,
        upload_id=upload_id,
        n_results=top_k,
        slide_filter=slide_filter,
    )

    return [
        {
            "slide_number": r.metadata.slide_number,
            "title": r.metadata.title,
            "content": r.content[:500],  # Truncate for context window
            "score": round(r.score, 4),
            "content_type": r.metadata.content_type,
        }
        for r in results
    ]


async def get_slide_content(upload_id: str, slide_number: int) -> dict[str, Any]:
    """
    Retrieve the full content of a specific slide by number.

    Returns slide text, title, images status, and speaker notes.
    """
    from prisma import Prisma

    db = Prisma()
    await db.connect()

    try:
        slide = await db.slide.find_first(
            where={"uploadId": upload_id, "slideNumber": slide_number}
        )

        if not slide:
            return {"error": f"Slide {slide_number} not found"}

        return {
            "slide_number": slide.slideNumber,
            "title": slide.title,
            "text_content": slide.textContent,
            "has_images": slide.hasImages,
            "metadata": slide.metadata,
        }
    finally:
        await db.disconnect()


async def generate_quiz_question(
    topic: str,
    difficulty: Literal["easy", "medium", "hard"] = "medium",
    question_type: Literal["multiple_choice", "short_answer", "true_false"] = "multiple_choice",
    upload_id: str = "",
    context: str = "",
) -> dict[str, Any]:
    """
    Generate a comprehension question about the given topic.

    Creates a quiz question grounded in the slide content,
    with an answer key and explanation.
    """
    from backend.llm.client import LLMClient

    client = LLMClient()

    format_instructions = {
        "multiple_choice": "Provide 4 options labeled A, B, C, D. Include the correct answer letter.",
        "short_answer": "Provide the expected answer in 1-2 sentences.",
        "true_false": "Make a statement that is either true or false. Include whether it's True or False.",
    }

    prompt = f"""Generate a {difficulty} {question_type} question about: {topic}

Based on this slide content:
{context[:1000] if context else 'Use your knowledge of the topic.'}

{format_instructions[question_type]}

Respond in this exact JSON format:
{{
    "question": "the question text",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "correct_answer": "the correct answer",
    "explanation": "brief explanation of why this is correct"
}}

For true/false, set options to ["True", "False"].
For short_answer, set options to null."""

    response = await client.chat(
        messages=[{"role": "user", "content": prompt}],
        model=settings.routing_model,  # Haiku for cheap generation
        temperature=0.8,
        max_tokens=500,
    )

    content = response["choices"][0]["message"]["content"]

    try:
        # Parse JSON from response
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            question_data = json.loads(content[json_start:json_end])
        else:
            question_data = {"question": content, "correct_answer": "", "explanation": ""}
    except json.JSONDecodeError:
        question_data = {"question": content, "correct_answer": "", "explanation": ""}

    return {
        "question": question_data.get("question", ""),
        "options": question_data.get("options"),
        "correct_answer": question_data.get("correct_answer", ""),
        "explanation": question_data.get("explanation", ""),
        "difficulty": difficulty,
        "question_type": question_type,
        "topic": topic,
    }


async def evaluate_student_answer(
    student_answer: str,
    correct_answer: str,
    context: str = "",
    question: str = "",
) -> dict[str, Any]:
    """
    Evaluate whether the student's answer is correct.

    Provides partial credit, feedback, and detailed explanation.
    """
    from backend.llm.client import LLMClient

    client = LLMClient()

    prompt = f"""Evaluate this student's answer:

Question: {question}
Correct answer: {correct_answer}
Student's answer: {student_answer}
Context: {context[:500]}

Respond in this exact JSON format:
{{
    "is_correct": true/false,
    "partial_credit": 0.0 to 1.0,
    "feedback": "encouraging feedback for the student",
    "explanation": "why the correct answer is correct"
}}

Be encouraging regardless of whether they got it right or wrong."""

    response = await client.chat(
        messages=[{"role": "user", "content": prompt}],
        model=settings.routing_model,  # Haiku
        temperature=0.3,
        max_tokens=300,
    )

    content = response["choices"][0]["message"]["content"]

    try:
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            eval_data = json.loads(content[json_start:json_end])
        else:
            eval_data = {}
    except json.JSONDecodeError:
        eval_data = {}

    return {
        "is_correct": eval_data.get("is_correct", False),
        "partial_credit": eval_data.get("partial_credit", 0.0),
        "feedback": eval_data.get("feedback", "Let's review this together."),
        "correct_answer": correct_answer,
        "explanation": eval_data.get("explanation", ""),
    }


async def get_student_progress(session_id: str) -> dict[str, Any]:
    """
    Retrieve the student's progress in the current session.

    Returns topics covered, quiz scores, and confidence level.
    """
    from prisma import Prisma

    db = Prisma()
    await db.connect()

    try:
        progress = await db.studentprogress.find_unique(
            where={"sessionId": session_id}
        )

        if not progress:
            return {
                "topics_covered": [],
                "quiz_scores": {},
                "total_questions": 0,
                "correct_answers": 0,
                "confidence_level": 0.5,
            }

        return {
            "topics_covered": progress.topicsCovered,
            "quiz_scores": progress.quizScores,
            "total_questions": progress.totalQuestions,
            "correct_answers": progress.correctAnswers,
            "confidence_level": progress.confidenceLevel,
        }
    finally:
        await db.disconnect()


async def extract_slide_image(
    upload_id: str,
    slide_number: int,
    image_index: int = 0,
) -> dict[str, Any]:
    """
    Extract and describe an image/diagram from a slide using VLM.

    Returns the image path and a text description of its content.
    """
    from prisma import Prisma

    db = Prisma()
    await db.connect()

    try:
        slide = await db.slide.find_first(
            where={"uploadId": upload_id, "slideNumber": slide_number}
        )

        if not slide or not slide.imagePaths:
            return {"error": "No images found on this slide"}

        paths = slide.imagePaths if isinstance(slide.imagePaths, list) else []
        if image_index >= len(paths):
            return {"error": f"Image index {image_index} out of range (has {len(paths)})"}

        image_path = paths[image_index]

        # Try VLM description (Phase 3 will add real implementation)
        try:
            from backend.llm.vision import VisionClient

            vision = VisionClient()
            description = await vision.describe_image(
                image_path, context=slide.textContent or ""
            )
        except (ImportError, Exception):
            description = (
                f"Image from slide {slide_number}. "
                "Visual description will be available after VLM integration."
            )

        return {
            "slide_number": slide_number,
            "image_index": image_index,
            "image_path": image_path,
            "description": description,
        }
    finally:
        await db.disconnect()


async def lookup_prerequisite(
    concept: str,
    upload_id: str,
) -> dict[str, Any]:
    """
    Check if a concept has prerequisites the student hasn't covered.

    Searches earlier slides for foundational concepts.
    """
    results = await search_slides(
        query=f"introduction to {concept} basics fundamentals",
        upload_id=upload_id,
        top_k=3,
    )

    prerequisites = []
    for r in results:
        if r["slide_number"] > 0:
            prerequisites.append({
                "topic": r["title"] or f"Slide {r['slide_number']}",
                "slide_number": r["slide_number"],
                "brief": r["content"][:200],
            })

    return {
        "concept": concept,
        "prerequisites": prerequisites,
        "has_prerequisites": len(prerequisites) > 0,
    }


# ── Tool Registry ─────────────────────────────────────────────────────────────

# Map of tool name → function
TOOLS_BY_NAME: dict[str, Any] = {
    "search_slides": search_slides,
    "get_slide_content": get_slide_content,
    "generate_quiz_question": generate_quiz_question,
    "evaluate_student_answer": evaluate_student_answer,
    "get_student_progress": get_student_progress,
    "extract_slide_image": extract_slide_image,
    "lookup_prerequisite": lookup_prerequisite,
}

# OpenAI-format tool schemas for function calling
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_slides",
            "description": "Search the slide deck for content relevant to a query. Uses hybrid semantic + keyword search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "top_k": {"type": "integer", "description": "Number of results to return", "default": 5},
                    "slide_range_start": {"type": "integer", "description": "Start slide number filter (optional)"},
                    "slide_range_end": {"type": "integer", "description": "End slide number filter (optional)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_slide_content",
            "description": "Get the full content of a specific slide by number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "slide_number": {"type": "integer", "description": "The slide number to retrieve"},
                },
                "required": ["slide_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_quiz_question",
            "description": "Create a comprehension question about a topic from the slides.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "The topic to quiz on"},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"], "default": "medium"},
                    "question_type": {"type": "string", "enum": ["multiple_choice", "short_answer", "true_false"], "default": "multiple_choice"},
                    "context": {"type": "string", "description": "Relevant slide content for grounding"},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_student_answer",
            "description": "Evaluate a student's answer against the expected answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "student_answer": {"type": "string", "description": "The student's answer"},
                    "correct_answer": {"type": "string", "description": "The expected correct answer"},
                    "question": {"type": "string", "description": "The original question"},
                    "context": {"type": "string", "description": "Additional context"},
                },
                "required": ["student_answer", "correct_answer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_student_progress",
            "description": "Get the student's progress: topics covered, quiz scores, confidence level.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_slide_image",
            "description": "Extract and describe an image or diagram from a specific slide.",
            "parameters": {
                "type": "object",
                "properties": {
                    "slide_number": {"type": "integer", "description": "Slide containing the image"},
                    "image_index": {"type": "integer", "description": "Index of the image on the slide", "default": 0},
                },
                "required": ["slide_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_prerequisite",
            "description": "Check if a concept has prerequisite topics the student hasn't covered yet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string", "description": "The concept to check prerequisites for"},
                },
                "required": ["concept"],
            },
        },
    },
]


async def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    session_id: str = "",
    upload_id: str = "",
) -> str:
    """
    Execute a tool by name with given arguments.

    Injects session_id and upload_id into tool calls that need them.
    Logs execution time and returns result as JSON string.
    """
    start_time = time.perf_counter()

    func = TOOLS_BY_NAME.get(tool_name)
    if not func:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    # Inject session/upload context
    if "upload_id" in func.__code__.co_varnames and "upload_id" not in arguments:
        arguments["upload_id"] = upload_id
    if "session_id" in func.__code__.co_varnames and "session_id" not in arguments:
        arguments["session_id"] = session_id

    try:
        result = await func(**arguments)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "tool_executed",
            tool=tool_name,
            latency_ms=round(elapsed_ms, 1),
            session_id=session_id,
        )

        return json.dumps(result, default=str)

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "tool_execution_failed",
            tool=tool_name,
            error=str(e),
            latency_ms=round(elapsed_ms, 1),
        )
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})
