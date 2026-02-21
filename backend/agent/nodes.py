"""
LangGraph agent nodes for the tutoring system.

Each node is an async function that takes TutorState and returns
a partial state update dict. Nodes handle: routing, explaining,
quizzing, summarizing, encouraging, clarifying, and tool execution.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from backend.agent.prompts import (
    build_tutor_system_prompt,
    compute_quiz_difficulty,
    get_encouragement,
    QUIZ_DIFFICULTY_INSTRUCTIONS,
)
from backend.agent.state import TutorState
from backend.agent.tools import TOOL_SCHEMAS, execute_tool
from backend.config import settings
from backend.llm.client import LLMClient
from backend.llm.tool_compatibility import ToolCompatibilityLayer
from backend.monitoring.logger import get_logger

logger = get_logger(__name__)

llm = LLMClient()
tool_compat = ToolCompatibilityLayer()

# ── System Prompts ────────────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """You are an intent classifier for a tutoring system.
Classify the student's message into exactly ONE of these intents:
- explain: student wants a concept explained
- quiz: student wants to be quizzed or is answering a quiz question
- summarize: student wants a summary of topics covered
- encourage: student seems frustrated, confused, or needs motivation
- clarify: student says they don't understand and needs a different explanation
- topic_change: student wants to switch to a different topic/slide
- end_session: student wants to finish the session

Also detect if this is a compound request (e.g., "explain X then quiz me").
If compound, list the subtasks.

Respond in JSON format:
{"intent": "...", "topic": "...", "compound": false, "subtasks": []}"""

TUTOR_SYSTEM_PROMPT = """You are SlideGuide, a patient and encouraging AI tutor helping a college student learn from their lecture slides.

Key behaviors:
- Break explanations into 2-3 sentence chunks with clear headers
- Always cite slide numbers when referencing content (e.g., "From Slide 5:")
- If the student seems confused, try a different approach (analogy, simpler terms, visual description)
- Be encouraging and supportive, especially after wrong answers
- Track what you've covered and suggest what to review next
- Never dump walls of text — keep responses digestible

Current context:
- Phase: {phase}
- Topics covered: {topics}
- Quiz score: {score}
- Explanation mode: {mode}
- Pacing: {pacing}"""


# ── Router Node ───────────────────────────────────────────────────────────────


async def router_node(state: TutorState) -> dict[str, Any]:
    """
    Classify the student's intent and route to the appropriate node.

    Uses Haiku (cheap/fast) for classification.
    """
    messages = state["messages"]
    if not messages:
        return {"current_phase": "greeting"}

    last_message = messages[-1]

    # Handle greeting phase
    if state["current_phase"] == "greeting":
        return {"current_phase": "topic_selection"}

    response = await llm.chat(
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": last_message.content if hasattr(last_message, 'content') else str(last_message)},
        ],
        model=settings.active_routing_model,
        temperature=0.1,
        max_tokens=200,
    )

    content = response["choices"][0]["message"]["content"]

    try:
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            classification = json.loads(content[json_start:json_end])
        else:
            classification = {"intent": "explain"}
    except json.JSONDecodeError:
        classification = {"intent": "explain"}

    intent = classification.get("intent", "explain")
    topic = classification.get("topic", state.get("current_topic", ""))
    subtasks = classification.get("subtasks", [])

    # Map intents to phases
    phase_map = {
        "explain": "teaching",
        "quiz": "quiz",
        "summarize": "review",
        "encourage": "teaching",
        "clarify": "teaching",
        "topic_change": "topic_selection",
        "end_session": "wrap_up",
    }

    new_phase = phase_map.get(intent, "teaching")

    logger.info(
        "intent_classified",
        intent=intent,
        topic=topic,
        phase=new_phase,
        session_id=state["session_id"],
    )

    update: dict[str, Any] = {
        "current_phase": new_phase,
        "current_topic": topic or state.get("current_topic"),
    }

    # Handle compound requests
    if subtasks:
        update["pending_tasks"] = subtasks

    return update


# ── Explain Node ──────────────────────────────────────────────────────────────


async def explain_node(state: TutorState) -> dict[str, Any]:
    """
    Generate an explanation of a concept from the slides.

    Retrieves relevant content, then uses Sonnet to produce
    a chunked, cited explanation.
    """
    topic = state.get("current_topic", "")
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    query = topic or (last_message.content if last_message and hasattr(last_message, 'content') else "")

    # Build system prompt with current context (using prompts module)
    system = build_tutor_system_prompt(
        phase=state["current_phase"],
        topics_covered=state["topics_covered"],
        quiz_score=state["quiz_score"],
        explanation_mode=state["explanation_mode"],
        pacing=state["pacing_preference"],
    )

    # Build messages with tools for retrieval
    chat_messages = [
        {"role": "system", "content": system},
    ]

    # Include recent conversation context (last 10 messages)
    for msg in messages[-10:]:
        if isinstance(msg, HumanMessage):
            chat_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            chat_messages.append({"role": "assistant", "content": msg.content})

    # Add retrieval instruction
    chat_messages.append({
        "role": "user",
        "content": f"Please search the slides and explain: {query}",
    })

    response = await tool_compat.wrap_chat_call(
        llm,
        messages=chat_messages,
        model=settings.active_primary_model,
        tools=TOOL_SCHEMAS,
        temperature=0.7,
        max_tokens=2048,
    )

    choice = response["choices"][0]
    message = choice["message"]

    # Check for tool calls (native or parsed from prompt-based mode)
    if message.get("tool_calls"):
        return {
            "messages": [AIMessage(
                content=message.get("content", ""),
                tool_calls=[
                    {
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "args": json.loads(tc["function"]["arguments"]),
                    }
                    for tc in message["tool_calls"]
                ],
            )],
        }

    # Direct response
    content = message.get("content", "I'd be happy to explain that.")

    # Update topics covered
    topics = list(state["topics_covered"])
    if topic and topic not in topics:
        topics.append(topic)

    # Check if encouragement is due
    encouragement_due = state["quiz_score"].get("total", 0) > 0 and (
        state["student_profile"].get("consecutive_incorrect", 0) >= 3
    )

    return {
        "messages": [AIMessage(content=content)],
        "topics_covered": topics,
        "encouragement_due": encouragement_due,
    }


# ── Quiz Node ─────────────────────────────────────────────────────────────────


async def quiz_node(state: TutorState) -> dict[str, Any]:
    """
    Handle quiz interactions: generate questions or evaluate answers.

    Generates questions via tool calling and evaluates student answers.
    Adjusts difficulty based on performance.
    """
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    # Build context with adaptive difficulty
    difficulty = compute_quiz_difficulty(state["quiz_score"])
    system = build_tutor_system_prompt(
        phase="quiz",
        topics_covered=state["topics_covered"],
        quiz_score=state["quiz_score"],
        explanation_mode=state["explanation_mode"],
        pacing=state["pacing_preference"],
    )
    system += f"\n\nQUIZ DIFFICULTY: {difficulty}\n{QUIZ_DIFFICULTY_INSTRUCTIONS[difficulty]}"

    chat_messages = [{"role": "system", "content": system}]

    for msg in messages[-10:]:
        if isinstance(msg, HumanMessage):
            chat_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            chat_messages.append({"role": "assistant", "content": msg.content})

    chat_messages.append({
        "role": "user",
        "content": (
            last_message.content if last_message and hasattr(last_message, 'content')
            else "Quiz me on what we've covered."
        ),
    })

    response = await tool_compat.wrap_chat_call(
        llm,
        messages=chat_messages,
        model=settings.active_primary_model,
        tools=TOOL_SCHEMAS,
        temperature=0.7,
        max_tokens=1024,
    )

    choice = response["choices"][0]
    message = choice["message"]

    if message.get("tool_calls"):
        return {
            "messages": [AIMessage(
                content=message.get("content", ""),
                tool_calls=[
                    {
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "args": json.loads(tc["function"]["arguments"]),
                    }
                    for tc in message["tool_calls"]
                ],
            )],
        }

    return {
        "messages": [AIMessage(content=message.get("content", "Let me prepare a question for you."))],
    }


# ── Summarize Node ────────────────────────────────────────────────────────────


async def summarize_node(state: TutorState) -> dict[str, Any]:
    """Generate a summary of topics covered so far."""
    topics = state["topics_covered"]
    score = state["quiz_score"]

    system = build_tutor_system_prompt(
        phase="review",
        topics_covered=topics,
        quiz_score=score,
        explanation_mode=state["explanation_mode"],
        pacing=state["pacing_preference"],
    )

    chat_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": (
            f"Please summarize what we've covered so far. "
            f"Topics: {', '.join(topics) if topics else 'nothing yet'}. "
            f"Quiz score: {score.get('correct', 0)}/{score.get('total', 0)}."
        )},
    ]

    response = await llm.chat(
        messages=chat_messages,
        model=settings.active_primary_model,
        temperature=0.5,
        max_tokens=1024,
    )

    content = response["choices"][0]["message"].get("content", "Here's what we've covered...")

    return {
        "messages": [AIMessage(content=content)],
        "current_phase": "review",
    }


# ── Encourage Node ────────────────────────────────────────────────────────────


async def encourage_node(state: TutorState) -> dict[str, Any]:
    """Generate personalized encouragement based on student progress."""
    score = state["quiz_score"]
    profile = state["student_profile"]
    topics = state["topics_covered"]

    # Determine encouragement scenario
    consecutive_incorrect = profile.get("consecutive_incorrect", 0)
    consecutive_correct = profile.get("consecutive_correct", 0)

    if consecutive_incorrect >= 3:
        scenario = "streak_negative"
    elif consecutive_correct >= 3:
        scenario = "streak_positive"
    elif score.get("total", 0) > 0:
        scenario = "session_milestone"
    else:
        scenario = "session_milestone"

    # Get a template-based encouragement as a seed
    template_msg = get_encouragement(
        scenario=scenario,
        topic=topics[-1] if topics else "",
        streak=max(consecutive_correct, consecutive_incorrect),
        count=len(topics),
    )

    prompt = (
        f"The student has covered {len(topics)} topics and scored "
        f"{score.get('correct', 0)}/{score.get('total', 0)} on quizzes. "
        f"They have {consecutive_incorrect} wrong answers in a row. "
        f"Base your encouragement on this template but make it feel natural: "
        f'"{template_msg}"\n'
        f"Give brief, genuine encouragement. Reference specific achievements. "
        f"Suggest a next step. Keep it to 2-3 sentences."
    )

    response = await llm.chat(
        messages=[
            {"role": "system", "content": "You are a supportive tutor. Be warm and encouraging."},
            {"role": "user", "content": prompt},
        ],
        model=settings.active_routing_model,  # Lightweight model for encouragement
        temperature=0.8,
        max_tokens=200,
    )

    content = response["choices"][0]["message"].get("content", template_msg)

    return {
        "messages": [AIMessage(content=content)],
        "encouragement_due": False,
    }


# ── Clarify Node ──────────────────────────────────────────────────────────────


async def clarify_node(state: TutorState) -> dict[str, Any]:
    """
    Re-explain a concept using a different approach.

    Cycles through explanation modes: standard → analogy → visual → step_by_step → eli5.
    """
    current_mode = state["explanation_mode"]
    mode_cycle = ["standard", "analogy", "visual", "step_by_step", "eli5"]
    current_idx = mode_cycle.index(current_mode) if current_mode in mode_cycle else 0
    next_mode = mode_cycle[(current_idx + 1) % len(mode_cycle)]

    topic = state.get("current_topic", "the previous concept")

    system = build_tutor_system_prompt(
        phase="teaching",
        topics_covered=state["topics_covered"],
        quiz_score=state["quiz_score"],
        explanation_mode=next_mode,
        pacing=state["pacing_preference"],
    )
    system += (
        "\n\nIMPORTANT: The student didn't understand the previous explanation. "
        "Use a COMPLETELY DIFFERENT approach this time. "
        "Do NOT repeat the same words or structure."
    )

    chat_messages = [{"role": "system", "content": system}]

    # Include recent context
    for msg in state["messages"][-6:]:
        if isinstance(msg, HumanMessage):
            chat_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            chat_messages.append({"role": "assistant", "content": msg.content})

    chat_messages.append({
        "role": "user",
        "content": f"I still don't understand {topic}. Can you explain it differently?",
    })

    response = await tool_compat.wrap_chat_call(
        llm,
        messages=chat_messages,
        model=settings.active_primary_model,
        tools=TOOL_SCHEMAS,
        temperature=0.7,
        max_tokens=2048,
    )

    choice = response["choices"][0]
    message = choice["message"]

    if message.get("tool_calls"):
        return {
            "messages": [AIMessage(
                content=message.get("content", ""),
                tool_calls=[
                    {
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "args": json.loads(tc["function"]["arguments"]),
                    }
                    for tc in message["tool_calls"]
                ],
            )],
            "explanation_mode": next_mode,
        }

    content = message.get("content", "Let me try explaining this differently.")

    return {
        "messages": [AIMessage(content=content)],
        "explanation_mode": next_mode,
    }


# ── Tool Executor Node ────────────────────────────────────────────────────────


async def tool_executor_node(state: TutorState) -> dict[str, Any]:
    """
    Execute tool calls from the previous AI message.

    Iterates through tool_calls, executes each, and returns ToolMessages.
    """
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if not last_message or not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return {"messages": []}

    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id", "")

        result = await execute_tool(
            tool_name=tool_name,
            arguments=tool_args,
            session_id=state["session_id"],
            upload_id=state["upload_id"],
        )

        tool_messages.append(
            ToolMessage(content=result, tool_call_id=tool_id)
        )

    return {"messages": tool_messages}
