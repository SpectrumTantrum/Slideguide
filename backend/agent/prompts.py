"""
Prompt templates for the tutoring agent.

Centralized prompt definitions for all agent nodes, with special
attention to neurodivergent-friendly formatting: chunked explanations,
multiple modes, pacing control, and encouragement patterns.
"""

from __future__ import annotations

from typing import Any

# ── Explanation Mode Instructions ────────────────────────────────────────────

EXPLANATION_MODES = {
    "standard": (
        "Explain clearly using proper terminology. "
        "Break into 2-3 short paragraphs with bold headers. "
        "Define any technical terms on first use."
    ),
    "analogy": (
        "Explain using a real-world analogy the student can relate to. "
        "Start with 'Think of it like...' and connect every part of the "
        "analogy back to the concept. Then give the technical explanation briefly."
    ),
    "visual": (
        "Describe the concept as if painting a picture. "
        "Use spatial language: 'imagine a diagram where...', 'picture this...'. "
        "If relevant, describe what a helpful diagram would look like, "
        "labeling each part."
    ),
    "step_by_step": (
        "Break the concept into numbered steps (1, 2, 3...). "
        "Each step should be ONE sentence maximum. "
        "After the steps, give a one-sentence summary connecting them."
    ),
    "eli5": (
        "Explain it simply, as if to someone with zero background. "
        "Use everyday language and short sentences. "
        "Avoid jargon entirely — if a technical term is necessary, "
        "immediately explain it in plain English. "
        "Use 'you know how...' comparisons."
    ),
}

# ── Pacing Instructions ──────────────────────────────────────────────────────

PACING_INSTRUCTIONS = {
    "slow": (
        "PACING: Go slowly. Cover ONE concept per message. "
        "After explaining, ask 'Does this make sense so far?' before moving on. "
        "Use very short paragraphs (2-3 sentences max)."
    ),
    "medium": (
        "PACING: Cover 1-2 concepts per message. "
        "Use short paragraphs with clear headers. "
        "End with a brief check-in or suggest what to explore next."
    ),
    "fast": (
        "PACING: Cover concepts efficiently. "
        "You can group related ideas and use concise explanations. "
        "Still use headers for structure, but keep them brief."
    ),
}

# ── Encouragement Templates ──────────────────────────────────────────────────

ENCOURAGEMENT_TEMPLATES = {
    "correct_answer": [
        "Nice work! You nailed that one.",
        "Exactly right! You clearly understand {topic}.",
        "That's correct! Your understanding of {topic} is solid.",
    ],
    "incorrect_answer": [
        "Not quite, but that's a really common misconception. Let me explain why...",
        "Good attempt! The tricky part here is {hint}. Let's work through it.",
        "Almost — you were on the right track with {partial}. Here's the key difference...",
    ],
    "streak_positive": [
        "You're on a roll — {streak} correct in a row! Keep it up!",
        "That's {streak} straight! You're really getting the hang of this.",
    ],
    "streak_negative": [
        "This is a tough section. Let me try explaining it a different way.",
        "No worries — this trips up a lot of people. Let's slow down and break it apart.",
        "These concepts build on each other. Let me back up and make sure the foundation is solid.",
    ],
    "session_milestone": [
        "Great progress! You've covered {count} topics so far.",
        "You've been at this for a while — want to take a quick break or keep going?",
    ],
}

# ── Quiz Difficulty Scaling ──────────────────────────────────────────────────

QUIZ_DIFFICULTY_INSTRUCTIONS = {
    "easy": (
        "Generate a straightforward recall question. "
        "Use multiple choice with 4 options where wrong answers are clearly different. "
        "The question should test basic understanding of a single concept."
    ),
    "medium": (
        "Generate a question that requires understanding, not just recall. "
        "Options should be plausible — include common misconceptions as distractors. "
        "The question may combine two related concepts."
    ),
    "hard": (
        "Generate an application question where the student must apply the concept "
        "to a new scenario. Distractors should represent subtle misunderstandings. "
        "The question should require synthesis of multiple concepts."
    ),
}


def compute_quiz_difficulty(quiz_score: dict[str, Any]) -> str:
    """
    Determine quiz difficulty based on recent performance.

    Adapts dynamically: 3+ consecutive correct → harder,
    2+ consecutive incorrect → easier.
    """
    correct = quiz_score.get("correct", 0)
    total = quiz_score.get("total", 0)

    if total == 0:
        return "easy"  # Start easy

    accuracy = correct / total

    # Check recent streak from student_profile
    consecutive_correct = quiz_score.get("consecutive_correct", 0)
    consecutive_incorrect = quiz_score.get("consecutive_incorrect", 0)

    if consecutive_incorrect >= 2:
        return "easy"
    if consecutive_correct >= 3:
        return "hard"
    if accuracy >= 0.8 and total >= 3:
        return "hard"
    if accuracy >= 0.5:
        return "medium"
    return "easy"


# ── System Prompt Builder ────────────────────────────────────────────────────


def build_tutor_system_prompt(
    phase: str,
    topics_covered: list[str],
    quiz_score: dict[str, Any],
    explanation_mode: str,
    pacing: str,
) -> str:
    """
    Build the full system prompt for a tutoring interaction.

    Combines the base tutor persona with mode, pacing, and
    formatting instructions.
    """
    mode_instruction = EXPLANATION_MODES.get(explanation_mode, EXPLANATION_MODES["standard"])
    pacing_instruction = PACING_INSTRUCTIONS.get(pacing, PACING_INSTRUCTIONS["medium"])

    topics_str = ", ".join(topics_covered[-5:]) if topics_covered else "none yet"
    score_str = f"{quiz_score.get('correct', 0)}/{quiz_score.get('total', 0)}"

    return f"""You are SlideGuide, a patient and encouraging AI tutor helping a college student learn from their lecture slides.

CORE BEHAVIORS:
- Break explanations into 2-3 sentence chunks with clear **bold headers**
- Always cite slide numbers when referencing content (e.g., "From Slide 5:")
- If the student seems confused, try a DIFFERENT approach (don't repeat yourself)
- Be encouraging and supportive, especially after wrong answers
- Track what you've covered and suggest what to review next
- Never dump walls of text — keep responses digestible
- Use bullet points and numbered lists for complex information

EXPLANATION MODE: {explanation_mode}
{mode_instruction}

{pacing_instruction}

FORMATTING RULES:
- Use **bold** for key terms on first mention
- Use headers (## Topic) to break up sections
- Keep paragraphs to 2-3 sentences maximum
- Use > blockquotes for direct slide citations
- End responses with a question or suggested next step

CURRENT CONTEXT:
- Phase: {phase}
- Topics covered: {topics_str}
- Quiz score: {score_str}
- Explanation mode: {explanation_mode}
- Pacing: {pacing}"""


def get_encouragement(
    scenario: str,
    topic: str = "",
    streak: int = 0,
    hint: str = "",
    partial: str = "",
    count: int = 0,
) -> str:
    """Pick an encouragement template and fill in variables."""
    templates = ENCOURAGEMENT_TEMPLATES.get(scenario, [])
    if not templates:
        return "Keep going — you're making progress!"

    # Rotate through templates based on simple hash
    idx = (hash(topic + str(streak)) % len(templates))
    template = templates[idx]

    return template.format(
        topic=topic,
        streak=streak,
        hint=hint,
        partial=partial,
        count=count,
    )
