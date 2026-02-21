"""Tests for the agent state and graph components."""

import pytest
from backend.agent.state import TutorState, create_initial_state
from backend.agent.prompts import (
    build_tutor_system_prompt,
    compute_quiz_difficulty,
    get_encouragement,
    EXPLANATION_MODES,
    PACING_INSTRUCTIONS,
)


class TestTutorState:
    """Tests for agent state management."""

    def test_create_initial_state(self):
        """Initial state has correct defaults."""
        state = create_initial_state("session-1", "upload-1")

        assert state["session_id"] == "session-1"
        assert state["upload_id"] == "upload-1"
        assert state["current_phase"] == "greeting"
        assert state["messages"] == []
        assert state["topics_covered"] == []
        assert state["quiz_score"]["correct"] == 0
        assert state["quiz_score"]["total"] == 0
        assert state["explanation_mode"] == "standard"
        assert state["pacing_preference"] == "medium"
        assert state["encouragement_due"] is False
        assert state["error_count"] == 0

    def test_initial_state_student_profile(self):
        """Student profile starts with neutral values."""
        state = create_initial_state("s1", "u1")
        profile = state["student_profile"]

        assert profile["confidence_level"] == 0.5
        assert profile["preferred_mode"] == "standard"
        assert profile["consecutive_correct"] == 0
        assert profile["consecutive_incorrect"] == 0


class TestGraphRouting:
    """Tests for graph routing functions."""

    def test_route_from_router_teaching(self):
        """Teaching phase routes to explain node."""
        from backend.agent.graph import route_from_router

        state = create_initial_state("s", "u")
        state["current_phase"] = "teaching"
        result = route_from_router(state)
        assert result == "explain"

    def test_route_from_router_quiz(self):
        """Quiz phase routes to quiz node."""
        from backend.agent.graph import route_from_router

        state = create_initial_state("s", "u")
        state["current_phase"] = "quiz"
        result = route_from_router(state)
        assert result == "quiz"

    def test_route_from_router_wrap_up(self):
        """Wrap-up phase routes to END."""
        from backend.agent.graph import route_from_router

        state = create_initial_state("s", "u")
        state["current_phase"] = "wrap_up"
        result = route_from_router(state)
        assert result == "__end__"

    def test_route_from_router_encouragement_override(self):
        """When encouragement is due, route overrides to encourage."""
        from backend.agent.graph import route_from_router

        state = create_initial_state("s", "u")
        state["current_phase"] = "teaching"
        state["encouragement_due"] = True
        result = route_from_router(state)
        assert result == "encourage"

    def test_should_use_tools_no_messages(self):
        """No messages means no tools needed."""
        from backend.agent.graph import should_use_tools

        state = create_initial_state("s", "u")
        result = should_use_tools(state)
        assert result == "__end__"


class TestPrompts:
    """Tests for the prompts module."""

    def test_all_explanation_modes_exist(self):
        """All 5 explanation modes are defined."""
        expected = ["standard", "analogy", "visual", "step_by_step", "eli5"]
        for mode in expected:
            assert mode in EXPLANATION_MODES

    def test_all_pacing_levels_exist(self):
        """All 3 pacing levels are defined."""
        expected = ["slow", "medium", "fast"]
        for pacing in expected:
            assert pacing in PACING_INSTRUCTIONS

    def test_build_tutor_system_prompt_includes_mode(self):
        """System prompt includes the current explanation mode."""
        prompt = build_tutor_system_prompt(
            phase="teaching",
            topics_covered=["loops", "arrays"],
            quiz_score={"correct": 3, "total": 5},
            explanation_mode="analogy",
            pacing="slow",
        )
        assert "analogy" in prompt.lower()
        assert "slow" in prompt.lower() or "PACING" in prompt

    def test_compute_quiz_difficulty_starts_easy(self):
        """With no quiz history, difficulty starts at easy."""
        assert compute_quiz_difficulty({"correct": 0, "total": 0}) == "easy"

    def test_compute_quiz_difficulty_scales_up(self):
        """High accuracy with enough attempts returns hard."""
        result = compute_quiz_difficulty({
            "correct": 9, "total": 10,
            "consecutive_correct": 3, "consecutive_incorrect": 0,
        })
        assert result == "hard"

    def test_compute_quiz_difficulty_scales_down(self):
        """Consecutive incorrect answers return easy."""
        result = compute_quiz_difficulty({
            "correct": 2, "total": 10,
            "consecutive_correct": 0, "consecutive_incorrect": 3,
        })
        assert result == "easy"

    def test_get_encouragement_returns_string(self):
        """Encouragement function returns a non-empty string."""
        msg = get_encouragement("correct_answer", topic="loops")
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_get_encouragement_unknown_scenario(self):
        """Unknown scenario returns fallback message."""
        msg = get_encouragement("unknown_scenario")
        assert "progress" in msg.lower()
