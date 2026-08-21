"""Recording stand-ins for ``cursor_sdk.Agent`` network calls.

Process e2e tests use the real ``AgentOptions`` / ``ModelSelection`` types
and only replace the Agent factory so no Cursor usage is billed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cursor_sdk import SDKModel


@dataclass
class RecordedCall:
    options: Any
    message: Any


@dataclass
class FakeUsage:
    input_tokens: int = 8
    output_tokens: int = 3
    total_tokens: int = 11


@dataclass
class FakeResult:
    status: str = "finished"
    result: str = "Osmosis is water moving across a membrane."
    id: str = "run-e2e"
    model: Any = field(default_factory=lambda: type("M", (), {"id": "grok-4.6"})())
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeRun:
    def __init__(
        self,
        result: FakeResult,
        tokens: tuple[str, ...] = ("Osmosis ", "works."),
    ) -> None:
        self._result = result
        self._tokens = tokens

    def wait(self) -> FakeResult:
        return self._result

    def iter_text(self):
        yield from self._tokens


class RecordingAgent:
    """Context-manager Agent that records ``create`` / ``send`` arguments."""

    last: RecordedCall | None = None

    def __init__(self, options: Any) -> None:
        self.options = options

    def __enter__(self) -> RecordingAgent:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def send(self, message: Any) -> FakeRun:
        RecordingAgent.last = RecordedCall(options=self.options, message=message)
        return FakeRun(FakeResult())

    @staticmethod
    def create(options: Any, **_kwargs: Any) -> RecordingAgent:
        return RecordingAgent(options)


class FakeCursorModels:
    """``Cursor.models.list`` stand-in that returns official ``SDKModel`` rows."""

    def list(self, *, api_key: str | None = None, client: Any = None) -> list[SDKModel]:
        del client
        assert api_key == "crsr_e2e_test"
        return [
            SDKModel(id="gpt-5.5", display_name="GPT 5.5"),
            SDKModel(id="composer-2.5", display_name="Composer 2.5"),
            SDKModel(id="auto-smart", display_name="Cursor Router"),
            SDKModel(id="grok-4.6", display_name="Grok 4.6"),
        ]


def model_id_and_params(model: Any) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Normalize a string or ``ModelSelection`` into id + params."""
    if hasattr(model, "id"):
        params = tuple((p.id, p.value) for p in getattr(model, "params", ()) or ())
        return model.id, params
    return str(model), ()
