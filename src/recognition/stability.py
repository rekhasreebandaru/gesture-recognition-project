"""Consecutive-frame confirmation for live gesture predictions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StabilityResult:
    """The current candidate and the gesture confirmed for downstream layers."""

    candidate_label: str | None
    consecutive_frames: int
    confirmed_label: str | None


class ConsecutiveFrameStabilizer:
    """Confirm a label only after it is predicted on consecutive frames."""

    def __init__(self, required_consecutive_frames: int = 5) -> None:
        if required_consecutive_frames < 1:
            raise ValueError("required_consecutive_frames must be at least 1.")
        self.required_consecutive_frames = required_consecutive_frames
        self._candidate_label: str | None = None
        self._consecutive_frames = 0
        self._confirmed_label: str | None = None

    def update(self, label: str) -> StabilityResult:
        """Process one raw prediction and return the confirmed gesture state."""
        if not isinstance(label, str) or not label.strip():
            raise ValueError("label must be a non-empty string.")
        if label == self._candidate_label:
            self._consecutive_frames += 1
        else:
            self._candidate_label = label
            self._consecutive_frames = 1

        if self._consecutive_frames >= self.required_consecutive_frames:
            self._confirmed_label = label
        return self.current_result()

    def reset(self) -> StabilityResult:
        """Clear all state when no hand is detected."""
        self._candidate_label = None
        self._consecutive_frames = 0
        self._confirmed_label = None
        return self.current_result()

    def current_result(self) -> StabilityResult:
        """Return a read-only snapshot of the current state."""
        return StabilityResult(
            candidate_label=self._candidate_label,
            consecutive_frames=self._consecutive_frames,
            confirmed_label=self._confirmed_label,
        )
