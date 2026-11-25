"""Stable MediaPipe handedness parsing and hand-slot assignment."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional


HAND_SLOTS = {"left": 0, "right": 1}


def handedness_label(categories: Sequence[Any] | None) -> Optional[str]:
    """Return ``left``/``right`` from a MediaPipe handedness category list."""
    if not categories:
        return None

    best = max(
        categories,
        key=lambda category: float(getattr(category, "score", 0.0) or 0.0),
    )
    for attr in ("category_name", "display_name"):
        value = getattr(best, attr, None)
        if not isinstance(value, str):
            continue
        normalized = value.strip().lower()
        for label in HAND_SLOTS:
            if (
                normalized == label
                or normalized.startswith(label + "_")
                or normalized.startswith(label + " ")
            ):
                return label
    return None


def assign_hand_slots(
    labels: Sequence[Optional[str]], max_hands: int = 2
) -> list[Optional[int]]:
    """Assign left/right hands to stable slots, falling back to free slots."""
    max_hands = int(max_hands)
    if max_hands <= 0:
        return [None] * len(labels)

    slots: list[Optional[int]] = [None] * len(labels)
    used: set[int] = set()

    for i, label in enumerate(labels):
        slot = HAND_SLOTS.get(label)
        if slot is not None and slot < max_hands and slot not in used:
            slots[i] = slot
            used.add(slot)

    free_slots = [slot for slot in range(max_hands) if slot not in used]
    for i, slot in enumerate(slots):
        if slot is None and free_slots:
            slots[i] = free_slots.pop(0)

    return slots