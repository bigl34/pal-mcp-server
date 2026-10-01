"""Per-call reasoning-effort flags for clink CLI agents."""

from __future__ import annotations

REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")


def effort_args(client_name: str, effort: str) -> list[str]:
    """Return the CLI flags that request ``effort`` from ``client_name``."""
    if effort not in REASONING_EFFORTS:
        raise ValueError(f"reasoning_effort must be one of {', '.join(REASONING_EFFORTS)}; got {effort!r}")
    client_key = client_name.lower()
    if client_key == "codex":
        return ["-c", f'model_reasoning_effort="{effort}"']
    if client_key == "claude":
        return ["--effort", effort]
    raise ValueError(f"CLI '{client_name}' does not support reasoning_effort")
