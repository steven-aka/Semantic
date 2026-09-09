from __future__ import annotations

from collections.abc import Sequence

from src.data.schemas import SemanticPacket


def build_representation(packets: Sequence[SemanticPacket], states: Sequence[int]) -> str:
    if len(packets) != len(states):
        raise ValueError("packets and states must have equal length")
    output: list[str] = []
    previous_unit_id = -1
    for packet, state in zip(packets, states):
        if packet.unit_id <= previous_unit_id:
            raise ValueError("packets must be in strictly increasing source-unit order")
        previous_unit_id = packet.unit_id
        if state not in (0, 1, 2):
            raise ValueError("each state must be 0, 1, or 2")
        if state >= 1 and packet.gist.strip():
            output.append(packet.gist.strip())
        if state >= 2 and packet.residual.strip():
            output.append(packet.residual.strip())
    return "\n".join(output)


def state_from_threshold(c: float, tau_g: float, tau_r: float) -> int:
    if not 0 <= c <= 1 or not 0 <= tau_g <= tau_r <= 1:
        raise ValueError("require c in [0,1] and 0 <= tau_g <= tau_r <= 1")
    if c < tau_g:
        return 0
    if c < tau_r:
        return 1
    return 2


def states_at_fidelity(
    c: float, tau_g: Sequence[float], tau_r: Sequence[float]
) -> tuple[int, ...]:
    if len(tau_g) != len(tau_r):
        raise ValueError("tau_g and tau_r must have equal length")
    return tuple(state_from_threshold(c, gist, residual) for gist, residual in zip(tau_g, tau_r))

