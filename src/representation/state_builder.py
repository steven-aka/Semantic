from __future__ import annotations

from collections.abc import Sequence

from src.data.schemas import AtomicPacket, SemanticPacket


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


def build_atomic_representation(
    packets: Sequence[AtomicPacket], states: Sequence[int]
) -> str:
    if len(packets) != len(states):
        raise ValueError("packets and states must have equal length")
    if [packet.packet_id for packet in packets] != list(range(len(packets))):
        raise ValueError("atomic packets must have contiguous source-order ids")
    if any(state not in (0, 1) for state in states):
        raise ValueError("each atomic state must be 0 or 1")
    return "\n\n".join(
        packet.text.strip() for packet, state in zip(packets, states) if state == 1
    )


def atomic_states_at_fidelity(
    c: float, reveal_thresholds: Sequence[float]
) -> tuple[int, ...]:
    if not 0 <= c <= 1:
        raise ValueError("require c in [0,1]")
    if any(not 0 <= threshold <= 1 for threshold in reveal_thresholds):
        raise ValueError("reveal thresholds must be in [0,1]")
    return tuple(int(c >= threshold) for threshold in reveal_thresholds)
