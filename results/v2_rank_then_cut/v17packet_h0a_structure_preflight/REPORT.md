# PACKET-H0A source-structure preflight

## Result

The proposed source-restoration interpretation of PACKET-H0 does not match the
implemented V8 pipeline.

Across all 2,032 lineage-clean training queries:

- every query has exactly 12 V8 packets;
- all 24,384 packets contain exactly one `Document title` and one
  `Source evidence` block;
- joining the 12 packets reproduces the stored source context exactly for
  2,032/2,032 queries;
- every packet exactly matches one pre-packetization source block;
- the upstream 6-unit representation contains exactly two source blocks per
  unit, and carries no title, source-sentence, table-row, or list-item metadata
  beyond the text already present in each block.

The base packetizer therefore does not split sentences into the short atoms
seen in later SEM/R1 experiments. Those fragments are post-hoc candidate
actions created inside diagnostic scripts. The earlier `15/64 -> 40/64`
eligibility change shows that joining those diagnostic fragments restores
more complete relations; it is not evidence that V8's 12 packet boundaries
destroy relations.

Merging the two blocks in each upstream unit would join two distinct
title-backed source blocks. It would reduce granularity and cannot be described
as recovering a split proposition. Conversely, splitting the current packets
into sentences would introduce the fragmentation risk that PACKET-H0 was meant
to remove. Table-row and list-item reconstruction is unavailable because those
boundaries are not preserved in the canonical artifact.

Decision: `STOP_PACKET_H0_NO_DISTINCT_STRUCTURAL_REPACKETIZATION`.

PACKET-H0B Target calls are not authorized. This result does not prove that no
alternative text packetization could ever help; it shows that the proposed
source-preserving repair is not a distinct intervention in the actual V8 data
path. Testing it would either reproduce V8 or conflate unrelated source blocks.
No Teacher, Target, development, confirmation, or sealed-set access occurred.
