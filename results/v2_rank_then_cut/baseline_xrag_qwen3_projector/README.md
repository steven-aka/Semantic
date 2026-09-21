# xRAG-style Qwen3 projector baseline

The official xRAG implementation targets Mistral/Mixtral. This directory records a
clearly qualified Qwen3 port: a frozen 768-dimensional QAMPARI retriever embedding is
mapped by a two-layer GELU projector to one Qwen3-8B input embedding. Both retriever
and Target are frozen; only the 19,931,136 projector parameters are optimized.

The preflight observed finite projector gradients, zero Target parameter gradients,
an unchanged Target fingerprint, and 19,446,410,240 peak allocated CUDA bytes.
