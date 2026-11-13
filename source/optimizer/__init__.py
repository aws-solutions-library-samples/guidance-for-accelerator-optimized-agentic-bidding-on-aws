"""Model Optimizer — the design's model-optimization component (TensorRT-based).

Separate from Triton serving, per IAB Tech Lab ARTF: the real-time serving path
(ARTF containers -> in-cluster Triton) has no external dependency. This service
runs out-of-band during model promotion: it compiles a newly-trained ONNX model
into an optimized TensorRT engine plan that Triton then loads as a new version.

It is deliberately NOT called "NIM" and is NOT a stock NVIDIA NIM container (none
exists for these custom DLRM/NCF/Wide&Deep recommender models). It uses NVIDIA
TensorRT directly — the same engine NIM is built on — to produce the optimized
engine the design calls for.
"""
