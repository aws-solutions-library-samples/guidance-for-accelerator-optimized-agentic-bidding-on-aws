# NeMo-RL Training Container for ARTF Recommender Models
# ======================================================
#
# This container runs SageMaker training jobs that fine-tune the DLRM, NCF,
# and Wide&Deep models using NVIDIA NeMo Framework with reinforcement learning
# from bid outcome signals (CTR, win rate, ROI).
#
# Base image: NVIDIA NeMo Framework (includes PyTorch, Megatron-Core,
# NeMo-RL, CUDA toolkit, NCCL for multi-GPU training).
#
# Training loop:
#   1. Load base model checkpoint (ONNX → PyTorch weights)
#   2. Load labeled training data (Parquet from S3, prepared by Glue ETL)
#   3. Supervised pre-training pass on labeled click/conversion data
#   4. RL fine-tuning pass using NeMo-RL with custom reward (ROI-based)
#   5. Export trained model to ONNX
#   6. Upload artifact to S3 output path (SageMaker convention)
#
# SageMaker contract:
#   - Training data in /opt/ml/input/data/training/
#   - Model output to /opt/ml/model/
#   - Hyperparameters in /opt/ml/input/config/hyperparameters.json
#   - Resource config in /opt/ml/input/config/resourceconfig.json
#
# Usage:
#   Built and pushed to ECR, then referenced by source/training/pipeline.py
#   which creates SageMaker training jobs using this image.

FROM nvcr.io/nvidia/nemo:24.07

LABEL maintainer="ARTF Bidding Platform"
LABEL description="NeMo-RL training container for recommender model retraining"
LABEL nvidia.nemo.version="24.07"

# Install additional dependencies for ONNX export and data loading
RUN pip install --no-cache-dir \
    onnx==1.16.1 \
    onnxscript==0.1.0.dev20240601 \
    pyarrow==16.1.0 \
    pandas==2.2.2 \
    boto3==1.34.131 \
    scikit-learn==1.5.0

# Copy training source code
COPY train.py /opt/ml/code/train.py
COPY reward.py /opt/ml/code/reward.py
COPY models/ /opt/ml/code/models/

WORKDIR /opt/ml/code

# SageMaker expects the entrypoint to be the training script
ENV SAGEMAKER_PROGRAM=train.py

ENTRYPOINT ["python", "/opt/ml/code/train.py"]
