# Shared Dockerfile for all ARTF Python containers.
# Build context must be the repo root so shared/ is available.
#
# Usage:
#   docker build --build-arg CONTAINER=containers/dlrm_bid_shader -t dlrm-bid-shader .
#
# Each container exposes three protocols per ARTF spec:
#   - gRPC  :50051  RTBExtensionPoint.GetMutations
#   - MCP   :8081   JSON-RPC at /mcp (extend_rtb tool)
#   - Health :8080   /health/live, /health/ready

FROM python:3.12-slim AS base

ARG CONTAINER
ARG AGENT_NAME=artf-nvidia-agent
ARG VERSION=0.1.0

LABEL agent-manifest="{ \
  \"name\": \"${AGENT_NAME}\", \
  \"version\": \"${VERSION}\", \
  \"vendor\": \"NVIDIA\", \
  \"resources\": {\"cpu\": \"500m\", \"memory\": \"512Mi\"}, \
  \"intents\": [\"ACTIVATE_SEGMENTS\",\"ACTIVATE_DEALS\",\"SUPPRESS_DEALS\",\"BID_SHADE\",\"ADD_METRICS\"], \
  \"health\": { \
    \"livenessProbe\": {\"httpGet\": {\"path\": \"/health/live\", \"port\": 8080}}, \
    \"readinessProbe\": {\"httpGet\": {\"path\": \"/health/ready\", \"port\": 8080}} \
  } \
}"

RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] starlette httpx pydantic \
    grpcio protobuf boto3 \
    "python-jose[cryptography]==3.3.0" \
    && pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY shared/ /app/shared/
COPY ${CONTAINER}/ /app/container/

RUN adduser --disabled-password --no-create-home appuser
USER appuser

# gRPC + MCP + Health
EXPOSE 50051 8081 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health/ready')" || exit 1

# Run the multi-protocol ARTF server
CMD ["python", "-m", "container.app"]
