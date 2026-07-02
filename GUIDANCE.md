# Guidance for Accelerator-Optimized Agentic Bidding on AWS, Part 1

**Category:** Advertising &amp; Marketing Technology  
**Industry:** Advertising, Media &amp; Entertainment  
**Products:** Amazon EKS, NVIDIA Triton Inference Server, Amazon Bedrock AgentCore, Amazon ECS Fargate  
**Published:** June 2026

## Overview

In programmatic advertising, the bidder that evaluates more signals and responds fastest wins. This guidance shows how NVIDIA GPU-accelerated compute and deep learning with NVIDIA Triton Inference Server can reduce bid-response latency while increasing the breadth of features evaluated per impression. This contributes to higher win rates and improved return on ad spend (ROAS).

The solution provides four production-ready ARTF-compliant containers, with three using GPU-accelerated inference served NVIDIA Triton Inference Server. Future releases will include ISV (Independent Software Vendor) partner containers demonstrating the ecosystem extensibility. It also includes an orchestration layer for parallel fan-out using gRPC as the primary ARTF protocol for the production auction path. Optionally, Amazon Bedrock AgentCore with Model Context Protocol (MCP) support is available as a testing and simulation interface for AI agent integration.

> **Note for the reader:** This Guidance is published in two parts. Part 1 (this edition) demonstrates how to implement ARTF-compliant containers that act as agents in a bidstream, determining ARTF intents to apply to a bid request while adhering to response-time SLAs. The containers leverage GPU-accelerated inference via NVIDIA Triton to meet sub-millisecond latency requirements.
>
> Part 2 (forthcoming) will introduce the additional infrastructural assets required to implement a closed-loop optimization architecture. This includes offline training pipelines (using NVIDIA NeMo-RL to support reinforcement learning workflows that use auction outcomes signals) that improves bidding behavior over time, NVIDIA TensorRT for optimizing compatible model artifacts, and NVIDIA NIM for standardized GPU-accelerated inference microservices where applicable. Triton Inference Server remains the real-time serving layer; it does not perform training. The training pipelines operate separately, producing updated ONNX models that are hot-loaded into Triton to improve bid decisions over time based on actual campaign performance data.

## Use Cases

1. **Bid price optimization:** Use deep learning CTR prediction (DLRM) to compute optimal shaded bid prices in real time, reducing overspend while maintaining win rates
2. **Audience segment activation:** Score user-segment affinities using Wide &amp; Deep neural networks and activate high-value audience segments at the impression level
3. **Private marketplace deal management:** Predict user-deal relevance with Neural Collaborative Filtering to autonomously activate high-affinity deals and suppress poor matches
4. **Quality metrics enrichment:** Add viewability and brand safety scores to bid requests before auction execution
5. **[Future Version] Creative intelligence:** Score creative quality, visual attention, brand suitability, and fatigue signals (ISV container)
6. **[Future Version] Identity resolution:** Resolve fragmented user/device signals into cross-device and household IDs (ISV container)
7. **[Future Version] Location audience activation:** Activate location-derived audience segments from device geo and visitation patterns (ISV container)
8. **Agentic advertising:** Enable AI agents to invoke real-time bidding decisions via MCP tool calls, supporting the transition to autonomous campaign optimization

## Business Benefits

| Benefit | Description |
|---------|-------------|
| Reduced decision-making latency | Sub-millisecond inference increases conversion rates and improves Return on Ad Spend (ROAS) |
| Expanded models improve optimization | Deep learning architectures (DLRM, Wide &amp; Deep, NCF) outperform linear models and heuristic rules, delivering better advertising outcomes |
| ISV partner ecosystem [Future] | Pre-built ISV partner containers enable DSPs to obtain new functionality faster without custom development |
| Agentic-ready platform | MCP interfaces enable AI agents to participate in bidding decisions, delivering a future-ready platform for autonomous advertising |

## Technical Benefits

| Benefit | Description |
|---------|-------------|
| Sub-millisecond inference | Dynamic batching on Triton with CUDA EP delivers GPU-accelerated inference well within OpenRTB timeout budgets |
| GPU acceleration | NVIDIA A10G GPUs via Triton deliver significant throughput improvement over equivalent CPU-based inference for recommender models |
| Deep learning CTR | DLRM, Wide &amp; Deep, and NCF architectures outperform linear models and heuristic rules for advertising decisions |
| Dynamic batching | Triton's preferred batch sizes (8, 16, 32) and max queue delay (500μs) maximize GPU utilization under concurrent load |
| Modular and extensible | ARTF container specification allows DSPs to add, swap, or update models independently without pipeline changes |
| ISV ecosystem ready [Future] | Three ISV partner containers demonstrate how third-party data providers plug into the same pipeline |
| Dual deployment paths | ECS Fargate for rapid demos (no GPU required) and EKS with Triton for production GPU inference |
| Agentic-ready | MCP interfaces (optional) enable AI agents to participate in bidding decisions via the extend_rtb tool |

## How It Works

### Step-by-Step Flow

1. **Bid request ingestion:** An OpenRTB bid request arrives via Amazon CloudFront and is routed through the load balancer to the orchestrator (CloudFront is for the front end testing tool).

2. **Orchestration:** The orchestrator (Starlette/Python) receives the request and fans it out in parallel to all registered ARTF containers.

   **GPU-accelerated inference:** The three GPU-backed containers (DLRM, Wide &amp; Deep, NCF) extract features from the bid request, invoke their assigned model on NVIDIA Triton Inference Server via tritonclient.http, and receive predictions from the GPU (A10G). The Metrics Enricher applies rule-based logic on CPU.

3. **Mutation generation:** Each container translates model predictions into typed ARTF mutations (bid price adjustments, segment activations, deal decisions, quality metrics).

4. **Response assembly:** The orchestrator merges all mutations from all containers into a single RTBResponse.

5. **Mutation application:** The DSP host platform applies approved mutations atomically to the bidstream before the auction continues.

### Architecture

![Architecture](assets/images/architecture.svg)

### ARTF Container Protocol Stack

Each ARTF container exposes three interfaces per the IAB Tech Lab ARTF v1.0 specification. The gRPC interface is the primary production protocol for real-time auction integration; the MCP interface is an optional testing and simulation endpoint for AI agent experimentation, and is not required for the core bidding stack:

| Port | Protocol | Endpoint | Description |
|------|----------|----------|-------------|
| 50051 | gRPC | RTBExtensionPoint.GetMutations | Primary ARTF protocol (protobuf) |
| 8081 | MCP (JSON-RPC) | POST /mcp | extend_rtb tool for AI agents (Streamable HTTP transport) |
| 8080 | HTTP | /health/live, /health/ready | Kubernetes liveness and readiness probes |


### Model Inference with NVIDIA Triton Inference Server

This Guidance implements three GPU-accelerated deep learning models, each chosen for its suitability to a specific advertising decision, plus a rule-based container demonstrating ARTF flexibility.

#### DLRM: Bid Shading (BID_SHADE)

**Architecture:** The Deep Learning Recommendation Model processes sparse categorical features (user IDs, site categories, device types) and dense numerical features (bid floors, time of day, user age, video presence). Bottom MLPs (4→32→16) transform dense features into embedding space; 3 EmbeddingBag tables (vocab=1000, dim=16) encode sparse features; dot-product interaction layers capture feature crosses; a top MLP (22→64→32→1→sigmoid) produces a CTR prediction.

**Triton config:** Model `dlrm_bid_shader`, ONNX backend, max batch size 64, 2× GPU instances, dynamic batching with preferred sizes [8, 16, 32] and 500μs max queue delay.

**Decision logic:** `shaded_price = min(original_bid, predicted_CTR × $12 conversion_value × 0.65 shade_factor)`, floored at the publisher's bidfloor.

#### Wide & Deep: Segment Activation (ACTIVATE_SEGMENTS)

**Architecture:** The Wide & Deep architecture combines a wide linear model (8→15) that memorizes specific feature cross-products (age×category, gender×domain, geo×device, floor×video) with a deep neural network (6→64→32→15 with BatchNorm) that generalizes to unseen feature combinations. Sigmoid fusion produces per-segment scores.

**Triton config:** Model `widedeep_segment_activator`, ONNX backend, dynamic batching, GPU instances.

**Decision logic:** Score 15 audience segments; activate those exceeding a 0.55 confidence threshold.

#### NCF: Deal Management (ACTIVATE_DEALS / SUPPRESS_DEALS)

**Architecture:** Neural Collaborative Filtering replaces traditional matrix factorization with a neural network, learning non-linear user-item interactions. In the RTB context, "items" are private marketplace deals and "users" are bidstream profiles. GMF path (user_embed × deal_embed, dim=64) captures linear interactions; MLP path (concat → 256→128→64) captures non-linear patterns; NeuMF fusion layer (128→1→sigmoid) produces relevance scores.

**Triton config:** Model `ncf_deal_manager`, ONNX backend, dynamic batching, GPU instances.

**Decision logic:** Per-deal relevance score; activate deals ≥0.499, suppress deals <0.497.


#### Metrics Enricher: Quality Scores (ADD_METRICS)

**Architecture:** Rule-based container (no GPU required). Unlike the three preceding containers (DLRM, Wide & Deep, NCF), which are powered by NVIDIA Triton Inference Server with GPU acceleration, the Metrics Enricher uses deterministic rules on CPU. This demonstrates that ARTF's container model is flexible enough to mix ML and deterministic logic in the same pipeline.

**Decision logic:** Viewability = f(ad position, banner dimensions, video presence); Brand safety = 0.60 + 0.40 × (safe_categories / total_categories).

### Container → Model → Intent Mapping

| Container | Model Architecture | Triton Model Name | ARTF Intent | Output |
|-----------|-------------------|-------------------|-------------|--------|
| DLRM Bid Shader | DLRM | dlrm_bid_shader | BID_SHADE | Optimal shaded bid price |
| Wide & Deep Activator | Wide & Deep | widedeep_segment_activator | ACTIVATE_SEGMENTS | Audience segments |
| NCF Deal Manager | NCF / NeuMF | ncf_deal_manager | ACTIVATE_DEALS / SUPPRESS_DEALS | Deal activations / suppressions |
| Metrics Enricher | Rule engine (CPU) | N/A | ADD_METRICS | Viewability + brand safety |
| [Future] Creative Enricher (ISV) | ViT/CLIP mock (CPU) | N/A | ADD_METRICS | Creative quality, attention, suitability, fatigue |
| [Future] Identity Resolver (ISV) | Graph NN mock (CPU) | N/A | ADD_CIDS | Cross-device + household IDs |
| [Future] Location Activator (ISV) | Blueprints™ mock (CPU) | N/A | ACTIVATE_SEGMENTS | Location-derived audience segments |


### AWS Services in This Guidance

| AWS Service | Role in This Guidance |
|-------------|----------------------|
| Amazon Elastic Kubernetes Service (EKS) | Orchestrates GPU and CPU node groups; manages container lifecycle, scaling, and health |
| Amazon EC2 (g5.xlarge) | Provides NVIDIA A10G GPU instances for Triton Inference Server |
| Amazon EC2 (c5.xlarge) | Runs ARTF containers, orchestrator, and metrics enricher on CPU |
| Amazon ECS Fargate | Alternative serverless deployment for demos without GPU (CPU-only inference) |
| Amazon S3 | Stores ONNX model repository (Triton) and static frontend assets |
| Amazon CloudFront | HTTPS edge delivery for testing frontend; proxies API requests to the cluster |
| Amazon ECR | Stores container images for all ARTF containers, orchestrator, and AgentCore bundle |
| Elastic Load Balancing (NLB/ALB) | NLB for EKS path (TCP pass-through); ALB for ECS path (HTTP routing) |
| AWS IAM (IRSA) | IAM Roles for Service Accounts grants Triton S3 read access without long-lived credentials |
| AWS CloudMap | Service discovery for ECS Fargate containers (internal DNS) |
| Amazon Bedrock AgentCore | Hosts the MCP runtime for AI agent integration via the extend_rtb tool |


### NVIDIA Acceleration and Integration Components

| Component | Version | Purpose |
|-----------|---------|---------|
| NVIDIA Triton Inference Server | nvcr.io/nvidia/tritonserver:24.08-py3 | Multi-model serving with dynamic batching on GPU |
| ONNX Runtime | Built into Triton | Backend execution for ONNX-exported models |
| CUDA Execution Provider | CUDA 12.x | GPU-accelerated inference |
| NVIDIA Kubernetes Device Plugin | v0.15.0 | Exposes GPU resources to Kubernetes scheduler |
| NVIDIA DCGM Exporter | Latest | GPU metrics for Prometheus/Grafana monitoring |
| tritonclient[http] | Latest | Python client SDK in ARTF containers |

### Part 2: NVIDIA Software Roadmap

Part 2 will extend the NVIDIA software stack with two additional components that complement the Triton inference path established in Part 1:

- **NVIDIA NeMo-RL** can support reinforcement learning workflows that use auction outcome signals to improve bidding behavior over time. Training runs offline on GPU clusters; outputs from those workflows can inform updates to trained bidding models, such as DLRM, Wide & Deep, or NCF-based implementations, which can be exported as ONNX models and deployed to Triton.

- **NVIDIA NIM** provides packaged, GPU-accelerated inference microservices for supported models. NIM can complement the Triton inference path for approved model artifacts where applicable, while model versioning, A/B deployment, traffic shifting, and deployment automation should be handled by the platform governance and deployment workflow.


## Well-Architected Pillars

### Operational Excellence

- **Infrastructure as Code:** The entire solution deploys via a single `deploy.sh` script that provisions EKS (GPU + CPU node groups), ONNX model export, S3 model upload, Kubernetes manifests, CloudFront distribution, and AgentCore runtime
- **Idempotent deployments:** Re-running `deploy.sh` reuses existing resources, re-exports models, rebuilds images, and applies manifests in place with zero manual intervention
- **Observability:** NVIDIA DCGM Exporter provides GPU utilization metrics; Triton exposes Prometheus metrics on port 8002; Kubernetes health probes ensure container readiness; CloudFront access logs capture request patterns
- **Container lifecycle:** Amazon ECR stores versioned images tagged with git commit SHA; Kubernetes rolling updates enable zero-downtime deployments

### Security

- **Least privilege:** IAM Roles for Service Accounts (IRSA) grant only S3 read access to the Triton pod; no long-lived credentials in containers
- **Non-root execution:** All ARTF containers run as appuser (non-root) per the ARTF specification
- **Security hardening:** no-new-privileges security option and read-only filesystem in container runtime
- **Network isolation:** ARTF containers communicate only within the cluster; external traffic enters exclusively through CloudFront → NLB/ALB
- **Image provenance:** Base images sourced from NVIDIA NGC (authenticated registry) and Python official images; application containers stored in private ECR

### Reliability

- **Multi-AZ deployment:** EKS node groups span multiple Availability Zones for fault tolerance
- **Health probes:** Every container implements /health/live and /health/ready endpoints for automatic pod replacement on failure
- **Graceful degradation:** The orchestrator continues with available container responses if one container times out (respects tmax budget)
- **Startup probes:** Triton uses startup probes with 30 retries to handle model loading time without false-positive restarts

### Performance Efficiency

- **GPU-accelerated inference:** NVIDIA A10G GPUs with CUDA Execution Provider for deep learning recommender models
- **Dynamic batching:** Triton batches concurrent requests (preferred sizes 8/16/32, max 500μs queue delay) to maximize GPU throughput
- **Multi-model serving:** A single Triton instance serves all three ONNX models (DLRM, Wide & Deep, NCF) with 2 GPU instances each
- **Parallel fan-out:** The orchestrator invokes all ARTF containers simultaneously; total latency equals the slowest container, not the sum

### Cost Optimization

- **Right-sized instances:** GPU nodes (g5.xlarge) run only Triton; lightweight ARTF containers run on cost-effective CPU nodes (c5.xlarge)
- **Horizontal Pod Autoscaler:** Kubernetes HPA scales pods based on actual request load, avoiding over-provisioning
- **Dual deployment model:** ECS Fargate path enables cost-effective demos (~$0/hr when idle) without provisioning GPU infrastructure
- **Deterministic naming:** Resource names include sha256(stack:account:region)[:8] suffix enabling multiple isolated stacks in one account without collision

### Sustainability

- **GPU efficiency:** Dynamic batching and multi-model serving maximize compute utilization per watt of GPU power consumed
- **Serverless edge:** CloudFront handles static assets and TLS termination without dedicated compute
- **Right-sized scaling:** Autoscaling ensures resources match demand rather than provisioning for peak at all times

## Plan Your Deployment

### Prerequisites

- An AWS account with permissions to create EKS clusters, EC2 instances (including g5 GPU instances), S3 buckets, ECR repositories, and IAM roles
- AWS CLI v2 with valid credentials
- Docker with buildx (for ARM64 cross-compilation)
- Python 3.11+ with boto3, torch, onnx, onnxscript
- jq, eksctl, kubectl
- Access to NVIDIA NGC registry for the Triton Inference Server image (`nvcr.io/nvidia/tritonserver:24.08-py3`)
- Service quota for at least one g5.xlarge instance in the target region

### Supported Regions

This Guidance can be deployed in any AWS Region that supports Amazon EKS and NVIDIA A10G instances (g5 family), including:

- US East (N. Virginia): `us-east-1`
- US West (Oregon): `us-west-2`
- Europe (Ireland): `eu-west-1`
- Europe (Frankfurt): `eu-central-1`
- Asia Pacific (Tokyo): `ap-northeast-1`
- Asia Pacific (Sydney): `ap-southeast-2`

### Deployment Steps

Full deployment (EKS + Triton + Frontend + AgentCore):

```bash
./deploy.sh
```

This single command provisions all infrastructure:

| Step | What | AWS Resources |
|------|------|---------------|
| 1 | ECR repositories | 9 repos (7 containers + orchestrator + agentcore) |
| 2 | Export ONNX models | PyTorch → ONNX via triton/export_models.py |
| 3 | Upload models to S3 | S3 model repository bucket |
| 4 | Build & push images | AMD64 for EKS, ARM64 for AgentCore |
| 5 | EKS cluster | g5.xlarge GPU nodes + c5.xlarge CPU nodes |
| 6 | NVIDIA Device Plugin | GPU scheduling in Kubernetes |
| 7 | IRSA configuration | IAM role for Triton S3 access |
| 8 | Kubernetes manifests | Triton server, ARTF containers, orchestrator, HPA |
| 9 | Frontend | S3 + CloudFront distribution |
| 10 | AgentCore | MCP runtime registration |

Deploy options:

```bash
./deploy.sh --prefix v1                    # namespaced resources (v1-nvidia-artf-*)
./deploy.sh --prefix prod --skip-agentcore # EKS + frontend only
./deploy.sh --skip-images                  # reuse existing images
./deploy.sh --skip-cluster                 # reuse existing EKS cluster
./deploy.sh --export-only                  # just export ONNX models
./deploy.sh --ui-only                      # redeploy frontend only
./deploy.sh --destroy                      # tear down everything
./deploy.sh --destroy --prefix v1          # tear down a specific stack
AWS_REGION=us-west-2 ./deploy.sh           # different region
```

Local development:

```bash
docker compose up --build
# Frontend at http://localhost:8081
# Orchestrator at http://localhost:8080/v1/mutations
```

### Integration with a DSP

1. **Model export:** Export your proprietary bidding models to ONNX format using `triton/export_models.py` as reference
2. **Upload to S3:** Place models in the Triton model repository following the `model_name/version/model.onnx` convention with a `config.pbtxt`
3. **Custom ARTF containers:** Implement the ARTF mutation logic specific to your bidding decisions, using the provided containers as reference implementations
4. **Register with orchestrator:** Add your containers to the orchestrator fan-out configuration
5. **Connect to bid pipeline:** Route OpenRTB bid requests through the orchestrator before auction execution

## Scaling Scenarios

The default deployment (1 GPU node) is designed for demos and development. Production DSPs handling real auction traffic need to scale horizontally. The solution includes built-in autoscaling at multiple layers.

### Scaling Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Scaling Layers                                                      │
│                                                                      │
│  Layer 1: Pod Autoscaling (HPA)                                      │
│    Orchestrator:  2→10 pods  (CPU utilization > 70%)                 │
│    Triton:        1→4 pods   (GPU utilization > 75% or CPU > 80%)    │
│                                                                      │
│  Layer 2: Node Autoscaling (Cluster Autoscaler)                      │
│    GPU nodes:     1→3 g5.xlarge  (when Triton pods are pending)      │
│    CPU nodes:     1→4 c5.xlarge  (when ARTF/orchestrator pods pend)  │
│                                                                      │
│  Layer 3: Triton Dynamic Batching                                    │
│    Preferred batch sizes: [8, 16, 32], max batch: 64                 │
│    Max queue delay: 500μs                                            │
│    GPU instances per model: 2                                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Production Scaling Scenarios

| Scenario | QPS | GPU Nodes | CPU Nodes | Triton Pods | Orchestrator Pods | Est. Monthly |
|----------|-----|-----------|-----------|-------------|-------------------|--------------|
| Demo / Dev | <100 | 1× g5.xlarge | 2× c5.xlarge | 1 | 2 | ~$1,080 |
| Small DSP | 1K–10K | 2× g5.xlarge | 3× c5.xlarge | 2 | 4 | ~$2,000 |
| Mid-market DSP | 10K–100K | 3× g5.xlarge | 4× c5.xlarge | 3–4 | 6–8 | ~$3,000 |
| Large DSP | 100K–500K | 6× g5.2xlarge | 8× c5.2xlarge | 6 | 10+ | ~$8,500 |
| Enterprise DSP | 500K–1M+ | 12× g5.4xlarge | 16× c5.4xlarge | 12 | 20+ | ~$22,000 |

### Instance Type Selection Guide

| Instance | GPU | GPU Memory | vCPU | RAM | Use Case |
|----------|-----|------------|------|-----|----------|
| g5.xlarge | 1× A10G | 24 GB | 4 | 16 GB | Demo, dev, small DSP |
| g5.2xlarge | 1× A10G | 24 GB | 8 | 32 GB | Mid-market DSP (more CPU headroom for Triton) |
| g5.4xlarge | 1× A10G | 24 GB | 16 | 64 GB | Large DSP (high-concurrency Triton serving) |
| g5.12xlarge | 4× A10G | 96 GB | 48 | 192 GB | Enterprise (multi-GPU Triton, many models) |
| p4d.24xlarge | 8× A100 | 320 GB | 96 | 1152 GB | Extreme scale (large models, TensorRT-LLM) |

> **Future GPU path:** This Guidance currently deploys on NVIDIA A10G GPUs (g5 instance family). As NVIDIA Blackwell-architecture GPUs become available on AWS (g7e instance family), they will offer significantly higher inference throughput and improved power efficiency. The Triton-based serving architecture is GPU-generation agnostic, requiring only updated instance types in the cluster configuration.

## Cost Estimation

### EKS Production Path (GPU)

| Resource | Configuration | Est. Monthly Cost |
|----------|---------------|-------------------|
| EKS Cluster | 1 cluster | $73 |
| GPU Node (g5.xlarge) | 1 instance (On-Demand) | ~$727 |
| CPU Nodes (c5.xlarge) | 2 instances (On-Demand) | ~$245 |
| NAT Gateway | 1 gateway + data transfer | ~$32 |
| S3 (Models + Frontend) | ~50 MB storage | <$1 |
| CloudFront | Low-traffic demo | <$1 |
| ECR | Image storage | <$1 |
| **Estimated Total** | | **~$1,080/month** |

GPU costs dominate. For demos: deploy, test, then `./deploy.sh --destroy` immediately. A 2-hour demo session costs approximately $3.

### ECS Fargate Demo Path (No GPU)

| Resource | Configuration | Est. Monthly Cost |
|----------|---------------|-------------------|
| ECS Fargate | 5 tasks (0.25 vCPU, 512 MB each) | ~$45 |
| ALB | 1 load balancer | ~$25 |
| NAT Gateway | 1 gateway | ~$32 |
| S3 + CloudFront | Frontend hosting | <$5 |
| **Estimated Total** | | **~$107/month** |

## Amazon Bedrock AgentCore Integration

The AgentCore deployment is an optional component for testing and AI agent simulation. It bundles all ARTF containers into a single ARM64 image that runs inside an AgentCore microVM, exposing the `extend_rtb` MCP tool on port 8000 at `/mcp`. AgentCore is not required for the core production bidding stack, which uses gRPC exclusively for real-time auction integration.

### AgentCore Configuration

```json
{
  "agents": [
    {
      "name": "NvidiaArtfRecommenders",
      "language": "Python",
      "framework": "Custom",
      "type": "create",
      "codeLocation": "agentcore",
      "entrypoint": "artf_mcp_server.py",
      "build": "Container",
      "protocol": "MCP",
      "networkMode": "PUBLIC",
      "memory": "none"
    }
  ]
}
```

### Invoke via AgentCore

```python
import boto3, json

client = boto3.client("bedrock-agentcore", region_name="us-east-1")
response = client.invoke_agent_runtime(
    agentRuntimeArn="arn:aws:bedrock-agentcore:us-east-1:ACCOUNT:runtime/NvidiaArtfRecommenders",
    payload=json.dumps({
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {
            "name": "extend_rtb",
            "arguments": {
                "id": "auction-123",
                "tmax": 100,
                "applicable_intents": ["ACTIVATE_SEGMENTS", "ADD_METRICS"],
                "bid_request": {
                    "id": "auction-123",
                    "imp": [{"id": "imp-1", "banner": {"w": 300, "h": 250}, "bidfloor": 1.50}],
                    "site": {"domain": "espn.com", "cat": ["IAB17"]},
                    "user": {"id": "user-789", "yob": 1990}
                }
            }
        }
    }).encode()
)
```

### AgentCore-Only Deploy

```bash
./deploy-agentcore.sh
./deploy-agentcore.sh --prefix v1
./deploy-agentcore.sh --destroy --prefix v1
```

## ARTF Compliance

Each container meets the IAB Tech Lab ARTF v1.0 specification:

| Requirement | Implementation |
|-------------|----------------|
| Non-root user | `adduser appuser` + `USER appuser` in Dockerfile |
| agent-manifest Docker label | JSON label with name, version, vendor, intents, health probes |
| gRPC RTBExtensionPoint.GetMutations | Generic handler on port 50051 |
| MCP extend_rtb tool | JSON-RPC at /mcp with Streamable HTTP transport and Mcp-Session-Id session management |
| Health probes | /health/live and /health/ready on port 8080 |
| applicable_intents filtering | Each container checks `intent_applicable()` before processing |
| tmax timeout respect | Orchestrator passes tmax/1000 as HTTP timeout to containers |
| Read-only filesystem | `read_only: true` in container runtime |
| No-new-privileges | `no-new-privileges:true` security option |
| Typed mutations | Mutation objects with intent, op, path, and typed payloads |

## Related Content

### AWS Resources

- [Amazon Bedrock AgentCore Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/agentcore.html)
- [Amazon EKS Best Practices Guide: Running GPU Workloads](https://aws.github.io/aws-eks-best-practices/gpu/)
- [Scaling Inference with Triton on Amazon EKS](https://aws.amazon.com/blogs/machine-learning/)

### NVIDIA Resources

- [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server)
- [NVIDIA Triton Client Libraries](https://github.com/triton-inference-server/client)
- [NVIDIA Deep Learning Recommender Models (DLRM)](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Recommendation/DLRM)
- [NVIDIA Recommender Systems Collection](https://developer.nvidia.com/recommender-systems)
- [NVIDIA TensorRT: Model Optimization](https://developer.nvidia.com/tensorrt)
- [NVIDIA Triton Inference Server: Optimization Guide](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/optimization.html)

### Industry Standards

- [IAB Tech Lab: Agentic RTB Framework (ARTF) v1.0 Specification](https://iabtechlab.com/artf)
- [ARTF Reference Implementation (Go)](https://github.com/nicholasgasior/artf)
- [ARTF MCP Integration Guide](https://iabtechlab.com/artf-mcp)
- [OpenRTB 2.6 Specification](https://iabtechlab.com/openrtb)
- [Model Context Protocol (MCP) Specification](https://modelcontextprotocol.io)

### Academic Papers

- [Deep Learning Recommendation Model for Personalization and Recommendation Systems (Naumov et al. 2019)](https://arxiv.org/abs/1906.00091)
- [Wide & Deep Learning for Recommender Systems (Cheng et al. 2016)](https://arxiv.org/abs/1606.07792)
- [Neural Collaborative Filtering (He et al. 2017)](https://arxiv.org/abs/1708.05031)

## Source Code

The complete source code for this Guidance is available at:

**Repository:** [aws-solutions-library-samples/guidance-for-accelerator-optimized-agentic-bidding-on-aws](https://github.com/aws-solutions-library-samples/guidance-for-accelerator-optimized-agentic-bidding-on-aws)

### Includes

- ARTF container implementations (DLRM, Wide & Deep, NCF, Metrics Enricher)
- Orchestrator with parallel fan-out (gRPC primary, HTTP fallback)
- Model export scripts (PyTorch → ONNX via triton/export_models.py)
- Triton model repository with config.pbtxt configurations
- Kubernetes manifests with HPA (EKS deployment)
- ECS Fargate deployment with CloudMap service discovery
- Testing frontend (CloudFront + S3)
- Amazon Bedrock AgentCore MCP runtime (ARM64)
- Single-command deployment script (deploy.sh)
- Local development via `docker compose up --build`

---

*© 2026 Amazon Web Services, Inc. or its affiliates. All rights reserved.*
