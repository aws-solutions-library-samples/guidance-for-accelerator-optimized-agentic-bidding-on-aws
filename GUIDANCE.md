# Guidance for Accelerator-Optimized Agentic Bidding on AWS, Part 1

**Category:** Advertising &amp; Marketing Technology  
**Industry:** Advertising, Media &amp; Entertainment  
**Products:** Amazon EKS, NVIDIA Triton Inference Server, Amazon Bedrock AgentCore  
**Published:** June 2026

## Overview

In programmatic advertising, the bidder that evaluates more signals and responds fastest wins. This guidance shows how NVIDIA GPU-accelerated compute and deep learning with NVIDIA Triton Inference Server can reduce bid-response latency while increasing the breadth of features evaluated per impression. This contributes to higher win rates and improved return on ad spend (ROAS).

The solution provides five production-ready ARTF-compliant containers, each doing one job in the bidstream — pricing bids, activating audience segments, scoring private marketplace deals, enriching quality signals, and optimizing publisher yield. Three run GPU-accelerated inference on NVIDIA Triton Inference Server (two deep-learning models via ONNX/TensorRT, one tree model via Triton's Forest Inference Library backend); two use deterministic, rule-based logic on CPU. Future releases will include ISV (Independent Software Vendor) partner containers demonstrating the ecosystem extensibility — including a partner segment-activation model slated to replace the current rule-based audience activator. It also includes an orchestration layer for parallel fan-out over the ARTF extension point; each hop (Prebid hook to orchestrator, orchestrator to containers, containers to Triton) is switchable between HTTP and gRPC per deployment. Optionally, Amazon Bedrock AgentCore with Model Context Protocol (MCP) support is available as a testing and simulation interface for AI agent integration.

> **Note for the reader:** This Guidance is published in two parts. Part 1 (this edition) demonstrates how to implement ARTF-compliant containers that act as agents in a bidstream, determining ARTF intents to apply to a bid request while adhering to response-time SLAs. The containers leverage GPU-accelerated inference via NVIDIA Triton to meet sub-millisecond latency requirements.
>
> Part 2 introduces the additional infrastructural assets required to implement a closed-loop optimization architecture. This includes offline training pipelines (using NVIDIA NeMo-RL to support reinforcement learning workflows that use auction-outcome signals) that improve bidding behavior over time, and NVIDIA TensorRT for optimizing model artifacts — Part 2 upgrades Triton serving from ONNX Runtime to compiled `tensorrt_plan` engines, built by an in-cluster **Model Optimizer** microservice (a TensorRT optimizer, not a stock NVIDIA NIM — none exists for these custom recommender models). Triton Inference Server remains the real-time serving layer; it does not perform training. The training pipelines operate separately, producing updated ONNX models that are compiled to TensorRT engines and rolled out to Triton — via a live stable/canary router — to improve bid decisions over time based on actual campaign performance data.

## Use Cases

1. **Bid price optimization:** The bid pricer uses deep learning CTR prediction (a DLRM model) to compute optimal shaded bid prices in real time, reducing overspend while maintaining win rates
2. **Audience segment activation:** The audience activator activates high-value audience segments at the impression level from real bid-request signals (content category, demographics, existing DMP segments, contextual signals) via transparent rules — slated for a partner ISV neural-model implementation
3. **Private marketplace deal management:** The deal scorer predicts user-deal relevance with Neural Collaborative Filtering to autonomously activate high-affinity deals and suppress poor matches
4. **Quality metrics enrichment:** The signals enricher adds viewability and brand safety scores to bid requests before auction execution
5. **Publisher yield optimization:** The yield optimizer predicts a floor-price multiplier and margin adjustment per private marketplace deal from real deal/context signals, using an XGBoost model served by Triton's Forest Inference Library backend — an SSP/publisher-side example distinct from the buy-side bid pricer
6. **[Future Version] Creative intelligence:** Score creative quality, visual attention, brand suitability, and fatigue signals (ISV container)
7. **[Future Version] Identity resolution:** Resolve fragmented user/device signals into cross-device and household IDs (ISV container)
8. **[Future Version] Location audience activation:** Activate location-derived audience segments from device geo and visitation patterns (ISV container)
9. **Agentic advertising:** Enable AI agents to invoke real-time bidding decisions via MCP tool calls, supporting the transition to autonomous campaign optimization

## Business Benefits

| Benefit | Description |
|---------|-------------|
| Reduced decision-making latency | Sub-millisecond inference increases conversion rates and improves Return on Ad Spend (ROAS) |
| Expanded models improve optimization | Deep learning architectures (DLRM, NCF) and tree-based models (the yield optimizer's XGBoost) outperform linear models and heuristic rules for the decisions they cover, delivering better advertising outcomes |
| ISV partner ecosystem [Future] | Pre-built ISV partner containers enable DSPs to obtain new functionality faster without custom development |
| Agentic-ready platform | MCP interfaces enable AI agents to participate in bidding decisions, delivering a future-ready platform for autonomous advertising |

## Technical Benefits

| Benefit | Description |
|---------|-------------|
| Sub-millisecond inference | Dynamic batching on Triton with CUDA EP delivers GPU-accelerated inference well within OpenRTB timeout budgets |
| GPU acceleration | NVIDIA A10G GPUs via Triton deliver significant throughput improvement over equivalent CPU-based inference for recommender models |
| Deep learning CTR | The bid pricer and deal scorer's deep learning models (DLRM and NCF) outperform linear models and heuristic rules for the advertising decisions they cover |
| Tree-model yield optimization | The yield optimizer's XGBoost model, served by Triton's FIL backend, follows the published approach for reserve-price optimization — a tree/regression problem, not a deep-embedding one |
| Dynamic batching | Triton's preferred batch sizes (8, 16, 32) and max queue delay (500μs) maximize GPU utilization under concurrent load |
| Modular and extensible | ARTF container specification allows DSPs to add, swap, or update models independently without pipeline changes |
| ISV ecosystem ready [Future] | Three ISV partner containers demonstrate how third-party data providers plug into the same pipeline |
| Agentic-ready | MCP interfaces (optional) enable AI agents to participate in bidding decisions via the extend_rtb tool |

## How It Works

### Step-by-Step Flow

1. **Bid request ingestion:** An OpenRTB bid request reaches the orchestrator over in-cluster Service DNS. From the testing frontend (static files on CloudFront + S3), the browser invokes the UI API proxy Lambda with SigV4 credentials from the Cognito Identity Pool; the Lambda, attached to the cluster VPC, forwards the call to the orchestrator's internal Network Load Balancer (`orchestrator-internal`). From Prebid Server (optional second host) the hook calls the orchestrator directly in-cluster. The orchestrator has no public address.

2. **Orchestration:** The orchestrator (Starlette/Python) receives the request and fans it out in parallel to all registered ARTF containers.

   **GPU-accelerated inference:** The three GPU-backed containers — the bid pricer, the deal scorer, and the yield optimizer — extract features from the bid request, invoke their assigned model (DLRM, NCF, and XGBoost via FIL, respectively) on NVIDIA Triton Inference Server via tritonclient.http, and receive predictions from the GPU (A10G). The audience activator and signals enricher apply rule-based logic on CPU.

3. **Mutation generation:** Each container translates model predictions into typed ARTF mutations (bid price adjustments, segment activations, deal decisions, quality metrics).

4. **Response assembly:** The orchestrator merges all mutations from all containers into a single RTBResponse.

5. **Mutation application:** The DSP host platform applies approved mutations atomically to the bidstream before the auction continues.

### Architecture

![Architecture](assets/images/architecture.svg)

### ARTF Container Protocol Stack

Each ARTF container exposes three interfaces per the IAB Tech Lab ARTF v1.0 specification. The gRPC interface is the ARTF-specified wire protocol for real-time auction integration and the orchestrator can fan out over it or over the REST equivalent (`ARTF_CONTAINER_TRANSPORT`); the MCP interface is an optional testing and simulation endpoint for AI agent experimentation, and is not required for the core bidding stack:

| Port | Protocol | Endpoint | Description |
|------|----------|----------|-------------|
| 50051 | gRPC | RTBExtensionPoint.GetMutations | ARTF extension point (JSON payload over gRPC; protobuf schema vendored under `source/proto/`) |
| 8081 | MCP (JSON-RPC) | POST /mcp | extend_rtb tool for AI agents (Streamable HTTP transport) |
| 8080 | HTTP | /health/live, /health/ready | Kubernetes liveness and readiness probes |


### The Five Containers

Each container does one job in the bidstream. Three are GPU-accelerated on
NVIDIA Triton Inference Server (two deep-learning models, one tree model); two
are rule-based on CPU. The model architecture behind each GPU-accelerated
container is implementation detail, noted below for reference — see
[Container → Model → Intent Mapping](#container--model--intent-mapping)
for the full picture.

#### Bid Pricer: bid pricing (BID_SHADE)

**What it does:** Prices every bid at the lowest amount still likely to win —
shading down from the maximum bid based on how likely the impression is to
convert, so the platform doesn't overpay for impressions it would have won at a
lower price.

**How:** Two separate steps, deliberately kept apart — a prediction about the
world, then a decision about strategy.

First the model estimates **how likely the response the advertiser pays for is**.
Which response that is comes from the training objective: `cpa` (the default)
trains against conversions, `cpc` against clicks, `profitable_win` against a
profitable win. The objective is recorded in the model's manifest, so a served
artifact always states which response its probability refers to.

Then an explicit policy turns that probability into a price. Expected value is
`predicted_probability × conversion_value`, and the policy is a parametric curve
over it:

```
price = clamp(base + slope × expected_value ^ curvature, bid_floor, original_bid)
```

The three coefficients live in `source/shared/shading_policy.py` and are read at
serving time from Parameter Store, so pricing behaviour can be retuned without
redeploying the model. The shipped values — `base=0.0, slope=0.65, curvature=1.0`
— reduce the curve to `min(original_bid, expected_value × 0.65)` floored at the
publisher's bidfloor, which is the straight-line case. Raising `curvature` above
1 bids proportionally less on low-value impressions and more on high-value ones;
`base` sets a floor the policy will not shade below.

Separating the two matters because they are learned differently. The probability
is fitted by gradient descent and its calibration is measured; the coefficients
are searched against an ROI objective, and gradient descent is deliberately kept
away from the probability head so that search cannot quietly undo the
calibration.

**Features:** four dense (`bid_floor`, `hour_norm`, `is_weekend`, `has_video`)
and three categorical (`site_domain`, `device_type`, `geo_country`, with vocabs
of 10000 / 32 / 256). The vector is defined once in
`source/shared/dlrm_features.py` and imported by both the trainer and the serving
container, so the two cannot disagree about what position two means. It is
versioned: a served artifact carries its `feature_spec_version`, and the
promotion path refuses a model whose version it cannot interpret.

**Model (implementation detail):** A Deep Learning Recommendation Model (DLRM),
168,881 parameters. A bottom MLP (4→32→16) lifts the dense features into
embedding space; three `nn.Embedding` tables (dim=16) encode the categoricals; a
dot-product interaction layer captures feature crosses; a top MLP (22→64→32→1)
produces a logit. The sigmoid is applied by the export wrapper rather than inside
the top MLP, so the trained network and the served graph share one definition of
the probability. Served by Triton as model `dlrm_bid_shader` (ONNX backend, max
batch size 64, 2× GPU instances, dynamic batching with preferred sizes
[8, 16, 32] and 500μs max queue delay).

#### Audience Activator: audience segment activation (ACTIVATE_SEGMENTS)

**What it does:** Activates high-value audience segments on an impression from
signals already on the bid request — content category, demographics, existing
first/third-party DMP segments, and contextual signals (bid floor tier, video
inventory, device type) — so downstream targeting can act on segments the buy
side cares about.

**How:** A deterministic rule engine (no GPU, no Triton) scores candidate
segments against the signals above, and activates those exceeding a 0.55
confidence threshold (overridable via `segment_threshold`).

**Model (implementation detail):** This container previously scored segments
with a Wide & Deep neural network on Triton — a wide linear model (8→15)
combined with a deep network (6→64→32→15 with BatchNorm). That model's ONNX
graph could not be compiled to a TensorRT engine (the `BatchNorm1d` fusion is
unsupported for this shape by TensorRT 10.3.0's builder), so it was replaced
with the transparent rules described above, and this container is slated to be
replaced by a partner ISV neural-model implementation.

#### Deal Scorer: private marketplace deal management (ACTIVATE_DEALS / SUPPRESS_DEALS)

**What it does:** Scores how well each private marketplace (PMP) deal fits the
current impression, autonomously activating high-affinity deals and suppressing
poor matches — so PMP inventory converts at a higher rate without a human
reviewing every deal on every impression.

**How:** Produces a per-deal relevance score from the bid-request profile and
each candidate deal; activates deals scoring ≥0.499, suppresses deals scoring
<0.497.

**Model (implementation detail):** Neural Collaborative Filtering (NCF/NeuMF),
which learns non-linear user-item interactions rather than relying on
traditional matrix factorization. In the RTB context, "items" are PMP deals and
"users" are bidstream profiles. A GMF path (user_embed × deal_embed, dim=64)
captures linear interactions; an MLP path (concat → 256→128→64) captures
non-linear patterns; a NeuMF fusion layer (128→1→sigmoid) produces the relevance
score. Served by Triton as model `ncf_deal_manager` (ONNX backend, dynamic
batching, GPU instances).

#### Signals Enricher: quality signal enrichment (ADD_METRICS)

**What it does:** Adds viewability and brand-safety scores to the bid request
before auction execution, so downstream bidding logic (including the bid
pricer) can factor real quality signals into the decision rather than treating
every impression as equally brand-safe and viewable.

**How:** A rule-based container (no GPU required) — the same pattern used by the
audience activator, demonstrating that ARTF's container model mixes ML and
deterministic logic in one pipeline. Viewability = f(ad position, banner
dimensions, video presence); brand safety = 0.60 + 0.40 × (safe_categories /
total_categories).

#### Yield Optimizer: publisher deal floor & margin adjustment (ADJUST_DEAL_FLOOR / ADJUST_DEAL_MARGIN)

**What it does:** An SSP/publisher-side example — predicts a floor-price
multiplier and a margin adjustment per private marketplace deal, so a publisher
can raise floors on high-demand inventory and lower them on remnant inventory
without manual deal management, and set margins appropriately for the deal's
auction type. Distinct from the (buy-side) bid pricer: this adjusts what the
*seller* will accept, not what the buyer bids.

**Two containers, two models.** The floor and the margin are served by two
separate containers — `yield-optimizer-floor` and `yield-optimizer-margin` —
each owning one single-target XGBoost model. This is not an arbitrary division:
Triton's FIL backend cannot serve a multi-output tree model, so a floor and a
margin prediction were always two distinct models. Giving each its own container
means each can be retrained, rolled out, and scaled on its own, and one model
being down no longer takes the other's intent offline. `ADJUST_DEAL_FLOOR` and
`ADJUST_DEAL_MARGIN` are independent, atomic mutations, so nothing in the
bidstream contract couples them either.

**How:** Each container builds the same 7-feature vector per deal from real
signals already on the bid request (auction type, existing bidfloor, IAB
content-category tier, hour-of-day, day-of-week — no fabricated signals), sends
it to its own Triton-served XGBoost model, and emits its one intent per deal
when that model recommends a real change. A prediction of "no change" (a floor
multiplier of exactly 1.0, or a margin of exactly 0.0) never becomes a mutation.

**Model (implementation detail):** Two XGBoost tree ensembles, served by
NVIDIA Triton's Forest Inference Library (FIL) backend rather than the
ONNX/TensorRT path DLRM/NCF use — FIL is purpose-built for GPU-accelerated
tree-model inference and is already bundled in the same
`nvcr.io/nvidia/tritonserver:24.08-py3` image this Guidance uses, so no
additional Triton image is required. Tree/regression models are the approach
used in the published literature for reserve-price optimization, unlike the
deep embedding architectures (DLRM, NCF) used for CTR prediction and
relevance scoring. Served by Triton as models `deal_yield_manager_floor` and
`deal_yield_manager_margin` (FIL backend, GPU instance). Those two model names
predate the container split and are deliberately unchanged — they identify
already-registered Triton models and SageMaker Model Package Groups.

### Container → Model → Intent Mapping

| Container | What it does | Model (implementation detail) | Triton Model Name | ARTF Intent | Output |
|-----------|--------------|-------------------------------|-------------------|-------------|--------|
| Bid Pricer | Prices bids by shading down from a CTR prediction | DLRM | dlrm_bid_shader | BID_SHADE | Optimal shaded bid price |
| Audience Activator | Activates audience segments from bid-request signals | Rule engine (CPU) — slated for a partner ISV neural model | N/A | ACTIVATE_SEGMENTS | Audience segments |
| Deal Scorer | Scores and activates/suppresses PMP deals | NCF / NeuMF | ncf_deal_manager | ACTIVATE_DEALS / SUPPRESS_DEALS | Deal activations / suppressions |
| Signals Enricher | Adds viewability + brand-safety quality signals | Rule engine (CPU) | N/A | ADD_METRICS | Viewability + brand safety |
| Yield Optimizer — Floor | Predicts deal floor-price adjustments | XGBoost (Triton FIL backend) | deal_yield_manager_floor | ADJUST_DEAL_FLOOR | Floor multiplier |
| Yield Optimizer — Margin | Predicts deal margin adjustments | XGBoost (Triton FIL backend) | deal_yield_manager_margin | ADJUST_DEAL_MARGIN | Margin value |
| [Future] Creative Enricher (ISV) | Scores creative quality signals | ViT/CLIP mock (CPU) | N/A | ADD_METRICS | Creative quality, attention, suitability, fatigue |
| [Future] Identity Resolver (ISV) | Resolves cross-device identity | Graph NN mock (CPU) | N/A | ADD_CIDS | Cross-device + household IDs |
| [Future] Location Activator (ISV) | Activates location-derived segments | Blueprints™ mock (CPU) | N/A | ACTIVATE_SEGMENTS | Location-derived audience segments |


### AWS Services in This Guidance

| AWS Service | Role in This Guidance |
|-------------|----------------------|
| Amazon Elastic Kubernetes Service (EKS) | Orchestrates GPU and CPU node groups; manages container lifecycle, scaling, and health |
| Amazon EC2 (g5.xlarge; g5.2xlarge and g5.4xlarge as capacity fallbacks) | Provides NVIDIA A10G GPU instances for Triton Inference Server (`gpu-inference` node group, 1 node by default) |
| Amazon EC2 (c5.2xlarge) | Runs the ARTF containers, the orchestrator and the signals enricher on CPU (`cpu-services` node group, 3 nodes by default) |
| Amazon VPC (NAT gateways) | One NAT gateway per Availability Zone (three); all nodes run in private subnets |
| AWS CodeBuild | Builds the x86 and arm64 container images by default, so no local Docker is needed |
| Elastic Load Balancing (internal NLBs) | In-VPC endpoints for Triton (`triton-internal`) and for the orchestrator (`orchestrator-internal`, the UI API proxy's target); neither is reachable from the internet |
| Amazon S3 | Stores ONNX model repository (Triton) and static frontend assets |
| Amazon CloudFront | HTTPS edge delivery of the testing frontend's static assets (S3 origin only) |
| Amazon ECR | Stores container images for all ARTF containers, orchestrator, and AgentCore bundle |
| AWS Lambda (UI API proxy) | VPC-attached function the browser invokes with SigV4; forwards UI API calls to the orchestrator's internal NLB so nothing in the VPC is internet-facing |
| Amazon Cognito (User Pool + Identity Pool) | Signs users in; exchanges the ID token for temporary credentials scoped to invoking the UI API proxy and the closed-loop agents |
| AWS IAM (IRSA) | IAM Roles for Service Accounts grants Triton S3 read access without long-lived credentials |
| Amazon Bedrock AgentCore | Hosts the MCP runtime for AI agent integration via the extend_rtb tool (deployed by default; `--skip-agentcore` omits it) |
| Amazon DynamoDB | `loadtest-history` and `container-registry` tables (on-demand) |
| AWS Secrets Manager | Stores the NVIDIA NGC API key used by the Part 2 training-image build |


### NVIDIA Acceleration and Integration Components

| Component | Version | Purpose |
|-----------|---------|---------|
| NVIDIA Triton Inference Server | nvcr.io/nvidia/tritonserver:24.08-py3 | Multi-model serving with dynamic batching on GPU |
| ONNX Runtime | Built into Triton | Part 1 backend for ONNX-exported models (Part 2 upgrades serving to `tensorrt_plan`) |
| Triton FIL (Forest Inference Library) backend | Built into Triton | GPU-accelerated serving backend for the yield optimizer's XGBoost tree model — no ONNX/TensorRT conversion step |
| NVIDIA TensorRT (`tensorrt_plan`) | trtexec 24.08 | Part 2 serving backend — compiled TensorRT engine plans on Triton |
| Model Optimizer microservice | nvcr.io/nvidia/tensorrt:24.08-py3 | In-cluster ONNX→TensorRT engine compilation (FP16 default); a TensorRT optimizer, **not** a stock NVIDIA NIM |
| CUDA Execution Provider | CUDA 12.x | GPU-accelerated inference |
| NVIDIA Kubernetes Device Plugin | v0.15.0 | Exposes GPU resources to Kubernetes scheduler |
| NVIDIA DCGM Exporter | Latest | GPU metrics for Prometheus/Grafana monitoring |
| tritonclient[http] | Latest | Python client SDK in ARTF containers |

### Part 2: NVIDIA Software Stack

Part 2 extends the NVIDIA software stack with components that build on the Triton inference path established in Part 1, and upgrades that path from ONNX Runtime to compiled TensorRT engines:

- **NVIDIA NeMo-RL** supports reinforcement-learning workflows that use auction-outcome signals to improve bidding behavior over time. Training runs offline on GPU clusters; the resulting DLRM or NCF artifacts are exported to ONNX and registered for promotion. (The audience activator is rule-based and has no trainable artifact.)

- **NVIDIA TensorRT (Model Optimizer).** Part 2 upgrades Triton serving to `tensorrt_plan`: an in-cluster **Model Optimizer** microservice (built on `nvcr.io/nvidia/tensorrt:24.08-py3`) compiles each ONNX artifact into an optimized TensorRT engine (FP16 by default; INT8 only with a real calibration cache, else an honest `400`). This is a TensorRT optimizer, **not** a stock NVIDIA NIM — no stock NIM exists for these custom recommender architectures, and TensorRT is the same engine a NIM is built on. Model versioning and zero-downtime A/B rollout are handled by the Model Governance Agent driving a **Triton-side canary router**: an in-memory `<model>_stable`↔`<model>_canary` split set at model load/reload (control-plane), never a per-request lookup, so the real-time ARTF bidstream stays dependency-free and the ARTF containers are never modified.


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
- **Network isolation:** ARTF containers and the orchestrator communicate only within the cluster; the orchestrator has no public address (a ClusterIP Service in-cluster, an internal NLB for the UI API proxy). The only path in from the testing frontend is browser, then `lambda:InvokeFunction` (SigV4, Cognito Identity Pool), then the UI API proxy Lambda inside the VPC, then the orchestrator's internal NLB, with the user's Cognito bearer token forwarded for the orchestrator's own JWT check. This works unchanged in accounts running VPC Block Public Access in block-ingress mode
- **Image provenance:** Base images sourced from NVIDIA NGC (authenticated registry) and Python official images; application containers stored in private ECR

### Reliability

- **Multi-AZ deployment:** EKS node groups span multiple Availability Zones for fault tolerance
- **Health probes:** Every container implements /health/live and /health/ready endpoints for automatic pod replacement on failure
- **Graceful degradation:** The orchestrator continues with available container responses if one container times out (respects tmax budget)
- **Startup probes:** Triton uses startup probes with 30 retries to handle model loading time without false-positive restarts

### Performance Efficiency

- **GPU-accelerated inference:** NVIDIA A10G GPUs with CUDA Execution Provider for deep learning recommender models
- **Dynamic batching:** Triton batches concurrent requests (preferred sizes 8/16/32, max 500μs queue delay) to maximize GPU throughput
- **Multi-model serving:** A single Triton instance serves both ONNX models (DLRM, NCF) with 2 GPU instances each
- **Parallel fan-out:** The orchestrator invokes all ARTF containers simultaneously; total latency equals the slowest container, not the sum

### Cost Optimization

- **Right-sized instances:** GPU nodes (g5.xlarge) run only Triton; the ARTF containers run on CPU nodes (c5.2xlarge) by default (`--artf-node-role services`)
- **Scheduled GPU shutdown:** the deploy installs a scheduled action that scales the GPU node group to zero at 8:00 PM America/New_York every day; the UI's Start GPUs button brings it back on demand
- **Horizontal Pod Autoscaler:** Kubernetes HPA scales pods based on actual request load, avoiding over-provisioning
- **Deterministic naming:** Resource names include sha256(stack:account:region)[:8] suffix enabling multiple isolated stacks in one account without collision

### Sustainability

- **GPU efficiency:** Dynamic batching and multi-model serving maximize compute utilization per watt of GPU power consumed
- **Serverless edge:** CloudFront handles static assets and TLS termination without dedicated compute
- **Right-sized scaling:** Autoscaling ensures resources match demand rather than provisioning for peak at all times

## Plan Your Deployment

### Prerequisites

- An AWS account with permissions to create EKS clusters, EC2 instances (including g5 GPU instances), VPCs with NAT gateways, S3 buckets, ECR repositories, CodeBuild projects, CloudFront distributions, Cognito pools, DynamoDB tables, Lambda functions, Bedrock AgentCore runtimes, CloudFormation stacks and IAM roles (Part 2, on by default, adds Kinesis, Firehose, KMS, Glue, SageMaker, EventBridge Scheduler, Secrets Manager, SNS and Amazon Bedrock model access; see [GUIDANCE-part2.md](GUIDANCE-part2.md#prerequisites))
- AWS CLI v2 with valid credentials
- Python 3.11+ with boto3, torch, onnx, onnxscript (and sagemaker, which `deploy.sh` installs itself when Part 2 is enabled)
- jq, eksctl, kubectl
- Docker with buildx only if you pass `--local-build`; images build on AWS CodeBuild by default
- An NVIDIA NGC API key (`--ngc-key`) for the gated NeMo-RL training base image, unless you pass `--no-retraining`. The Triton Inference Server image (`nvcr.io/nvidia/tritonserver:24.08-py3`) is public and needs no key
- Service quota for at least one g5.xlarge instance in the target region (the node group also accepts g5.2xlarge and g5.4xlarge)

### Supported Regions

This Guidance can be deployed in any AWS Region that supports Amazon EKS and NVIDIA A10G instances (g5 family), including:

- US East (N. Virginia): `us-east-1`
- US West (Oregon): `us-west-2`
- Europe (Ireland): `eu-west-1`
- Europe (Frankfurt): `eu-central-1`
- Asia Pacific (Tokyo): `ap-northeast-1`
- Asia Pacific (Sydney): `ap-southeast-2`

### Deployment Steps

Full deployment (EKS + Triton + frontend + MCP runtime + Part 2 closed-loop learning, which is on by default):

```bash
cd deployment
./deploy.sh --prefix dv1 --ngc-key YOUR_NGC_API_KEY
```

`--prefix` is required on every run: exactly three characters, a letter followed by letters or digits. It names every resource (`dv1-nvidia-artf-recommenders-*`) and keys the local record of remembered settings. The region defaults to `us-east-1` (`AWS_REGION` overrides it); the AWS profile resolves as `--profile`, then `AWS_PROFILE`, then the profile remembered for the prefix, then `default`.

This single command runs five phases:

| Phase | What | AWS resources created (defaults) |
|-------|------|----------------------------------|
| 1 | Prepare models | 9 ECR repositories (`<stack>-*`), DynamoDB tables `<stack>-loadtest-history` and `<stack>-container-registry` (on-demand), ONNX export of DLRM/NCF and the genesis XGBoost models, S3 model bucket `<stack>-triton-models-<id>` |
| 2 | Build images and provision the cluster, in parallel | CodeBuild stack `<stack>-codebuild` (x86 and arm64 projects) and source bucket; EKS cluster `<stack>-triton` (Kubernetes 1.31, 3 AZs, private nodes, 3 NAT gateways) with node groups `gpu-inference` (1 × g5.xlarge, min 1, max `--maxGPUs`, default 3) and `cpu-services` (3 × c5.2xlarge, min 2, max 8); NVIDIA device plugin; IRSA roles for Triton and the Model Optimizer |
| 3 | Deploy workloads | Cognito user pool, app client and identity pool; Triton with an internal NLB; the six ARTF containers, the orchestrator (ClusterIP plus the `orchestrator-internal` internal NLB) and HPAs; a one-shot TensorRT bootstrap Job; the `<prefix>-ui-api-proxy` Lambda stack pointed at that NLB and verified with one request; the GPU node group's nightly scheduled shutdown (desired and minimum 0 at 8:00 PM America/New_York) |
| 4 | Set up access | Frontend bucket `<stack>-frontend-<id>`, CloudFront distribution (S3 origin only), demo Cognito user `admin@example.com` |
| 5 | Register agents | Bedrock AgentCore MCP runtime `<stack>_mcp` (unless `--skip-agentcore`); then, unless `--no-retraining`, `deploy_closed_loop.sh` provisions Part 2 ([GUIDANCE-part2.md](GUIDANCE-part2.md#deployment-steps)); then, only with `--with-prebid`, `deploy_prebid.sh` adds Prebid Server as a second ARTF host |

Deploy options (each flag's default is in parentheses):

```bash
./deploy.sh --prefix dv1 --profile prof     # AWS CLI profile (AWS_PROFILE, else remembered, else "default")
./deploy.sh --prefix dv1 --no-retraining    # Part 1 only (default: Part 2 on, --with-retraining)
./deploy.sh --prefix dv1 --skip-agentcore   # no MCP runtime (default: registered)
./deploy.sh --prefix dv1 --with-prebid      # add Prebid Server as a second host (default: off)
./deploy.sh --prefix dv1 --local-build      # build images with local Docker (default: --remote-build on CodeBuild)
./deploy.sh --prefix dv1 --skip-images      # reuse images already in ECR
./deploy.sh --prefix dv1 --skip-cluster     # reuse the existing EKS cluster
./deploy.sh --prefix dv1 --maxGPUs 5        # GPU node group maximum (default 3)
./deploy.sh --prefix dv1 --artf-on-gpu      # run the ARTF containers on the GPU node (default: CPU node group)
./deploy.sh --prefix dv1 --model-id ID      # Bedrock model for both agents (default: global.anthropic.claude-opus-4-8)
./deploy.sh --prefix dv1 --start-at 3       # force a starting phase, 1-5 (default: probe AWS, run what is missing)
./deploy.sh --prefix dv1 --export-only      # only export and upload the models
./deploy.sh --prefix dv1 --ui-only          # only redeploy the frontend
./deploy.sh --prefix dv1 --status           # read-only live status table
./deploy.sh --prefix dv1 --destroy          # tear the prefix down
AWS_REGION=us-west-2 ./deploy.sh --prefix dv1   # different region (default: us-east-1)
```

Re-running the same command is the way to resume: every phase is idempotent and the script probes AWS first and runs only what is missing. The full flag reference, including environment variables, is in [README.md](README.md#customizing-your-deployment).

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

The default deployment (1 GPU node, 3 CPU nodes) is designed for demos and development. Production DSPs handling real auction traffic need to scale horizontally. The solution scales at three layers, two of them automatic.

### Scaling Architecture

- **Pod autoscaling (automatic, Horizontal Pod Autoscalers).** The orchestrator runs 2 to 10 pods on CPU utilization above 70%. Each of the six ARTF containers runs 2 to 5 pods on CPU above 70% (the `artf-template` placeholder runs 1 to 5). Triton runs 1 to 4 pods on GPU utilization above 75% or CPU above 80%. HPAs place new pods only where node capacity already exists.
- **Node scaling (manual).** No Cluster Autoscaler or Karpenter is installed. The `gpu-inference` node group is created with 1 node (minimum 1, maximum `--maxGPUs`, default 3) and the `cpu-services` node group with 3 `c5.2xlarge` nodes (minimum 2, maximum 8). To add capacity, scale the node group (`eksctl scale nodegroup` or the Auto Scaling console) or edit `deployment/eks/cluster-config.yaml` before deploying; a Triton pod that needs a GPU no node offers stays Pending until you do. The only automatic node-count change in the deployment is the nightly scheduled action that sets the GPU group to 0 at 8:00 PM America/New_York.
- **Triton dynamic batching (automatic).** Preferred batch sizes 8, 16 and 32, maximum batch 64, maximum queue delay 500 microseconds, 2 GPU instances per model.

### Production Scaling Scenarios

Monthly figures are `us-east-1` on-demand compute only (EKS control plane $73, g5.xlarge $1.006/h, g5.2xlarge $1.212/h, g5.4xlarge $1.624/h, c5.2xlarge $0.34/h, c5.4xlarge $0.68/h, 730 h), with every node running 24/7 and three NAT gateways ($99). They exclude Part 2, data processing and storage.

| Scenario | QPS | GPU nodes | CPU nodes | Triton pods | Orchestrator pods | Est. monthly compute |
|----------|-----|-----------|-----------|-------------|-------------------|----------------------|
| Demo / Dev (shipped default) | <100 | 1 × g5.xlarge | 3 × c5.2xlarge | 1 | 2 | ~$1,650 |
| Small DSP | 1K to 10K | 2 × g5.xlarge | 4 × c5.2xlarge | 2 | 4 | ~$2,640 |
| Mid-market DSP | 10K to 100K | 3 × g5.xlarge | 6 × c5.2xlarge | 3 to 4 | 6 to 8 | ~$3,870 |
| Large DSP | 100K to 500K | 6 × g5.2xlarge | 8 × c5.2xlarge | 6 | 10+ | ~$7,470 |
| Enterprise DSP | 500K to 1M+ | 12 × g5.4xlarge | 16 × c5.4xlarge | 12 | 20+ | ~$22,340 |

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

### EKS Production Path (GPU) — Part 1 only

All line items below are Part 1 (the real-time bidding path) as the default
`deploy.sh` run creates it in `us-east-1`, at public on-demand rates current
when this was written (730 hours per month). The GPU line is shown both ways:
on the included nightly shutdown schedule (about 260 GPU hours per month if
the node is started each weekday morning) and running 24/7.
`deploy.sh` deploys Part 2 by default as well; see [README.md](README.md#cost)
for both parts in one table.

| Resource | Configuration | Part | Est. monthly cost |
|----------|---------------|------|-------------------|
| EKS cluster | 1 control plane | Part 1 | $73 |
| GPU node (g5.xlarge, $1.006/h) | 1 instance, ~260 h on the nightly shutdown schedule | Part 1 | $262 (24/7: $734) |
| CPU nodes (c5.2xlarge, $0.34/h) | 3 instances, 24/7 (`cpu-services` desired 3, min 2, max 8) | Part 1 | $745 |
| NAT gateways | 3 (one per AZ) at $0.045/h, plus $0.045 per GB processed | Part 1 | $99 + data |
| Network Load Balancer (internal, Triton) | 1 at $0.0225/h, plus LCU usage | Part 1 | $16 + LCU |
| EBS (gp3, $0.08/GB-mo) | 100 GB GPU node volume + 3 × 50 GB CPU node volumes | Part 1 | $20 |
| S3 (models + frontend), CloudFront, Cognito | Low-traffic demo | Part 1 | <$2 |
| DynamoDB (`loadtest-history`, `container-registry`) | On-demand | Part 1 | <$2 |
| Lambda (UI API proxy), CloudWatch Logs, ECR storage | Per-invocation and per-GB; idle when the UI is closed | Part 1 | <$5 |
| Bedrock AgentCore MCP runtime | Consumption-billed; idle unless an agent calls `extend_rtb` | Part 1 | <$2 |
| **Part 1 total (estimate)** | | | **~$1,225/month (24/7 GPU: ~$1,700)** |

The CPU node group, not the GPU, is the largest fixed line: the three
`c5.2xlarge` nodes run around the clock. Scale `cpu-services` to its minimum
of 2 to save about $248/month. For demos: deploy, test, then
`./deploy.sh --prefix <p> --destroy` immediately. A 2-hour demo session with
the default node groups costs approximately $5 of compute.

Part 2 (closed-loop learning, deployed by default alongside Part 1) adds
roughly $245/month at its default schedules (the Adaptive Bidding Agent runs
once every 24 hours by default; shortening that cadence is what moves the
Bedrock line); see
[GUIDANCE-part2.md](GUIDANCE-part2.md#cost-estimation) for its line items and
the knobs that reduce it, or [README.md](README.md#cost) for both parts
combined in one table.

## Amazon Bedrock AgentCore Integration

The AgentCore MCP runtime is deployed by default in Phase 5 (omit it with `--skip-agentcore`) and is an interoperability surface for testing and AI agent simulation. It bundles all ARTF containers into a single ARM64 image that runs inside an AgentCore microVM (runtime name `<prefix>-nvidia-artf-recommenders_mcp`), exposing the `extend_rtb` MCP tool on port 8000 at `/mcp`. The real-time bidding path does not depend on it: the orchestrator reaches the ARTF containers directly inside the cluster over the ARTF extension point, and the containers' EKS and Triton deployment is unchanged whether or not the runtime exists.

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

- [IAB Tech Lab: Agentic Real Time Framework (ARTF) v1.0 Specification](https://iabtechlab.com/standards/artf/)
- [ARTF schema and Go reference implementation](https://github.com/IABTechLab/agentic-realtime-framework) — IAB Tech Lab's own repository, and the source of the Protocol Buffers schema vendored in `source/proto/`
- [OpenRTB 2.6 Specification](https://iabtechlab.com/openrtb)
- [Model Context Protocol (MCP) Specification](https://modelcontextprotocol.io)

### Academic Papers

- [Deep Learning Recommendation Model for Personalization and Recommendation Systems (Naumov et al. 2019)](https://arxiv.org/abs/1906.00091)
- [Neural Collaborative Filtering (He et al. 2017)](https://arxiv.org/abs/1708.05031)

## Source Code

The complete source code for this Guidance is available at:

**Repository:** [aws-solutions-library-samples/guidance-for-accelerator-optimized-agentic-bidding-on-aws](https://github.com/aws-solutions-library-samples/guidance-for-accelerator-optimized-agentic-bidding-on-aws)

### Includes

- ARTF container implementations: bid pricer, audience activator, deal scorer, signals enricher, yield optimizer
- Orchestrator with parallel fan-out (gRPC or REST per deployment, MCP fallback)
- Model export scripts (PyTorch → ONNX via triton/export_models.py)
- Triton model repository with config.pbtxt configurations
- Kubernetes manifests with HPA (EKS deployment)
- Testing frontend (CloudFront + S3)
- Amazon Bedrock AgentCore MCP runtime (ARM64)
- Single-command deployment script (deploy.sh)
- Local development via `docker compose up --build`

---

*© 2026 Amazon Web Services, Inc. or its affiliates. All rights reserved.*
