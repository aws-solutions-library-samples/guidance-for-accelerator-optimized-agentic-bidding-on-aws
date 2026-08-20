# Guidance for Accelerator-Optimized Agentic Bidding on AWS

> 📄 **Full guidance document:** See [GUIDANCE.md](GUIDANCE.md) for the complete published guidance including architecture details, model specifications, scaling scenarios, cost estimation, and Well-Architected analysis.

## Table of Contents

1. [Overview](#overview)
    - [Architecture](#architecture)
    - [Cost](#cost)
2. [Prerequisites](#prerequisites)
    - [Operating System](#operating-system)
    - [Third-party tools](#third-party-tools)
    - [AWS account requirements](#aws-account-requirements)
    - [Supported Regions](#supported-regions)
3. [Deployment Steps](#deployment-steps)
4. [Deployment Validation](#deployment-validation)
5. [Running the Guidance](#running-the-guidance)
6. [Next Steps](#next-steps)
7. [Cleanup](#cleanup)
8. [Notices](#notices)

## Overview

This Guidance shows how demand-side platforms (DSPs) and advertising technology providers can run GPU-accelerated AI inference directly inside the OpenRTB programmatic bidding pipeline, following the IAB Tech Lab Agentic RTB Framework (ARTF) v1.0 specification.

ARTF defines agent-driven containers that receive an OpenRTB bid request, analyze it, and propose typed *mutations* to the bidstream — adjusting bid prices, activating audience segments, managing private marketplace deals, or adding quality metrics. The host platform applies approved mutations atomically before the auction continues. This Guidance replaces rule-based bidding heuristics with real-time neural network inference served on the GPU, so bidding decisions stay within OpenRTB timeout budgets.

The solution provides four ARTF-compliant containers, an orchestration layer for parallel fan-out, and AI agent integration through Amazon Bedrock AgentCore with Model Context Protocol (MCP) support. Two containers run neural network inference on NVIDIA Triton Inference Server; two are deterministic, rule-based:

- **DLRM Bid Shader** (`BID_SHADE`) — predicts click-through rate (CTR) and computes an optimal shaded bid price. GPU-accelerated (Triton).
- **Wide & Deep Segment Activator** (`ACTIVATE_SEGMENTS`) — activates audience segments from real bid-request signals (IAB content category, age bucket, existing DMP segments, bid floor, device/video context) via deterministic rules. This container previously scored segments with a Wide & Deep neural network on Triton; that model's ONNX graph could not be compiled to a TensorRT engine (a `BatchNorm1d` fusion limitation), so it was replaced with transparent rules and is slated to be swapped for a partner ISV implementation.
- **NCF Deal Manager** (`ACTIVATE_DEALS` / `SUPPRESS_DEALS`) — predicts user-deal relevance and activates or suppresses private marketplace deals. GPU-accelerated (Triton).
- **Metrics Enricher** (`ADD_METRICS`) — a rule-based container that adds viewability and brand-safety scores, demonstrating that the ARTF container model can mix neural and deterministic logic in one pipeline.

Two of the four containers (DLRM, NCF) run inference against models hosted on NVIDIA Triton Inference Server on NVIDIA A10G GPUs (`g5` family) — the Part 1 baseline serves the exported ONNX via ONNX Runtime, and **Part 2 upgrades this serving path to compiled TensorRT `tensorrt_plan` engines** built by the in-cluster Model Optimizer (see [Part 2](#part-2-closed-loop-learning--adaptive-bidding)); for higher throughput, the more powerful Amazon EC2 G7e instances — accelerated by NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs — are an alternative. The orchestrator (a Starlette application exposing gRPC as the primary protocol plus MCP and REST) fans each `RTBRequest` out to all four containers in parallel, merges the resulting mutations, and returns a single `RTBResponse`. A React testing frontend (served from Amazon S3 through Amazon CloudFront, authenticated by Amazon Cognito) lets builders submit sample payloads, inspect mutations, and exercise the MCP interface. Test run history is persisted in Amazon DynamoDB.

> **Note on the models.** The bundled DLRM and NCF implementations are industry-standard deep learning recommender model examples served through NVIDIA Triton Inference Server; they use seeded random weights to exercise the GPU-accelerated inference path but do not make meaningful predictions. The Wide & Deep Segment Activator and Metrics Enricher are CPU/rule-based containers. The two Triton-served models are defined in `source/triton/export_models.py` and exported to ONNX with **randomly initialized (seeded) weights** — they are **not pretrained or production-trained models**, and their predictions are not meaningful until you train the architectures on your own data (or substitute your own ONNX models — see [Next Steps](#next-steps)). The NVIDIA Triton Inference Server image itself (`nvcr.io/nvidia/tritonserver:24.08-py3`) is the genuine upstream NVIDIA NGC container.

### How it works

1. **Bid request ingestion** — An OpenRTB bid request arrives through Amazon CloudFront and is routed to the orchestrator.
2. **Orchestration** — The orchestrator receives the request and fans it out in parallel to the four ARTF containers.
3. **Inference** — The DLRM and NCF containers extract features, invoke their assigned model on Triton via `tritonclient.http`, and receive predictions from the GPU. The Wide & Deep Segment Activator and Metrics Enricher apply rule-based logic on CPU.
4. **Mutation generation** — Each container translates its result into typed ARTF mutations.
5. **Response assembly** — The orchestrator merges all mutations into a single `RTBResponse`.
6. **Mutation application** — The DSP host platform applies the approved mutations to the bidstream before the auction continues.

### Architecture

![Architecture](assets/images/architecture.svg)

A detailed component-by-component description of the architecture is available at [assets/images/architecture.md](assets/images/architecture.md).

The primary deployment path provisions an Amazon EKS cluster:

- **CPU node group (`c5.xlarge`)** runs the orchestrator behind a Network Load Balancer, the two Triton-backed model containers (DLRM, NCF), which hold no GPU and call Triton over the cluster network, and the two rule-based containers (Wide & Deep Segment Activator, Metrics Enricher), which have no Triton dependency at all.
- **GPU node group (`g5` family, NVIDIA A10G)** runs NVIDIA Triton Inference Server (`nvcr.io/nvidia/tritonserver:24.08-py3`) serving two models — `dlrm_bid_shader` and `ncf_deal_manager` — loaded from Amazon S3. **Triton is the only steady-state GPU consumer, so the example runs on a single GPU node.** The node group is instance-type diversified across `g5.xlarge` / `g5.2xlarge` / `g5.4xlarge` (all one A10G, same GPU architecture) so a capacity shortage of one size falls back to another instead of leaving the node group stuck `Pending`. For higher throughput, the more powerful Amazon EC2 G7e instances are an alternative. Pass `--artf-node-role inference` at deploy to co-locate the Triton-backed model containers (DLRM, NCF) on the GPU node instead of CPU.
- **Amazon CloudFront + Amazon S3** serve the React frontend; **Amazon Cognito** authenticates users; the orchestrator verifies Cognito-issued JWTs.
- **Amazon Bedrock AgentCore** optionally hosts an MCP runtime (ARM64 microVM) that exposes the `extend_rtb` tool for AI agent integration.

An ECS Fargate path is also provided as an explicit alternative for CPU-only demos without GPU infrastructure (see [Deployment Steps](#deployment-steps)).

### Cost

You are responsible for the cost of the AWS services used while running this Guidance. As of June 2026, the cost for running this Guidance with the default settings in the US East (N. Virginia) Region is approximately **$592 per month** for a single-GPU demo using the included daytime-only GPU schedule, rising to about **$1,080 per month** if the GPU node runs 24/7.

We recommend creating a [Budget](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html) through [AWS Cost Explorer](https://aws.amazon.com/aws-cost-management/aws-cost-explorer/) to help manage costs. Prices are subject to change. For full details, refer to the pricing webpage for each AWS service used in this Guidance.

#### Sample cost table

The following table provides a sample, illustrative cost breakdown for deploying this Guidance with the default demo parameters in the US East (N. Virginia) Region for one month. Figures are example estimates only; actual costs depend on traffic, instance hours, and data transfer. The GPU node group is the dominant cost — this estimate assumes the included scheduled shutdown keeps it running roughly 12 hours per business day (~260 hours/month) rather than 24/7.

| AWS service | Dimensions | Cost [USD/month] |
| ----------- | ---------- | ---------------- |
| Amazon EKS | 1 cluster control plane, 730 hrs @ $0.10/hr | $73 |
| Amazon EC2 (GPU) | 1 × g5.xlarge (NVIDIA A10G), ~260 hrs/month with scheduled shutdown @ ~$1.006/hr | $262 |
| Amazon EC2 (CPU) | 2 × c5.xlarge, 730 hrs each @ $0.17/hr | $248 |
| Amazon S3 | ~1 GB model + frontend storage and requests | $1 |
| Amazon CloudFront | Low-traffic demo, < 10 GB egress | $2 |
| Amazon Cognito | < 1,000 monthly active users | $0 |
| Amazon DynamoDB | On-demand, low read/write volume for test history | $1 |
| Amazon Bedrock AgentCore | Optional MCP runtime, intermittent invocation | $5 |
| **Total (example estimate)** | | **~$592** |

> The ~$1,080/month figure reflects running the GPU node 24/7 plus headroom; enabling the included scheduled GPU shutdown brings a typical demo month closer to the ~$592 shown above. For a short demo, deploy, validate, and tear down immediately — a 2-hour session costs only a few dollars. The A10G `g5.xlarge` line item can be substituted with the more powerful Amazon EC2 G7e instances, which deliver higher performance at higher cost.

## Prerequisites

### Operating System

These deployment instructions are optimized to best work on **macOS** or a **Linux** workstation (for example Amazon Linux 2023 or Ubuntu). Deployment on Windows may require additional steps (for example, running the deployment scripts inside WSL2). The deployment scripts are POSIX shell and Python and assume a Unix-like environment.

### Third-party tools

Install the following before running the deployment, with the listed minimum versions:

- **AWS CLI v2** with valid credentials configured (`aws configure`)
- **Python 3.11+** with `boto3`, `torch`, `onnx`, and `onnxscript` (used to export PyTorch models to ONNX)
- **jq** (JSON processor)
- **eksctl** (EKS cluster management)
- **kubectl** (Kubernetes CLI)
- **Docker** with **buildx** — *only required for local builds* (`--local-build`). Not needed when using the default remote build via AWS CodeBuild. See [Container Image Build Process](#container-image-build-process) below for details.

```bash
# AWS CLI v2
aws --version

# Python 3.11+ with export/deploy dependencies
pip install boto3 torch onnx onnxscript

# jq, eksctl, kubectl (macOS via Homebrew shown; see each tool's docs for Linux)
brew install jq eksctl kubectl

# Docker (only if using --local-build)
docker buildx version
```

The deployment script checks for all of these at startup and fails with a clear message if any are missing.

### AWS account requirements

This deployment requires:

- An AWS account with permissions to create Amazon EKS clusters, Amazon EC2 instances (including `g5` GPU instances), Amazon S3 buckets, Amazon ECR repositories, Amazon CloudFront distributions, Amazon Cognito user pools, Amazon DynamoDB tables, and AWS IAM roles.
- **NVIDIA NGC registry access** to pull the NVIDIA Triton Inference Server image (`nvcr.io/nvidia/tritonserver:24.08-py3`).
- **NVIDIA GPU service quota** for at least one `g5.xlarge` (NVIDIA A10G) instance in your target Region — a more powerful alternative is the Amazon EC2 G7e instance family. In `us-east-1`, confirm the *Running On-Demand G and VT instances* quota is large enough before deploying. If not, request an increase through the [Service Quotas console](https://console.aws.amazon.com/servicequotas/).

### Supported Regions

This Guidance can be deployed in any AWS Region that supports Amazon EKS and NVIDIA A10G (`g5` family) instances — or the more powerful Amazon EC2 G7e instances, where available — including:

- US East (N. Virginia): `us-east-1` (default)
- US West (Oregon): `us-west-2`
- Europe (Ireland): `eu-west-1`
- Europe (Frankfurt): `eu-central-1`
- Asia Pacific (Tokyo): `ap-northeast-1`
- Asia Pacific (Sydney): `ap-southeast-2`

## Deployment Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/aws-solutions-library-samples/guidance-for-accelerator-optimized-agentic-bidding-on-aws.git
   ```

2. Change into the repository directory:

   ```bash
   cd guidance-for-accelerator-optimized-agentic-bidding-on-aws
   ```

3. Confirm your AWS credentials and target Region:

   ```bash
   aws sts get-caller-identity
   export AWS_REGION=us-east-1
   ```

4. Deploy the full solution on the **primary EKS path** using the deployment entry-point script. Pass `--prefix <prefix>` to namespace every resource (stack, cluster, buckets, IAM, CloudFront, AgentCore) so multiple environments can coexist in one account/Region:

   ```bash
   cd deployment
   ./deploy.sh --prefix stg
   ```

   With `--prefix stg`, resources are named `stg-nvidia-artf-recommenders-*`. Omit the flag to deploy with the default (unprefixed) stack name.

   `deploy.sh` provisions the entire EKS stack end to end:

   | Step | Action |
   |------|--------|
   | 1 | Create Amazon ECR repositories |
   | 2 | Export PyTorch models to ONNX (`../source/triton/export_models.py`) |
   | 3 | Upload the ONNX model repository to Amazon S3 |
   | 4 | Build and push container images via AWS CodeBuild (AMD64 for EKS, ARM64 for AgentCore) |
   | 5 | Create the Amazon EKS cluster (`g5`-family GPU — `g5.xlarge`/`2xlarge`/`4xlarge` A10G, or the more powerful Amazon EC2 G7e — + `c5.xlarge` CPU node groups) |
   | 6 | Install the NVIDIA Kubernetes Device Plugin |
   | 7 | Configure IAM Roles for Service Accounts (IRSA) for Triton + Model Optimizer S3 access |
   | 8 | Build base TensorRT engines via a one-shot optimizer **bootstrap Job** (frees the GPU on exit), then apply the Kubernetes manifests in `deployment/eks/` (Triton, ARTF containers on CPU by default, orchestrator, HPA) |
   | 9 | Deploy the frontend (Amazon S3 + Amazon CloudFront) and Amazon Cognito user pool |
   | 10 | Register the Amazon Bedrock AgentCore MCP runtime |

   Useful options:

   ```bash
   ./deploy.sh --prefix v1                    # namespaced resources (v1-*)
   ./deploy.sh --prefix prod --skip-agentcore # EKS + frontend only
   ./deploy.sh --skip-images                  # reuse existing images
   ./deploy.sh --skip-cluster                 # reuse an existing EKS cluster
   ./deploy.sh --local-build                  # build images locally with Docker (requires ~30 GB free disk)
   ./deploy.sh --maxGPUs 5                     # cap GPU node group max size (default 3)
   ./deploy.sh --artf-node-role inference      # co-locate ARTF model containers on the GPU node (default: services / CPU nodes)
   AWS_REGION=us-west-2 ./deploy.sh           # deploy to a different Region
   ```

   ### Container Image Build Process

   By default, `deploy.sh` builds all container images remotely using **AWS CodeBuild**. This is the recommended approach because:

   - **No local Docker required** — you don't need Docker installed on your workstation.
   - **No disk space concerns** — CodeBuild runs on `BUILD_GENERAL1_2XLARGE` instances with 824 GB SSD, easily accommodating the NeMo-RL training container (~20 GB base image from NVIDIA NGC).
   - **Faster ECR pushes** — builds run in-region with high-bandwidth access to ECR.
   - **ARM64 native builds** — ARM containers (AgentCore, bid-shading agent) build natively on Graviton without slow cross-compilation.

   The first deploy automatically provisions a CodeBuild CloudFormation stack (`<stack-name>-codebuild`) with IAM roles and two build projects (x86_64 and ARM64). Subsequent deploys reuse the existing projects.

   **Part 1 images** (Triton ARTF containers, orchestrator, metrics enricher, AgentCore) are lightweight `python:3.12-slim` images that build in **5–10 minutes** on CodeBuild. The script waits for this build to complete before continuing — it's fast enough that no special handling is needed.

   **The NeMo-RL training image** (`--with-retraining`) is the only long build (~20 GB base image, 15–50 minutes). This build fires **asynchronously** — deployment continues with the remaining closed-loop infrastructure (DynamoDB, Glue, EventBridge, AgentCore agents) while NeMo builds in the background. At the end of deployment, the script checks whether it finished:

   - **Build finished** → writes `.nemo-outputs.json`, deployment complete.
   - **Build still running** → prints a message explaining that everything is deployed and functional, and tells you how to check on it.

   The NeMo image is only used by SageMaker retraining jobs. The inference pipeline, UI, and agents all function without it.

   To check build status independently:

   ```bash
   # One-time check
   ./check_builds.sh --prefix stg

   # Watch continuously (refreshes every 30s)
   ./check_builds.sh --prefix stg --watch
   ```

   Output looks like:

   ```
   ══════════════════════════════════════════════════════════════
     Container Image Build Status
     Stack: stg-nvidia-artf-recommenders  Tag: abc1234  Region: us-east-1
   ══════════════════════════════════════════════════════════════

     Part 1 Images (needed for: kubectl apply, AgentCore):
     ─────────────────────────────────────────────────────
       ✓  stg-nvidia-artf-recommenders-dlrm-bid-shader:abc1234
       ✓  stg-nvidia-artf-recommenders-orchestrator:abc1234
       ...

     NeMo-RL Training Image (needed for: SageMaker retraining):
     ───────────────────────────────────────────────────────────
       CodeBuild:  ⧗ IN PROGRESS
       ✗  artf-nemo-rl-training:dlrm  (missing)

     Part 1 is ready. NeMo build is still in progress.

     Re-check with:
       ./check_builds.sh --prefix stg
   ```

   Once the NeMo image appears, `check_builds.sh` writes `.nemo-outputs.json` automatically. No further action needed — SageMaker retraining jobs will pick it up when triggered.

   On subsequent runs, `deploy.sh` detects the images already exist in ECR and skips the build entirely — no rebuild unless source code changes (tracked by the git SHA used as the image tag) or you explicitly delete `.image-outputs.json`.

   For **NeMo-RL builds** (when deploying with `--with-retraining`), the CodeBuild project needs to pull the `nvcr.io/nvidia/nemo:24.07` base image from NVIDIA NGC. Pass your API key directly and the script stores it in Secrets Manager automatically:

   ```bash
   ./deploy.sh --prefix stg --with-retraining --ngc-key YOUR_NGC_API_KEY
   ```

   Or if you've already stored the key:

   ```bash
   # One-time: store your NGC API key
   aws secretsmanager create-secret --name artf-ngc-api-key \
     --secret-string "YOUR_NGC_API_KEY" --region us-east-1

   # Then deploy with NGC auth
   ./deploy.sh --prefix stg --with-retraining --ngc-secret artf-ngc-api-key
   ```

   **Switching to local Docker builds:** If you prefer to build images on your own machine (for example, to iterate on Dockerfiles without waiting for CodeBuild), pass `--local-build`:

   ```bash
   ./deploy.sh --prefix stg --local-build
   ```

   > **Storage requirements for local builds:** Building all images locally requires approximately **30 GB of free Docker disk space** for Part 1 images (python:3.12-slim based). If you also build the NeMo-RL training container (`--with-retraining`), you need an additional **~25 GB** for the NVIDIA NeMo base image — totalling roughly **55 GB**. On macOS, this means increasing the Docker Desktop disk image size in Settings > Resources. Docker with **buildx** is required for the ARM64 cross-compilation (AgentCore and bid-shading agent images).

   The GPU node group starts at a single node (desired capacity 1, `g5` family) and can scale out to `--maxGPUs` nodes (default 3) — under inference load, or to briefly add a second GPU for the base-engine bootstrap Job and on-demand model-optimization Jobs. At steady state only Triton holds a GPU, so a single GPU node is enough for the demo.

   During deployment, the initial password for the admin-created demo user is either taken from the `DEMO_USER_PASSWORD` environment variable (if you set one) or generated as a strong, policy-compliant temporary password. The frontend has no self-signup; users are created by an administrator. The demo username and password are printed once in the **Demo Login** section of the deployment summary at the end of the run — copy them then, keep them private, and do not commit them. (For a pre-existing user, the summary shows the username and a reset command instead of a password, since the existing password is not recoverable.)

5. **(Alternative) ECS Fargate path.** For a CPU-only demo without provisioning GPU infrastructure, deploy the alternative ECS Fargate stack instead of `deploy.sh`. This path runs the containers with inline CPU inference behind an Application Load Balancer:

   ```bash
   cd deployment
   python scripts/deploy_ecs.py
   ```

   The ECS Fargate path is intended for rapid demos and does not provide GPU-accelerated Triton inference. The EKS path (`deploy.sh`) remains the primary, production-oriented deployment.

## Deployment Validation

After `deploy.sh` completes, validate the deployment:

1. Confirm the EKS cluster is reachable and nodes are ready:

   ```bash
   kubectl get nodes
   ```

   You should see at least one GPU node (labeled `role: inference`) and the CPU nodes in `Ready` state.

2. Confirm all workloads are running:

   ```bash
   kubectl get pods
   ```

   The Triton server pod (on the GPU node), the four ARTF container pods (the three model containers on CPU nodes by default, plus the rule-based metrics enricher), and the orchestrator pod should all report `Running` with ready containers. You will also see a **completed** `model-optimizer-bootstrap` Job — it builds the base TensorRT engines on the GPU node and exits, freeing the GPU for Triton. Triton may take a few minutes to reach `Ready` while it loads its models from S3 (startup probes allow for this).

3. Confirm the orchestrator health endpoints respond:

   ```bash
   kubectl port-forward deploy/orchestrator 8000:8000 &
   curl -s localhost:8000/health/ready
   ```

4. Confirm the frontend distribution is deployed. The deployment output prints the CloudFront domain; open `https://<CLOUDFRONT_DOMAIN>/` in a browser and confirm the login screen loads.

**Security note:** The frontend is served through Amazon CloudFront using Origin Access Control (OAC) over a private Amazon S3 bucket — the bucket is not publicly readable. All orchestrator API calls require a valid Amazon Cognito JWT: the orchestrator verifies the RS256 signature, issuer, and expiry against the Cognito JWKS on every non-health endpoint. There are no open or unauthenticated application endpoints.

## Running the Guidance

1. Open the frontend at `https://<CLOUDFRONT_DOMAIN>/` and sign in with the admin-created demo user — the username and password shown in the **Demo Login** section of the deployment summary. On first login you may be prompted to set a new password (the Cognito `newPasswordRequired` challenge).

2. **Before running scenarios, confirm the containers are available.** Click the **Containers** link in the navigation to open the Container Health panel. The GPU node group is scaled to zero when idle to save costs, so you may need to start it:

   - **GPUs stopped** — If GPU Inference shows as stopped, Triton shows *offline* and the three model containers show a *gpu offline* inference badge. Because those containers now run on CPU, the containers themselves stay reachable — only Triton needs the GPU — so "gpu offline" means Triton is down, not the container. Click **Start GPUs**; starting takes approximately 3–5 minutes.

     <img src="assets/images/containers-starting.png" alt="Containers starting" height="300">

   - **GPUs running, all containers ready** — Once the GPU node group is running and Triton has loaded all models, each container will show a green **ready** badge. Click **Refresh** to update the status.

     <img src="assets/images/containers-ready.png" alt="Containers ready" height="300">

   Wait until all four containers (dlrm-bid-shader, widedeep-segment-activator, ncf-deal-manager, metrics-enricher) and the Triton Inference Server (serving `dlrm_bid_shader` and `ncf_deal_manager`) show **ready** before proceeding.

3. Click on one of the pre-built **Scenarios** to run it through the pipeline. Each scenario card shows which ARTF intents (containers) it exercises:

   <img src="assets/images/scenarios-ui.png" alt="Scenarios UI" height="400">

   | Scenario | What it exercises | Containers triggered |
   |----------|-------------------|---------------------|
   | **Banner Ad — Segment Activation** | ESPN sports page with a 300×250 banner | `ACTIVATE_SEGMENTS`, `ADD_METRICS` |
   | **Bid Shading — DLRM Price Optimization** | Nike DSP bid response at $7.50; DLRM predicts CTR and shades the bid | `BID_SHADE` |
   | **Video + PMP Deals — NCF Scoring** | Video impression with 3 private marketplace deals | `ACTIVATE_DEALS`, `SUPPRESS_DEALS`, `ADD_METRICS` |
   | **SSP Enrichment — 3 Containers** | CNN sports page triggering the SSP-side enrichment intents: segments, deals, and metrics. Bid shading is excluded — it's a DSP-side decision, not something an SSP would request | `ACTIVATE_SEGMENTS`, `ACTIVATE_DEALS`, `ADD_METRICS` |

   Clicking a scenario submits the corresponding OpenRTB payload to the orchestrator and displays the results:

   <img src="assets/images/scenario-result.png" alt="Scenario result — Banner Ad" height="350">

   - **Right panel (Request + Mutations Applied)** — The full OpenRTB request with all accepted mutations merged in, showing the final mutated state that would be passed downstream in a live bidstream.
   - **Center (latency breakdown)** — A per-container timing waterfall showing how long each ARTF agent took to produce its mutations. The containers run in parallel; the orchestrator overhead is the fan-out/merge cost. The **"Network (round-trip)"** segment represents the CloudFront CDN and internet hop between the browser and the EKS cluster — this latency is an artifact of the testing UI and would not be present in a production integration where the caller is co-located with the orchestrator (for example, an SSP exchange calling the orchestrator directly over a VPC or RTB Fabric link).
   - **Bottom (Mutations)** — The individual mutations proposed by each container, showing the intent, operation, JSON path, and payload for each bidstream modification.

   The bid shader computes its shaded price as `min(original_bid, predicted_CTR × conversion_value × shade_factor)`, floored at the publisher's `bidfloor`.

4. (Optional) Invoke the Amazon Bedrock AgentCore MCP runtime directly. The runtime exposes the `extend_rtb` tool over MCP and can be called with the `invoke_agent_runtime` API from a Bedrock-enabled agent or the AWS SDK.

## Part 2: Closed-Loop Learning & Adaptive Bidding

Part 2 extends the core inference pipeline with a full closed-loop learning system: the models observe bid outcomes, an AI agent adapts bidding parameters in real time, and a governance pipeline retrains and promotes model versions — all orchestrated through AWS and NVIDIA infrastructure.

### Additional Components

| Component | Purpose | AWS Service |
|-----------|---------|-------------|
| **Adaptive Bidding Strategy Agent** | Reads market signals (CloudWatch), computes parameter adjustments, writes to DynamoDB | Amazon Bedrock AgentCore (Firecracker microVM) |
| **Model Governance Agent** | Runs A/B evaluation (Welch's t-test + SPRT), promotes/rejects challenger models | Amazon Bedrock AgentCore |
| **NeMo-RL Training Pipeline** | Two-phase retraining: supervised + reinforcement learning with bid outcome rewards | Amazon SageMaker + NVIDIA NeMo Framework |
| **DynamoDB Parameter Store** | Online feature store for bidding parameters (shade_factor, conversion_value) with sub-5ms reads | Amazon DynamoDB (+ optional DAX) |
| **DynamoDB User Feature Table** | Per-user feature vectors (win_rate, CTR, spend) materialized from bid outcomes with TTL-based expiry | Amazon DynamoDB |
| **Glue Feature ETL** | Transforms raw bid outcomes into labeled training datasets + materializes user features into DynamoDB | AWS Glue (Spark) |
| **EventBridge Scheduler** | Invokes the Bid Shading Agent every 5 minutes and triggers retraining on a 6-hour cadence | Amazon EventBridge Scheduler |
| **SageMaker Model Registry** | Versioned model packages with approval workflow (governance agent promotes/rejects). Seeded at deploy time with a **genesis (v1) version** per model type — the same seeded-weights ONNX artifact the ARTF containers serve — tagged `genesis=true` so it is never mistaken for a trained result; this gives the first scheduled retraining job a real `base_model_version` to fine-tune from | Amazon SageMaker Model Registry |
| **Model Optimizer** | Compiles ONNX → TensorRT engine plans (`trtexec`, FP16 by default) so Triton serves `tensorrt_plan`. Runs **on-demand as one-shot Kubernetes Jobs** — a deploy-time *bootstrap* Job builds the base engines and a promotion launches an *optimize* Job — so **no optimizer pod holds a GPU at steady state**. A TensorRT optimizer — **not** a stock NVIDIA NIM (none exists for these custom DLRM/NCF/Wide&Deep models; TensorRT is the engine a NIM is built on) | NVIDIA TensorRT (`nvcr.io/nvidia/tensorrt:24.08-py3`) Job on a GPU node |
| **Triton canary router** | Per-model Python-backend router that splits live traffic between `<model>_stable` and `<model>_canary`; the split % is an in-memory model parameter set at load/reload — never a per-request external lookup — so the ARTF bidstream stays dependency-free | NVIDIA Triton (Python backend) |
| **VPC proxy Lambda** | Bridges the PUBLIC Governance runtime to the cluster: forwards Triton readiness/model calls to the internal NLB, and for `POST /v1/optimize` **launches an on-demand optimizer Job via the Kubernetes API** (then returns the result) — so promotions build engines without an always-on GPU service, and the runtime never enters VPC mode | AWS Lambda (VPC-attached; IAM role mapped into cluster RBAC) |

### Architecture (Closed-Loop)

```
Deploy time: genesis (v1) model registered per type ──► Model Registry
                                                              │
Bid Outcomes (CloudWatch) ──► Glue ETL ──► S3 Training Data   │
                                              │                │
                                              ▼                │
EventBridge Schedule (6h) ──► SageMaker Training (NeMo-RL + NVIDIA NeMo Framework)
                                              │
                                              ▼
                              Model Registry (versioned ONNX artifacts)
                                              │
                                              ▼
                              Governance Agent (A/B gate: promote / reject / extend)
                                              │
                               ┌──────────────┴──────────────┐
                               ▼                              ▼
           Model Optimizer Job (on-demand: ONNX → TensorRT)   Reject (keep incumbent)
                               │
                               ▼
        Triton canary router:  <model>_stable  ⇄  <model>_canary
        (in-memory split set at load/reload — real-time ARTF bidstream untouched)
                               │
                               ▼
CloudWatch Metrics ──► Bid Shading Agent (AgentCore) ──► DynamoDB Parameters
                                                              │
                                                              ▼
                                                    ARTF Containers (read at bid time)
```

### Deployment

Part 2 is deployed via a single flag added to the main deployment:

```bash
cd deployment
./deploy.sh --prefix dv --with-retraining
```

This runs the full Part 1 stack (EKS + Triton + frontend) and then automatically deploys the closed-loop infrastructure:

- Builds and pushes the NeMo-RL training container (`nvcr.io/nvidia/nemo:24.07` base) to ECR
- Creates DynamoDB tables (parameter-store, audit-trail, user-features) with KMS encryption
- Creates SageMaker Model Package Groups for each model type, a SageMaker training execution role, and registers a **genesis (v1) model version** per type from the already-uploaded ONNX artifact (idempotent — safe to re-run)
- Deploys Glue ETL jobs (feature engineering + user feature materialization)
- Creates EventBridge schedules for the Bid Shading Agent (every 5 min) and for retraining (every 6h: resolves the latest Approved model version, starts a SageMaker training job, and — on completion — registers the trained model and notifies the Governance Agent)
- Deploys AgentCore runtimes for the Bid Shading and Governance agents

To deploy Part 2 separately on an existing Part 1 stack:

```bash
cd deployment
./deploy_closed_loop.sh --prefix dv
```

### Disabling Expensive Components

The scheduled components (agent invocations, retraining jobs) run on fixed cadences and incur ongoing costs. Disable them without tearing down infrastructure:

```bash
# Disable scheduled retraining via the API
curl -X POST https://<CLOUDFRONT_DOMAIN>/api/v1/closed-loop/schedule \
  -H "Authorization: Bearer <TOKEN>" \
  -d '{"enabled": false}'

# Re-enable when needed
curl -X POST https://<CLOUDFRONT_DOMAIN>/api/v1/closed-loop/schedule \
  -H "Authorization: Bearer <TOKEN>" \
  -d '{"enabled": true}'
```

The UI also exposes this toggle on the **Adaptive Bidding** page.

### Model serving upgrade: TensorRT engines + live canary

Part 2 is a **progression** of Part 1, not a separate stack: deploying it upgrades the Part 1 serving path from ONNX Runtime to compiled **`tensorrt_plan`** engines, and the closed loop rolls out new model versions as live canaries behind the same model names the ARTF containers already call.

**Model Optimizer (on-demand, not a stock NIM).** Engine compilation is performed by the **Model Optimizer** (`source/optimizer/`, image `nvcr.io/nvidia/tensorrt:24.08-py3`) which runs `trtexec` on a GPU node to turn an ONNX artifact into a `model.plan`. Rather than an always-on GPU service, it runs **on demand as one-shot Kubernetes Jobs**, so no optimizer pod holds a GPU at steady state — Triton is then the only steady-state GPU consumer and **a single GPU node runs the demo**. A deploy-time **bootstrap Job** (`--bootstrap`) builds the base engines and exits; each promotion launches an **optimize Job** (`--optimize-once`). FP16 is the default; INT8 is refused with an honest `400` unless a real calibration cache is supplied (no silently-uncalibrated engines). It is deliberately **not** called "NIM": no stock NVIDIA NIM container exists for these custom DLRM/NCF/Wide & Deep architectures, and TensorRT is the same engine a NIM is built on. The repo layout is `s3://<bucket>/onnx-source/<model>/model.onnx` (source) → `s3://<bucket>/triton-models/<model>_stable/1/model.plan` (served), and Triton runs in poll mode so it auto-loads engines as they appear. The bootstrap Job runs **before** Triton, so on a single GPU node it builds the engines and frees the GPU before Triton claims it.

**Live canary via the Triton router (preserves ARTF).** Each recommender model name the ARTF container calls (e.g. `dlrm_bid_shader`) is a lightweight Triton **Python-backend router** that forwards each request to either `<model>_stable` or `<model>_canary` (separate `tensorrt_plan` models). The traffic split is an **in-memory model parameter set at model load/reload** — a control-plane config change the governance agent makes — and is **never a per-request external lookup**, so the real-time bidstream gains no new dependency and the four ARTF containers and their entrypoints are left untouched (steering `artf-container-architecture`). Staging a canary writes its engine to S3 and sets the router split; **promote** publishes a new `<model>_stable` version and zeroes the split; **rollback** zeroes the split and removes the canary.

**Governance runtime networking (PUBLIC runtime + VPC proxy Lambda).** The Model Promotion Governance Agent runs as a **PUBLIC** AgentCore runtime and reaches the cluster through a small **VPC-attached proxy Lambda** (`deployment/vpc_proxy_cfn.yaml`) — the runtime itself never enters VPC mode. The agent's HTTP client tunnels each call through `lambda:InvokeFunction` → the proxy Lambda (attached to the cluster's private subnets + security group). Triton-readiness calls are forwarded to the internal NLB; a `POST /v1/optimize` call is instead handled by the Lambda **launching a one-shot optimizer Job through the Kubernetes API** and returning the engine's `output_uri` — the same response shape the old always-on service returned, so the governance agent code is unchanged. To do this the Lambda's IAM role is mapped into the cluster's `aws-auth` (via `eksctl create iamidentitymapping`) and bound to a **minimal namespaced Role** that can only create/watch Jobs and read pods — no cluster-wide access. S3 engine reads/writes go direct since S3 is reachable from PUBLIC. This deliberately sidesteps the AgentCore-supported-AZ constraint (a Lambda can attach to any private subnet, unlike an AgentCore VPC runtime). It is wired via the `VPC_PROXY_LAMBDA_ARN` env var on the runtime; if no cluster VPC is resolvable at deploy time the runtime still comes up PUBLIC and the optimize/canary step fails honestly rather than being faked. Nothing in the cluster ever calls *into* the agent — invocation is always UI-direct or EventBridge→Lambda→runtime.

**Honest scope of per-variant A/B metrics.** The served variant is chosen *inside* Triton, below the ARTF container, so the container never learns which variant served a given bid. Serving guardrails (p99 latency, error rate) are real and per-variant: Triton exposes per-model Prometheus metrics on `:8002` (reachable over the internal NLB), and the `GuardrailMonitor` forces an automatic rollback on breach. The primary **business** metric (revenue-per-bid) is **not** attributable per variant without modifying an ARTF container, which the architecture forbids. The statistical promote gate therefore reads per-variant samples from CloudWatch (namespace `ARTF/ABTest`, `Variant` dimension) and honestly reports *inconclusive* when they are absent — it never fabricates samples. Wiring an ADOT / CloudWatch-agent scrape of Triton `:8002` that maps each Triton model to a `Variant` dimension is the documented step required to populate the business-metric gate; until then ROI is validated in aggregate, post-promotion.

### Cost (Part 2 additional)

The following costs are **in addition to** the Part 1 base (~$592/month):

| AWS Service | Dimensions | Cost [USD/month] |
|-------------|-----------|------------------|
| Amazon DynamoDB | 3 tables (parameter-store, audit-trail, user-features), on-demand | ~$5 |
| Amazon DynamoDB DAX (optional) | 1 × dax.t3.small cluster | ~$36 |
| Amazon Bedrock AgentCore | Bid Shading Agent: ~8,640 invocations/month (5-min cadence) | ~$15 |
| Amazon Bedrock AgentCore | Governance Agent: triggered on model registration | ~$2 |
| Amazon SageMaker Training | 1 × ml.g5.xlarge, ~4 retraining jobs/day × 15 min each | ~$60 |
| AWS Glue | 10 DPU-hours/day for feature engineering + materialization | ~$44 |
| Amazon EventBridge Scheduler | 2 schedules, negligible | <$1 |
| **Part 2 total (estimate)** | | **~$125–160** |

> With both Part 1 and Part 2 running (daytime GPU schedule), expect approximately **$720–750/month**. Disable the EventBridge schedules and Glue jobs to drop Part 2 costs to near-zero (only DynamoDB storage remains). DAX is optional — only needed if you require sub-millisecond parameter reads at very high request rates.

## Next Steps

You can adapt this Guidance to your own bidding pipeline:

- **Bring your own models.** The bundled DLRM, Wide & Deep, and NCF models are reference architectures exported with seeded random weights (see the note in the [Overview](#overview)); they exercise the inference path but do not make meaningful predictions. To move toward production:
  1. **Train real weights.** These are dataset-specific recommender models — there is no portable "pretrained" checkpoint to drop in, because the embedding tables are keyed to a particular feature vocabulary. Reuse applicable training recipes rather than expecting reusable weights. For DLRM and NCF/NeuMF, use [NVIDIA DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples) where applicable; for Wide & Deep, use framework-native PyTorch or TensorFlow implementations. Training on a public benchmark (e.g., Criteo for DLRM, MovieLens for NCF) yields legitimately trained weights, but predictions are only meaningful once you train on data representative of *your* bidstream and outcomes (clicks, conversions, deal acceptance).
  2. **Align the feature contract.** A real model's input schema will differ from the small synthetic tensors used here (e.g., DLRM's `dense_features [4]` + three sparse inputs over a vocabulary of 1000). Update each container's feature-engineering in `source/containers/<name>/app.py` so the tensors it builds match the trained model's expected inputs, and update the corresponding `config.pbtxt` I/O (`name`, `dims`, `data_type`) to match the new ONNX signature. Plan for recsys-specific concerns: large sparse embedding tables, a hashing/collision strategy for high-cardinality IDs, train/serve feature skew, and periodic retraining as your catalog drifts.
  3. **Export and serve.** Produce the new `model.onnx`, place it under `source/triton/model_repository/<model_name>/<version>/model.onnx` with its `config.pbtxt`, and re-run `deployment/deploy.sh` (or `./deploy.sh --export-only` to regenerate models only). Triton loads from the S3 model repository at startup. Point the relevant container at the new model name. The integration plumbing is model-agnostic — swapping in a trained model is a config-and-upload step; the training and feature alignment above are where the real effort lives.
- **Add or swap ARTF containers.** Implement additional ARTF intent logic using the four containers under `source/containers/` as reference implementations, then register them with the orchestrator fan-out.
- **Scale for production traffic.** The included Horizontal Pod Autoscalers and Cluster Autoscaler scale the orchestrator and Triton pods (and their nodes) with load. The GPU node group defaults to a single node and scales out to `--maxGPUs` (default 3); raise that ceiling at deploy time (e.g. `./deploy.sh --maxGPUs 5`) or edit `deployment/eks/cluster-config.yaml`, and adjust the HPA targets in `deployment/eks/triton-hpa.yaml`. Consider EC2 Spot Instances or Savings Plans for the GPU node group, and NVIDIA TensorRT optimization of compatible ONNX models to improve inference efficiency.
- **Connect to a live DSP.** Route OpenRTB bid requests through the orchestrator before auction execution, and apply the returned mutations to your bidstream.

## Cleanup

Tear down all deployed resources using the destroy flag on the deployment script:

```bash
cd deployment
./deploy.sh --destroy
```

The script prompts for confirmation before deleting anything — type `destroy` when asked to proceed.

To tear down a specific namespaced stack, include the same prefix used at deploy time:

```bash
./deploy.sh --destroy --prefix v1
```

`--destroy` removes, in order:

- **Kubernetes workloads** — the manifests under `deployment/eks/` and the `artf` namespace.
- **The Amazon EKS cluster** (GPU and CPU node groups, and their Auto Scaling groups including the scheduled GPU shutdown action) via `eksctl delete cluster`. Because eksctl enables CloudFormation termination protection on the stacks it creates, the script first disables it on each `eksctl-<cluster>-*` stack.
- **Amazon S3 Triton model bucket** (`--force` empties it first).
- **Amazon DynamoDB** load-test history table.
- **Amazon CloudFront distribution + frontend S3 bucket** (via `deploy_frontend.py`).
- **Amazon Bedrock AgentCore** MCP runtime.
- **Amazon Cognito** user pool.
- **AWS IAM** customer-managed policies (Triton S3, DynamoDB load-test, EKS GPU scaling) and the AgentCore execution role.

**Required permission for teardown:** disabling termination protection needs `cloudformation:UpdateTerminationProtection`. If your session denies that action (for example, a restricted Isengard/SSO session policy), the script logs a warning and the EKS stacks will not delete — re-run `--destroy` from a session that allows the action.

**Retained by design:** Amazon ECR repositories are kept so cached images survive between deployments. Delete them manually from the Amazon ECR console or CLI if no longer needed. If you want to stop all storage charges, also confirm the S3 model and frontend buckets are fully emptied and removed.

## Notices

*Customers are responsible for making their own independent assessment of the information in this Guidance. This Guidance: (a) is for informational purposes only, (b) represents AWS current product offerings and practices, which are subject to change without notice, and (c) does not create any commitments or assurances from AWS and its affiliates, suppliers or licensors. AWS products or services are provided "as is" without warranties, representations, or conditions of any kind, whether express or implied. AWS responsibilities and liabilities to its customers are controlled by AWS agreements, and this Guidance is not part of, nor does it modify, any agreement between AWS and its customers.*
