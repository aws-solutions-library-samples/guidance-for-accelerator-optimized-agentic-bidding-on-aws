# Guidance for Accelerator-Optimized Agentic Bidding on AWS

Run AI models that price bids, activate audience segments, and manage private marketplace deals in real time — accelerated by NVIDIA GPUs, deployed on Amazon EKS.

> 📄 **Full guidance document:** See [GUIDANCE.md](GUIDANCE.md) for the complete published guidance, including architecture details, model specifications, scaling scenarios, and Well-Architected analysis.

## Table of Contents

1. [Quick start](#quick-start)
2. [What just happened?](#what-just-happened)
3. [Try it](#try-it)
4. [Go deeper](#go-deeper)
    - [Architecture](#architecture)
    - [Cost](#cost)
    - [Prerequisites](#prerequisites)
    - [Customizing your deployment](#customizing-your-deployment)
    - [Part 2: closed-loop learning](#part-2-closed-loop-learning)
    - [Next steps](#next-steps)
5. [Cleanup](#cleanup)
6. [Notices](#notices)

## Quick start

Before you start, you'll need an AWS account with permission to create Amazon EKS, EC2 (including `g5` GPU instances), S3, ECR, CloudFront, Cognito, and IAM resources, and a `g5.xlarge` GPU quota in your target Region. `deploy.sh` checks for the required CLI tools at startup and tells you exactly what's missing — see [Prerequisites](#prerequisites) for the full list if you want to check ahead of time.

You'll also need an **NVIDIA NGC API key**. `deploy.sh` deploys the closed-loop retraining stack by default ([Part 2](#part-2-closed-loop-learning)), which builds an NVIDIA NeMo-RL training container from a gated NGC image — the key is what authenticates that pull. Get a free one:

1. Create an account (or sign in) at [ngc.nvidia.com/signin](https://ngc.nvidia.com/signin).
2. Click your account icon (top right) → **Setup** → **Generate Personal Key** (older accounts: **Account Settings** → **Generate API Key**).
3. Give the key a name, leave the default expiration/services, and click **Generate Personal Key**. Copy it now — NGC only shows it once.

```bash
git clone https://github.com/aws-solutions-library-samples/guidance-for-accelerator-optimized-agentic-bidding-on-aws.git
cd guidance-for-accelerator-optimized-agentic-bidding-on-aws/deployment
./deploy.sh --ngc-key YOUR_NGC_API_KEY
```

`deploy.sh` stores the key in AWS Secrets Manager and reuses it automatically on later runs of the same stack — you only need to pass `--ngc-key` once. It provisions everything: an EKS cluster with GPU and CPU node groups, NVIDIA Triton Inference Server, the bidding containers, the orchestrator, and a React frontend behind CloudFront. It takes roughly 30–40 minutes. When it finishes, it prints a URL and a demo login — open the URL in your browser.

Don't want the closed-loop stack (and don't want to create an NGC account)? Skip it — the core real-time bidding pipeline doesn't need NGC credentials:

```bash
./deploy.sh --no-retraining
```

Want a separate, namespaced deployment (for example to run more than one environment in the same account)? Add `--prefix`:

```bash
./deploy.sh --ngc-key YOUR_NGC_API_KEY --prefix stg
```

## What just happened?

```
Browser (React UI)
    |
    |  sign in (Cognito) + HTTPS
    v
+--------------------------------+
|  Amazon CloudFront + S3        |
|  (serves the UI)               |
+--------------------------------+
    |
    |  POST /api/*
    v
+--------------------------------+
|  Orchestrator (EKS)            |
|  verifies the login, fans      |
|  out the bid request           |
+--------------------------------+
    |
    |  parallel calls
    v
+--------------------------------+
|  5 ARTF containers (EKS)       |
|  bid pricer, audience          |
|  activator, deal scorer,       |
|  signals enricher, yield       |
|  optimizer                     |
+--------------------------------+
    |
    |  3 of the 5 call Triton
    v
+--------------------------------+
|  NVIDIA Triton (GPU node)      |
|  runs the AI models            |
+--------------------------------+
```

`deploy.sh` deployed an Amazon EKS cluster with two node groups: a GPU node running NVIDIA Triton Inference Server, and CPU nodes running the orchestrator and five bidding containers, each responsible for one job in the pipeline:

| Container | What it does |
|-----------|---------------|
| **Bid Pricer** | Shades every bid down to the lowest price still likely to win |
| **Audience Activator** | Activates audience segments from bid-request signals |
| **Deal Scorer** | Scores and activates/suppresses private marketplace deals |
| **Signals Enricher** | Adds viewability and brand-safety quality signals |
| **Yield Optimizer** | Publisher/SSP-side counterpart — adjusts deal floors and margins using an XGBoost model on Triton |

See [Architecture](#architecture) for the full picture, including which containers run GPU inference on Triton and which run rule-based logic on CPU.

The orchestrator fans out every incoming bid request to all five containers in parallel, merges their mutations, and returns a single response — the same fan-out list includes the yield optimizer, so it's exercised whenever a scenario carries a deal with a floor/margin-adjustable intent. A React frontend, served through CloudFront and authenticated by Cognito, lets you submit sample payloads and inspect the results — that's the "Try it" step below.

> **Note on the models.** The bundled models (DLRM, NCF, XGBoost) ship with **seeded, untrained weights**. They exercise the real GPU inference path but don't make meaningful predictions until you train them on your own data — see [Next steps](#next-steps). The audience activator and signals enricher are deterministic rule engines, not models.

### The 5 deployment phases

`deploy.sh` prints its progress as 5 numbered phases (`Phase N/5: ...`). Pass `--verbose` to see every underlying command instead of just the phase summaries; use `--start-at N` to resume from a given phase (for example after fixing a one-off failure) without redoing earlier ones.

| Phase | What happens |
|-------|---------------|
| **1/5 — Preparing models** | Creates the ECR repositories and the DynamoDB load-test-history table; exports the PyTorch DLRM/NCF models to ONNX and the yield optimizer's genesis XGBoost models (best-effort — the deploy continues even if this step is skipped); uploads all of it, plus the Triton router/model-config repository, to the S3 model bucket. |
| **2/5 — Building containers & provisioning infrastructure** | Builds and pushes any container image whose source has changed (remotely via AWS CodeBuild by default, or locally with `--local-build`) **at the same time** it creates the EKS cluster (GPU + CPU node groups) — the two don't depend on each other, so running them in parallel is roughly half the wait of doing them one after another. Then installs the NVIDIA Kubernetes device plugin and sets up the IAM/IRSA roles Triton, the Model Optimizer, and the orchestrator need. |
| **3/5 — Deploying workloads** | Provisions the Cognito user pool, then applies the Kubernetes manifests for Triton, the five ARTF containers, and the orchestrator. Kicks off a one-shot Kubernetes Job that compiles the base TensorRT engines on the GPU node — this runs in the background and does **not** block the rest of the deploy; Triton picks the compiled engines up automatically once they're ready (see the "Confirm everything is healthy" note below). Also configures the included daily GPU-node scheduled shutdown. |
| **4/5 — Setting up access** | Deploys the React frontend (S3 + CloudFront) and creates the demo admin user in Cognito. |
| **5/5 — Registering agents** | Registers the Amazon Bedrock AgentCore MCP runtime, then — unless you passed `--no-retraining` — deploys the entire Part 2 closed-loop stack: the bid-outcome feedback pipeline, the Glue ETL job, the SageMaker Model Registry groups (seeded with genesis model versions), the NeMo-RL training container (built asynchronously — this is the long build the NGC key is for), the Adaptive Bidding and Governance AgentCore agent runtimes, their EventBridge invocation schedules, and a final frontend rebuild wired with the real agent ARNs. |

**Cost while it's running:** approximately **$592/month** with the included daytime-only GPU schedule (about **$1,080/month** if the GPU node runs 24/7). See [Cost](#cost) for the breakdown. Deploy, try it, and [tear it down](#cleanup) when you're done — a short session costs a few dollars.

**Confirm everything is healthy:**

```bash
kubectl get nodes    # at least one GPU node + CPU nodes, all Ready
kubectl get pods     # Triton, the 5 containers, and the orchestrator all Running
```

Triton and the model-optimizer bootstrap Job may take a few minutes to finish loading/compiling models after the script exits — `deploy.sh`'s final summary tells you whether that finished and how to check.

## Try it

1. Open the frontend URL from the deployment summary and sign in with the demo username and password shown there. On first login, Cognito will prompt you to set a new password.

2. Click **Containers** in the navigation to check status. The GPU node group scales to zero when idle to save cost — if Triton shows *offline*, click **Start GPUs** (takes about 3–5 minutes).

   <img src="assets/images/containers-starting.png" alt="Containers starting" height="300">

   Once Triton has loaded its models, every container shows a green **ready** badge.

   <img src="assets/images/containers-ready.png" alt="Containers ready" height="300">

3. Click one of the pre-built **Scenarios** to run it through the pipeline:

   <img src="assets/images/scenarios-ui.png" alt="Scenarios UI" height="400">

   | Scenario | Containers exercised |
   |----------|----------------------|
   | Banner Ad — Segment Activation | Audience Activator, Signals Enricher |
   | Bid Shading — Price Optimization | Bid Pricer |
   | Video + PMP Deals — Deal Scoring | Deal Scorer, Signals Enricher, Yield Optimizer |
   | SSP Enrichment — 3 Containers | Audience Activator, Deal Scorer, Signals Enricher |

   <img src="assets/images/scenario-result.png" alt="Scenario result" height="350">

   The result view shows the mutated bid request, a per-container latency breakdown, and the individual mutations each container proposed.

4. (Optional) Call the same pipeline over MCP. An Amazon Bedrock AgentCore runtime exposes an `extend_rtb` tool that any Bedrock-hosted agent can invoke — see [GUIDANCE.md](GUIDANCE.md#amazon-bedrock-agentcore-integration) for a working example.

## Go deeper

### Architecture

![Architecture](assets/images/architecture.svg)

An Amazon EKS cluster runs two node groups: a `g5`-family GPU node group (NVIDIA A10G, or the more powerful Amazon EC2 G7e) running NVIDIA Triton Inference Server, and a `c5.xlarge` CPU node group running the orchestrator and the bidding containers. Two containers — the bid pricer and the deal scorer — call Triton for GPU-accelerated inference; the audience activator and signals enricher are deterministic rule engines on CPU; the yield optimizer calls Triton's Forest Inference Library backend for its XGBoost model. Amazon CloudFront + S3 serve the React frontend; Amazon Cognito authenticates users; an optional Amazon Bedrock AgentCore runtime exposes the same pipeline over MCP.

Full component-by-component detail, model specifications, and a request-flow diagram: [assets/images/architecture.md](assets/images/architecture.md). Container-naming history (what changed, what didn't, and why): [RENAME_MAP.md](RENAME_MAP.md).

### Cost

Sample estimate for the default settings in `us-east-1`, assuming the included scheduled GPU shutdown (~260 GPU-hours/month):

| AWS service | Cost [USD/month] |
| ----------- | ----------------- |
| Amazon EKS | $73 |
| Amazon EC2 (GPU, ~260 hrs/mo) | $262 |
| Amazon EC2 (CPU) | $248 |
| Everything else (S3, CloudFront, Cognito, DynamoDB, AgentCore) | ~$9 |
| **Total** | **~$592** |

Running the GPU node 24/7 instead of on the included schedule raises the total to roughly **$1,080/month**. The `g5.xlarge` line item can be swapped for the more powerful Amazon EC2 G7e instances at higher cost. Full breakdown and Part 2's additional cost: [GUIDANCE.md](GUIDANCE.md#cost-estimation) and [CLOSED_LOOP.md](CLOSED_LOOP.md#cost).

### Prerequisites

**Operating system:** macOS or Linux (for example Amazon Linux 2023 or Ubuntu). The deployment scripts are POSIX shell and Python; Windows users should run them inside WSL2.

**Third-party tools** (minimum versions; `deploy.sh` checks these at startup):

```bash
aws --version                                    # AWS CLI v2, with credentials configured
pip install boto3 torch onnx onnxscript          # Python 3.11+
brew install jq eksctl kubectl                   # or your Linux package manager
docker buildx version                            # only needed for --local-build
```

**AWS account requirements:** permission to create Amazon EKS clusters, EC2 instances (including `g5` GPU instances), S3 buckets, ECR repositories, CloudFront distributions, Cognito user pools, DynamoDB tables, and IAM roles. A `g5.xlarge` (NVIDIA A10G) service quota in your target Region — check the *Running On-Demand G and VT instances* quota in the [Service Quotas console](https://console.aws.amazon.com/servicequotas/) before deploying.

**NVIDIA NGC API key:** required by default, since Part 2 (closed-loop retraining) is on unless you pass `--no-retraining`. `nvcr.io/nvidia/tritonserver` itself is a public image and needs no key. See the [Quick start](#quick-start) above for how to get one.

**Supported Regions:** any Region with Amazon EKS and `g5`-family (or G7e) instances, including `us-east-1` (default), `us-west-2`, `eu-west-1`, `eu-central-1`, `ap-northeast-1`, and `ap-southeast-2`.

### Customizing your deployment

```bash
./deploy.sh --prefix v1                    # namespaced resources (v1-*)
./deploy.sh --skip-images                  # reuse existing images
./deploy.sh --skip-cluster                 # reuse an existing EKS cluster
./deploy.sh --local-build                  # build images locally with Docker instead of CodeBuild
./deploy.sh --no-retraining                # Part 1 only — skip the closed-loop stack (see below)
./deploy.sh --ngc-secret my-secret-name    # reuse an NGC key already stored in Secrets Manager
./deploy.sh --maxGPUs 5                    # raise the GPU node group's max size (default 3)
./deploy.sh --artf-node-role inference     # co-locate the model containers on the GPU node
./deploy.sh --start-at 3                   # resume from Phase 3 (see the breaking-change note below)
./deploy.sh --verbose                      # print full detailed logs, not just phase summaries
AWS_REGION=us-west-2 ./deploy.sh           # deploy to a different Region
```

By default, images build remotely on **AWS CodeBuild** — no local Docker required, and ARM64 images (AgentCore) build natively on Graviton. The first deploy provisions a CodeBuild stack automatically; later runs skip rebuilding images whose source hasn't changed. Pass `--local-build` if you'd rather build with Docker on your own machine (needs ~30 GB free disk, plus buildx for the ARM64 cross-compile).

The closed-loop stack (on by default) also builds an NVIDIA NeMo-RL training container from a gated NGC image, which is a long build (15–50 minutes) that runs asynchronously — the rest of the deployment doesn't wait on it. If you already stored your NGC key in Secrets Manager on a prior run (or via `aws secretsmanager create-secret --name my-secret-name --secret-string YOUR_NGC_API_KEY`), pass `--ngc-secret my-secret-name` instead of `--ngc-key` to reuse it. Check the NeMo build's status any time with:

```bash
./check_builds.sh --prefix stg          # one-time check
./check_builds.sh --prefix stg --watch  # refresh every 30s
```

**Breaking change:** `deploy.sh --start-at` now takes a phase number 1–5 instead of the old internal step numbers. If you have scripts referencing the old numbering, see the old-step-to-new-phase mapping table in [RENAME_MAP.md](RENAME_MAP.md).

### Part 2: closed-loop learning

The models above ship with static, untrained weights. Part 2 closes the loop: it watches real bid outcomes, retrains and validates new model versions, and tunes bidding parameters continuously — all without touching the real-time bidding path. It's deployed by default (`--with-retraining`) alongside Part 1.

See [CLOSED_LOOP.md](CLOSED_LOOP.md) for how to deploy it separately, try the Adaptive Bidding and Governance demos, disable the scheduled components to control cost, and its own cost breakdown. For the full architecture and Well-Architected analysis, see [GUIDANCE-part2.md](GUIDANCE-part2.md).

### Next steps

- **Bring your own models.** The bundled models exercise the inference path but aren't trained on real data — there's no portable "pretrained" checkpoint to drop in, since the embedding tables are keyed to a particular feature vocabulary. Train real weights on a representative dataset, align each container's feature-engineering (`source/containers/<name>/app.py`) and Triton `config.pbtxt` to the new ONNX signature, then re-run `deploy.sh --export-only` to regenerate and upload the models.
- **Add or swap containers.** Use the containers under `source/containers/` as reference implementations for additional ARTF intents, then register them with the orchestrator fan-out.
- **Scale for production traffic.** The included Horizontal Pod Autoscalers and Cluster Autoscaler scale with load; raise `--maxGPUs` or edit `deployment/eks/cluster-config.yaml` for higher ceilings. See [GUIDANCE.md](GUIDANCE.md#scaling-scenarios) for reference sizing at different QPS levels.
- **Connect to a live DSP.** Route real OpenRTB bid requests through the orchestrator before auction execution, and apply the returned mutations to your bidstream. See [GUIDANCE.md](GUIDANCE.md#integration-with-a-dsp) for the integration steps.

## Cleanup

```bash
cd deployment
./deploy.sh --destroy
```

The script prompts for confirmation — type `destroy` to proceed. Include the same `--prefix` used at deploy time to tear down a specific namespaced stack.

`--destroy` removes the Kubernetes workloads, the EKS cluster (both node groups), the S3 model bucket, the DynamoDB load-test table, the CloudFront distribution and frontend bucket, the AgentCore runtime, the Cognito user pool, and the IAM policies/roles this Guidance created.

**Retained by design:** Amazon ECR repositories persist across deployments so cached images survive between runs. Delete them manually from the ECR console or CLI if you no longer need them.

**Required permission for teardown:** disabling CloudFormation termination protection (which `eksctl` enables automatically) needs `cloudformation:UpdateTerminationProtection`. If your session denies that action, `--destroy` logs a warning and the EKS stacks won't delete — re-run it from a session that allows the action.

## Notices

*Customers are responsible for making their own independent assessment of the information in this Guidance. This Guidance: (a) is for informational purposes only, (b) represents AWS current product offerings and practices, which are subject to change without notice, and (c) does not create any commitments or assurances from AWS and its affiliates, suppliers or licensors. AWS products or services are provided "as is" without warranties, representations, or conditions of any kind, whether express or implied. AWS responsibilities and liabilities to its customers are controlled by AWS agreements, and this Guidance is not part of, nor does it modify, any agreement between AWS and its customers.*
