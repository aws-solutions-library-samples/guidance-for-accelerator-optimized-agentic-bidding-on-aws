# Architecture — Guidance for Accelerator-Optimized Agentic Bidding on AWS

This document describes the solution architecture and its request flow. It is the
text companion to the architecture visual. The canonical embeddable image for the
top-level `README.md` is **`assets/images/architecture.svg`** (a hand-authored SVG,
not a draw.io export). The Mermaid diagram below renders inline on GitHub and gives
the same picture in a maintainable, text-authored form.

## Solution overview

The solution implements **six** ARTF-compliant (IAB Tech Lab Agentic Real Time Framework)
containers, each doing one job in the bidstream. Four run GPU-accelerated inference
on **NVIDIA Triton Inference Server**; two are deterministic and rule-based on CPU:

| Container | What it does | ARTF intent(s) | Model (implementation detail) |
|-----------|--------------|-----------------|--------------------------------|
| **Bid Pricer** (`bid-pricer`) | Prices bids by shading down from a CTR prediction | `BID_SHADE` | DLRM (Triton) |
| **Audience Activator** (`audience-activator`) | Activates audience segments from bid-request signals | `ACTIVATE_SEGMENTS` | Rule-based (no Triton) |
| **Deal Scorer** (`deal-scorer`) | Scores and activates/suppresses PMP deals | `ACTIVATE_DEALS`, `SUPPRESS_DEALS` | NCF / NeuMF (Triton) |
| **Signals Enricher** (`signals-enricher`) | Adds viewability + brand-safety quality signals | `ADD_METRICS` | Rule-based (no Triton) |
| **Yield Optimizer — Floor** (`yield-optimizer-floor`) | SSP/publisher-side: adjusts the floor price on PMP deals | `ADJUST_DEAL_FLOOR` | XGBoost via Triton FIL (Triton) |
| **Yield Optimizer — Margin** (`yield-optimizer-margin`) | SSP/publisher-side: adjusts the margin taken on PMP deals | `ADJUST_DEAL_MARGIN` | XGBoost via Triton FIL (Triton) |

Triton serves **two deep-learning models** (DLRM for the bid pricer, NCF for the
deal scorer) plus **two independent XGBoost tree models** for the yield optimizer
(one for floor, one for margin, each served by its own container — Triton's Forest
Inference Library backend doesn't support multi-output regression, so these were
always two models) on a single NVIDIA **A10G GPU** (`g5.xlarge` by default;
the node group also accepts `g5.2xlarge` and `g5.4xlarge`) with the CUDA
Execution Provider. The TensorRT engines Part 2 compiles are specific to the
A10G, so the node group is pinned to the `g5` family. The audience activator and signals enricher
are rule-based and need no GPU model. The audience activator previously scored
segments with a Wide & Deep neural network on Triton — that model's ONNX graph
could not be compiled to a TensorRT engine (a `BatchNorm1d` fusion limitation), so
it was replaced with transparent rules over real bid-request signals, and is
slated to be replaced by a partner ISV implementation. See
[RENAME_MAP.md](../../RENAME_MAP.md) for the full old-name → new-name mapping.

> **Serving backend.** The Part 1 baseline serves the exported ONNX models via ONNX
> Runtime. Part 2 is a progression that upgrades Triton serving to compiled **TensorRT
> engine plans** (`tensorrt_plan`), built from the same ONNX by an in-cluster Model
> Optimizer microservice, and rolls new versions out through a Triton-side stable/canary
> router. See the top-level `README.md` (Part 2) for details.

> **Note on the models.** DLRM and NCF are *reference architectures* following the
> published NVIDIA DeepLearningExamples (DLRM, NeuMF/NCF) designs. They are defined
> in `source/triton/export_models.py` and exported to ONNX with **randomly
> initialized (seeded) weights** — they are **not pretrained or production-trained**.
> They exist to demonstrate the GPU inference path and the ARTF container/Triton
> integration; train the architectures on your own data (or supply your own ONNX
> models) before relying on their predictions. The Triton Inference Server image
> (`nvcr.io/nvidia/tritonserver:24.08-py3`) is the genuine upstream NVIDIA NGC
> container.

An **orchestrator** (Starlette) receives the OpenRTB request, verifies the caller's
Amazon Cognito JWT, and **fans out in parallel** to the six containers over the ARTF
extension point (gRPC or REST, selected per deployment) with an MCP path for AI-agent
and tool interoperability.
It merges the per-container mutations into a single `RTBResponse`.

A **React frontend** is hosted on Amazon S3 and delivered through Amazon CloudFront;
Amazon Cognito provides user-pool authentication (SRP, admin-created users, no
self-signup). An **Amazon Bedrock AgentCore MCP runtime**, registered by default
(`--skip-agentcore` omits it), exposes the same `extend_rtb` capability to
Bedrock-hosted AI agents.

## Architecture diagram (Mermaid)

```mermaid
graph TB
    subgraph User["End User"]
        Browser["Browser / MCP client"]
    end

    subgraph Edge["AWS Edge & Frontend"]
        CF["Amazon CloudFront<br/>HTTPS edge, static assets only"]
        S3F["Amazon S3<br/>React static frontend"]
        COG["Amazon Cognito<br/>User Pool + Identity Pool<br/>SRP / JWT / SigV4 credentials"]
    end

    subgraph VPC["Cluster VPC (private subnets, NAT egress; nothing internet-facing)"]
        PROXY["UI API proxy<br/>AWS Lambda, VPC-attached<br/>forwards /api/* to the orchestrator"]

    subgraph EKS["Compute (Amazon EKS)"]
        ORCH["Orchestrator (Starlette)<br/>ClusterIP in-cluster + internal NLB for the UI proxy<br/>no public address<br/>JWT verify + parallel fan-out<br/>gRPC / REST / MCP"]

        subgraph GPU["GPU node: g5.xlarge, NVIDIA A10G"]
            TRITON["NVIDIA Triton Inference Server<br/>ONNX Runtime + FIL + CUDA EP<br/>4 models on GPU"]
            DLRM_C["Bid Pricer<br/>BID_SHADE"]
            NCF_C["Deal Scorer<br/>ACTIVATE_DEALS / SUPPRESS_DEALS"]
            YIELD_F["Yield Optimizer — Floor<br/>ADJUST_DEAL_FLOOR"]
            YIELD_M["Yield Optimizer — Margin<br/>ADJUST_DEAL_MARGIN"]
        end

        WD_C["Audience Activator<br/>ACTIVATE_SEGMENTS (rule-based)"]
        MET_C["Signals Enricher<br/>ADD_METRICS (rule-based)"]
    end
    end

    subgraph Models["Model Storage"]
        S3M["Amazon S3 model repository<br/>dlrm_bid_shader · ncf_deal_manager<br/>deal_yield_manager_floor · deal_yield_manager_margin<br/>(model.onnx / xgboost.json)"]
    end

    subgraph Bedrock["Amazon Bedrock AgentCore"]
        AC["MCP Runtime (default; --skip-agentcore omits)<br/>extend_rtb tool"]
    end

    Browser -->|"SRP auth, ID token -> SigV4 creds"| COG
    COG -->|"JWT tokens + temporary credentials"| Browser
    Browser -->|"HTTPS GET /*"| CF
    CF --> S3F
    Browser -->|"lambda:InvokeFunction (SigV4)<br/>event carries the Bearer token"| PROXY
    PROXY -->|"HTTP to the internal NLB (orchestrator-internal)"| ORCH
    ORCH -->|"verify JWT via JWKS"| COG
    ORCH -->|"gRPC / MCP fan-out"| DLRM_C
    ORCH -->|"gRPC / MCP fan-out"| WD_C
    ORCH -->|"gRPC / MCP fan-out"| NCF_C
    ORCH -->|"gRPC / MCP fan-out"| MET_C
    ORCH -->|"gRPC / MCP fan-out"| YIELD_F
    ORCH -->|"gRPC / MCP fan-out"| YIELD_M
    DLRM_C -->|"tritonclient"| TRITON
    NCF_C -->|"tritonclient"| TRITON
    YIELD_F -->|"tritonclient"| TRITON
    YIELD_M -->|"tritonclient"| TRITON
    TRITON -.->|"load models at startup"| S3M
    AC -.->|"extend_rtb → orchestrator"| ORCH

    style TRITON fill:#76b900,color:#000
    style DLRM_C fill:#16a34a,color:#fff
    style WD_C fill:#6366f1,color:#fff
    style NCF_C fill:#d97706,color:#fff
    style MET_C fill:#0891b2,color:#fff
    style YIELD_F fill:#be185d,color:#fff
    style YIELD_M fill:#be185d,color:#fff
    style CF fill:#f59e0b,color:#000
    style S3F fill:#f59e0b,color:#000
    style PROXY fill:#f59e0b,color:#000
    style COG fill:#dd6b20,color:#fff
    style AC fill:#f59e0b,color:#000
```

## Request flow

1. The user authenticates against **Amazon Cognito** (SRP, email + password) and
   receives JWT access and ID tokens. The browser exchanges the ID token for
   temporary SigV4 credentials through the **Cognito Identity Pool**.
2. **CloudFront** serves the React app from **S3** and nothing else. For every
   `/api/*` call the browser packages the request as a JSON event (method, path,
   query, the `Bearer` token and content headers, body) and sends it with
   `lambda:InvokeFunction` to the **UI API proxy** Lambda. The Identity Pool's
   authenticated role may invoke that one function and nothing else.
3. The proxy, attached to the cluster's private subnets, forwards the request to
   the orchestrator's internal Network Load Balancer (`orchestrator-internal`,
   `deployment/eks/orchestrator-internal-nlb.yaml`) and returns the HTTP response.
   The in-cluster name `orchestrator.default.svc.cluster.local` is not usable here:
   a Lambda ENI resolves through the VPC resolver, which has no `cluster.local` zone,
   and a ClusterIP has no route from outside the nodes. The **orchestrator** keeps
   its ClusterIP Service for in-cluster callers and has no public address; it verifies
   the forwarded JWT (RS256, JWKS cached from Cognito) and rejects unauthenticated
   requests exactly as before. This path works in accounts running VPC Block Public
   Access in block-ingress mode, where an internet-facing load balancer would be
   dropped at the internet gateway.
4. The orchestrator **fans out the OpenRTB request in parallel** to the six
   containers, respecting the OpenRTB `tmax` timeout.
5. The bid pricer, deal scorer, and both yield optimizers call **Triton** via
   `tritonclient` for GPU inference; the audience activator and signals enricher
   return rule-based mutations.
6. **Triton** loads the DLRM/NCF ONNX models and the two yield optimizer XGBoost
   models from the **S3 model repository** at startup and runs inference on the
   A10G GPU.
7. The orchestrator **merges** all mutations into a single `RTBResponse` and returns
   it through the proxy Lambda to the browser.

The same fan-out is reachable as an MCP tool (`extend_rtb`) — either through the
orchestrator's `/api/mcp` endpoint or through the **Bedrock AgentCore** MCP runtime,
which `deploy.sh` registers by default (`--skip-agentcore` omits it).

## Deployment path

- **Amazon EKS** via `deployment/deploy.sh --prefix <p>` (the prefix is required;
  every resource is named `<p>-nvidia-artf-recommenders-*`). In five phases it
  provisions the ECR repositories and two DynamoDB tables, exports and uploads the
  ONNX and XGBoost models to S3, builds and pushes the images on AWS CodeBuild
  (default) or with local Docker (`--local-build`) while it creates or reuses the
  EKS cluster (Kubernetes 1.31, three AZs, private nodes behind one NAT gateway per
  AZ; `gpu-inference` node group of 1 `g5.xlarge`, min 1, max `--maxGPUs` default 3;
  `cpu-services` node group of 3 `c5.2xlarge`, min 2, max 8), installs the NVIDIA
  device plugin and IRSA roles, creates the Cognito pools, applies the manifests
  under `deployment/eks/` (Triton behind an internal NLB, six ARTF containers, the
  orchestrator with its own internal NLB for the UI API proxy, HPAs, the TensorRT
  bootstrap Job), deploys the UI API proxy Lambda
  stack (`deployment/ui_api_proxy_cfn.yaml`) and grants the Identity Pool role
  invoke on it, installs the nightly GPU scheduled shutdown (8:00 PM
  America/New_York), deploys the frontend (S3 + CloudFront), registers the AgentCore
  MCP runtime (default; `--skip-agentcore` omits it) and, unless `--no-retraining`,
  runs `deploy_closed_loop.sh` for Part 2. `--with-prebid` (off by default) adds
  Prebid Server to the same cluster as a second ARTF host via `deploy_prebid.sh`.
  Region defaults to `us-east-1` (`AWS_REGION`).
