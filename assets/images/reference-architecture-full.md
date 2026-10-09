# Reference architecture: full workload (Part 1 + Part 2)

Companion to `reference-architecture-full.drawio` (editable) and `reference-architecture-full.drawio.png` (export with embedded XML). Technical audience; one page.

## Layout

Left to right: operator browser and MCP client, then the edge and identity layer (Cognito, CloudFront, S3), then the cluster VPC with the EKS cluster (cpu-services node group with the orchestrator, six ARTF containers, optional Prebid Server, artfhouse demand service and simulator seat; gpu-inference node group with the Triton NLB, Triton, and the one-shot Model Optimizer Job), then the right column (build and shared security, S3 model repository, DynamoDB tables). Part 2 occupies the bottom band: the batch retraining loop on the left, the governance and agentic parameter-tuning loops on the right.

Solid edges are the real-time request path. Dashed edges are control plane, batch, async, and startup loads. Orange rounded boxes are Kubernetes workloads, not AWS service icons.

## Part 1 request flow

1. The browser signs in with Cognito (SRP) and exchanges the ID token for SigV4 credentials through the identity pool.
2. CloudFront serves the React bundle from S3 and nothing else.
3. Every `/api/*` call is a `lambda:InvokeFunction` to the VPC-attached UI API proxy Lambda, which forwards to the orchestrator's internal NLB. The orchestrator has no public address.
4. The orchestrator verifies the JWT against Cognito's JWKS and runs the ARTF pipeline in four stages: enrich (Signals Enricher, Audience Activator), deals (Deal Scorer), yield (Floor, Margin), price (Bid Pricer).
5. The Triton-backed containers (Deal Scorer, Floor, Margin, Bid Pricer) call Triton through its internal NLB. Triton's Python router splits traffic between stable and canary TensorRT engines; the two XGBoost models are served by the FIL backend. Engines load from the S3 model repository at startup.
6. The orchestrator reads container records from the DynamoDB container registry (refreshed every 30 s) and emits bid-outcome and deal-yield-outcome events to Kinesis.
7. The Bid Pricer reads `shade_factor` and `conversion_value` from the DynamoDB parameter store through a background refresh, never on the request hot path.
8. Optional second host (`--with-prebid`): Prebid Server runs in the same cluster. The Auction Theater posts `/v1/auction/run`; the ARTF hook calls the orchestrator's extension point over gRPC; the artfhouse adapter requests bids from the in-cluster artfhouse demand service (a Deployment and ClusterIP Service next to Prebid; it checks the Cognito M2M token itself and serves from an in-memory catalog of the deals table, refreshed every 30 s); the simulator seat returns static OpenRTB bids.
9. The Part 1 AgentCore MCP runtime exposes `extend_rtb` to AI agents and forwards to the orchestrator.

## Part 2 loops

Batch retraining: Kinesis Data Streams, Data Firehose, S3 raw outcomes (Parquet, Glue Data Catalog), two Glue ETL jobs every 6 hours, KMS-encrypted S3 training data, SageMaker Training (PyTorch on the NVIDIA NeMo Framework image for DLRM, built-in XGBoost for floor and margin; NCF is trained at deploy time on sample interactions), SageMaker Model Registry with four package groups. EventBridge Scheduler triggers retraining at `rate(24 hours)`. Training failures alert through SNS.

Governance: a model registration event hits an EventBridge rule, a Lambda shim invokes the Model Promotion Governance Agent (AgentCore). The agent reasons with Bedrock, launches a Model Optimizer Job and controls the Triton canary split through the VPC optimizer proxy Lambda (RBAC scoped to Jobs), evaluates with Welch's t-test and SPRT against CloudWatch per-variant metrics, polls guardrails every 10 s, writes Approved or Rejected to the registry, and records every decision in the append-only DynamoDB audit trail.

Agentic tuning: EventBridge Scheduler invokes the Adaptive Bidding Strategy Agent through a Lambda shim at `rate(24 hours)`. The agent reads win rate, ROI, and prices from CloudWatch, reasons with Bedrock, and writes bounded parameter updates (hard bounds, max delta, optimistic concurrency) to the parameter store.

## Services shown

Amazon Cognito, Amazon CloudFront, Amazon S3 (frontend, model repository, raw outcomes, training data), AWS Lambda (UI API proxy, VPC optimizer proxy, invocation shims), Elastic Load Balancing (two internal NLBs), Amazon EKS, Amazon ECR, AWS CodeBuild, AWS Secrets Manager, AWS IAM, Amazon DynamoDB (container registry, parameter store, deals table, audit trail), Amazon Kinesis Data Streams, Amazon Data Firehose, AWS Glue, Amazon SageMaker (Training, Model Registry), Amazon EventBridge (rules, Scheduler), Amazon Bedrock, Amazon Bedrock AgentCore (three runtimes), Amazon CloudWatch, AWS KMS, Amazon SNS. NVIDIA components: Triton Inference Server, TensorRT (Model Optimizer), NeMo Framework (training image base).

Not drawn: the DynamoDB user-features table (no edges in the request path), the nightly GPU scheduled shutdown (noted in the gpu node group label), and the NGC key secret flow (noted on Secrets Manager).

## Known rough edges in v1.0

- The `image pull` edge from ECR to EKS passes across the CodeBuild icon; swapping CodeBuild and ECR fixes it.
- The outcome-events edge into Kinesis crosses the Part 2 group title; moving that title to the bottom edge fixes it.
- The VPC optimizer proxy Lambda sits tight against the VPC's bottom border.

## Slide versions

`reference-architecture-slides.drawio` holds the same architecture as two 1920 x 1080 (16:9) pages, each exported at 2x for PowerPoint:

- `reference-architecture-part1-slide.drawio.png` (page 1): the Part 1 real-time path, with a single Kinesis icon standing in for Part 2.
- `reference-architecture-part2-slide.drawio.png` (page 2): the three Part 2 loops, with Part 1 collapsed to the orchestrator, the Bid Pricer, Triton, and the Model Optimizer.

Both pages drop the user-features table, NAT detail (page 2), and IAM icon (folded into the build group caption) to keep labels readable at slide scale.
