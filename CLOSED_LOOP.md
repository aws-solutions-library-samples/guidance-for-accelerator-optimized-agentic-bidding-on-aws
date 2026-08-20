# Part 2: Closed-Loop Learning & Adaptive Bidding

Part 1 runs bidding decisions on GPU inside the live auction. Part 2 makes those
decisions get better on their own, safely: it watches what actually happened in
each auction, retrains and validates new model versions from those outcomes, and
tunes bidding parameters in near real time — all without touching the real-time
bidding path.

For the full architecture, Well-Architected analysis, and business case, see
[GUIDANCE-part2.md](GUIDANCE-part2.md). This page is the practical "how to deploy
and use it" companion.

## What it adds

Two feedback loops run continuously on top of Part 1:

- **Fast loop (~5 min)** — the **Adaptive Bidding Strategy Agent** reads live
  market signals (win rate, ROI, prices paid) from CloudWatch and adjusts bid
  pricing parameters, subject to hard safety bounds it cannot override itself.
- **Slow loop (~6 hr)** — bid outcomes are transformed into labeled training
  data, a challenger model is retrained on NVIDIA NeMo-RL, and the **Model
  Promotion Governance Agent** runs a live canary + statistical A/B test before
  promoting it (or rejecting it, or rolling back automatically on a guardrail
  breach). Every decision is written to an append-only audit trail.

Both loops feed the same two containers that already run in Part 1 — **Bid
Pricer** and **Deal Scorer** — without changing their ARTF interface.

## Deploy it

Part 2 deploys alongside Part 1 with a single flag (on by default):

```bash
cd deployment
./deploy.sh --prefix dv --with-retraining
```

This builds and pushes the NeMo-RL training container, creates the DynamoDB
parameter/audit/feature tables, creates the SageMaker Model Registry groups and
registers a **genesis (v1)** starter version per model type, deploys the Glue ETL
jobs, sets up the EventBridge schedules, and deploys both AgentCore agent
runtimes.

To deploy Part 2 separately against an existing Part 1 stack:

```bash
cd deployment
./deploy_closed_loop.sh --prefix dv
```

To skip Part 2 entirely (Part 1 only):

```bash
./deploy.sh --prefix dv --no-retraining
```

## Try it

Open the frontend and go to the **Adaptive Bidding** page to watch the fast loop
adjust bid parameters, or the **Governance** page to watch a model promotion
cycle (retrain → canary → A/B test → promote/reject). Both pages show the agent's
real reasoning and real telemetry — never simulated data.

## Disabling scheduled components

The scheduled agent invocation and retraining jobs run on fixed cadences and
incur ongoing cost. Disable them independently of the underlying infrastructure,
without a teardown:

```bash
curl -X POST https://<CLOUDFRONT_DOMAIN>/api/v1/closed-loop/schedule \
  -H "Authorization: Bearer <TOKEN>" \
  -d '{"enabled": false}'

# Re-enable when needed
curl -X POST https://<CLOUDFRONT_DOMAIN>/api/v1/closed-loop/schedule \
  -H "Authorization: Bearer <TOKEN>" \
  -d '{"enabled": true}'
```

The UI also exposes this as a toggle on the **Adaptive Bidding** page.

## Cost

The following is in addition to Part 1's base cost (see [README.md](README.md#cost)):

| AWS service | Dimensions | Cost [USD/month] |
| --- | --- | --- |
| Amazon DynamoDB | 3 tables, on-demand | ~$5 |
| Amazon DynamoDB DAX (optional) | 1 × dax.t3.small | ~$36 |
| Amazon Bedrock AgentCore | Adaptive Bidding Agent, ~8,640 invocations/month | ~$15 |
| Amazon Bedrock AgentCore | Governance Agent, triggered on registration | ~$2 |
| Amazon SageMaker Training | ~4 retraining jobs/day × 15 min, ml.g5.xlarge | ~$60 |
| AWS Glue | ~10 DPU-hours/day | ~$44 |
| Amazon EventBridge Scheduler | 2 schedules | <$1 |
| **Part 2 total (estimate)** | | **~$125–160** |

With both parts running on the included daytime GPU schedule, expect roughly
**$720-750/month** total. Disabling the scheduled components above drops Part 2's
ongoing cost to near-zero (only DynamoDB storage remains). DAX is optional —
only needed for very high parameter-read request rates.

## Learn more

- Full architecture, the three feedback loops in detail, the governance decision
  pipeline, and Well-Architected analysis: [GUIDANCE-part2.md](GUIDANCE-part2.md)
- Rename map for the four containers: [RENAME_MAP.md](RENAME_MAP.md)
