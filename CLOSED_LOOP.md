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

Both loops feed three containers that already run in Part 1 — **Bid
Pricer**, **Deal Scorer**, and **Yield Optimizer** — without changing their ARTF
interface.

### Yield Optimizer: bootstrapping training data without a live signal path yet

The Yield Optimizer's floor/margin models don't have a downstream win/loss signal
path from real auctions yet (that's future work), so two mechanisms generate the
real, disclosed feedback its Glue ETL job needs to build labeled training data:

- **Load-test traffic as training data.** Running a load test against the Yield
  Optimizer (see [Try it](README.md#try-it)) doesn't just measure throughput — it
  emits real `DealYieldOutcomeEvent`s down the same Kinesis → Firehose → S3 → Glue
  pipeline production traffic uses, tagged `source=load_test` so it's never
  confused with a live outcome. This is how the container gets real training data
  before real auction outcomes exist to learn from.
- **Bounded exploration.** The container can optionally perturb its own
  prediction by a small, bounded random amount (`YIELD_EXPLORATION_EPSILON`,
  default `0.0`/disabled) instead of always returning the same recommendation —
  this is what breaks a model that has converged to "no change" out of a
  cold-start deadlock, since a constant recommendation never produces the
  mutation an outcome event needs to exist. Every exploratory response is
  disclosed via a `:explore` suffix on the returned `model_version`, so training
  data never silently blends an exploratory probe in as if it were a confident
  model recommendation.

### Bid Pricer: where the bid shader's labels come from

The slow loop retrains the bid shader on the consequences of its own decisions: it
chose a price, the auction resolved, and a response either followed or did not. In
a production DSP those facts arrive as exchange win notices and advertiser
conversion pixels. This guidance deploys no such feed, so the same question the
Yield Optimizer faces applies here — where does a real label come from before real
outcomes exist?

**Three label states, not two.** Downstream signals only ever report events that
*happened*. There is no "no click" message, from a real pixel or from anything
else, because a non-event cannot be observed directly. So an absent signal is not
evidence of a negative — it is an absence of evidence, and the ETL writes NULL
rather than 0. Writing 0 would make an unlabelled dataset look like a dataset of
negatives and train the model to predict zero at a healthy-looking loss.

**The conversion lag is what turns absence into evidence.** A win resolves in
seconds; a conversion can take days. `ConversionLagHours` (see
`deployment/glue_etl_cfn.yaml`, settable at deploy time via
`CONVERSION_LAG_HOURS`) declares how long a response may take to arrive, and it
does two jobs with that one number. It holds the ETL's rolling window back that
far, so a bid is only processed once its signals have had time to land; and it sets
each row's attribution deadline. A bid that was **won**, whose deadline has
passed, and which reported no conversion did not convert — and that is a real 0.
A row nothing has reported on at all stays NULL, because with no impression there
was nothing for a response to follow. Declaring `0` asserts responses are
immediate, which makes every won row with no response an immediate negative.

**The outcome simulator.** For a deployment with no win notice and no pixel, an
optional simulator produces synthetic win/impression/click/conversion outcomes so
the loop has labels to learn from. It is **off by default** and stays that way
unless a deployment deliberately wants synthetic labels:

```bash
# Off unless explicitly enabled.
OUTCOME_SIMULATOR_ENABLED=true ./deploy.sh --prefix dv

# The funnel is tunable; each rate defaults to the value in
# source/orchestrator/outcome_simulator.py when left unset.
OUTCOME_SIMULATOR_ENABLED=true \
OUTCOME_SIMULATOR_WIN_RATE=0.40 \
OUTCOME_SIMULATOR_IMPRESSION_RATE=0.95 \
OUTCOME_SIMULATOR_CLICK_RATE=0.02 \
OUTCOME_SIMULATOR_CONVERSION_RATE=0.05 \
OUTCOME_SIMULATOR_CONVERSION_VALUE=25.0 \
  ./deploy.sh --prefix dv
```

Outcomes are derived deterministically from the request ID, so the same request
always produces the same outcome and a training run is reproducible. The funnel
composes the way the real one does — a click requires an impression, a conversion
requires a click — and the shipped rates sit in the range display advertising
actually sees, so a simulated dataset is not trivially separable. A short demo can
raise the click and conversion rates well above realistic levels to reach the
trainer's dataset gate in minutes rather than days; that is a legitimate trade, but
rates set that way describe the simulator and not any campaign.

**What a simulated dataset cannot teach, and cannot prove.** The outcome depends on
the request ID and nothing else — in particular, not on the price the shader chose.
That is the property that makes a run reproducible, and it is also a real limit:
a simulated dataset contains no relationship between price and winning. So the model
cannot learn how its own pricing affects whether it wins, and a canary-vs-stable
comparison over simulated outcomes is not evidence about pricing, because both
variants meet the same outcomes. What the simulator does establish is that the whole
loop runs — outcomes land, the ETL labels them, the gate accepts the dataset, a
model trains, and governance evaluates it. Connect real win notices and conversion
pixels and the missing relationship arrives with them.

**Provenance travels with every row.** Each outcome carries an
`outcome_provenance` of `unresolved`, `simulated`, or `observed`, and it follows the
record through Kinesis, the Glue ETL, the training dataset and into the model's
manifest — which reports the mix it was trained on. A model trained on synthetic
labels is therefore always identifiable as one, and a partition written before the
column existed reports `unresolved` rather than claiming to be `observed`.

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
real reasoning and real telemetry — no agent decision, metric or model comparison
on these pages is fabricated to look better than it is.

The one synthetic input anywhere in the loop is the optional outcome simulator
described above, which is off by default and labels everything it produces
`provenance="simulated"` all the way into the model manifest — so when it is on,
the UI and the manifest both say so rather than presenting synthetic labels as
observed ones.

To generate real training data for the Yield Optimizer specifically, go to the
**Load Test** page and select either **Yield Optimizer — Floor** or **Yield
Optimizer — Margin** as the target, then run a batch. The floor and margin
models are separate containers with separate training targets, so a run captures
outcomes for the one you selected — run it twice, once per target, to produce
training data for both. See [Yield Optimizer: bootstrapping training
data](#yield-optimizer-bootstrapping-training-data-without-a-live-signal-path-yet)
above for why this matters and what it emits.

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

The following is in addition to Part 1's base cost (see [README.md](README.md#cost)). Two AWS Glue ETL jobs run in Part 2 — one for bid outcomes (DLRM/NCF), one for deal-yield outcomes (Yield Optimizer floor/margin):

| AWS service | Dimensions | Cost [USD/month] |
| --- | --- | --- |
| Amazon DynamoDB | 3 tables, on-demand | ~$5 |
| Amazon DynamoDB DAX (optional) | 1 × dax.t3.small | ~$36 |
| Amazon Bedrock AgentCore | Adaptive Bidding Agent, ~8,640 invocations/month | ~$15 |
| Amazon Bedrock AgentCore | Governance Agent, triggered on registration | ~$2 |
| Amazon SageMaker Training | NeMo-RL on ml.g5.2xlarge ($1.515/hr) + built-in XGBoost, ~4 retraining jobs/day × 15 min | ~$60 |
| AWS Glue | 2 scheduled ETL jobs, ~10 DPU-hours/day each | ~$88 |
| Amazon EventBridge Scheduler | 2 schedules | <$1 |
| **Part 2 total (estimate)** | | **~$170–200** |

With both parts running on the included daytime GPU schedule, expect roughly
**$762/month** total. Disabling the scheduled components above drops Part 2's
ongoing cost to near-zero (only DynamoDB storage remains). DAX is optional —
only needed for very high parameter-read request rates. See
[README.md](README.md#cost) for the full combined line-item table.

## Learn more

- Full architecture, the three feedback loops in detail, the governance decision
  pipeline, and Well-Architected analysis: [GUIDANCE-part2.md](GUIDANCE-part2.md)
- Rename map for the four containers: [RENAME_MAP.md](RENAME_MAP.md)
