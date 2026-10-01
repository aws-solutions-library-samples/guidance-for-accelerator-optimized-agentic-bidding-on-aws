# ARTF v1.0 — Intents & Mutations by Buy-Side vs. Sell-Side

Based on the [IAB Tech Lab Agentic Real Time Framework v1.0](https://github.com/IABTechLab/agentic-real-time-framework/blob/main/Agentic_Real_Time_Framework_Version_1_0_FINAL.md), as implemented in this guidance.

The spec's intent enum is vendored at `source/proto/agenticrtbframework.proto`. The wire names used throughout this document are the enum's exact values:

| Value | Intent | Spec comment |
|---|---|---|
| 1 | `ACTIVATE_SEGMENTS` | Activate user segments by their external segment IDs |
| 2 | `ACTIVATE_DEALS` | Activate deals by their external deal IDs |
| 3 | `SUPPRESS_DEALS` | Suppress deals by their external deal IDs |
| 4 | `ADJUST_DEAL_FLOOR` | Adjust the bid floor of a specific deal |
| 5 | `ADJUST_DEAL_MARGIN` | Adjust the deal margin of a specific deal |
| 6 | `BID_SHADE` | Adjust the bid price of a specific bid |
| 7 | `ADD_METRICS` | Add metrics to an impression |
| 8 | `ADD_CIDS` | Add extended content IDs |

Every mutation is `{intent, op, path, value}` where `op` is `ADD`, `REMOVE` or `REPLACE`, `path` names the OpenRTB location, and `value` is one of `ids`, `adjust_deal`, `adjust_bid`, `add_metrics` or `content_data`. The `RTBRequest` envelope carries a `lifecycle` (`LIFECYCLE_PUBLISHER_BID_REQUEST` or `LIFECYCLE_DSP_BID_RESPONSE`), the `bid_request`, optionally the `bid_response`, and the `applicable_intents` the host is willing to accept.

---

## How It Works

In ARTF, a **host platform** (SSP, DSP, or exchange) deploys agent containers into its own infrastructure. Each container declares one or more **intents**, the type of mutation it wants to make to the bidstream. The host's orchestrator decides whether to accept or reject each proposed mutation.

Which side hosts a container depends on **who controls the data being mutated and who benefits from the outcome**.

### How this guidance implements the pattern

The default deployment has **one ARTF host: the orchestrator, called directly** by the frontend (Auction Theater, flow pipeline, load test) and by the optional Bedrock AgentCore MCP runtime. That is enough to exercise every container, every intent and the closed loop.

```
+-----------------------------+       +----------------------+       +----------------------------------+
|  Callers                    |       |  ARTF Orchestrator   |       |  ARTF containers (EKS)           |
|  - Frontend / Theater       | ----> |  POST /v1/mutations  | ----> |  6 shipped + registry-attached   |
|  - Load test, MCP runtime   | <---- |  fan-out, precedence | <---- |  gRPC :50051  REST/MCP :8081     |
|  - Prebid Server (OPTIONAL) |       |  outcome events      |       |  4 of 6 call NVIDIA Triton (GPU) |
+-----------------------------+       +----------------------+       +----------------------------------+
```

- **Orchestrator** (`source/orchestrator/app.py`): fans out to every active container whose intents overlap `applicable_intents`, in parallel, within `tmax`. The container set is the six code-defined containers plus any records in the DynamoDB container registry (`${STACK_NAME}-container-registry`). Containers are called over REST `POST /mutate` with an MCP `extend_rtb` fallback; each container also serves the spec's gRPC `RTBExtensionPoint/GetMutations` on 50051.
- **Precedence**: two containers may claim the same intent and both are called. For each `(path, intent)` one mutation survives: highest `priority` on the registry record wins, then later registry order (store records follow the built-ins). With no priorities set this is last-write-wins. Losers are reported in `metadata.conflicts` and the losing container's `superseded` count, and its own `mutations` stay visible.
- **Feedback**: after responding, the orchestrator emits a bid-outcome event and a deal-yield-outcome event for every request. Those feed the closed loop described in `CLOSED_LOOP.md`.

### Optional: Prebid Server as a second host, for testing containers inside a real auction

Prebid Server is **not part of the default deployment**. It is an opt-in, additive variant (`./deploy.sh --with-prebid`, or `./deploy_prebid.sh --prefix <prefix>` onto an existing deployment) whose purpose is to test the ARTF containers in an integrated, real bidding scenario: the mutations are applied inside a live OpenRTB auction that Prebid then resolves, with floor enforcement, currency, deal handling, top-bid selection and `ext.seatnonbid`. Without the flag nothing else in this guidance changes; the orchestrator remains the only host, and the Auction Theater shows a captured fixture in place of the live auction.

It is built from the **[Guidance for Deploying a Prebid Server on AWS](https://github.com/aws-solutions-library-samples/prebid-server-deployment-on-aws)**, consumed as a pinned, unforked release (`v1.4.0`, Prebid Server Java `3.43.0`). The deploy fetches that release, adds this guidance's hook module and adapter through the one extension point the upstream Dockerfile provides (`diff -rq` against the pristine release: 0 modified files, additions only), and runs the result as pods **in the EKS cluster this guidance already creates**, rather than on the upstream guidance's ECS Fargate topology. The second competing seat is that same release's `amt` adapter pointed at its bidder simulator, running in-cluster (`--no-simulator` turns it off). Nothing upstream is vendored; the upstream guidance remains deployable on its own if you want Prebid in its own VPC.

When deployed, the Prebid host works like this:

- **Hook**: the `artf-orchestrator` module (`source/prebid/artf-hook/`) runs at the `processed-auction-request` stage, before any bidder is called, with the lifecycle fixed to the publisher bid request. It calls the orchestrator with a Cognito client-credentials token, never the containers, and inherits whatever the container registry resolves.
- **Acceptance**: the orchestrator resolves *which* mutation is returned; the Prebid applier (`ArtfMutationApplier.java`) decides whether it is *applied*. Every mutation gets a disposition, application is all-or-nothing per mutation, and a rejection with a reason is a normal outcome. The applier writes to four locations only (`user.data`, `imp.pmp.deals`, `imp.bidfloor`/deal floor, `imp.metric`) and can never address a single bidder, which is what keeps the module distributable. A hook timeout or fault never rejects an auction; the auction proceeds unmutated.
- **Demand**: the `artfhouse` adapter forwards the enriched request unchanged to a demand endpoint holding an authored campaign catalog, which matches deals, compares CPM against the resolved floor, and bids or reports a per-campaign exclusion reason. The auction mechanism and the contest are real; the prices are fixtures you control.

The "Host write" rows and the "Applied in Prebid host" column below describe this optional variant. Every intent except `BID_SHADE` is exercised there; all eight are exercised on the default orchestrator path.

---

## Intents That Benefit SSPs / Publishers

These intents enrich or modify the **bid request** before it reaches buyers, making supply more valuable, addressable, and trustworthy. All of them run on the default orchestrator path, and all of them are applied to a live auction when the optional Prebid host is deployed.

### `ACTIVATE_SEGMENTS`

| | |
|---|---|
| **What it mutates** | Adds audience segment IDs to the user object in the bid request |
| **Why it benefits SSPs/Publishers** | Publishers hold first-party data (logins, subscriptions, behavioral signals) but need to express it in a buyer-readable format. An identity or data partner's container resolves raw signals into standard segments and patches them into the request, making inventory more targetable and increasing CPMs without exposing raw user data to buyers. |
| **Example** | A publisher's logged-in user is resolved into segments like `"sports-enthusiast"` or `"high-income-household"` before the request fans out to DSPs. |
| **Implemented by** | **Audience Activator** (`widedeep-segment-activator`, `source/containers/widedeep_segment_activator/`). Rule-based on CPU (`segment-rules-v2-iab-taxonomy`): maps `site.cat`/`app.cat` with `cattax` to IAB interest segments, `user.yob` to an IAB Audience Taxonomy age bucket, reclassifies asserted `user.data[].segment[].name` values, and adds contextual signals (floor tier, mobile UA, video). Activation threshold 0.55, overridable per request via `model_params.segment_threshold`. The Wide & Deep Triton model it replaced is slated for a partner ISV implementation. |
| **Mutation** | `op=ADD`, `path=/user/data/segment`, `ids=[...]` |
| **Host write** | Appends a `user.data` entry named `artf` carrying the segments. Rejected if no ids are supplied. |

---

### `ACTIVATE_DEALS` / `SUPPRESS_DEALS`

| | |
|---|---|
| **What it mutates** | Adds or removes deal IDs on impressions in-flight |
| **Why it benefits SSPs/Publishers** | SSPs and publishers curate private marketplaces (PMPs) dynamically. A curation partner's container can activate deals for impressions that match buyer criteria, or suppress deals that are paused/expired, in real time and without platform code changes. This drives PMP fill rates and premium pricing. |
| **Example** | A curation platform activates a `"holiday-travel-pmp"` deal on travel-content pages during Q4, then suppresses it automatically in January. |
| **Implemented by** | **Deal Scorer** (`ncf-deal-manager`, `source/containers/ncf_deal_manager/`). NCF / NeuMF collaborative-filtering model served by Triton (`ncf-neumf-triton-v1`); scores the hashed user id against each `imp.pmp.deals[].id`, activation threshold 0.499. The two intents are gated independently, so a request that asks only for `SUPPRESS_DEALS` never receives an activation. Retrained on the slow loop (NeMo-RL on SageMaker, ONNX to TensorRT). |
| **Mutation** | `ACTIVATE_DEALS`: `op=ADD`, `path=/imp/{impId}`, `ids=[dealId...]`. `SUPPRESS_DEALS`: `op=REMOVE`, same path and payload. |
| **Host write** | Activation appends `Deal{id}` to `imp.pmp.deals` if not already present. Suppression **marks** the deal with `deal.ext.artf.suppressed = true` rather than deleting it, so the auction record (and the Auction Theater) can show a deal was considered and suppressed instead of never offered. The `artfhouse` demand endpoint reads that marker and excludes the campaign with reason `deal_suppressed`. |

---

### `ADJUST_DEAL_FLOOR` / `ADJUST_DEAL_MARGIN`

| | |
|---|---|
| **What it mutates** | Changes floor price or margin parameters on existing deals |
| **Why it benefits SSPs/Publishers** | Yield optimization. A container can dynamically raise floors on high-demand inventory or lower them on remnant, maximizing publisher revenue without manual deal management. The SSP keeps control because it can reject any floor change that violates business rules. |
| **Example** | A yield-optimization agent raises the floor on a sports-content deal during a live game when demand spikes. |
| **Implemented by** | Two containers, one per intent, so a request asking for only one intent never fans out to the other model and each has its own health, load-test target and outcome attribution. **Yield Optimizer — Floor** (`yield-optimizer-floor`, `source/containers/yield_optimizer_floor/`): XGBoost on Triton's FIL backend (`deal_yield_manager_floor`, `deal-yield-floor-xgboost-v1`) predicts a floor multiplier; the CPU fallback returns 1.0, meaning no change. **Yield Optimizer — Margin** (`yield-optimizer-margin`, `source/containers/yield_optimizer_margin/`): XGBoost on FIL (`deal_yield_manager_margin`, `deal-yield-margin-xgboost-v1`) predicts a margin value. Both clamp their outputs and run bounded epsilon-greedy exploration during load tests to bootstrap training data. Retrained with SageMaker built-in XGBoost from a single Glue ETL job whose dedup key includes the intent. |
| **Mutation** | `op=REPLACE`, `path=/imp/{impId}/deals/{dealId}`. Floor: `adjust_deal.bidfloor` (absolute, non-negative). Margin: `adjust_deal.margin={value, calculation_type}` where `CPM` (0) is absolute and `PERCENT` (1) is relative; the container picks `CPM` when `deal.at == 1`, else `PERCENT`. The payload carries no deal id and no currency; the deal is identified only by path. |
| **Host write** | Resolves the new floor (absolute for FLOOR; `base + value` or `base + base*value/100` for MARGIN) and rejects an unknown calculation type, a missing imp or deal, or any result that is not strictly positive, because Prebid ignores non-positive floors. Writes `deal.bidfloor` and `deal.bidfloorcur` (currency from deal, then imp, then `request.cur[0]`, then `USD`), and **also** raises `imp.bidfloor` to `max(existing, new)`. A mutation never lowers a floor the publisher already set. Prebid's own price-floors module is disabled so the hook is the single floor authority. |
| **Attachable** | An externally built container may claim `ADJUST_DEAL_FLOOR`; see [Attaching third-party containers](#attaching-third-party-containers). |

---

### `ADD_CIDS`

| | |
|---|---|
| **What it mutates** | Adds content-level identifiers (content taxonomy, content IDs, contextual classifications) to the bid request |
| **Why it benefits SSPs/Publishers** | Publishers can enrich their supply with standardized content signals that help buyers target contextually without relying on user-level identifiers. This is especially valuable in cookieless environments where contextual relevance replaces behavioral targeting. |
| **Example** | A contextual AI container classifies a page as `"IAB-607: Electric Vehicles"` and patches that taxonomy ID into the request. |
| **Implemented by** | No shipped container. This is the one intent in the enum that this repository leaves open on purpose: the **ARTF template container** (`source/containers/artf_template/`) defaults to `ADD_CIDS` so that a partner container fills a gap rather than shadowing working code. The template is deployed with an inactive registry record and returns no mutations until its `mutate()` is implemented. GUIDANCE.md lists a future ISV Identity Resolver on this intent. |
| **Mutation** | Suggested shape in the template: `op=ADD`, `path=/imp/{impId}/ext/cids`, `ids=[...]`; the spec also provides `content_data` (`DataPayload` of OpenRTB `Data`). |
| **Host write** | The Prebid hook requests `ADD_CIDS` by default and the applier accepts it at `/imp/{impId}/metric` alongside `ADD_METRICS`. |

---

### `ADD_METRICS` (Pre-Bid Verification)

| | |
|---|---|
| **What it mutates** | Inserts quality/trust signals: viewability predictions, fraud scores, brand-safety classifications |
| **Why it benefits SSPs/Publishers** | Pre-bid verification signals stamp supply as trustworthy *before* buyers see it. Publishers with clean metrics get higher fill rates and CPMs. The SSP hosts the verification container so signals are computed on first-party page data without leaking that data to third parties. |
| **Example** | An IVT detection container scores each request and adds `"fraud_risk": 0.02`, so buyers bid more confidently on verified supply. |
| **Implemented by** | **Signals Enricher** (`metrics-enricher`, `source/containers/metrics_enricher/`). Rule-based on CPU (`metrics-rules-v2-iab-taxonomy`): a viewability estimate from `imp.pos` and a brand-safety score from `site.cat`/`cattax` scaled to [0.60, 1.00]. GUIDANCE.md lists a future ISV Creative Enricher on the same intent. |
| **Mutation** | `op=ADD`, `path=/imp/{impId}/metric`, `add_metrics.metric=[{type:"viewability", value, vendor:"nvidia-artf"}, {type:"brand_safety", value, vendor:"nvidia-artf"}]`. This guidance's proto references `BidRequest.Imp.Metric` for this payload, the one local change to the vendored schema. |
| **Host write** | Appends to `imp.metric`. Each metric needs a non-blank `type` and a `value` in [0.0, 1.0], the OpenRTB range Prebid's own validator enforces; out-of-range values are rejected with a reason rather than clamped. In the measured run of the optional Prebid variant in README.md this is the intent that applies on every auction. |

---

## Intents That Benefit DSPs / Agencies / Advertisers

These intents modify the **bid response** or influence buying decisions, optimizing spend efficiency, audience precision, and campaign performance.

### `BID_SHADE`

| | |
|---|---|
| **What it mutates** | Adjusts the bid price downward on the response before it's submitted to the auction |
| **Why it benefits DSPs/Advertisers** | Bid shading saves advertisers money in first-price auctions by predicting the minimum bid needed to win. A specialized ML container can analyze auction dynamics (win rates, floor patterns, competitive density) and shade bids optimally, reducing CPMs without sacrificing win rate. |
| **Example** | A bid-shading agent reduces a $12 bid to $8.40 based on historical clearing prices for that inventory, saving the advertiser 30%. |
| **Implemented by** | **Bid Pricer** (`dlrm-bid-shader`, `source/containers/dlrm_bid_shader/`). DLRM CTR model on Triton (`dlrm-nvidia-triton-v1`). For each bid it computes expected value `predicted_ctr * conversion_value` and prices it through a `ShadingPolicy` (slope from `shade_factor`, optional `base`/`curvature`) clamped to `[imp.bidfloor, original_price]`. A mutation is emitted only when the shaded price moves more than $0.01. If no CTR prediction is available the container **abstains** with `abstained_reason: inference_unavailable` rather than shading from a placeholder. `shade_factor` and `conversion_value` come from `model_params` on the request, else from the DynamoDB Parameter Store. Full detail in `DLRM_BID_SHADING.md`. |
| **Mutation** | `op=REPLACE`, `path=/seatbid/{seat}/bid/{bidId}`, `adjust_bid.price` |
| **Where it runs** | This is the only response-side intent and it requires a `bid_response` in the `RTBRequest`. It runs on the default orchestrator path used by the Auction Theater, the load test and the MCP proxy. It is **excluded from the optional Prebid hook's intent set**: `processed-auction-request` runs before any `seatbid` exists, so the hook resolves a `/seatbid/...` path as unresolvable and reports the Bid Pricer as `skipped`. Left in, it would be a silent no-op. |
| **Closed loop** | The fast loop (~5 min) has the Adaptive Bidding Strategy Agent on Bedrock AgentCore read win rate, ROI and prices paid from CloudWatch and write `shade_factor`/`conversion_value` to the Parameter Store within hard safety bounds; the container reads them through a TTL cache. The slow loop (~6 hr) retrains the DLRM with NeMo-RL on SageMaker from bid-outcome events, and the Model Promotion Governance Agent runs canary and A/B before promoting. Per-request `metadata.model_version` names the served variant so outcomes are attributable. |

---

### `ACTIVATE_SEGMENTS` (Buy-Side Use)

| | |
|---|---|
| **What it mutates** | Resolves or enriches audience identifiers on the bid request to improve match rates |
| **Why it benefits DSPs/Advertisers** | While this intent also serves the sell-side, a DSP can host identity-resolution containers that enrich incoming bid requests with additional user graph data, improving addressability and match rates against advertiser audience lists. The DSP keeps data control because the container runs in its own infra. |
| **Example** | An identity container resolves a publisher's first-party ID into a unified ID that maps to the advertiser's CRM segments, enabling precision targeting. |
| **In this guidance** | With the optional Prebid variant deployed, the same Audience Activator mutation is consumed on the buy side by the `artfhouse` demand endpoint, which targets campaigns against the enriched request (`not_targeted` is one of its exclusion reasons). GUIDANCE.md lists a future ISV Location Activator on this intent. |

---

### `ADD_METRICS` (Buy-Side Use — Post-Bid Analytics)

| | |
|---|---|
| **What it mutates** | Appends measurement or attribution signals to bid responses/events |
| **Why it benefits DSPs/Advertisers** | Advertisers need closed-loop measurement. A measurement container within the DSP can tag bids with attribution metadata, attention scores, or incrementality signals, enriching campaign analytics without pixel-based tracking. |
| **Example** | An attention-measurement container appends predicted attention scores to each bid, enabling the DSP to optimize toward attention rather than just viewability. |
| **In this guidance** | Not implemented as a response-side mutation. Measurement here flows through the orchestrator's outcome events (bid outcomes and deal-yield outcomes to Kinesis Firehose, S3 and Glue) rather than through `ADD_METRICS` on a `bid_response`. |

---

## Both Sides — Shared Benefit

| Intent | Sell-Side Benefit | Buy-Side Benefit |
|--------|-------------------|------------------|
| `ACTIVATE_SEGMENTS` | Makes supply more addressable → higher CPMs | Improves audience match rates → better targeting precision |
| `ADD_METRICS` | Stamps supply as trustworthy → higher fill rates | Enriches campaign analytics → better optimization signals |

---

## Attaching third-party containers

The registry table accepts containers this repository did not build, with no orchestrator code change and no Java change, because every host (the frontend, the MCP runtime, and the optional Prebid hook) calls the orchestrator and inherits whatever the registry resolves. `deployment/attach_artf_container.sh` validates the producer's `artf-registry-record.json`, copies the image into this account's ECR, renders the Kubernetes manifest, probes `POST /mutate`, and writes the record inactive; activation is a flip of `active` on the record or in the Container Health panel. The endpoint must be cluster-internal, `host:port` with no path and no IP.

The worked example is `contextual-yield-agent` from Guidance for Containerized Advertising Context on AWS. It claims `ADJUST_DEAL_FLOOR`, so at the default priority it wins the floor decision over the built-in Yield Optimizer — Floor; `--priority -1` keeps the built-in. Built-ins are pinned at priority 0 and cannot be renamed, re-targeted or deactivated from the UI or the table, so priority on the attached record is the one lever. The cross-account ECR pull permission is the producer's to grant.

---

## Summary

| Intent | Primary Beneficiary | Core Value | Container here | Model | Applied in optional Prebid host |
|--------|-------------------|------------|----------------|-------|---------------------------------|
| `ACTIVATE_SEGMENTS` | **Both** (leans sell-side) | Audience enrichment / identity resolution | Audience Activator | Rule engine (CPU) | Yes → `user.data` |
| `ACTIVATE_DEALS` | **SSP / Publisher** | Dynamic PMP activation | Deal Scorer | NCF / NeuMF on Triton | Yes → `imp.pmp.deals` |
| `SUPPRESS_DEALS` | **SSP / Publisher** | Remove stale/paused deals in real time | Deal Scorer | NCF / NeuMF on Triton | Yes → `deal.ext.artf.suppressed` |
| `ADJUST_DEAL_FLOOR` | **SSP / Publisher** | Dynamic yield optimization | Yield Optimizer — Floor | XGBoost on Triton FIL | Yes → `deal.bidfloor`, `imp.bidfloor` |
| `ADJUST_DEAL_MARGIN` | **SSP / Publisher** | Margin management on curated deals | Yield Optimizer — Margin | XGBoost on Triton FIL | Yes → `deal.bidfloor`, `imp.bidfloor` |
| `ADD_CIDS` | **SSP / Publisher** | Contextual enrichment for cookieless targeting | Template only (partner slot) | None | Accepted → `imp.metric` |
| `ADD_METRICS` | **Both** | Pre-bid verification (sell) / measurement (buy) | Signals Enricher | Rule engine (CPU) | Yes → `imp.metric` |
| `BID_SHADE` | **DSP / Advertiser** | Spend efficiency in first-price auctions | Bid Pricer | DLRM on Triton | No (response-side; orchestrator path only) |

---

## Key Takeaway

**Sell-side intents** enrich the bid request to make supply more valuable, addressable, and trustworthy, driving CPMs and fill rates. In this guidance all seven request-side intents run on the default orchestrator path, and with the optional Prebid variant (built from the AWS Prebid Server guidance) they flow through one hook that applies or rejects each mutation with a reason inside a real, contested auction.

**Buy-side intents** optimize the bid response to improve spend efficiency, targeting precision, and measurement, driving ROAS and campaign performance. Here that is `BID_SHADE`, the one intent that needs a `bid_response`, and the one whose parameters and model are tuned by the closed loop.

The ARTF model works because both sides benefit from the same infrastructure pattern: the host keeps data control and SLA governance, while the container provider focuses purely on its algorithmic value-add. The container registry and precedence rules are what let a third party's model take over a single intent without the host changing any code.
