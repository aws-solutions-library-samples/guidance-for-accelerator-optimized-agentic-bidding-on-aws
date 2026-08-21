# ARTF v1.0 — Intents & Mutations by Buy-Side vs. Sell-Side

Based on the [IAB Tech Lab Agentic Real-Time Framework v1.0](https://github.com/IABTechLab/agentic-real-time-framework/blob/main/Agentic_Real_Time_Framework_Version_1_0_FINAL.md).

---

## How It Works

In ARTF, a **host platform** (SSP, DSP, or exchange) deploys agent containers into its own infrastructure. Each container declares one or more **intents** — the type of mutation it wants to make to the bidstream. The host's orchestrator decides whether to accept or reject each proposed mutation.

Which side hosts a container depends on **who controls the data being mutated and who benefits from the outcome**.

---

## Intents That Benefit SSPs / Publishers

These intents enrich or modify the **bid request** before it reaches buyers — making supply more valuable, addressable, and trustworthy.

### `ACTIVATE_SEGMENTS`

| | |
|---|---|
| **What it mutates** | Adds audience segment IDs to the user object in the bid request |
| **Why it benefits SSPs/Publishers** | Publishers hold first-party data (logins, subscriptions, behavioral signals) but need to express it in a buyer-readable format. An identity or data partner's container resolves raw signals into standard segments and patches them into the request — making inventory more targetable and increasing CPMs without exposing raw user data to buyers. |
| **Example** | A publisher's logged-in user is resolved into segments like `"sports-enthusiast"` or `"high-income-household"` before the request fans out to DSPs. |

---

### `ACTIVATE_DEALS` / `SUPPRESS_DEALS`

| | |
|---|---|
| **What it mutates** | Adds or removes deal IDs on impressions in-flight |
| **Why it benefits SSPs/Publishers** | SSPs and publishers curate private marketplaces (PMPs) dynamically. A curation partner's container can activate deals for impressions that match buyer criteria, or suppress deals that are paused/expired — in real-time, without requiring platform code changes. This drives PMP fill rates and premium pricing. |
| **Example** | A curation platform activates a `"holiday-travel-pmp"` deal on travel-content pages during Q4, then suppresses it automatically in January. |

---

### `ADJUST_DEAL_FLOOR` / `ADJUST_DEAL_MARGIN`

| | |
|---|---|
| **What it mutates** | Changes floor price or margin parameters on existing deals |
| **Why it benefits SSPs/Publishers** | Yield optimization. A container can dynamically raise floors on high-demand inventory or lower them on remnant — maximizing publisher revenue without manual deal management. The SSP maintains control because it can reject any floor change that violates business rules. |
| **Example** | A yield-optimization agent raises the floor on a sports-content deal during a live game when demand spikes. |

---

### `ADD_CONTENT_IDS`

| | |
|---|---|
| **What it mutates** | Adds content-level identifiers (content taxonomy, content IDs, contextual classifications) to the bid request |
| **Why it benefits SSPs/Publishers** | Publishers can enrich their supply with standardized content signals that help buyers target contextually — without relying on user-level identifiers. This is especially valuable in cookieless environments where contextual relevance replaces behavioral targeting. |
| **Example** | A contextual AI container classifies a page as `"IAB-607: Electric Vehicles"` and patches that taxonomy ID into the request. |

---

### `ADD_METRICS` (Pre-Bid Verification)

| | |
|---|---|
| **What it mutates** | Inserts quality/trust signals — viewability predictions, fraud scores, brand-safety classifications |
| **Why it benefits SSPs/Publishers** | Pre-bid verification signals stamp supply as trustworthy *before* buyers see it. Publishers with clean metrics get higher fill rates and CPMs. The SSP hosts the verification container so signals are computed on first-party page data without leaking that data to third parties. |
| **Example** | An IVT detection container scores each request and adds `"fraud_risk": 0.02` — buyers bid more confidently on verified supply. |

---

## Intents That Benefit DSPs / Agencies / Advertisers

These intents modify the **bid response** or influence buying decisions — optimizing spend efficiency, audience precision, and campaign performance.

### `BID_SHADE`

| | |
|---|---|
| **What it mutates** | Adjusts the bid price downward on the response before it's submitted to the auction |
| **Why it benefits DSPs/Advertisers** | Bid shading saves advertisers money in first-price auctions by predicting the minimum bid needed to win. A specialized ML container can analyze auction dynamics (win rates, floor patterns, competitive density) and shade bids optimally — reducing CPMs without sacrificing win rate. |
| **Example** | A bid-shading agent reduces a $12 bid to $8.40 based on historical clearing prices for that inventory, saving the advertiser 30%. |

---

### `ACTIVATE_SEGMENTS` (Buy-Side Use)

| | |
|---|---|
| **What it mutates** | Resolves or enriches audience identifiers on the bid request to improve match rates |
| **Why it benefits DSPs/Advertisers** | While this intent also serves the sell-side, a DSP can host identity-resolution containers that enrich incoming bid requests with additional user graph data — improving addressability and match rates against advertiser audience lists. The DSP maintains data control because the container runs in its own infra. |
| **Example** | An identity container resolves a publisher's first-party ID into a unified ID that maps to the advertiser's CRM segments, enabling precision targeting. |

---

### `ADD_METRICS` (Buy-Side Use — Post-Bid Analytics)

| | |
|---|---|
| **What it mutates** | Appends measurement or attribution signals to bid responses/events |
| **Why it benefits DSPs/Advertisers** | Advertisers need closed-loop measurement. A measurement container within the DSP can tag bids with attribution metadata, attention scores, or incrementality signals — enriching campaign analytics without requiring pixel-based tracking. |
| **Example** | An attention-measurement container appends predicted attention scores to each bid, enabling the DSP to optimize toward attention rather than just viewability. |

---

## Both Sides — Shared Benefit

| Intent | Sell-Side Benefit | Buy-Side Benefit |
|--------|-------------------|------------------|
| `ACTIVATE_SEGMENTS` | Makes supply more addressable → higher CPMs | Improves audience match rates → better targeting precision |
| `ADD_METRICS` | Stamps supply as trustworthy → higher fill rates | Enriches campaign analytics → better optimization signals |

---

## Summary

| Intent | Primary Beneficiary | Core Value |
|--------|-------------------|------------|
| `ACTIVATE_SEGMENTS` | **Both** (leans sell-side) | Audience enrichment / identity resolution |
| `ACTIVATE_DEALS` | **SSP / Publisher** | Dynamic PMP activation |
| `SUPPRESS_DEALS` | **SSP / Publisher** | Remove stale/paused deals in real-time |
| `ADJUST_DEAL_FLOOR` | **SSP / Publisher** | Dynamic yield optimization |
| `ADJUST_DEAL_MARGIN` | **SSP / Publisher** | Margin management on curated deals |
| `ADD_CONTENT_IDS` | **SSP / Publisher** | Contextual enrichment for cookieless targeting |
| `ADD_METRICS` | **Both** | Pre-bid verification (sell) / measurement (buy) |
| `BID_SHADE` | **DSP / Advertiser** | Spend efficiency in first-price auctions |

---

## Key Takeaway

**Sell-side intents** enrich the bid request to make supply more valuable, addressable, and trustworthy — driving CPMs and fill rates.

**Buy-side intents** optimize the bid response to improve spend efficiency, targeting precision, and measurement — driving ROAS and campaign performance.

The ARTF model works because both sides benefit from the same infrastructure pattern: the host maintains data control and SLA governance, while the container provider focuses purely on its algorithmic value-add.
