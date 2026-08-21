# ARTF v1.0 — Intents & Mutations Relevant to SSPs/Publishers

Based on the [IAB Tech Lab Agentic Real-Time Framework v1.0](https://github.com/IABTechLab/agentic-real-time-framework/blob/main/Agentic_Real_Time_Framework_Version_1_0_FINAL.md).

---

## Overview

The ARTF spec is side-agnostic — any "orchestrating entity" (SSP, DSP, or exchange) can host containers. For **SSPs and Publishers**, the relevant intents are those that mutate the **bid request** (supply-side enrichment) before it fans out to buyers. The orchestrator (SSP) evaluates each proposed mutation and accepts/rejects it.

---

## Intents SSPs/Publishers Would Host Containers For

### 1. `activateSegments` / `audienceSegmentation`

| | |
|---|---|
| **What it does** | Adds audience cohorts/segments to the user object in the bid request before it goes to buyers. |
| **Why SSP/Publisher hosts it** | Publishers have first-party data. An identity or data partner deploys a container that enriches the bid request with segments (e.g. `"18-35-age-segment"`, `"soccer-watchers"`) *before* the request reaches DSPs — increasing bid density and CPMs. |
| **Op** | `add` on `/user/data/segment` |
| **Typical container provider** | LiveRamp, ID5, publisher DMP |

---

### 2. `activateDeals` / `expireDeals` / `adjustDeals` (Dynamic Deal Curation)

| | |
|---|---|
| **What it does** | Adds new deal IDs, removes expired deals, or adjusts deal parameters (floor price, `wadomain` allowlists) on impressions in-flight. |
| **Why SSP/Publisher hosts it** | Publishers/SSPs curate PMPs dynamically. A curation partner's container can activate deals for specific buyer segments, expire stale deals, or adjust floors — all in real-time without platform code changes. |
| **Ops** | `add`, `remove`, `replace` on `/imp/{n}` and `/imp/{n}/deals/{id}` |
| **Payload** | `IDsPayload` (for activate/expire) or `AdjustDealPayload` (for floor/domain changes) |
| **Typical container provider** | Curation platforms (Audigent, Peer39), yield management tools |

---

### 3. `metadataEnhancement`

| | |
|---|---|
| **What it does** | Inserts or modifies auction metadata — fraud scores, viewability signals, brand-safety classifications, content taxonomy. |
| **Why SSP/Publisher hosts it** | Pre-bid verification. A fraud detection or viewability vendor deploys a container that stamps each request with signals *before* it reaches buyers. Publishers benefit from higher trust and fill rates. |
| **Op** | `add` or `replace` on metadata paths |
| **Typical container provider** | IAS, DoubleVerify, Pixalate, Adelaide |

---

### 4. `bidRequestModification`

| | |
|---|---|
| **What it does** | General-purpose mutations to the bid request prior to auction execution. |
| **Why SSP/Publisher hosts it** | Catch-all for supply-side enrichment — contextual signals, page-quality scores, identity resolution (e.g. UID2/ID5 resolution container that resolves identifiers and patches them into the request). |
| **Typical container provider** | ID vendors, contextual AI providers |

---

### 5. `auctionOrchestration`

| | |
|---|---|
| **What it does** | Routes or prioritizes bid requests across multiple buyers. |
| **Why SSP/Publisher hosts it** | The SSP *is* the orchestrator. This intent lets a container influence routing logic — e.g. a yield-optimization agent that decides which demand paths to activate for a given impression. |
| **Typical container provider** | Yield analytics, traffic shaping tools |

---

## Intents More Relevant to DSP/Buy-Side (NOT typically hosted by SSPs)

| Intent | Why it's buyer-side |
|--------|-------------------|
| `bidResponseGeneration` | Generating the actual bid — that's the DSP's job |
| `bidValuation` | Evaluating how much to bid — buyer decisioning |
| `adjustBid` | Modifying bid price on the response — buyer-side |

---

## Summary Table

| Intent | Use Case | Typical Container Provider |
|--------|----------|--------------------------|
| `activateSegments` | First-party data / identity enrichment | LiveRamp, ID5, publisher DMP |
| `activateDeals` | Dynamic PMP activation | Audigent, Peer39, curation platforms |
| `expireDeals` | Remove stale/inactive deals | Curation or yield tools |
| `adjustDeals` | Dynamic floor pricing, domain allowlists | Yield management, SSP-side optimization |
| `metadataEnhancement` | Fraud, viewability, brand safety signals | IAS, DV, Pixalate, Adelaide |
| `bidRequestModification` | Identity resolution, contextual enrichment | ID vendors, contextual AI |
| `auctionOrchestration` | Demand routing, header bidding optimization | Yield analytics, traffic shaping |

---

## Key Architectural Note

On the SSP/publisher side, containers mutate the **bid request** (supply-side enrichment) before it fans out to buyers. The orchestrator (SSP) evaluates each proposed mutation and accepts/rejects it. The spec's "least-data" principle matters most here — containers only see the fields the SSP exposes, protecting publisher first-party data.
