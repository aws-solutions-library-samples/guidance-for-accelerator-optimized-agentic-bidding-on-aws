# Training the Bid Shading Model

A bid shader is trained on data it generated. It chose a price, the auction resolved, and a response
either followed or did not — so every training row is the consequence of a decision the model made.
That has two implications the pipeline is built around.

**The decision has to be recorded with the outcome.** Knowing an impression converted is not enough;
you need the price that was bid, the features that produced it, and the model version that served.
Otherwise the row cannot be attributed to anything.

**Outcomes arrive later than decisions, and at different speeds.** A win resolves in seconds. A
conversion can take days. A pipeline that reads too early sees the absence of a conversion and records
it as a negative.

## The loop

```mermaid
flowchart TB
  subgraph COLLECT["1 · COLLECT — one event per shading decision"]
    direction TB
    C1["What we decided<br/>original price, shaded price, floor,<br/>the features used, the parameters applied,<br/>the model version that served"]
    C2["What happened<br/>won → impression → click → conversion<br/>each landing on its own timescale"]
    C3[("Event store<br/>keyed by request id · origin marked")]
    C1 --> C3
    C2 --> C3
  end

  subgraph PREP["2 · PREPARE — batch pipeline"]
    direction TB
    P1["De-duplicate by request id,<br/>keep the most recent version"]
    P2["Window = the conversion lag<br/>of the objective being modelled"]
    P3["Label = the response the<br/>advertiser pays for"]
    P4["Derived features published as<br/>a lookup the container carries"]
    P5{"Gate: does the data carry signal?<br/>categorical variation + positive rate"}
    P1 --> P2 --> P3 --> P4 --> P5
  end

  subgraph TRAIN["3 · TRAIN — two stages, two parameter sets"]
    direction TB
    T1["Stage A — the probability head<br/>supervised on the label, against raw logits"]
    T2["Measure calibration on held-out data<br/>and record it in the manifest"]
    T3["Stage B — the policy<br/>searched against realised ROI<br/>with the head FROZEN"]
    T1 --> T2 --> T3
  end

  subgraph SHIP["4 · SHIP"]
    direction TB
    S1["Held-out evaluation<br/>unseen periods and unseen keys"]
    S2["Export + register<br/>label, objective, calibration,<br/>feature spec version, data origin"]
    S3["Canary against the incumbent"]
    S4["Governed promotion,<br/>rolling deployment"]
    S1 --> S2 --> S3 --> S4
  end

  C3 --> P1
  P5 -->|"passes"| T1
  P5 -->|"fails — named diagnosis, no weights"| STOP["Run ends"]
  T3 --> S1
  S4 -->|"serves the next requests"| C1

  classDef collect fill:#F0F9FF,stroke:#BAE6FD,color:#0F172A
  classDef prep fill:#ECFDF5,stroke:#A7F3D0,color:#0F172A
  classDef train fill:#F5F3FF,stroke:#DDD6FE,color:#0F172A
  classDef ship fill:#FFFBEB,stroke:#FDE68A,color:#78350F
  classDef stop fill:#F1F5F9,stroke:#CBD5E1,color:#334155
  class C1,C2,C3 collect
  class P1,P2,P3,P4,P5 prep
  class T1,T2,T3 train
  class S1,S2,S3,S4 ship
  class STOP stop
```

