"""Closed-Loop Demo — controllable synthetic-outcome generation for Part 2.

This package is an explicit **demo data generator**. It produces *synthetic
bid-outcome market data* and feeds it as **input** to the real closed-loop
learning components (the Bid Shading Strategy Agent and the A/B evaluator).

Design contract (per the repo's no-fabricated-data rule):

- The *only* synthetic thing produced here is the **input** market data
  (CloudWatch metric datapoints and A/B metric samples). This is realistic
  test input the user deliberately controls.
- The **decisions** (parameter updates, promote/reject recommendations) are
  computed by the real, unmodified Part 2 code and persisted to the real
  DynamoDB parameter store / audit trail. Nothing about the *outcome* of a
  decision is faked, hardcoded, or randomized to "look real".
- Every generated batch returns an evidence record describing exactly what was
  emitted (metric names, values, timestamps) so a consumer can verify it.

Modules:
- ``scenarios``  — deterministic presets mapping a scenario to target input
  metrics and the decision it is expected to drive.
- ``generator``  — emits the synthetic input metrics to real CloudWatch.
- ``invoker``    — runs the real decision path and returns the real decision.
- ``readers``    — reads back real Part 2 state for visualization.
"""
