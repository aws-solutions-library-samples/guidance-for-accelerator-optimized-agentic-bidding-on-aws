# Test Suite

## Running Tests

```bash
pytest source/tests/
```

Run a specific file:

```bash
pytest source/tests/test_reward.py -v
```

## Test Categories

### Unit Tests (Requirement 13.2)

| File | What it covers | Requirements |
|------|---------------|-------------|
| `test_reward.py` | Reward function with known outcomes (profitable win, overpayment, lost high-value, lost low-value, bounds) | 3.3 |
| `test_adaptive_bidding_agent.py` | Adjustment policy with synthetic MarketState (below/above target, within tolerance, bounded, min_samples) | 7.2, 7.3, 7.4, 7.5, 7.7, 8.5, 10.1 |
| `test_ab_evaluator.py` | A/B evaluator with known distributions (clear winner, clear loser, identical, SPRT, guardrails) | 4.3, 4.4 |
| `test_canary_deployer.py` | Canary routing with deterministic hashes (same request_id same version, traffic distribution, promote/rollback) | 5.4, 5.5, 5.6, 5.7 |

### Supporting Unit Tests

| File | What it covers |
|------|---------------|
| `test_feedback_models.py` | BidOutcomeRecord validation rules |
| `test_feedback_collector.py` | Kinesis emission, backpressure, error handling |
| `test_parameter_store.py` | DynamoDB parameter CRUD, bounds enforcement, optimistic locking |
| `test_parameter_cache.py` | DAX/in-memory cache, TTL, graceful degradation |
| `test_model_deployer.py` | NIM optimization and Triton model loading |
| `test_governance_agent.py` | Governance pipeline (validate, promote, reject) |
| `test_guardrail_monitor.py` | Latency/error-rate guardrail detection and rollback |
| `test_training_pipeline.py` | Retraining trigger, at-most-one job, registration |
| `test_failure_handler.py` | Consecutive failure tracking and pause logic |
| `test_register_genesis_models.py` | Genesis (v1 unretrained-starter) model registration: idempotency, honest skip on missing ONNX artifact, CLI exit code |
| `test_glue_etl.py` | Glue ETL feature engineering and labeling |
| `test_signal_associator.py` | Late-arriving signal association by request_id |
| `test_signal_receiver.py` | Signal ingestion handling |
| `test_bid_shading_handler.py` | AgentCore HTTP handler for bid shading agent |
| `test_latency_stats.py` | Latency percentile computation |
| `test_sse_streaming.py` | Server-sent events streaming |

### Integration Tests (Requirement 13.4)

| File | What it covers |
|------|---------------|
| `test_feedback_integration.py` | outcome to Kinesis to S3 to Glue path |

## Conventions

- All tests use declared, deterministic fixtures. No fabricated success/failure/telemetry values in surfaces presented as real (Requirement 13.5).
- Test files live alongside their subjects via `sys.path.insert` to `source/`.
- Async tests use `pytest.mark.asyncio`.
- Mocks are limited to external service boundaries (CloudWatch, DynamoDB, Triton HTTP).
