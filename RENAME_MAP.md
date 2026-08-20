# Rename Map

This Guidance renamed its four ARTF containers from model-architecture names to
job-oriented names that describe what each container does in the bidstream. This
document is the single reference for what changed, what didn't, and why.

## The four containers

| Old name | New name | What it does |
| --- | --- | --- |
| `dlrm-bid-shader` | **`bid-pricer`** | Shades every bid down to the lowest price still likely to win, using a CTR prediction model |
| `widedeep-segment-activator` | **`audience-activator`** | Activates audience segments on an impression from bid-request signals |
| `ncf-deal-manager` | **`deal-scorer`** | Scores private marketplace (PMP) deal fit and activates or suppresses deals per impression |
| `metrics-enricher` | **`signals-enricher`** | Adds viewability and brand-safety quality signals to the bid request |

The underlying model architectures (DLRM, NCF/NeuMF, Wide & Deep) are unchanged —
they're implementation detail, documented alongside each container, not the
container's name. See [GUIDANCE.md](GUIDANCE.md) for the model-level detail.

## Surfaces that changed

- **README.md, CLOSED_LOOP.md, GUIDANCE.md, GUIDANCE-part2.md** — prose and tables
  now lead with the job-oriented name; the model architecture is mentioned as
  implementation detail.
- **Kubernetes manifests** (`deployment/eks/artf-containers-deployment.yaml`) —
  `Deployment`/`Service`/`HorizontalPodAutoscaler` names use the new names (e.g.
  `bid-pricer` Service, reachable at `http://bid-pricer:8081` from the orchestrator).
- **`deployment/eks/orchestrator-deployment.yaml`** — the `DLRM_URL`/`WIDEDEEP_URL`/
  `NCF_URL`/`METRICS_URL` env var **values** point at the renamed Service DNS names
  (env var *names* are unchanged internal identifiers).
- **`source/docker-compose.yml`** — service names for local development.
- **ECR image names / `deploy.sh`** — the `REPOS` array and printed deployment
  summary use the new names via a `display_name()` lookup.
- **`deployment/codebuild/remote_build.sh`, `deployment/codebuild/buildspec.yml`** —
  ECR repo names use the new names; internal build keys (see below) are unchanged.
- **`deployment/check_builds.sh`** — image-status output uses the new names.
- **Frontend display labels** — `FlowPipeline.jsx` node labels, `RawPanel.jsx`
  `AGENT_LABELS`, `LoadTestPanel.jsx`'s container health panel (new
  `CONTAINER_LABELS` lookup), `ScenarioCard.jsx` descriptions,
  `ClosedLoopPanel.jsx`/`AdaptiveBiddingPanel.jsx`'s model-type dropdown labels.

## Surfaces intentionally left unchanged, and why

These are internal identifiers bound to already-registered AWS resources, a live
model-serving contract, or pure internal plumbing — renaming them would break
continuity for anyone who has already deployed, or require re-registering real
AWS resources for no functional benefit:

- **`source/containers/{dlrm_bid_shader,widedeep_segment_activator,ncf_deal_manager,metrics_enricher}/`**
  — container source directory names.
- **Triton model repository names and `config.pbtxt` model names**
  (`dlrm_bid_shader`, `ncf_deal_manager`) — renaming breaks the TensorRT engines'
  model-name binding and Triton's poll-mode repository layout.
- **SageMaker Model Package Group names** (`artf-dlrm-bid-shader`,
  `artf-ncf-deal-manager`) — real, already-registered SageMaker resources.
  Renaming would orphan any Model Registry history for an existing deployment.
- **Orchestrator internal identifiers** (`source/orchestrator/app.py`'s
  `CONTAINERS[].name` list and `container_to_model` map) — internal lookup keys,
  never displayed directly to a user.
- **Frontend internal node IDs / `MODEL_TYPES` keys** (`dlrm`, `ncf`, `widedeep`,
  `metrics`, and the underscore-form `dlrm_bid_shader` etc.) — internal lookup
  keys; only their **displayed labels** changed.
- **`deploy.sh`'s internal `key` values** (used for `_source_hash()` and
  `STEP4_KEYS`) — resolve the container **directory path**
  (`source/containers/${key//-/_}`) and Dockerfile selection; never printed to a
  user. The ECR **repository name** built from each `key` is the only thing
  translated to the new display name (via `display_name()`), consistently across
  `deploy.sh`, `remote_build.sh`, and `buildspec.yml`, so `--only`/`BUILD_ONLY`
  matching keeps working across all three files.
- **Existing test fixtures** referencing internal node IDs (e.g.
  `DemoSequenceOrchestrator.test.js`, `FocusAreaRegistry.test.js`) — these
  reference the unchanged internal keys and were not modified.

## Breaking change: `deploy.sh --start-at`

`--start-at` now takes a **phase number 1-5** instead of the old ad-hoc step
numbers (1, 1.5, 2, 3a-3c, 4, 5, 5.5, 6, 7.x, 8.x, 8.5, 9.x, 10, 11). If you have
scripts or notes referencing the old numbering, use this mapping:

| New phase | Label | Old step(s) absorbed |
| --- | --- | --- |
| 1 | Preparing models | 1, 1.5, 2, 3, 3a, 3b, 3c |
| 2 | Building containers & provisioning infrastructure | 4, 5, 5.5, 6, 7, 7.1, 7.5, 7.6, 7.7 |
| 3 | Deploying workloads | 8, 8a, 8b, 8.5 |
| 4 | Setting up access | 9, 9c, 9d |
| 5 | Registering agents | 10, 11 |

Example: `./deploy.sh --start-at=8` (old) becomes `./deploy.sh --start-at=3` (new).

## Upgrading an existing deployment

Because the Kubernetes Service names changed, an already-deployed stack cannot be
upgraded in place. Redeploy fresh:

```bash
./deploy.sh --destroy --prefix <your-prefix>
./deploy.sh --prefix <your-prefix>
```

ECR repositories are retained by design (see README's Cleanup section) — the new
deploy will create new repos under the new names alongside the old ones. Delete
the old, now-unused repos manually once you've confirmed the new deployment
works.
