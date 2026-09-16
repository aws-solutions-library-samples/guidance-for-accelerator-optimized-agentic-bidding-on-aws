# ARTF Template Container

A deployed, wired, inactive ARTF container you can turn into your own.

Everything except the logic is already done: the gRPC/MCP/health server, the ECR
repository, the Kubernetes Deployment/Service/HPA, orchestrator registration, a
row in the Container Health panel, and an activation switch. You write
`mutate()`, rebuild one image, restart one Deployment, and switch it on.

## Why this exists

Adding a container to this stack by hand means editing nine places across
`source/orchestrator/app.py`, `deployment/eks/*.yaml`, `deployment/deploy.sh` and
`deployment/codebuild/*` — and then redeploying the orchestrator so it knows the
container exists. This container is already in all nine, so you don't.

## The four steps

### 1. Write your logic

Edit `app.py`. There is one marked block:

```python
# =======================================================================
# >>> IMPLEMENT YOUR LOGIC HERE <<<
# =======================================================================
```

It carries a worked example, the mapping from intent to payload field, and
pointers to the two containers worth copying. `shared/artf_types.py` has the
types.

### 2. Rebuild this one image

```bash
cd deployment
./deploy.sh --start-at 2
```

Or build it directly, from `source/`:

```bash
docker build --build-arg CONTAINER=containers/artf_template \
  -t "$REGISTRY/$STACK_NAME-artf-template:$TAG" .
docker push "$REGISTRY/$STACK_NAME-artf-template:$TAG"
```

There is no Dockerfile in this directory by design — `source/Dockerfile` is
shared by every ARTF container and selects one with `--build-arg CONTAINER=`.

### 3. Restart this one Deployment

**Do not skip this.** `deploy.sh` reuses the previous image tag when the registry
already has it, so `kubectl apply` is a no-op when only the image *content*
changed, and `imagePullPolicy: Always` only applies to newly created pods. Your
old code keeps serving until the pod is replaced.

```bash
kubectl rollout restart deployment/artf-template
kubectl rollout status  deployment/artf-template
```

### 4. Activate it

Open the Container Health panel in the UI and switch it on. That writes the
`active` flag to the registry table. No redeploy. Every orchestrator replica
picks it up within the registry cache TTL — 30 seconds by default, and the UI
tells you the window.

Verify it is being called: submit a request from the main view and look for your
container in the flow, or check `GET /api/v1/containers` for its `active` and
`status` values.

## Configuration

Two places, and they must agree.

| What | Where | Default |
|---|---|---|
| Intent the orchestrator **filters on** | `intents` on the registry record — set via the UI or `aws dynamodb update-item` | `["ADD_CIDS"]` |
| Intent this container **guards on** | `ARTF_TEMPLATE_INTENT` env var in `deployment/eks/artf-containers-deployment.yaml` | `ADD_CIDS` |
| Display name and description | `display_name` / `description` on the registry record | seeded by `deploy.sh` |
| Endpoint | `endpoint` on the record; `ARTF_TEMPLATE_URL` on the orchestrator as fallback | `http://artf-template:8081` |
| Registry cache TTL | `CONTAINER_REGISTRY_TTL` on the orchestrator | `30` |

The registry record is authoritative for routing. Changing only the env var
leaves the orchestrator filtering on the old intent, so your container is never
called.

`ADD_CIDS` is the default because it is the one intent in the ARTF enum that no
container here implements — the template fills a gap rather than shadowing
working code. Two containers *may* claim one intent; both get called and both
sets of mutations are merged. The UI flags a shared intent because merge order
then decides which value survives downstream.

## What the statuses mean

The Container Health panel and every bid response report one of these for your
container:

| Status | Meaning |
|---|---|
| `disabled` | Inactive. Not called. Not a failure |
| `unreachable` | Active, but nothing answered — pod down, no Service endpoints, wrong port |
| `error` | Answered, but the response was unusable (non-200, or unparseable) |
| `timeout` | Took longer than the request's `tmax` budget |
| `no_mutations` | Ran, and returned nothing. What the unmodified template reports |
| `ok` | Ran, and returned mutations |

`no_mutations` is a legitimate answer. You never need to fabricate a mutation to
look healthy, and doing so would breach this repo's no-fabricated-data rule.

An inactive, absent, broken or slow container cannot break the flow: the other
containers' mutations are still returned, the HTTP status is unchanged, and the
response shape is unchanged.

## Constraints worth knowing before you write code

- **Latency.** This is the live bid path, inside the request's `tmax` (100ms by
  default), shared with every other container since they run concurrently.
  `mutate()` is synchronous and offloaded to a thread pool by
  `shared/server.py`, so blocking I/O is safe but counts against the budget.
- **Node placement.** Runs on the `services` (CPU) node group, `minReplicas: 1`.
  It holds no GPU. If you call Triton, do it over the network the way
  `dlrm_bid_shader` does rather than asking for a GPU here.
- **Health probes are unconditional.** `shared/server.py` answers
  `/health/ready` with `ok` regardless of whether your dependencies are up. If
  your container needs a real readiness signal, that is yours to add.
- **Don't fabricate.** If you cannot compute a value, return fewer mutations.

## Adding a second container

The mechanism is generic: a second store-defined container is a second record in
the registry table, with no orchestrator code change. Its Kubernetes objects
(Deployment, Service, HPA) and ECR repository still come from a deploy, so copy
this directory and its nine registration sites — `RENAME_MAP.md` lists them, and
with the reason each exists.
