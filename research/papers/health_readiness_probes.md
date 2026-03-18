---
topic: Health and Readiness Probes for Production Deployment
status: implemented
priority: high
estimated_complexity: small
researched_at: 2026-03-18T14:30:00Z
---

# Health and Readiness Probes for Production Deployment

## Problem Statement

Agent Zero runs on RunPod (container orchestration) but has **no health or readiness endpoints**. The existing `/api/status` endpoint (agent_zero_server.py:1139) returns application feature status (model capabilities, tool list, version) -- not operational health. There is no way for RunPod, a load balancer, or a monitoring system to determine:

1. **Is the process alive?** (liveness) -- If the Python event loop is blocked or deadlocked, no external system can detect it
2. **Can it serve traffic?** (readiness) -- If the database connection pool is exhausted or the model failed to load, requests will fail but traffic keeps routing
3. **Is startup complete?** (startup) -- During cold start, migrations run and model warms up; premature traffic causes errors

Without these probes, a degraded Agent Zero instance continues receiving traffic, producing errors that reach users.

## Current State in Agent Zero

### Existing status endpoint (not a health check):
- `GET /api/status` (agent_zero_server.py:1139-1170) -- Returns JSON with model status, voice config, framework features, and system info. Does NOT check DB connectivity, pool health, or model inference readiness. No HTTP status code variation (always 200).

### Startup sequence (agent_zero_server.py:126-142):
```python
@asynccontextmanager
async def _lifespan(application: FastAPI):
    try:
        await run_migrations()     # DB connection + schema
        _log.info("Database ready")
    except Exception as e:
        _log.warning("Running in offline mode")

    inference = get_inference()
    if not inference.loaded and _configured_model_available(...):
        asyncio.create_task(_warm_local_model())  # async model warm

    yield
    await close_pool()
```

The startup does NOT track whether DB or model initialization succeeded. The `_warm_local_model()` task runs in background with no completion tracking.

### Database pool (database.py:56-62):
```python
_pool = await asyncpg.create_pool(
    dsn, ssl=ssl_ctx,
    min_size=2, max_size=10, command_timeout=30,
)
```
Pool has `.get_size()`, `.get_idle_size()`, `.get_min_size()`, `.get_max_size()` methods for health reporting. There is already a `health_check()` function in database.py for circuit breaker probing (added in Session 291).

### Resilience layer (resilience.py):
Circuit breaker `db_circuit` tracks open/closed/half-open state. This state is valuable for readiness reporting -- if the DB circuit is open, the service cannot serve most requests.

### Config (config.py):
No health check configuration fields exist yet.

## Industry Standard / Research Findings

### Kubernetes Probe Architecture (Kubernetes Docs, 2026)
Kubernetes defines three probe types: **liveness** (restart on failure), **readiness** (remove from service on failure), and **startup** (delay other probes until ready). The critical principle: "Never add external dependency checks to the liveness probe. A database outage should remove the pod from the load balancer (readiness), not trigger a restart cycle (liveness) that makes recovery harder."
**URL:** https://oneuptime.com/blog/post/2026-02-20-kubernetes-liveness-readiness-startup-probes/view

### FastAPI Production Health Checks (Golabek, 2025)
The FastAPI Chassis framework implements a two-probe architecture: liveness is a synchronous endpoint with zero I/O (`return {"status": "healthy"}`), readiness uses a `ReadinessRegistry` that runs async dependency checks with automatic latency measurement. Each check gets a timeout shorter than the Kubernetes probe timeout to avoid cascading failures.
**URL:** https://patrykgolabek.dev/guides/fastapi-production/health-checks/

### Health Check Microservice Pattern (DEV Community, 2025)
Recommends structured health responses with per-dependency status and latency: `{"status": "healthy", "checks": {"database": {"status": "up", "latency_ms": 4.2}, "model": {"status": "up"}}}`. Returns 200 when all checks pass, 503 when any critical dependency fails.
**URL:** https://dev.to/lisan_al_gaib/building-a-health-check-microservice-with-fastapi-26jo

### Python Health Checks for Kubernetes (OneUptime, 2025)
"The liveness check is a pure function with no I/O, no state, and no dependencies that always returns 200 unless the process itself is broken. The readiness check, on the other hand, is async and queries the registry, returning 503 along with a detailed breakdown if any dependency is unhealthy."
**URL:** https://oneuptime.com/blog/post/2025-01-06-python-health-checks-kubernetes/view

### Langflow Health Check Issue (GitHub, 2025)
Langflow's `/health` liveness probe was incorrectly used as a readiness probe, hiding DB failures from orchestrators. This was filed as a production incident (issue #8921), demonstrating why separation matters.
**URL:** https://github.com/langflow-ai/langflow/issues/8921

### asyncpg Pool Health Monitoring
asyncpg connection pools expose `get_size()`, `get_idle_size()`, `get_min_size()`, `get_max_size()` for monitoring. A deep health check acquires a connection and runs `SELECT 1` to verify end-to-end connectivity.
**URL:** https://pypi.org/project/fastapi-healthchecks/ (PostgreSQL check reference)

## Proposed Implementation

### Changes to `agent_zero/agent_zero_server.py`

Add three new endpoints and a startup readiness flag.

**Step 1:** Add startup tracking in `_lifespan` (after line 127):

```python
# At module level (after app definition, ~line 146):
_startup_complete = False
_db_available = False

@asynccontextmanager
async def _lifespan(application: FastAPI):
    global _startup_complete, _db_available
    _log.info("Connecting to NeonDB")
    try:
        await run_migrations()
        _db_available = True
        _log.info("Database ready")
    except Exception as e:
        _db_available = False
        _log.error("Database connection failed", exc_info=True)
        _log.warning("Running in offline mode (no auth, no persistence)")

    inference = get_inference()
    if not inference.loaded and _configured_model_available(AGENT_ZERO_MODEL_PATH, AGENT_ZERO_MODEL_SOURCE):
        asyncio.create_task(_warm_local_model())

    _startup_complete = True
    _log.info("Startup complete")

    yield

    _startup_complete = False
    await close_pool()
```

**Step 2:** Add health check endpoints (after `/api/status` endpoint, ~line 1170):

```python
@app.get("/health/live")
async def liveness():
    """Liveness probe: Is the process alive?

    Pure function, no I/O, no external dependencies.
    Returns 200 if the event loop can serve a request.
    Kubernetes: restarts pod on failure.
    """
    return JSONResponse({"status": "alive"})


@app.get("/health/ready")
async def readiness():
    """Readiness probe: Can this instance serve traffic?

    Checks: database connectivity, model availability, startup completion.
    Returns 200 if all critical dependencies are healthy, 503 otherwise.
    Kubernetes: removes pod from service on failure (no restart).
    """
    checks = {}
    all_healthy = True

    # Check 1: Startup completion
    checks["startup"] = {"status": "up" if _startup_complete else "down"}
    if not _startup_complete:
        all_healthy = False

    # Check 2: Database connectivity
    try:
        pool = await get_pool()
        pool_size = pool.get_size()
        pool_idle = pool.get_idle_size()
        pool_max = pool.get_max_size()

        # Deep check: actually query the database
        start = time.monotonic()
        async with asyncio.timeout(_cfg.health_check_timeout):
            await pool.fetchval("SELECT 1")
        latency_ms = (time.monotonic() - start) * 1000

        checks["database"] = {
            "status": "up",
            "latency_ms": round(latency_ms, 1),
            "pool_size": pool_size,
            "pool_idle": pool_idle,
            "pool_max": pool_max,
        }
    except Exception as e:
        checks["database"] = {"status": "down", "error": str(e)}
        all_healthy = False

    # Check 3: Circuit breaker state
    cb_state = db_circuit.state  # "closed", "open", or "half_open"
    checks["circuit_breaker"] = {"status": "up" if cb_state == "closed" else "degraded", "state": cb_state}
    if cb_state == "open":
        all_healthy = False

    # Check 4: Model availability
    try:
        inference = await get_inference_async()
        model_status = inference.get_status()
        model_ready = model_status.get("loaded", False) or model_status.get("mode") == "vllm"
        checks["model"] = {"status": "up" if model_ready else "down", "detail": model_status}
        if not model_ready:
            all_healthy = False
    except Exception as e:
        checks["model"] = {"status": "down", "error": str(e)}
        all_healthy = False

    status_code = 200 if all_healthy else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_healthy else "not_ready",
            "checks": checks,
        },
    )


@app.get("/health/startup")
async def startup_probe():
    """Startup probe: Has initialization completed?

    Returns 200 once DB migrations and initial setup are done.
    Kubernetes: delays liveness/readiness probes until this passes.
    """
    if _startup_complete:
        return JSONResponse({"status": "started"})
    return JSONResponse(status_code=503, content={"status": "starting"})
```

**Step 3:** Add config fields to `agent_zero/config.py`:

```python
# --- Health Checks ---
health_check_timeout: float = Field(
    default=2.0, ge=0.5, le=10.0,
    description="Timeout in seconds for database health check query"
)
```

**Step 4:** Import `time` if not already imported (it is -- line 14 of agent_zero_server.py).

### RunPod / Docker Configuration

Document in `DEPLOYMENT_RUNBOOK.md`:

```yaml
# RunPod health check configuration
HEALTHCHECK_ENDPOINT: /health/live
READINESS_ENDPOINT: /health/ready
STARTUP_ENDPOINT: /health/startup

# Kubernetes probe configuration (if using K8s directly):
livenessProbe:
  httpGet:
    path: /health/live
    port: 8888
  initialDelaySeconds: 5
  periodSeconds: 10
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8888
  initialDelaySeconds: 10
  periodSeconds: 15
  failureThreshold: 2

startupProbe:
  httpGet:
    path: /health/startup
    port: 8888
  initialDelaySeconds: 0
  periodSeconds: 5
  failureThreshold: 30  # 150s max startup time
```

## Test Specifications

### File: `agent_zero/test_health_probes.py`

```
test_liveness_always_200 -- GET /health/live returns 200 with {"status": "alive"}.
test_readiness_200_when_healthy -- With DB pool active and model loaded, GET /health/ready returns 200 with all checks "up".
test_readiness_503_when_db_down -- Mock get_pool to raise exception. GET /health/ready returns 503 with database check "down".
test_readiness_503_when_circuit_open -- Set db_circuit to open state. GET /health/ready returns 503 with circuit_breaker "degraded".
test_readiness_503_before_startup -- Set _startup_complete = False. GET /health/ready returns 503.
test_readiness_includes_pool_metrics -- Verify response includes pool_size, pool_idle, pool_max fields.
test_readiness_includes_latency -- Verify database check includes latency_ms field (float).
test_readiness_timeout -- Mock pool.fetchval to sleep > health_check_timeout. Verify returns "down" with timeout error.
test_startup_503_during_init -- Set _startup_complete = False. GET /health/startup returns 503 with {"status": "starting"}.
test_startup_200_after_init -- Set _startup_complete = True. GET /health/startup returns 200 with {"status": "started"}.
test_config_health_check_timeout -- Verify Agent ZeroConfig.health_check_timeout defaults to 2.0 with range [0.5, 10.0].
test_health_endpoints_no_auth -- Verify all three health endpoints work without JWT token (no Depends(get_current_user)).
```

## Estimated Impact

- **Deployment reliability:** RunPod can detect and route around degraded instances
- **Recovery speed:** Failed DB connections trigger readiness failure -> traffic stops -> circuit breaker recovers -> readiness passes -> traffic resumes (automatic healing)
- **Observability:** `/health/ready` response provides real-time pool metrics and latency for monitoring dashboards
- **Startup safety:** Startup probe prevents premature traffic during cold start (migration + model warm)
- **Zero risk:** All endpoints are read-only, unauthenticated (standard for health probes), and add no latency to normal request path
- **Minimal code:** ~60 lines of new endpoint code + 1 config field + 2 module-level flags
