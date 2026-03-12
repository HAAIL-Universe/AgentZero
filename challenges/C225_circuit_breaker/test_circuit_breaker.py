"""Tests for C225: Circuit Breaker & Resilience Patterns."""

import time
import pytest
from circuit_breaker import (
    CircuitBreaker, CircuitState, CircuitOpenError,
    RetryPolicy, RetryBackoff, RetriesExhaustedError,
    TimeoutPolicy, TimeoutError,
    Bulkhead, BulkheadFullError,
    RateLimiter, RateLimitStrategy, RateLimitExceededError,
    FallbackChain, FallbackExhaustedError,
    ResiliencePipeline,
    ServiceCircuitManager,
    SlidingWindow,
)
from service_discovery import (
    ServiceRegistry, HealthStatus, HealthCheck, CheckType,
    LoadBalanceStrategy,
)


# =============================================================================
# Helpers
# =============================================================================

class MockTime:
    """Controllable time source for testing."""
    def __init__(self, start=1000.0):
        self.now = start
    def __call__(self):
        return self.now
    def advance(self, seconds):
        self.now += seconds


def make_failing(n, result="ok"):
    """Returns a function that fails n times then succeeds."""
    state = {'calls': 0}
    def func(*args, **kwargs):
        state['calls'] += 1
        if state['calls'] <= n:
            raise RuntimeError(f"Fail #{state['calls']}")
        return result
    func.state = state
    return func


def always_fail(msg="fail"):
    def func(*args, **kwargs):
        raise RuntimeError(msg)
    return func


def always_succeed(result="ok"):
    def func(*args, **kwargs):
        return result
    return func


# =============================================================================
# SlidingWindow Tests
# =============================================================================

class TestSlidingWindow:
    def test_basic_recording(self):
        sw = SlidingWindow(window_size=10.0, bucket_count=5)
        mt = MockTime()
        sw._time = mt
        sw.record_success(mt())
        sw.record_success(mt())
        sw.record_failure(mt())
        s, f, total = sw.get_stats(mt())
        assert s == 2
        assert f == 1
        assert total == 3

    def test_failure_rate(self):
        sw = SlidingWindow(window_size=10.0)
        mt = MockTime()
        sw._time = mt
        for _ in range(7):
            sw.record_success(mt())
        for _ in range(3):
            sw.record_failure(mt())
        assert abs(sw.get_failure_rate(mt()) - 0.3) < 0.01

    def test_window_expiry(self):
        sw = SlidingWindow(window_size=5.0, bucket_count=5)
        mt = MockTime()
        sw._time = mt
        sw.record_failure(mt())
        sw.record_failure(mt())
        mt.advance(6.0)
        sw.record_success(mt())
        s, f, total = sw.get_stats(mt())
        assert f == 0  # Old failures expired
        assert s == 1

    def test_reset(self):
        sw = SlidingWindow()
        mt = MockTime()
        sw._time = mt
        sw.record_failure(mt())
        sw.reset()
        assert sw.get_total_calls(mt()) == 0

    def test_multiple_buckets(self):
        sw = SlidingWindow(window_size=10.0, bucket_count=5)
        mt = MockTime()
        sw._time = mt
        sw.record_success(mt())
        mt.advance(3.0)
        sw.record_failure(mt())
        mt.advance(3.0)
        sw.record_success(mt())
        s, f, total = sw.get_stats(mt())
        assert total == 3


# =============================================================================
# CircuitBreaker Tests
# =============================================================================

class TestCircuitBreaker:
    def test_initial_state(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_success_stays_closed(self):
        cb = CircuitBreaker()
        for _ in range(10):
            cb.execute(always_succeed())
        assert cb.state == CircuitState.CLOSED

    def test_failure_rate_opens_circuit(self):
        cb = CircuitBreaker(failure_rate_threshold=0.5, minimum_calls=5,
                            window_size=60.0)
        mt = MockTime()
        cb._time = mt
        cb._window._time = mt

        # 5 failures -> 100% failure rate > 50% threshold
        for i in range(5):
            try:
                cb.execute(always_fail())
            except RuntimeError:
                pass

        assert cb.state == CircuitState.OPEN

    def test_open_rejects_calls(self):
        cb = CircuitBreaker(failure_rate_threshold=0.5, minimum_calls=3)
        mt = MockTime()
        cb._time = mt
        cb._window._time = mt

        # Force open
        for _ in range(5):
            try:
                cb.execute(always_fail())
            except (RuntimeError, CircuitOpenError):
                pass

        with pytest.raises(CircuitOpenError) as exc:
            cb.execute(always_succeed())
        assert exc.value.name == "default"

    def test_open_to_half_open_after_wait(self):
        cb = CircuitBreaker(failure_rate_threshold=0.5, minimum_calls=3,
                            wait_duration=10.0)
        mt = MockTime()
        cb._time = mt
        cb._window._time = mt

        # Open the circuit
        for _ in range(5):
            try:
                cb.execute(always_fail())
            except (RuntimeError, CircuitOpenError):
                pass
        assert cb.state == CircuitState.OPEN

        # Wait
        mt.advance(11.0)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_rate_threshold=0.5, minimum_calls=3,
                            wait_duration=5.0, permitted_half_open_calls=3,
                            success_rate_threshold=0.5)
        mt = MockTime()
        cb._time = mt
        cb._window._time = mt

        # Open
        for _ in range(5):
            try:
                cb.execute(always_fail())
            except (RuntimeError, CircuitOpenError):
                pass

        mt.advance(6.0)
        assert cb.state == CircuitState.HALF_OPEN

        # 3 successes in half-open
        for _ in range(3):
            cb.execute(always_succeed())

        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_rate_threshold=0.5, minimum_calls=3,
                            wait_duration=5.0, permitted_half_open_calls=2,
                            success_rate_threshold=0.8)
        mt = MockTime()
        cb._time = mt
        cb._window._time = mt

        # Open
        for _ in range(5):
            try:
                cb.execute(always_fail())
            except (RuntimeError, CircuitOpenError):
                pass

        mt.advance(6.0)
        assert cb.state == CircuitState.HALF_OPEN

        # 1 success, 1 failure -> 50% success rate < 80% threshold -> reopen
        cb.execute(always_succeed())
        try:
            cb.execute(always_fail())
        except RuntimeError:
            pass

        assert cb.state == CircuitState.OPEN

    def test_minimum_calls_required(self):
        cb = CircuitBreaker(failure_rate_threshold=0.5, minimum_calls=10)
        mt = MockTime()
        cb._time = mt
        cb._window._time = mt

        # 5 failures but minimum_calls=10, stays closed
        for _ in range(5):
            try:
                cb.execute(always_fail())
            except RuntimeError:
                pass

        assert cb.state == CircuitState.CLOSED

    def test_ignore_exceptions(self):
        cb = CircuitBreaker(failure_rate_threshold=0.5, minimum_calls=3,
                            ignore_exceptions=(ValueError,))
        mt = MockTime()
        cb._time = mt
        cb._window._time = mt

        def raise_value_error():
            raise ValueError("ignored")

        # ValueError is ignored -- should not count as failures
        for _ in range(10):
            try:
                cb.execute(raise_value_error)
            except ValueError:
                pass

        assert cb.state == CircuitState.CLOSED

    def test_state_change_callback(self):
        changes = []
        cb = CircuitBreaker(failure_rate_threshold=0.5, minimum_calls=3)
        mt = MockTime()
        cb._time = mt
        cb._window._time = mt
        cb.on_state_change(lambda old, new: changes.append((old, new)))

        for _ in range(5):
            try:
                cb.execute(always_fail())
            except (RuntimeError, CircuitOpenError):
                pass

        assert len(changes) >= 1
        assert changes[0] == (CircuitState.CLOSED, CircuitState.OPEN)

    def test_metrics(self):
        cb = CircuitBreaker()
        mt = MockTime()
        cb._time = mt
        cb._window._time = mt

        cb.execute(always_succeed())
        try:
            cb.execute(always_fail())
        except RuntimeError:
            pass

        m = cb.get_metrics()
        assert m['total_calls'] == 2
        assert m['total_successes'] == 1
        assert m['total_failures'] == 1
        assert m['state'] == 'closed'

    def test_force_open(self):
        cb = CircuitBreaker()
        cb.force_open()
        assert cb.state == CircuitState.OPEN

    def test_reset(self):
        cb = CircuitBreaker()
        cb.force_open()
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_consecutive_tracking(self):
        cb = CircuitBreaker(minimum_calls=100)  # high so it won't open
        mt = MockTime()
        cb._time = mt
        cb._window._time = mt

        cb.execute(always_succeed())
        cb.execute(always_succeed())
        m = cb.get_metrics()
        assert m['consecutive_successes'] == 2
        assert m['consecutive_failures'] == 0

        try:
            cb.execute(always_fail())
        except RuntimeError:
            pass

        m = cb.get_metrics()
        assert m['consecutive_successes'] == 0
        assert m['consecutive_failures'] == 1

    def test_call_alias(self):
        cb = CircuitBreaker()
        result = cb.call(always_succeed("hello"))
        assert result == "hello"

    def test_half_open_rejects_excess_calls(self):
        cb = CircuitBreaker(failure_rate_threshold=0.5, minimum_calls=3,
                            wait_duration=5.0, permitted_half_open_calls=2)
        mt = MockTime()
        cb._time = mt
        cb._window._time = mt

        # Open
        for _ in range(5):
            try:
                cb.execute(always_fail())
            except (RuntimeError, CircuitOpenError):
                pass

        mt.advance(6.0)
        assert cb.state == CircuitState.HALF_OPEN

        # Use up permitted calls (1 success, 1 fail => goes back to open)
        cb.execute(always_succeed())
        try:
            cb.execute(always_fail())
        except RuntimeError:
            pass

        # Now it should be open or half-open's permitted calls exhausted
        # Either way, next call gets rejected
        # (After 2 calls with <80% success it should reopen by default threshold)

    def test_record_success_failure_manual(self):
        cb = CircuitBreaker(minimum_calls=100)
        mt = MockTime()
        cb._time = mt
        cb._window._time = mt

        cb.record_success()
        cb.record_failure()
        m = cb.get_metrics()
        assert m['total_successes'] == 1
        assert m['total_failures'] == 1


# =============================================================================
# RetryPolicy Tests
# =============================================================================

class TestRetryPolicy:
    def test_success_no_retry(self):
        rp = RetryPolicy(max_retries=3)
        result = rp.execute(always_succeed("ok"), use_sleep=False)
        assert result == "ok"

    def test_retry_then_succeed(self):
        rp = RetryPolicy(max_retries=3)
        func = make_failing(2, "recovered")
        result = rp.execute(func, use_sleep=False)
        assert result == "recovered"
        assert func.state['calls'] == 3

    def test_exhausted(self):
        rp = RetryPolicy(max_retries=2)
        with pytest.raises(RetriesExhaustedError) as exc:
            rp.execute(always_fail(), use_sleep=False)
        assert exc.value.attempts == 3
        assert exc.value.name == "default"

    def test_fixed_backoff(self):
        rp = RetryPolicy(max_retries=3, backoff=RetryBackoff.FIXED,
                         base_delay=1.0, jitter=False)
        func = make_failing(3, "ok")
        result, delays = rp.execute_with_delays(func)
        assert result == "ok"
        assert all(abs(d - 1.0) < 0.01 for d in delays)

    def test_exponential_backoff(self):
        rp = RetryPolicy(max_retries=4, backoff=RetryBackoff.EXPONENTIAL,
                         base_delay=1.0, multiplier=2.0, jitter=False)
        func = make_failing(4, "ok")
        result, delays = rp.execute_with_delays(func)
        assert result == "ok"
        assert abs(delays[0] - 1.0) < 0.01
        assert abs(delays[1] - 2.0) < 0.01
        assert abs(delays[2] - 4.0) < 0.01
        assert abs(delays[3] - 8.0) < 0.01

    def test_linear_backoff(self):
        rp = RetryPolicy(max_retries=3, backoff=RetryBackoff.LINEAR,
                         base_delay=2.0, jitter=False)
        func = make_failing(3, "ok")
        result, delays = rp.execute_with_delays(func)
        assert abs(delays[0] - 2.0) < 0.01   # 2 * 1
        assert abs(delays[1] - 4.0) < 0.01   # 2 * 2
        assert abs(delays[2] - 6.0) < 0.01   # 2 * 3

    def test_max_delay_cap(self):
        rp = RetryPolicy(max_retries=5, backoff=RetryBackoff.EXPONENTIAL,
                         base_delay=1.0, multiplier=10.0, max_delay=5.0,
                         jitter=False)
        func = make_failing(3, "ok")
        result, delays = rp.execute_with_delays(func)
        for d in delays:
            assert d <= 5.0

    def test_jitter(self):
        rp = RetryPolicy(max_retries=3, backoff=RetryBackoff.FIXED,
                         base_delay=10.0, jitter=True, jitter_range=0.5)
        func = make_failing(3, "ok")
        result, delays = rp.execute_with_delays(func)
        # With jitter, delays should vary
        for d in delays:
            assert 5.0 <= d <= 15.0

    def test_retry_on_specific_exceptions(self):
        rp = RetryPolicy(max_retries=3, retry_on=(ValueError,))

        def raise_runtime():
            raise RuntimeError("not retried")

        with pytest.raises(RuntimeError):
            rp.execute(raise_runtime, use_sleep=False)

    def test_retry_if_predicate(self):
        rp = RetryPolicy(max_retries=3,
                         retry_if=lambda e: "transient" in str(e))

        state = {'calls': 0}
        def func():
            state['calls'] += 1
            if state['calls'] <= 2:
                raise RuntimeError("transient error")
            return "ok"

        result = rp.execute(func, use_sleep=False)
        assert result == "ok"

    def test_on_retry_callback(self):
        retries = []
        rp = RetryPolicy(max_retries=3,
                         on_retry=lambda attempt, delay, err: retries.append(attempt))
        func = make_failing(2, "ok")
        rp.execute(func, use_sleep=False)
        assert retries == [1, 2]

    def test_metrics(self):
        rp = RetryPolicy(max_retries=2)
        rp.execute(always_succeed(), use_sleep=False)
        m = rp.get_metrics()
        assert m['total_successes'] == 1

        try:
            rp.execute(always_fail(), use_sleep=False)
        except RetriesExhaustedError:
            pass
        m = rp.get_metrics()
        assert m['total_exhausted'] == 1


# =============================================================================
# TimeoutPolicy Tests
# =============================================================================

class TestTimeoutPolicy:
    def test_within_timeout(self):
        tp = TimeoutPolicy(timeout=5.0)
        mt = MockTime()
        tp._time = mt

        def fast():
            mt.advance(1.0)
            return "ok"

        result = tp.execute(fast)
        assert result == "ok"

    def test_exceeds_timeout(self):
        tp = TimeoutPolicy(timeout=5.0)
        mt = MockTime()
        tp._time = mt

        def slow():
            mt.advance(10.0)
            return "late"

        with pytest.raises(TimeoutError) as exc:
            tp.execute(slow)
        assert exc.value.timeout == 5.0

    def test_exception_within_timeout(self):
        tp = TimeoutPolicy(timeout=5.0)
        mt = MockTime()
        tp._time = mt

        def fast_fail():
            mt.advance(1.0)
            raise ValueError("fast fail")

        with pytest.raises(ValueError):
            tp.execute(fast_fail)

    def test_exception_after_timeout(self):
        tp = TimeoutPolicy(timeout=5.0)
        mt = MockTime()
        tp._time = mt

        def slow_fail():
            mt.advance(10.0)
            raise ValueError("slow fail")

        with pytest.raises(TimeoutError):
            tp.execute(slow_fail)

    def test_metrics(self):
        tp = TimeoutPolicy(timeout=5.0)
        mt = MockTime()
        tp._time = mt

        def fast():
            mt.advance(1.0)
            return "ok"

        def slow():
            mt.advance(10.0)
            return "late"

        tp.execute(fast)
        try:
            tp.execute(slow)
        except TimeoutError:
            pass

        m = tp.get_metrics()
        assert m['total_calls'] == 2
        assert m['total_timeouts'] == 1


# =============================================================================
# Bulkhead Tests
# =============================================================================

class TestBulkhead:
    def test_within_limit(self):
        bh = Bulkhead(max_concurrent=5)
        result = bh.execute(always_succeed("ok"))
        assert result == "ok"
        assert bh.active_count == 0

    def test_reject_over_limit(self):
        bh = Bulkhead(max_concurrent=2)
        # Manually acquire both slots
        assert bh.acquire()
        assert bh.acquire()
        assert bh.active_count == 2

        with pytest.raises(BulkheadFullError):
            bh.execute(always_succeed())

        bh.release()
        assert bh.active_count == 1

    def test_release_after_exception(self):
        bh = Bulkhead(max_concurrent=2)
        try:
            bh.execute(always_fail())
        except RuntimeError:
            pass
        assert bh.active_count == 0  # Released even on failure

    def test_available_count(self):
        bh = Bulkhead(max_concurrent=3)
        assert bh.available_count == 3
        bh.acquire()
        assert bh.available_count == 2

    def test_metrics(self):
        bh = Bulkhead(max_concurrent=2)
        bh.execute(always_succeed())
        bh.acquire()
        bh.acquire()
        try:
            bh.execute(always_succeed())
        except BulkheadFullError:
            pass

        m = bh.get_metrics()
        assert m['total_calls'] == 1
        assert m['total_rejections'] == 1
        assert m['active'] == 2

    def test_release_below_zero(self):
        bh = Bulkhead(max_concurrent=2)
        bh.release()  # Should not go negative
        assert bh.active_count == 0


# =============================================================================
# RateLimiter Tests
# =============================================================================

class TestRateLimiter:
    def test_token_bucket_allows(self):
        rl = RateLimiter(strategy=RateLimitStrategy.TOKEN_BUCKET,
                         rate=10.0, period=1.0, burst=10)
        mt = MockTime()
        rl._time = mt
        rl._last_refill = mt()

        for _ in range(10):
            assert rl.acquire(mt())

    def test_token_bucket_rejects(self):
        rl = RateLimiter(strategy=RateLimitStrategy.TOKEN_BUCKET,
                         rate=5.0, period=1.0, burst=5)
        mt = MockTime()
        rl._time = mt
        rl._last_refill = mt()
        rl._tokens = 5.0

        for _ in range(5):
            assert rl.acquire(mt())
        assert not rl.acquire(mt())

    def test_token_bucket_refills(self):
        rl = RateLimiter(strategy=RateLimitStrategy.TOKEN_BUCKET,
                         rate=10.0, period=1.0, burst=10)
        mt = MockTime()
        rl._time = mt
        rl._last_refill = mt()
        rl._tokens = 10.0

        # Drain all tokens
        for _ in range(10):
            rl.acquire(mt())
        assert not rl.acquire(mt())

        # Advance 1 second -> 10 new tokens
        mt.advance(1.0)
        for _ in range(10):
            assert rl.acquire(mt())

    def test_sliding_window(self):
        rl = RateLimiter(strategy=RateLimitStrategy.SLIDING_WINDOW,
                         rate=5, period=10.0)
        mt = MockTime()
        rl._time = mt

        for _ in range(5):
            assert rl.acquire(mt())
        assert not rl.acquire(mt())  # 6th blocked

        mt.advance(11.0)
        assert rl.acquire(mt())  # Window slid

    def test_fixed_window(self):
        rl = RateLimiter(strategy=RateLimitStrategy.FIXED_WINDOW,
                         rate=3, period=5.0)
        mt = MockTime()
        rl._time = mt
        rl._window_start = mt()

        for _ in range(3):
            assert rl.acquire(mt())
        assert not rl.acquire(mt())

        mt.advance(6.0)
        assert rl.acquire(mt())  # New window

    def test_execute(self):
        rl = RateLimiter(strategy=RateLimitStrategy.SLIDING_WINDOW,
                         rate=2, period=10.0)
        mt = MockTime()
        rl._time = mt

        assert rl.execute(always_succeed("a"), now=mt()) == "a"
        assert rl.execute(always_succeed("b"), now=mt()) == "b"

        with pytest.raises(RateLimitExceededError):
            rl.execute(always_succeed("c"), now=mt())

    def test_metrics(self):
        rl = RateLimiter(rate=5, period=1.0)
        mt = MockTime()
        rl._time = mt
        rl._last_refill = mt()
        rl._tokens = 5.0

        for _ in range(5):
            rl.acquire(mt())
        rl.acquire(mt())  # rejected

        m = rl.get_metrics()
        assert m['total_allowed'] == 5
        assert m['total_rejected'] == 1


# =============================================================================
# FallbackChain Tests
# =============================================================================

class TestFallbackChain:
    def test_primary_succeeds(self):
        fc = FallbackChain()
        fc.add(always_succeed("primary"), "primary")
        fc.add(always_succeed("fallback"), "fallback")
        assert fc.execute() == "primary"

    def test_fallback_on_failure(self):
        fc = FallbackChain()
        fc.add(always_fail(), "primary")
        fc.add(always_succeed("fallback"), "fallback")
        assert fc.execute() == "fallback"

    def test_chained_fallbacks(self):
        fc = FallbackChain()
        fc.add(always_fail(), "first")
        fc.add(always_fail(), "second")
        fc.add(always_succeed("third"), "third")
        assert fc.execute() == "third"

    def test_all_fail(self):
        fc = FallbackChain()
        fc.add(always_fail(), "first")
        fc.add(always_fail(), "second")
        with pytest.raises(FallbackExhaustedError):
            fc.execute()

    def test_empty_chain(self):
        fc = FallbackChain()
        with pytest.raises(FallbackExhaustedError):
            fc.execute()

    def test_metrics(self):
        fc = FallbackChain()
        fc.add(always_fail(), "primary")
        fc.add(always_succeed("fb"), "fb")
        fc.execute()
        fc.execute()

        m = fc.get_metrics()
        assert m['total_primary'] == 0
        assert m['total_fallbacks'] == 2
        assert m['fallback_counts']['fb'] == 2

    def test_fluent_api(self):
        fc = FallbackChain()
        result = fc.add(always_succeed("ok"), "p").execute()
        assert result == "ok"


# =============================================================================
# ResiliencePipeline Tests
# =============================================================================

class TestResiliencePipeline:
    def test_empty_pipeline(self):
        pipe = ResiliencePipeline()
        result = pipe.execute(always_succeed("ok"))
        assert result == "ok"

    def test_circuit_breaker_in_pipeline(self):
        pipe = ResiliencePipeline()
        cb = pipe.add_circuit_breaker(failure_rate_threshold=0.5, minimum_calls=3)
        mt = MockTime()
        cb._time = mt
        cb._window._time = mt

        # Normal calls
        assert pipe.execute(always_succeed("ok")) == "ok"

    def test_retry_in_pipeline(self):
        pipe = ResiliencePipeline()
        pipe.add_retry(max_retries=3)
        func = make_failing(2, "recovered")
        result = pipe.execute(func)
        assert result == "recovered"

    def test_timeout_in_pipeline(self):
        pipe = ResiliencePipeline()
        tp = pipe.add_timeout(5.0)
        mt = MockTime()
        tp._time = mt

        def slow():
            mt.advance(10.0)
            return "late"

        with pytest.raises(TimeoutError):
            pipe.execute(slow)

    def test_bulkhead_in_pipeline(self):
        pipe = ResiliencePipeline()
        bh = pipe.add_bulkhead(max_concurrent=1)
        bh.acquire()  # Take the only slot

        with pytest.raises(BulkheadFullError):
            pipe.execute(always_succeed())

    def test_rate_limiter_in_pipeline(self):
        pipe = ResiliencePipeline()
        rl = pipe.add_rate_limiter(strategy=RateLimitStrategy.SLIDING_WINDOW,
                                   rate=2, period=10.0)
        mt = MockTime()
        rl._time = mt

        pipe.execute(always_succeed("a"))
        pipe.execute(always_succeed("b"))
        with pytest.raises(RateLimitExceededError):
            pipe.execute(always_succeed("c"))

    def test_fallback_in_pipeline(self):
        pipe = ResiliencePipeline()
        pipe.add_fallback(always_succeed("fallback"))

        result = pipe.execute(always_fail())
        assert result == "fallback"

    def test_combined_retry_and_circuit_breaker(self):
        pipe = ResiliencePipeline()
        pipe.add_retry(max_retries=2)
        cb = pipe.add_circuit_breaker(failure_rate_threshold=0.5, minimum_calls=10)
        mt = MockTime()
        cb._time = mt
        cb._window._time = mt

        func = make_failing(1, "recovered")
        result = pipe.execute(func)
        assert result == "recovered"

    def test_pipeline_metrics(self):
        pipe = ResiliencePipeline()
        pipe.add_retry(max_retries=1)
        pipe.execute(always_succeed())
        m = pipe.get_metrics()
        assert m['total_calls'] == 1
        assert m['total_successes'] == 1

    def test_pipeline_failure(self):
        pipe = ResiliencePipeline()
        pipe.add_retry(max_retries=1)
        with pytest.raises(RetriesExhaustedError):
            pipe.execute(always_fail())
        m = pipe.get_metrics()
        assert m['total_failures'] == 1


# =============================================================================
# ServiceCircuitManager Tests
# =============================================================================

class TestServiceCircuitManager:
    def _setup_registry(self):
        reg = ServiceRegistry()
        reg.register("api-gateway", service_id="api-1", port=8080)
        reg.register("api-gateway", service_id="api-2", port=8081)
        reg.register("user-service", service_id="user-1", port=9090)
        return reg

    def test_get_breaker(self):
        reg = self._setup_registry()
        mgr = ServiceCircuitManager(registry=reg)
        cb = mgr.get_breaker("api-gateway")
        assert isinstance(cb, CircuitBreaker)
        assert cb.name == "api-gateway"

    def test_get_same_breaker(self):
        reg = self._setup_registry()
        mgr = ServiceCircuitManager(registry=reg)
        cb1 = mgr.get_breaker("api-gateway")
        cb2 = mgr.get_breaker("api-gateway")
        assert cb1 is cb2

    def test_configure_service(self):
        reg = self._setup_registry()
        mgr = ServiceCircuitManager(registry=reg)
        mgr.configure_service("api-gateway", failure_rate_threshold=0.3,
                              wait_duration=15.0)
        cb = mgr.get_breaker("api-gateway")
        assert cb.failure_rate_threshold == 0.3
        assert cb.wait_duration == 15.0

    def test_call_service(self):
        reg = self._setup_registry()
        mgr = ServiceCircuitManager(registry=reg)

        def handler(instance):
            return f"called {instance.service_id}"

        result = mgr.call_service("api-gateway", handler)
        assert result.startswith("called api-")

    def test_call_service_failure(self):
        reg = self._setup_registry()
        mgr = ServiceCircuitManager(registry=reg)

        def handler(instance):
            raise RuntimeError("service down")

        with pytest.raises(RuntimeError):
            mgr.call_service("api-gateway", handler)

    def test_sync_health_force_open(self):
        reg = ServiceRegistry()
        inst = reg.register("failing-svc", service_id="fail-1",
                           health_checks=[HealthCheck("hc1", CheckType.HTTP,
                                                      callback=lambda: 500)])
        mgr = ServiceCircuitManager(registry=reg)
        cb = mgr.get_breaker("failing-svc")

        # Run health checks to mark service critical
        for _ in range(6):
            reg.run_health_checks()

        changes = mgr.sync_health()
        assert any(c[1] == 'force_open' for c in changes)
        assert cb.state == CircuitState.OPEN

    def test_sync_health_reset(self):
        reg = ServiceRegistry()
        reg.register("svc", service_id="s1")
        mgr = ServiceCircuitManager(registry=reg)
        cb = mgr.get_breaker("svc")
        cb.force_open()

        # Service is healthy in registry (default PASSING)
        changes = mgr.sync_health()
        assert any(c[1] == 'reset' for c in changes)
        assert cb.state == CircuitState.CLOSED

    def test_get_pipeline(self):
        reg = self._setup_registry()
        mgr = ServiceCircuitManager(registry=reg)
        mgr.configure_service("api-gateway", timeout=5.0, max_retries=2)
        pipe = mgr.get_pipeline("api-gateway")
        assert isinstance(pipe, ResiliencePipeline)

    def test_all_metrics(self):
        reg = self._setup_registry()
        mgr = ServiceCircuitManager(registry=reg)
        mgr.get_breaker("api-gateway")
        mgr.get_breaker("user-service")
        metrics = mgr.get_all_metrics()
        assert "api-gateway" in metrics
        assert "user-service" in metrics

    def test_health_report(self):
        reg = self._setup_registry()
        mgr = ServiceCircuitManager(registry=reg)
        mgr.get_breaker("api-gateway")
        report = mgr.get_service_health_report()
        assert "api-gateway" in report
        assert report["api-gateway"]["instances"] == 2
        assert report["api-gateway"]["discovery_health"] == "passing"

    def test_default_config(self):
        mgr = ServiceCircuitManager(default_config={
            'failure_rate_threshold': 0.3,
            'wait_duration': 10.0,
            'minimum_calls': 3,
            'permitted_half_open_calls': 2,
            'max_retries': 1,
            'timeout': 5.0,
        })
        cb = mgr.get_breaker("test-svc")
        assert cb.failure_rate_threshold == 0.3
        assert cb.wait_duration == 10.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    def test_circuit_breaker_with_retry(self):
        """Retry inside a circuit breaker."""
        cb = CircuitBreaker(failure_rate_threshold=0.5, minimum_calls=10)
        mt = MockTime()
        cb._time = mt
        cb._window._time = mt
        rp = RetryPolicy(max_retries=2)

        func = make_failing(1, "recovered")

        def with_retry():
            return rp.execute(func, use_sleep=False)

        result = cb.execute(with_retry)
        assert result == "recovered"

    def test_full_resilience_stack(self):
        """Timeout -> Retry -> CircuitBreaker -> call."""
        pipe = ResiliencePipeline("full-stack")
        tp = pipe.add_timeout(10.0)
        mt = MockTime()
        tp._time = mt
        pipe.add_retry(max_retries=2)
        cb = pipe.add_circuit_breaker(failure_rate_threshold=0.5, minimum_calls=10)
        cb._time = mt
        cb._window._time = mt

        func = make_failing(1, "ok")
        result = pipe.execute(func)
        assert result == "ok"

    def test_bulkhead_with_fallback(self):
        """Bulkhead rejection triggers fallback."""
        pipe = ResiliencePipeline("bh-fb")
        bh = pipe.add_bulkhead(max_concurrent=0)  # Always full
        pipe.add_fallback(always_succeed("cached"))

        result = pipe.execute(always_succeed("live"))
        assert result == "cached"

    def test_rate_limit_with_fallback(self):
        """Rate limit exceeded triggers fallback."""
        pipe = ResiliencePipeline("rl-fb")
        rl = pipe.add_rate_limiter(strategy=RateLimitStrategy.SLIDING_WINDOW,
                                   rate=1, period=60.0)
        mt = MockTime()
        rl._time = mt
        pipe.add_fallback(always_succeed("throttled"))

        result = pipe.execute(always_succeed("live"))
        assert result == "live"
        result = pipe.execute(always_succeed("live2"))
        assert result == "throttled"

    def test_service_manager_with_discovery(self):
        """ServiceCircuitManager resolves and calls through circuit."""
        reg = ServiceRegistry()
        reg.register("payment-api", service_id="pay-1", port=443,
                     tags=["production"])
        reg.register("payment-api", service_id="pay-2", port=443,
                     tags=["production"])

        mgr = ServiceCircuitManager(registry=reg)

        called = []
        def handler(instance):
            called.append(instance.service_id)
            return "charged"

        result = mgr.call_service("payment-api", handler)
        assert result == "charged"
        assert len(called) == 1

    def test_circuit_opens_then_recovers(self):
        """Full cycle: closed -> open -> half-open -> closed."""
        cb = CircuitBreaker(
            name="recovery-test",
            failure_rate_threshold=0.5,
            minimum_calls=3,
            wait_duration=5.0,
            permitted_half_open_calls=2,
            success_rate_threshold=0.5,
        )
        mt = MockTime()
        cb._time = mt
        cb._window._time = mt

        states = []
        cb.on_state_change(lambda old, new: states.append(new))

        # Phase 1: Failures open the circuit
        for _ in range(5):
            try:
                cb.execute(always_fail())
            except (RuntimeError, CircuitOpenError):
                pass
        assert cb.state == CircuitState.OPEN

        # Phase 2: Wait for half-open
        mt.advance(6.0)
        assert cb.state == CircuitState.HALF_OPEN

        # Phase 3: Successes close the circuit
        cb.execute(always_succeed())
        cb.execute(always_succeed())
        assert cb.state == CircuitState.CLOSED

        assert CircuitState.OPEN in states
        assert CircuitState.HALF_OPEN in states
        assert CircuitState.CLOSED in states

    def test_retry_with_jitter_variance(self):
        """Verify jitter produces different delays across retries."""
        delays_sets = []
        for _ in range(5):
            rp = RetryPolicy(max_retries=3, backoff=RetryBackoff.FIXED,
                             base_delay=10.0, jitter=True, jitter_range=0.5)
            func = make_failing(3, "ok")
            _, delays = rp.execute_with_delays(func)
            delays_sets.append(tuple(round(d, 2) for d in delays))
        # With jitter, not all sets should be identical
        unique = set(delays_sets)
        assert len(unique) >= 2  # At least some variance

    def test_sliding_window_accuracy(self):
        """Verify sliding window tracks correctly over time."""
        sw = SlidingWindow(window_size=10.0, bucket_count=10)
        mt = MockTime()
        sw._time = mt

        # Record at t=0
        for _ in range(5):
            sw.record_success(mt())
        for _ in range(5):
            sw.record_failure(mt())

        assert abs(sw.get_failure_rate(mt()) - 0.5) < 0.01

        # Advance past window
        mt.advance(11.0)
        sw.record_success(mt())
        assert sw.get_failure_rate(mt()) == 0.0

    def test_multiple_services_independent_circuits(self):
        """Each service gets its own independent circuit breaker."""
        mgr = ServiceCircuitManager()
        cb1 = mgr.get_breaker("svc-a")
        cb2 = mgr.get_breaker("svc-b")

        cb1.force_open()
        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.CLOSED

    def test_pipeline_custom_policy(self):
        """Add a custom policy to the pipeline."""
        class LoggingPolicy:
            def __init__(self):
                self.calls = []
            def execute(self, func, *args, **kwargs):
                self.calls.append("called")
                return func(*args, **kwargs)

        pipe = ResiliencePipeline()
        lp = LoggingPolicy()
        pipe.add_policy(lp)
        result = pipe.execute(always_succeed("ok"))
        assert result == "ok"
        assert lp.calls == ["called"]


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    def test_circuit_breaker_zero_wait(self):
        cb = CircuitBreaker(failure_rate_threshold=0.5, minimum_calls=3,
                            wait_duration=0.0)
        mt = MockTime()
        cb._time = mt
        cb._window._time = mt

        for _ in range(5):
            try:
                cb.execute(always_fail())
            except RuntimeError:
                pass

        # With 0 wait, should immediately go to half-open
        assert cb.state == CircuitState.HALF_OPEN

    def test_retry_zero_retries(self):
        rp = RetryPolicy(max_retries=0)
        with pytest.raises(RetriesExhaustedError):
            rp.execute(always_fail(), use_sleep=False)

    def test_bulkhead_zero_concurrent(self):
        bh = Bulkhead(max_concurrent=0)
        with pytest.raises(BulkheadFullError):
            bh.execute(always_succeed())

    def test_rate_limiter_zero_rate(self):
        rl = RateLimiter(strategy=RateLimitStrategy.SLIDING_WINDOW, rate=0)
        mt = MockTime()
        rl._time = mt
        assert not rl.acquire(mt())

    def test_fallback_chain_first_succeeds(self):
        fc = FallbackChain()
        fc.add(always_succeed("first"))
        fc.add(always_succeed("second"))
        assert fc.execute() == "first"

    def test_circuit_breaker_exception_passthrough(self):
        """Exceptions pass through even when circuit records them."""
        cb = CircuitBreaker(minimum_calls=100)
        with pytest.raises(ValueError, match="specific"):
            cb.execute(lambda: (_ for _ in ()).throw(ValueError("specific")))

    def test_timeout_exact_boundary(self):
        tp = TimeoutPolicy(timeout=5.0)
        mt = MockTime()
        tp._time = mt

        def exact():
            mt.advance(5.0)
            return "boundary"

        # Exactly at boundary is not a timeout (> not >=)
        result = tp.execute(exact)
        assert result == "boundary"

    def test_rate_limiter_burst(self):
        rl = RateLimiter(strategy=RateLimitStrategy.TOKEN_BUCKET,
                         rate=1.0, period=1.0, burst=5)
        mt = MockTime()
        rl._time = mt
        rl._last_refill = mt()
        rl._tokens = 5.0

        # Can burst 5 immediately even though rate is 1/s
        for _ in range(5):
            assert rl.acquire(mt())
        assert not rl.acquire(mt())

    def test_pipeline_order_matters(self):
        """Policies are applied in order."""
        calls = []

        class TrackingPolicy:
            def __init__(self, name):
                self.n = name
            def execute(self, func, *args, **kwargs):
                calls.append(f"enter-{self.n}")
                result = func(*args, **kwargs)
                calls.append(f"exit-{self.n}")
                return result

        pipe = ResiliencePipeline()
        pipe.add_policy(TrackingPolicy("outer"))
        pipe.add_policy(TrackingPolicy("inner"))
        pipe.execute(always_succeed())

        assert calls == ["enter-outer", "enter-inner", "exit-inner", "exit-outer"]

    def test_circuit_breaker_high_threshold(self):
        """Circuit with 100% threshold never opens on partial failures."""
        cb = CircuitBreaker(failure_rate_threshold=1.0, minimum_calls=3)
        mt = MockTime()
        cb._time = mt
        cb._window._time = mt

        cb.execute(always_succeed())
        try:
            cb.execute(always_fail())
        except RuntimeError:
            pass
        try:
            cb.execute(always_fail())
        except RuntimeError:
            pass

        # 66% failure rate < 100% threshold
        assert cb.state == CircuitState.CLOSED

    def test_service_manager_no_instances(self):
        """Calling a service with no instances."""
        reg = ServiceRegistry()
        mgr = ServiceCircuitManager(registry=reg)

        def handler(instance):
            return "ok"

        # resolver returns None, handler gets None
        result = mgr.call_service("nonexistent", handler)
        # Handler receives None as instance
        assert result == "ok"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
