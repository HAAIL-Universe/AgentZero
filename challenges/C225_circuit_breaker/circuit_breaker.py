"""
C225: Circuit Breaker & Resilience Patterns
Composing C222 (Service Discovery)

A comprehensive resilience library inspired by Netflix Hystrix / resilience4j:
- CircuitBreaker: 3-state (closed/open/half-open) with failure rate tracking
- RetryPolicy: configurable retry with exponential backoff, jitter, predicates
- Timeout: wraps calls with configurable timeout
- Bulkhead: limits concurrent executions (semaphore-based)
- RateLimiter: token bucket and sliding window rate limiting
- FallbackChain: ordered fallback execution
- ResiliencePipeline: composable pipeline of resilience policies
- ServiceCircuitManager: per-service circuit breakers using C222 health info
"""

import sys
import os
import time
import math
import random
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable, List
from collections import deque, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C222_service_discovery'))
from service_discovery import (
    ServiceRegistry, ServiceResolver, HealthStatus, HealthCheck, CheckType,
    LoadBalanceStrategy, ServiceCatalog
)


# =============================================================================
# Enums
# =============================================================================

class CircuitState(Enum):
    CLOSED = "closed"          # Normal operation, tracking failures
    OPEN = "open"              # Failing fast, rejecting calls
    HALF_OPEN = "half_open"    # Testing if service recovered


class RetryBackoff(Enum):
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"


class RateLimitStrategy(Enum):
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


# =============================================================================
# Exceptions
# =============================================================================

class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and rejects calls."""
    def __init__(self, name, remaining_time=0):
        self.name = name
        self.remaining_time = remaining_time
        super().__init__(f"Circuit '{name}' is open (retry in {remaining_time:.1f}s)")


class TimeoutError(Exception):
    """Raised when a call exceeds the timeout."""
    def __init__(self, name, timeout):
        self.name = name
        self.timeout = timeout
        super().__init__(f"Call '{name}' timed out after {timeout}s")


class BulkheadFullError(Exception):
    """Raised when bulkhead rejects due to max concurrency."""
    def __init__(self, name, max_concurrent):
        self.name = name
        self.max_concurrent = max_concurrent
        super().__init__(f"Bulkhead '{name}' full ({max_concurrent} concurrent)")


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""
    def __init__(self, name, limit):
        self.name = name
        self.limit = limit
        super().__init__(f"Rate limit exceeded for '{name}' (limit: {limit})")


class RetriesExhaustedError(Exception):
    """Raised when all retry attempts fail."""
    def __init__(self, name, attempts, last_error):
        self.name = name
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"All {attempts} retries exhausted for '{name}': {last_error}")


class FallbackExhaustedError(Exception):
    """Raised when all fallbacks fail."""
    def __init__(self, name, errors):
        self.name = name
        self.errors = errors
        super().__init__(f"All fallbacks exhausted for '{name}'")


# =============================================================================
# SlidingWindow - shared utility for tracking events in a time window
# =============================================================================

class SlidingWindow:
    """Tracks events within a sliding time window."""

    def __init__(self, window_size: float = 60.0, bucket_count: int = 10):
        self.window_size = window_size
        self.bucket_count = bucket_count
        self.bucket_duration = window_size / bucket_count
        self._buckets = deque()  # (start_time, success_count, failure_count)
        self._current_bucket = None
        self._current_start = 0.0
        self._time = time.time

    def record_success(self, now=None):
        now = now or self._time()
        self._ensure_bucket(now)
        bucket = self._current_bucket
        bucket[1] += 1

    def record_failure(self, now=None):
        now = now or self._time()
        self._ensure_bucket(now)
        bucket = self._current_bucket
        bucket[2] += 1

    def get_stats(self, now=None):
        """Returns (total_success, total_failure, total_calls) in the window."""
        now = now or self._time()
        self._cleanup(now)
        total_success = 0
        total_failure = 0
        for bucket in self._buckets:
            total_success += bucket[1]
            total_failure += bucket[2]
        if self._current_bucket:
            total_success += self._current_bucket[1]
            total_failure += self._current_bucket[2]
        total = total_success + total_failure
        return total_success, total_failure, total

    def get_failure_rate(self, now=None):
        """Returns failure rate as a float 0.0-1.0."""
        s, f, total = self.get_stats(now)
        if total == 0:
            return 0.0
        return f / total

    def get_total_calls(self, now=None):
        _, _, total = self.get_stats(now)
        return total

    def reset(self):
        self._buckets.clear()
        self._current_bucket = None
        self._current_start = 0.0

    def _ensure_bucket(self, now):
        if self._current_bucket is None or now - self._current_start >= self.bucket_duration:
            if self._current_bucket is not None:
                self._buckets.append(self._current_bucket)
            self._current_bucket = [now, 0, 0]  # [start, success, failure]
            self._current_start = now
            self._cleanup(now)

    def _cleanup(self, now):
        cutoff = now - self.window_size
        while self._buckets and self._buckets[0][0] < cutoff:
            self._buckets.popleft()
        if self._current_bucket and self._current_bucket[0] < cutoff:
            self._current_bucket = None


# =============================================================================
# CircuitBreaker
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker with 3 states: CLOSED -> OPEN -> HALF_OPEN -> CLOSED.

    In CLOSED state: tracks failures. Opens when failure_rate > threshold
                     within the sliding window AND minimum_calls reached.
    In OPEN state: rejects all calls. After wait_duration, transitions to HALF_OPEN.
    In HALF_OPEN state: allows permitted_calls through. If success_rate >= threshold,
                        transitions to CLOSED. If any fails, back to OPEN.
    """

    def __init__(self, name: str = "default",
                 failure_rate_threshold: float = 0.5,
                 wait_duration: float = 60.0,
                 minimum_calls: int = 5,
                 permitted_half_open_calls: int = 3,
                 window_size: float = 60.0,
                 success_rate_threshold: float = 0.5,
                 record_exceptions: tuple = (Exception,),
                 ignore_exceptions: tuple = ()):
        self.name = name
        self.failure_rate_threshold = failure_rate_threshold
        self.wait_duration = wait_duration
        self.minimum_calls = minimum_calls
        self.permitted_half_open_calls = permitted_half_open_calls
        self.success_rate_threshold = success_rate_threshold
        self.record_exceptions = record_exceptions
        self.ignore_exceptions = ignore_exceptions

        self._state = CircuitState.CLOSED
        self._window = SlidingWindow(window_size)
        self._opened_at = 0.0
        self._half_open_calls = 0
        self._half_open_successes = 0
        self._half_open_failures = 0
        self._state_change_callbacks = []
        self._time = time.time
        self._total_calls = 0
        self._total_successes = 0
        self._total_failures = 0
        self._total_rejections = 0
        self._consecutive_successes = 0
        self._consecutive_failures = 0

    @property
    def state(self):
        # Check if OPEN should transition to HALF_OPEN
        if self._state == CircuitState.OPEN:
            now = self._time()
            if now - self._opened_at >= self.wait_duration:
                self._transition(CircuitState.HALF_OPEN)
        return self._state

    def execute(self, func, *args, **kwargs):
        """Execute a function through the circuit breaker."""
        self._check_state()

        self._total_calls += 1
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            if self._should_record(e):
                self._on_failure()
            else:
                self._on_success()  # Ignored exceptions count as success
            raise

    def call(self, func, *args, **kwargs):
        """Alias for execute."""
        return self.execute(func, *args, **kwargs)

    def record_success(self):
        """Manually record a success."""
        self._on_success()

    def record_failure(self):
        """Manually record a failure."""
        self._on_failure()

    def reset(self):
        """Reset the circuit breaker to CLOSED state."""
        self._transition(CircuitState.CLOSED)
        self._window.reset()
        self._consecutive_successes = 0
        self._consecutive_failures = 0

    def force_open(self):
        """Force the circuit to OPEN state."""
        self._transition(CircuitState.OPEN)
        self._opened_at = self._time()

    def get_metrics(self):
        """Get current metrics."""
        now = self._time()
        s, f, total = self._window.get_stats(now)
        return {
            'state': self._state.value,
            'failure_rate': self._window.get_failure_rate(now),
            'total_calls': self._total_calls,
            'total_successes': self._total_successes,
            'total_failures': self._total_failures,
            'total_rejections': self._total_rejections,
            'window_success': s,
            'window_failure': f,
            'window_total': total,
            'consecutive_successes': self._consecutive_successes,
            'consecutive_failures': self._consecutive_failures,
        }

    def on_state_change(self, callback):
        """Register a callback for state changes: callback(old_state, new_state)."""
        self._state_change_callbacks.append(callback)

    def _check_state(self):
        state = self.state  # Triggers OPEN->HALF_OPEN check
        if state == CircuitState.OPEN:
            remaining = self.wait_duration - (self._time() - self._opened_at)
            self._total_rejections += 1
            raise CircuitOpenError(self.name, max(0, remaining))
        if state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.permitted_half_open_calls:
                self._total_rejections += 1
                raise CircuitOpenError(self.name, 0)

    def _on_success(self):
        now = self._time()
        self._total_successes += 1
        self._consecutive_successes += 1
        self._consecutive_failures = 0

        if self._state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            total_ho = self._half_open_successes + self._half_open_failures
            if total_ho >= self.permitted_half_open_calls:
                rate = self._half_open_successes / total_ho
                if rate >= self.success_rate_threshold:
                    self._transition(CircuitState.CLOSED)
                    self._window.reset()
                else:
                    self._transition(CircuitState.OPEN)
                    self._opened_at = now
        else:
            self._window.record_success(now)

    def _on_failure(self):
        now = self._time()
        self._total_failures += 1
        self._consecutive_failures += 1
        self._consecutive_successes = 0

        if self._state == CircuitState.HALF_OPEN:
            self._half_open_failures += 1
            total_ho = self._half_open_successes + self._half_open_failures
            if total_ho >= self.permitted_half_open_calls:
                rate = self._half_open_successes / total_ho
                if rate >= self.success_rate_threshold:
                    self._transition(CircuitState.CLOSED)
                    self._window.reset()
                else:
                    self._transition(CircuitState.OPEN)
                    self._opened_at = now
            # Even single failure in half-open can trip back to open
            # if we haven't reached permitted_calls yet, keep going
        else:
            self._window.record_failure(now)
            s, f, total = self._window.get_stats(now)
            if total >= self.minimum_calls:
                failure_rate = f / total
                if failure_rate >= self.failure_rate_threshold:
                    self._transition(CircuitState.OPEN)
                    self._opened_at = now

    def _should_record(self, exception):
        """Check if exception should be recorded as a failure."""
        if self.ignore_exceptions and isinstance(exception, self.ignore_exceptions):
            return False
        return isinstance(exception, self.record_exceptions)

    def _transition(self, new_state):
        old_state = self._state
        if old_state == new_state:
            return
        self._state = new_state
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._half_open_successes = 0
            self._half_open_failures = 0
        for cb in self._state_change_callbacks:
            try:
                cb(old_state, new_state)
            except Exception:
                pass


# =============================================================================
# RetryPolicy
# =============================================================================

class RetryPolicy:
    """
    Configurable retry policy with backoff strategies.

    Supports:
    - Fixed, exponential, and linear backoff
    - Jitter (full, equal, decorrelated)
    - Retry predicates (retry only specific exceptions)
    - Max delay cap
    - Retry budget (max retries per time window)
    """

    def __init__(self, name: str = "default",
                 max_retries: int = 3,
                 backoff: RetryBackoff = RetryBackoff.EXPONENTIAL,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 multiplier: float = 2.0,
                 jitter: bool = True,
                 jitter_range: float = 0.5,
                 retry_on: tuple = (Exception,),
                 retry_if: Optional[Callable] = None,
                 on_retry: Optional[Callable] = None):
        self.name = name
        self.max_retries = max_retries
        self.backoff = backoff
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.jitter_range = jitter_range
        self.retry_on = retry_on
        self.retry_if = retry_if
        self.on_retry = on_retry

        self._total_retries = 0
        self._total_successes = 0
        self._total_exhausted = 0
        self._time = time.time
        self._sleep = time.sleep

    def execute(self, func, *args, use_sleep=True, **kwargs):
        """Execute a function with retry logic."""
        last_error = None
        delays = []

        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self._total_retries += attempt
                self._total_successes += 1
                return result
            except Exception as e:
                last_error = e
                if not self._should_retry(e):
                    raise
                if attempt >= self.max_retries:
                    break

                delay = self._calculate_delay(attempt)
                delays.append(delay)

                if self.on_retry:
                    try:
                        self.on_retry(attempt + 1, delay, e)
                    except Exception:
                        pass

                if use_sleep and delay > 0:
                    self._sleep(delay)

        self._total_exhausted += 1
        raise RetriesExhaustedError(self.name, self.max_retries + 1, last_error)

    def execute_with_delays(self, func, *args, **kwargs):
        """Execute returning (result, delays_used). Does not actually sleep."""
        last_error = None
        delays = []

        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                return result, delays
            except Exception as e:
                last_error = e
                if not self._should_retry(e):
                    raise
                if attempt >= self.max_retries:
                    break
                delay = self._calculate_delay(attempt)
                delays.append(delay)

        raise RetriesExhaustedError(self.name, self.max_retries + 1, last_error)

    def get_metrics(self):
        return {
            'total_retries': self._total_retries,
            'total_successes': self._total_successes,
            'total_exhausted': self._total_exhausted,
        }

    def _should_retry(self, exception):
        if self.retry_if and not self.retry_if(exception):
            return False
        return isinstance(exception, self.retry_on)

    def _calculate_delay(self, attempt):
        if self.backoff == RetryBackoff.FIXED:
            delay = self.base_delay
        elif self.backoff == RetryBackoff.EXPONENTIAL:
            delay = self.base_delay * (self.multiplier ** attempt)
        elif self.backoff == RetryBackoff.LINEAR:
            delay = self.base_delay * (attempt + 1)
        else:
            delay = self.base_delay

        delay = min(delay, self.max_delay)

        if self.jitter:
            jitter_amount = delay * self.jitter_range
            delay = delay + random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay)

        return delay


# =============================================================================
# Timeout
# =============================================================================

class TimeoutPolicy:
    """
    Wraps a call with a timeout. Uses simulated time for testability.
    """

    def __init__(self, name: str = "default", timeout: float = 30.0):
        self.name = name
        self.timeout = timeout
        self._total_calls = 0
        self._total_timeouts = 0
        self._time = time.time

    def execute(self, func, *args, **kwargs):
        """
        Execute with timeout. For testability, we check duration after
        the call completes (simulated timeout).
        """
        self._total_calls += 1
        start = self._time()
        try:
            result = func(*args, **kwargs)
        except Exception:
            elapsed = self._time() - start
            if elapsed > self.timeout:
                self._total_timeouts += 1
                raise TimeoutError(self.name, self.timeout)
            raise
        elapsed = self._time() - start
        if elapsed > self.timeout:
            self._total_timeouts += 1
            raise TimeoutError(self.name, self.timeout)
        return result

    def get_metrics(self):
        return {
            'total_calls': self._total_calls,
            'total_timeouts': self._total_timeouts,
        }


# =============================================================================
# Bulkhead
# =============================================================================

class Bulkhead:
    """
    Limits concurrent executions to prevent resource exhaustion.
    Semaphore-based bulkhead pattern.
    """

    def __init__(self, name: str = "default",
                 max_concurrent: int = 10,
                 max_wait: float = 0.0):
        self.name = name
        self.max_concurrent = max_concurrent
        self.max_wait = max_wait
        self._active = 0
        self._total_calls = 0
        self._total_rejections = 0
        self._lock = threading.Lock()

    def execute(self, func, *args, **kwargs):
        """Execute within the bulkhead."""
        if not self.acquire():
            raise BulkheadFullError(self.name, self.max_concurrent)

        try:
            self._total_calls += 1
            return func(*args, **kwargs)
        finally:
            self.release()

    def acquire(self):
        """Try to acquire a slot."""
        with self._lock:
            if self._active < self.max_concurrent:
                self._active += 1
                return True
            self._total_rejections += 1
            return False

    def release(self):
        """Release a slot."""
        with self._lock:
            self._active = max(0, self._active - 1)

    @property
    def active_count(self):
        return self._active

    @property
    def available_count(self):
        return max(0, self.max_concurrent - self._active)

    def get_metrics(self):
        return {
            'active': self._active,
            'available': self.available_count,
            'max_concurrent': self.max_concurrent,
            'total_calls': self._total_calls,
            'total_rejections': self._total_rejections,
        }


# =============================================================================
# RateLimiter
# =============================================================================

class RateLimiter:
    """
    Rate limiter supporting token bucket, sliding window, and fixed window.
    """

    def __init__(self, name: str = "default",
                 strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET,
                 rate: float = 10.0,
                 period: float = 1.0,
                 burst: int = 0):
        self.name = name
        self.strategy = strategy
        self.rate = rate          # requests per period
        self.period = period      # seconds
        self.burst = burst or int(rate)  # max burst (token bucket)

        self._time = time.time

        # Token bucket state
        self._tokens = float(self.burst)
        self._last_refill = self._time()

        # Sliding window state
        self._request_times = deque()

        # Fixed window state
        self._window_start = self._time()
        self._window_count = 0

        self._total_allowed = 0
        self._total_rejected = 0

    def acquire(self, now=None):
        """Try to acquire permission. Returns True if allowed."""
        now = now or self._time()

        if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._token_bucket_acquire(now)
        elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._sliding_window_acquire(now)
        elif self.strategy == RateLimitStrategy.FIXED_WINDOW:
            return self._fixed_window_acquire(now)
        return True

    def execute(self, func, *args, now=None, **kwargs):
        """Execute if rate limit allows."""
        if not self.acquire(now=now):
            raise RateLimitExceededError(self.name, self.rate)
        return func(*args, **kwargs)

    def get_metrics(self):
        now = self._time()
        metrics = {
            'total_allowed': self._total_allowed,
            'total_rejected': self._total_rejected,
            'strategy': self.strategy.value,
        }
        if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            self._refill_tokens(now)
            metrics['available_tokens'] = self._tokens
        return metrics

    def _token_bucket_acquire(self, now):
        self._refill_tokens(now)
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            self._total_allowed += 1
            return True
        self._total_rejected += 1
        return False

    def _refill_tokens(self, now):
        elapsed = now - self._last_refill
        new_tokens = elapsed * (self.rate / self.period)
        self._tokens = min(self.burst, self._tokens + new_tokens)
        self._last_refill = now

    def _sliding_window_acquire(self, now):
        cutoff = now - self.period
        while self._request_times and self._request_times[0] <= cutoff:
            self._request_times.popleft()

        if len(self._request_times) < self.rate:
            self._request_times.append(now)
            self._total_allowed += 1
            return True
        self._total_rejected += 1
        return False

    def _fixed_window_acquire(self, now):
        if now - self._window_start >= self.period:
            self._window_start = now
            self._window_count = 0

        if self._window_count < self.rate:
            self._window_count += 1
            self._total_allowed += 1
            return True
        self._total_rejected += 1
        return False


# =============================================================================
# FallbackChain
# =============================================================================

class FallbackChain:
    """
    Executes a chain of functions, falling back to the next on failure.
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self._chain = []  # List of (func, description)
        self._total_primary = 0
        self._total_fallbacks = 0
        self._fallback_counts = defaultdict(int)

    def add(self, func, description: str = ""):
        """Add a function to the fallback chain."""
        self._chain.append((func, description or f"fallback-{len(self._chain)}"))
        return self

    def execute(self, *args, **kwargs):
        """Execute the chain until one succeeds."""
        if not self._chain:
            raise FallbackExhaustedError(self.name, ["No fallbacks configured"])

        errors = []
        for i, (func, desc) in enumerate(self._chain):
            try:
                result = func(*args, **kwargs)
                if i == 0:
                    self._total_primary += 1
                else:
                    self._total_fallbacks += 1
                    self._fallback_counts[desc] += 1
                return result
            except Exception as e:
                errors.append((desc, e))

        raise FallbackExhaustedError(self.name, errors)

    def get_metrics(self):
        return {
            'total_primary': self._total_primary,
            'total_fallbacks': self._total_fallbacks,
            'fallback_counts': dict(self._fallback_counts),
            'chain_length': len(self._chain),
        }


# =============================================================================
# ResiliencePipeline
# =============================================================================

class ResiliencePipeline:
    """
    Composable pipeline of resilience policies.
    Policies are applied in order (outermost first).

    Example: pipeline = ResiliencePipeline("api")
             pipeline.add_timeout(5.0)
             pipeline.add_retry(max_retries=3)
             pipeline.add_circuit_breaker(failure_rate=0.5)
             result = pipeline.execute(api_call)

    Execution order: Timeout -> Retry -> CircuitBreaker -> actual call
    (Timeout wraps retry, retry wraps circuit breaker, CB wraps call)
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self._policies = []  # List of policy objects
        self._total_calls = 0
        self._total_successes = 0
        self._total_failures = 0

    def add_circuit_breaker(self, **kwargs):
        """Add a circuit breaker to the pipeline."""
        cb = CircuitBreaker(name=f"{self.name}-cb", **kwargs)
        self._policies.append(('circuit_breaker', cb))
        return cb

    def add_retry(self, **kwargs):
        """Add a retry policy to the pipeline."""
        rp = RetryPolicy(name=f"{self.name}-retry", **kwargs)
        self._policies.append(('retry', rp))
        return rp

    def add_timeout(self, timeout=30.0, **kwargs):
        """Add a timeout policy to the pipeline."""
        tp = TimeoutPolicy(name=f"{self.name}-timeout", timeout=timeout, **kwargs)
        self._policies.append(('timeout', tp))
        return tp

    def add_bulkhead(self, **kwargs):
        """Add a bulkhead to the pipeline."""
        bh = Bulkhead(name=f"{self.name}-bh", **kwargs)
        self._policies.append(('bulkhead', bh))
        return bh

    def add_rate_limiter(self, **kwargs):
        """Add a rate limiter to the pipeline."""
        rl = RateLimiter(name=f"{self.name}-rl", **kwargs)
        self._policies.append(('rate_limiter', rl))
        return rl

    def add_fallback(self, fallback_func, description="fallback"):
        """Add a fallback function."""
        self._policies.append(('fallback', (fallback_func, description)))
        return self

    def add_policy(self, policy):
        """Add a custom policy object (must have execute method)."""
        self._policies.append(('custom', policy))
        return self

    def execute(self, func, *args, **kwargs):
        """Execute through the pipeline."""
        self._total_calls += 1

        # Build the execution chain (last added is innermost)
        wrapped = func
        fallback_func = None

        # Find fallback if any
        for ptype, policy in self._policies:
            if ptype == 'fallback':
                fallback_func = policy[0]

        # Build chain from innermost to outermost
        chain_policies = [(pt, p) for pt, p in self._policies if pt != 'fallback']

        # Reverse so we wrap from innermost out
        for ptype, policy in reversed(chain_policies):
            inner = wrapped
            if ptype == 'circuit_breaker':
                wrapped = self._wrap_cb(policy, inner)
            elif ptype == 'retry':
                wrapped = self._wrap_retry(policy, inner)
            elif ptype == 'timeout':
                wrapped = self._wrap_timeout(policy, inner)
            elif ptype == 'bulkhead':
                wrapped = self._wrap_bulkhead(policy, inner)
            elif ptype == 'rate_limiter':
                wrapped = self._wrap_rate_limiter(policy, inner)
            elif ptype == 'custom':
                wrapped = self._wrap_custom(policy, inner)

        try:
            result = wrapped(*args, **kwargs)
            self._total_successes += 1
            return result
        except Exception as e:
            if fallback_func:
                try:
                    result = fallback_func(*args, **kwargs)
                    self._total_successes += 1
                    return result
                except Exception:
                    pass
            self._total_failures += 1
            raise

    def get_metrics(self):
        metrics = {
            'total_calls': self._total_calls,
            'total_successes': self._total_successes,
            'total_failures': self._total_failures,
            'policies': [],
        }
        for ptype, policy in self._policies:
            if hasattr(policy, 'get_metrics'):
                metrics['policies'].append({
                    'type': ptype,
                    'metrics': policy.get_metrics(),
                })
        return metrics

    def _wrap_cb(self, cb, inner):
        def wrapped(*args, **kwargs):
            return cb.execute(inner, *args, **kwargs)
        return wrapped

    def _wrap_retry(self, rp, inner):
        def wrapped(*args, **kwargs):
            return rp.execute(inner, *args, use_sleep=False, **kwargs)
        return wrapped

    def _wrap_timeout(self, tp, inner):
        def wrapped(*args, **kwargs):
            return tp.execute(inner, *args, **kwargs)
        return wrapped

    def _wrap_bulkhead(self, bh, inner):
        def wrapped(*args, **kwargs):
            return bh.execute(inner, *args, **kwargs)
        return wrapped

    def _wrap_rate_limiter(self, rl, inner):
        def wrapped(*args, **kwargs):
            return rl.execute(inner, *args, **kwargs)
        return wrapped

    def _wrap_custom(self, policy, inner):
        def wrapped(*args, **kwargs):
            return policy.execute(inner, *args, **kwargs)
        return wrapped


# =============================================================================
# ServiceCircuitManager
# =============================================================================

class ServiceCircuitManager:
    """
    Manages per-service circuit breakers using C222 Service Discovery.
    Integrates health check information from ServiceRegistry to inform
    circuit breaker decisions.
    """

    def __init__(self, registry=None, default_config=None):
        self.registry = registry or ServiceRegistry()
        self.resolver = ServiceResolver(self.registry)
        self._breakers = {}          # service_name -> CircuitBreaker
        self._pipelines = {}         # service_name -> ResiliencePipeline
        self._service_configs = {}   # service_name -> config dict
        self._default_config = default_config or {
            'failure_rate_threshold': 0.5,
            'wait_duration': 30.0,
            'minimum_calls': 5,
            'permitted_half_open_calls': 3,
            'max_retries': 2,
            'timeout': 10.0,
        }
        self._call_log = []  # (timestamp, service, success, latency)

    def configure_service(self, service_name, **config):
        """Configure resilience settings for a specific service."""
        self._service_configs[service_name] = {**self._default_config, **config}

    def get_breaker(self, service_name):
        """Get or create a circuit breaker for a service."""
        if service_name not in self._breakers:
            config = self._service_configs.get(service_name, self._default_config)
            cb = CircuitBreaker(
                name=service_name,
                failure_rate_threshold=config.get('failure_rate_threshold', 0.5),
                wait_duration=config.get('wait_duration', 30.0),
                minimum_calls=config.get('minimum_calls', 5),
                permitted_half_open_calls=config.get('permitted_half_open_calls', 3),
            )
            self._breakers[service_name] = cb
        return self._breakers[service_name]

    def get_pipeline(self, service_name):
        """Get or create a full resilience pipeline for a service."""
        if service_name not in self._pipelines:
            config = self._service_configs.get(service_name, self._default_config)
            pipeline = ResiliencePipeline(service_name)

            if config.get('timeout'):
                pipeline.add_timeout(config['timeout'])
            if config.get('max_retries', 0) > 0:
                pipeline.add_retry(max_retries=config['max_retries'])
            pipeline.add_circuit_breaker(
                failure_rate_threshold=config.get('failure_rate_threshold', 0.5),
                wait_duration=config.get('wait_duration', 30.0),
                minimum_calls=config.get('minimum_calls', 5),
            )

            self._pipelines[service_name] = pipeline
        return self._pipelines[service_name]

    def call_service(self, service_name, func, *args,
                     strategy=LoadBalanceStrategy.ROUND_ROBIN, **kwargs):
        """
        Call a service through its circuit breaker.
        Resolves an instance via service discovery, then calls through the breaker.
        """
        breaker = self.get_breaker(service_name)
        instance = self.resolver.resolve(service_name, strategy=strategy)

        start = time.time()
        try:
            result = breaker.execute(func, instance, *args, **kwargs)
            elapsed = time.time() - start
            self._call_log.append((time.time(), service_name, True, elapsed))
            return result
        except Exception as e:
            elapsed = time.time() - start
            self._call_log.append((time.time(), service_name, False, elapsed))
            raise

    def sync_health(self):
        """
        Sync circuit breaker states with service discovery health.
        If a service is CRITICAL in the registry, force-open its circuit.
        If it recovers to PASSING, allow the circuit to test.
        """
        catalog = ServiceCatalog(self.registry)
        changes = []

        for service_name in self.registry.get_all_services():
            health = catalog.service_health(service_name)
            breaker = self._breakers.get(service_name)
            if not breaker:
                continue

            if health == HealthStatus.CRITICAL and breaker.state != CircuitState.OPEN:
                breaker.force_open()
                changes.append((service_name, 'force_open', 'health_critical'))
            elif health == HealthStatus.PASSING and breaker.state == CircuitState.OPEN:
                breaker.reset()
                changes.append((service_name, 'reset', 'health_passing'))

        return changes

    def get_all_metrics(self):
        """Get metrics for all managed services."""
        metrics = {}
        for name, breaker in self._breakers.items():
            metrics[name] = breaker.get_metrics()
        return metrics

    def get_service_health_report(self):
        """Combined report of circuit breaker state + service discovery health."""
        report = {}
        catalog = ServiceCatalog(self.registry)

        for name in set(list(self._breakers.keys()) + self.registry.get_all_services()):
            entry = {
                'circuit_state': None,
                'discovery_health': None,
                'instances': 0,
                'healthy_instances': 0,
            }
            if name in self._breakers:
                entry['circuit_state'] = self._breakers[name].state.value
            instances = self.registry.get_services(name)
            entry['instances'] = len(instances)
            entry['healthy_instances'] = sum(
                1 for i in instances if i.health_status == HealthStatus.PASSING
            )
            entry['discovery_health'] = catalog.service_health(name).value
            report[name] = entry
        return report
