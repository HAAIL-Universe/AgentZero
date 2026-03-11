"""
C139: Signal Processing
FFT, spectral analysis, convolution, correlation, digital filters, windowing.
New domain: frequency-domain analysis and signal manipulation.

Standalone -- no composition dependencies (pure math + complex arithmetic).
"""

import math
import cmath


# ============================================================
# Complex arithmetic helpers
# ============================================================

def _ensure_complex(x):
    """Convert to complex if not already."""
    if isinstance(x, complex):
        return x
    return complex(float(x), 0.0)


# ============================================================
# FFT Core (Cooley-Tukey radix-2 DIT)
# ============================================================

def fft(x):
    """Compute the Discrete Fourier Transform using Cooley-Tukey radix-2.

    Input length must be a power of 2. Returns list of complex values.
    """
    n = len(x)
    if n == 0:
        return []
    if n == 1:
        return [_ensure_complex(x[0])]

    # Pad to power of 2 if needed
    if n & (n - 1) != 0:
        raise ValueError(f"FFT length must be power of 2, got {n}")

    # Bit-reversal permutation
    result = [_ensure_complex(x[i]) for i in range(n)]
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            result[i], result[j] = result[j], result[i]

    # Butterfly stages
    length = 2
    while length <= n:
        angle = -2.0 * math.pi / length
        wn = cmath.exp(complex(0, angle))
        for start in range(0, n, length):
            w = complex(1, 0)
            for k in range(length // 2):
                u = result[start + k]
                v = w * result[start + k + length // 2]
                result[start + k] = u + v
                result[start + k + length // 2] = u - v
                w *= wn
        length *= 2

    return result


def ifft(X):
    """Compute the Inverse Discrete Fourier Transform.

    Returns list of complex values.
    """
    n = len(X)
    if n == 0:
        return []
    if n == 1:
        return [_ensure_complex(X[0])]

    # Conjugate, FFT, conjugate, divide by N
    conj = [x.conjugate() for x in X]
    result = fft(conj)
    return [x.conjugate() / n for x in result]


def rfft(x):
    """FFT of real-valued input. Returns only positive frequencies (n//2 + 1 values)."""
    X = fft(x)
    return X[:len(x) // 2 + 1]


def irfft(X, n=None):
    """Inverse FFT returning real values from half-spectrum."""
    if n is None:
        n = 2 * (len(X) - 1)
    # Reconstruct full spectrum from half
    full = list(X)
    for i in range(n // 2 + 1, n):
        full.append(X[n - i].conjugate())
    result = ifft(full)
    return [x.real for x in result]


def dft_naive(x):
    """Naive O(n^2) DFT for reference/testing."""
    n = len(x)
    X = []
    for k in range(n):
        s = complex(0, 0)
        for j in range(n):
            angle = -2.0 * math.pi * k * j / n
            s += _ensure_complex(x[j]) * cmath.exp(complex(0, angle))
        X.append(s)
    return X


# ============================================================
# Windowing Functions
# ============================================================

def hann_window(n):
    """Hann (raised cosine) window."""
    if n <= 1:
        return [1.0]
    return [0.5 * (1.0 - math.cos(2.0 * math.pi * i / (n - 1))) for i in range(n)]


def hamming_window(n):
    """Hamming window."""
    if n <= 1:
        return [1.0]
    return [0.54 - 0.46 * math.cos(2.0 * math.pi * i / (n - 1)) for i in range(n)]


def blackman_window(n):
    """Blackman window."""
    if n <= 1:
        return [1.0]
    return [
        0.42 - 0.5 * math.cos(2.0 * math.pi * i / (n - 1))
        + 0.08 * math.cos(4.0 * math.pi * i / (n - 1))
        for i in range(n)
    ]


def bartlett_window(n):
    """Bartlett (triangular) window."""
    if n <= 1:
        return [1.0]
    return [1.0 - abs(2.0 * i / (n - 1) - 1.0) for i in range(n)]


def rectangular_window(n):
    """Rectangular (no) window."""
    return [1.0] * n


def kaiser_window(n, beta=8.6):
    """Kaiser window with parameter beta."""
    if n <= 1:
        return [1.0]

    def _i0(x):
        """Modified Bessel function of the first kind, order 0."""
        s = 1.0
        term = 1.0
        for k in range(1, 50):
            term *= (x / 2.0) ** 2 / (k * k)
            s += term
            if abs(term) < 1e-15:
                break
        return s

    denom = _i0(beta)
    result = []
    for i in range(n):
        alpha = 2.0 * i / (n - 1) - 1.0
        result.append(_i0(beta * math.sqrt(max(0, 1.0 - alpha * alpha))) / denom)
    return result


def apply_window(signal, window):
    """Apply a window function to a signal."""
    if len(signal) != len(window):
        raise ValueError("Signal and window must have same length")
    return [s * w for s, w in zip(signal, window)]


# ============================================================
# Spectral Analysis
# ============================================================

def power_spectrum(x):
    """Compute power spectrum |X[k]|^2 / N."""
    X = fft(x)
    n = len(X)
    return [abs(xk) ** 2 / n for xk in X]


def power_spectral_density(x, fs=1.0, window=None):
    """Compute PSD using Welch-like single-segment approach.

    Args:
        x: signal
        fs: sampling frequency
        window: window function values (or None for Hann)

    Returns:
        (freqs, psd) tuple
    """
    n = len(x)
    if window is None:
        window = hann_window(n)

    xw = apply_window(x, window)
    X = fft(xw)

    # Window energy normalization
    S1 = sum(w ** 2 for w in window)

    # One-sided PSD
    n_freq = n // 2 + 1
    psd = []
    for k in range(n_freq):
        p = abs(X[k]) ** 2 / (fs * S1)
        if 0 < k < n - k:  # Not DC or Nyquist
            p *= 2.0
        psd.append(p)

    freqs = [k * fs / n for k in range(n_freq)]
    return freqs, psd


def magnitude_spectrum(x):
    """Compute magnitude spectrum |X[k]|."""
    X = fft(x)
    return [abs(xk) for xk in X]


def phase_spectrum(x):
    """Compute phase spectrum arg(X[k])."""
    X = fft(x)
    return [cmath.phase(xk) for xk in X]


def spectrogram(x, nperseg, noverlap=None, window_fn=None, fs=1.0):
    """Compute spectrogram via STFT.

    Args:
        x: signal
        nperseg: samples per segment (must be power of 2)
        noverlap: overlap between segments (default nperseg//2)
        window_fn: window function (default Hann)
        fs: sampling frequency

    Returns:
        (times, freqs, Sxx) where Sxx[t][f] is power at time t, freq f
    """
    n = len(x)
    if noverlap is None:
        noverlap = nperseg // 2
    if window_fn is None:
        window_fn = hann_window(nperseg)

    step = nperseg - noverlap
    n_segments = max(0, (n - nperseg) // step + 1)
    n_freq = nperseg // 2 + 1

    times = []
    freqs = [k * fs / nperseg for k in range(n_freq)]
    Sxx = []

    for i in range(n_segments):
        start = i * step
        segment = x[start:start + nperseg]
        xw = apply_window(segment, window_fn)
        X = fft(xw)

        S1 = sum(w ** 2 for w in window_fn)
        power = []
        for k in range(n_freq):
            p = abs(X[k]) ** 2 / (fs * S1)
            if 0 < k < nperseg - k:
                p *= 2.0
            power.append(p)

        Sxx.append(power)
        times.append((start + nperseg / 2) / fs)

    return times, freqs, Sxx


def cepstrum(x):
    """Compute real cepstrum of signal."""
    X = fft(x)
    log_mag = [math.log(max(abs(xk), 1e-300)) for xk in X]
    # Pad log_mag to complex for ifft
    log_complex = [complex(lm, 0) for lm in log_mag]
    c = ifft(log_complex)
    return [ci.real for ci in c]


# ============================================================
# Convolution and Correlation
# ============================================================

def convolve(a, b):
    """Linear convolution using FFT.

    Result length = len(a) + len(b) - 1.
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return []

    n_out = na + nb - 1
    # Pad to next power of 2
    n_fft = 1
    while n_fft < n_out:
        n_fft *= 2

    # Zero-pad
    ap = list(a) + [0.0] * (n_fft - na)
    bp = list(b) + [0.0] * (n_fft - nb)

    A = fft(ap)
    B = fft(bp)
    C = [a_k * b_k for a_k, b_k in zip(A, B)]
    c = ifft(C)

    return [ci.real for ci in c[:n_out]]


def correlate(a, b):
    """Cross-correlation using FFT.

    Result length = len(a) + len(b) - 1.
    Correlation at lag k = sum(a[i] * b[i+k]).
    """
    # correlate(a, b) = convolve(a, reverse(b))
    return convolve(a, list(reversed(b)))


def autocorrelate(x):
    """Autocorrelation of signal (correlation with itself)."""
    return correlate(x, x)


def convolve_direct(a, b):
    """Direct (time-domain) convolution O(n*m)."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return []
    n_out = na + nb - 1
    result = [0.0] * n_out
    for i in range(na):
        for j in range(nb):
            result[i + j] += a[i] * b[j]
    return result


# ============================================================
# Signal Generation
# ============================================================

def sine_wave(freq, duration, fs, amplitude=1.0, phase=0.0):
    """Generate a sine wave.

    Args:
        freq: frequency in Hz
        duration: duration in seconds
        fs: sampling frequency
        amplitude: peak amplitude
        phase: initial phase in radians
    """
    n = int(duration * fs)
    return [amplitude * math.sin(2.0 * math.pi * freq * i / fs + phase) for i in range(n)]


def cosine_wave(freq, duration, fs, amplitude=1.0, phase=0.0):
    """Generate a cosine wave."""
    n = int(duration * fs)
    return [amplitude * math.cos(2.0 * math.pi * freq * i / fs + phase) for i in range(n)]


def square_wave(freq, duration, fs, amplitude=1.0, duty=0.5):
    """Generate a square wave."""
    n = int(duration * fs)
    period = fs / freq
    result = []
    for i in range(n):
        t_in_period = (i % period) / period
        result.append(amplitude if t_in_period < duty else -amplitude)
    return result


def sawtooth_wave(freq, duration, fs, amplitude=1.0):
    """Generate a sawtooth wave."""
    n = int(duration * fs)
    period = fs / freq
    return [amplitude * (2.0 * (i % period) / period - 1.0) for i in range(n)]


def triangle_wave(freq, duration, fs, amplitude=1.0):
    """Generate a triangle wave."""
    n = int(duration * fs)
    period = fs / freq
    result = []
    for i in range(n):
        t = (i % period) / period
        result.append(amplitude * (4.0 * abs(t - 0.5) - 1.0))
    return result


def chirp(f0, f1, duration, fs, amplitude=1.0):
    """Generate a linear chirp (frequency sweep).

    Args:
        f0: start frequency
        f1: end frequency
        duration: duration in seconds
        fs: sampling frequency
    """
    n = int(duration * fs)
    result = []
    for i in range(n):
        t = i / fs
        freq_t = f0 + (f1 - f0) * t / duration
        phase = 2.0 * math.pi * (f0 * t + (f1 - f0) * t * t / (2.0 * duration))
        result.append(amplitude * math.sin(phase))
    return result


def impulse(n, position=0, amplitude=1.0):
    """Generate a unit impulse (Dirac delta)."""
    result = [0.0] * n
    if 0 <= position < n:
        result[position] = amplitude
    return result


def white_noise(n, amplitude=1.0, seed=None):
    """Generate white noise using a simple LCG PRNG."""
    if seed is None:
        seed = 42
    state = seed
    result = []
    for _ in range(n):
        # LCG: simple but reproducible
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        # Map to [-amplitude, amplitude]
        val = (state / 0x7FFFFFFF) * 2.0 * amplitude - amplitude
        result.append(val)
    return result


def step_function(n, position=0, amplitude=1.0):
    """Generate a step function (Heaviside)."""
    return [amplitude if i >= position else 0.0 for i in range(n)]


# ============================================================
# Digital Filters
# ============================================================

class FIRFilter:
    """Finite Impulse Response filter.

    y[n] = sum(b[k] * x[n-k] for k in range(len(b)))
    """

    def __init__(self, coefficients):
        self.b = list(coefficients)
        self.order = len(self.b) - 1

    def apply(self, x):
        """Apply filter to signal (using convolution)."""
        return convolve(x, self.b)[:len(x)]

    def frequency_response(self, n_points=512):
        """Compute frequency response H(e^jw).

        Returns (frequencies_normalized, magnitude, phase) where
        frequencies are in [0, pi].
        """
        # Pad coefficients to power of 2
        n_fft = 1
        while n_fft < max(n_points, len(self.b)):
            n_fft *= 2

        padded = list(self.b) + [0.0] * (n_fft - len(self.b))
        H = fft(padded)

        n_freq = n_fft // 2 + 1
        freqs = [math.pi * k / (n_fft // 2) for k in range(n_freq)]
        mags = [abs(H[k]) for k in range(n_freq)]
        phases = [cmath.phase(H[k]) for k in range(n_freq)]

        return freqs, mags, phases

    @staticmethod
    def moving_average(length):
        """Create a moving average filter."""
        return FIRFilter([1.0 / length] * length)

    @staticmethod
    def low_pass_sinc(cutoff, length, window_fn=None):
        """Design FIR low-pass filter using windowed sinc method.

        Args:
            cutoff: normalized cutoff frequency (0 to 1, where 1 = Nyquist)
            length: filter length (odd recommended)
            window_fn: window function values (default Hamming)
        """
        if window_fn is None:
            window_fn = hamming_window(length)

        fc = cutoff / 2.0  # cutoff in [0,1] where 1=Nyquist=0.5*fs, so fc in [0, 0.5]
        mid = (length - 1) / 2.0
        coeffs = []
        for i in range(length):
            if i == mid:
                coeffs.append(2.0 * fc)
            else:
                x = i - mid
                coeffs.append(math.sin(2.0 * math.pi * fc * x) / (math.pi * x))

        # Apply window
        coeffs = [c * w for c, w in zip(coeffs, window_fn)]

        # Normalize
        s = sum(coeffs)
        if abs(s) > 1e-10:
            coeffs = [c / s for c in coeffs]

        return FIRFilter(coeffs)

    @staticmethod
    def high_pass_sinc(cutoff, length, window_fn=None):
        """Design FIR high-pass filter (spectral inversion of low-pass)."""
        lp = FIRFilter.low_pass_sinc(cutoff, length, window_fn)
        mid = (length - 1) // 2
        hp_coeffs = [-c for c in lp.b]
        hp_coeffs[mid] += 1.0
        return FIRFilter(hp_coeffs)

    @staticmethod
    def band_pass_sinc(low_cutoff, high_cutoff, length, window_fn=None):
        """Design FIR band-pass filter."""
        lp1 = FIRFilter.low_pass_sinc(high_cutoff, length, window_fn)
        lp2 = FIRFilter.low_pass_sinc(low_cutoff, length, window_fn)
        bp_coeffs = [a - b for a, b in zip(lp1.b, lp2.b)]
        return FIRFilter(bp_coeffs)


class IIRFilter:
    """Infinite Impulse Response filter.

    y[n] = sum(b[k]*x[n-k]) - sum(a[k]*y[n-k]) for k >= 1
    a[0] is assumed to be 1 (normalized).
    """

    def __init__(self, b, a):
        self.b = list(b)
        self.a = list(a)
        # Normalize so a[0] = 1
        if abs(self.a[0]) > 1e-15 and self.a[0] != 1.0:
            norm = self.a[0]
            self.b = [bi / norm for bi in self.b]
            self.a = [ai / norm for ai in self.a]

    def apply(self, x):
        """Apply IIR filter using Direct Form I."""
        n = len(x)
        y = [0.0] * n
        nb = len(self.b)
        na = len(self.a)

        for i in range(n):
            # Feedforward
            s = 0.0
            for k in range(nb):
                if i - k >= 0:
                    s += self.b[k] * x[i - k]
            # Feedback
            for k in range(1, na):
                if i - k >= 0:
                    s -= self.a[k] * y[i - k]
            y[i] = s

        return y

    def frequency_response(self, n_points=512):
        """Compute frequency response H(e^jw) = B(e^jw) / A(e^jw)."""
        freqs = []
        mags = []
        phases = []

        for i in range(n_points):
            w = math.pi * i / (n_points - 1)
            # Evaluate B(e^jw) and A(e^jw)
            B = complex(0, 0)
            for k, bk in enumerate(self.b):
                B += bk * cmath.exp(complex(0, -w * k))
            A = complex(0, 0)
            for k, ak in enumerate(self.a):
                A += ak * cmath.exp(complex(0, -w * k))

            H = B / A if abs(A) > 1e-15 else complex(0, 0)
            freqs.append(w)
            mags.append(abs(H))
            phases.append(cmath.phase(H))

        return freqs, mags, phases

    @staticmethod
    def butterworth_lowpass(cutoff, order=2):
        """Design Butterworth low-pass IIR filter (bilinear transform).

        Args:
            cutoff: normalized cutoff frequency (0 to 1, where 1 = Nyquist)
            order: filter order (1-8)
        """
        # Pre-warp cutoff
        wc = math.tan(math.pi * cutoff / 2.0)

        # Analog Butterworth poles
        poles = []
        for k in range(order):
            angle = math.pi * (2 * k + order + 1) / (2 * order)
            poles.append(wc * cmath.exp(complex(0, angle)))

        # Bilinear transform: s = 2*(z-1)/(z+1)
        # Map each analog pole to digital
        z_poles = []
        for p in poles:
            z_poles.append((2.0 + p) / (2.0 - p))

        # Build polynomial from roots
        def poly_from_roots(roots):
            coeffs = [complex(1, 0)]
            for r in roots:
                new_coeffs = [complex(0, 0)] * (len(coeffs) + 1)
                for i, c in enumerate(coeffs):
                    new_coeffs[i] += c
                    new_coeffs[i + 1] -= c * r
                coeffs = new_coeffs
            return [c.real for c in coeffs]

        a = poly_from_roots(z_poles)

        # Gain at DC: H(z=1) = 1
        # b is all-zero at z = -1 with gain matching
        z_zeros = [complex(-1, 0)] * order
        b = poly_from_roots(z_zeros)

        # Normalize gain at DC (z=1)
        gain_a = sum(a)
        gain_b = sum(b)

        if abs(gain_b) > 1e-15:
            scale = gain_a / gain_b
            b = [bi * scale for bi in b]

        return IIRFilter(b, a)

    @staticmethod
    def butterworth_highpass(cutoff, order=2):
        """Design Butterworth high-pass IIR filter."""
        # Pre-warp
        wc = math.tan(math.pi * cutoff / 2.0)

        # HP poles: transform LP pole p -> wc^2/p
        # But simpler: use LP->HP transform in analog then bilinear
        poles = []
        for k in range(order):
            angle = math.pi * (2 * k + order + 1) / (2 * order)
            p_lp = cmath.exp(complex(0, angle))  # Unit Butterworth pole
            p_hp = wc / p_lp  # LP to HP frequency transform
            poles.append(p_hp)

        z_poles = []
        for p in poles:
            z_poles.append((2.0 + p) / (2.0 - p))

        def poly_from_roots(roots):
            coeffs = [complex(1, 0)]
            for r in roots:
                new_coeffs = [complex(0, 0)] * (len(coeffs) + 1)
                for i, c in enumerate(coeffs):
                    new_coeffs[i] += c
                    new_coeffs[i + 1] -= c * r
                coeffs = new_coeffs
            return [c.real for c in coeffs]

        a = poly_from_roots(z_poles)

        # Zeros at z = 1 (DC) for high-pass
        z_zeros = [complex(1, 0)] * order
        b = poly_from_roots(z_zeros)

        # Normalize gain at Nyquist (z=-1): H(z=-1) = 1
        gain_a = sum(ai * ((-1) ** i) for i, ai in enumerate(a))
        gain_b = sum(bi * ((-1) ** i) for i, bi in enumerate(b))

        if abs(gain_b) > 1e-15:
            scale = gain_a / gain_b
            b = [bi * scale for bi in b]

        return IIRFilter(b, a)

    @staticmethod
    def first_order_lowpass(alpha):
        """Simple first-order IIR low-pass: y[n] = alpha*x[n] + (1-alpha)*y[n-1]."""
        return IIRFilter([alpha], [1.0, -(1.0 - alpha)])

    @staticmethod
    def first_order_highpass(alpha):
        """Simple first-order IIR high-pass."""
        return IIRFilter([(1.0 + alpha) / 2.0, -(1.0 + alpha) / 2.0], [1.0, -alpha])

    @staticmethod
    def notch(freq, Q=10.0):
        """Design a notch (band-reject) filter.

        Args:
            freq: normalized center frequency (0 to 1)
            Q: quality factor (higher = narrower notch)
        """
        w0 = math.pi * freq
        bw = w0 / Q
        alpha = math.sin(bw) / 2.0 if bw > 0 else 0.01

        cos_w0 = math.cos(w0)
        b = [1.0, -2.0 * cos_w0, 1.0]
        a = [1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha]

        return IIRFilter(b, a)


# ============================================================
# Utility Functions
# ============================================================

def zero_pad(x, length):
    """Zero-pad signal to specified length."""
    if len(x) >= length:
        return list(x)
    return list(x) + [0.0] * (length - len(x))


def next_power_of_2(n):
    """Find the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


def db(x, ref=1.0):
    """Convert to decibels: 20 * log10(x / ref)."""
    if x <= 0:
        return -math.inf
    return 20.0 * math.log10(x / ref)


def db_power(x, ref=1.0):
    """Convert power to decibels: 10 * log10(x / ref)."""
    if x <= 0:
        return -math.inf
    return 10.0 * math.log10(x / ref)


def normalize(x):
    """Normalize signal to [-1, 1] range."""
    mx = max(abs(v) for v in x)
    if mx == 0:
        return list(x)
    return [v / mx for v in x]


def rms(x):
    """Root mean square of signal."""
    if len(x) == 0:
        return 0.0
    return math.sqrt(sum(v * v for v in x) / len(x))


def energy(x):
    """Signal energy: sum of squares."""
    return sum(v * v for v in x)


def peak_frequency(x, fs=1.0):
    """Find the dominant frequency in a signal."""
    X = fft(x)
    n = len(X)
    # Only look at positive frequencies
    mags = [abs(X[k]) for k in range(1, n // 2)]
    peak_bin = mags.index(max(mags)) + 1
    return peak_bin * fs / n


def snr(signal, noise):
    """Signal-to-noise ratio in dB."""
    sig_power = energy(signal) / len(signal)
    noise_power = energy(noise) / len(noise)
    if noise_power == 0:
        return math.inf
    return 10.0 * math.log10(sig_power / noise_power)


def resample(x, factor):
    """Simple resampling by integer factor.

    factor > 1: upsample (insert zeros + lowpass)
    factor < 1: not supported (use decimation)
    factor == 1: identity
    """
    if factor == 1:
        return list(x)
    if factor <= 0:
        raise ValueError("Resample factor must be positive")

    if isinstance(factor, int) and factor > 1:
        # Upsample: insert zeros
        upsampled = []
        for v in x:
            upsampled.append(v)
            upsampled.extend([0.0] * (factor - 1))

        # Low-pass filter at 1/factor of Nyquist
        filt = FIRFilter.low_pass_sinc(1.0 / factor, factor * 4 + 1)
        filtered = filt.apply(upsampled)

        # Scale by factor to maintain amplitude
        return [v * factor for v in filtered]

    raise ValueError("Only integer upsampling supported")


def decimate(x, factor):
    """Decimate signal by integer factor (low-pass then downsample)."""
    if factor <= 0 or not isinstance(factor, int):
        raise ValueError("Decimation factor must be positive integer")
    if factor == 1:
        return list(x)

    # Anti-aliasing low-pass filter
    filt = FIRFilter.low_pass_sinc(1.0 / factor, factor * 4 + 1)
    filtered = filt.apply(x)

    # Downsample
    return [filtered[i] for i in range(0, len(filtered), factor)]


def hilbert_transform(x):
    """Compute the analytic signal using the Hilbert transform.

    Returns list of complex values where real part = original signal,
    imaginary part = Hilbert transform.
    """
    n = len(x)
    X = fft(x)

    # Double positive frequencies, zero negative frequencies
    H = [complex(0, 0)] * n
    H[0] = X[0]
    if n > 1:
        H[n // 2] = X[n // 2]
    for k in range(1, n // 2):
        H[k] = 2.0 * X[k]

    return ifft(H)


def envelope(x):
    """Compute the amplitude envelope of a signal via Hilbert transform."""
    analytic = hilbert_transform(x)
    return [abs(z) for z in analytic]


def instantaneous_frequency(x, fs=1.0):
    """Compute instantaneous frequency from analytic signal."""
    analytic = hilbert_transform(x)
    n = len(analytic)
    if n < 2:
        return []

    phase = [cmath.phase(z) for z in analytic]
    freq = []
    for i in range(1, n):
        dp = phase[i] - phase[i - 1]
        # Unwrap
        while dp > math.pi:
            dp -= 2.0 * math.pi
        while dp < -math.pi:
            dp += 2.0 * math.pi
        freq.append(dp * fs / (2.0 * math.pi))

    return freq


def goertzel(x, target_freq, fs=1.0):
    """Goertzel algorithm: efficient single-frequency DFT.

    Returns the complex DFT coefficient at the target frequency.
    More efficient than full FFT when only one frequency is needed.
    """
    n = len(x)
    k = round(target_freq * n / fs)
    w = 2.0 * math.pi * k / n
    coeff = 2.0 * math.cos(w)

    s0 = 0.0
    s1 = 0.0
    s2 = 0.0

    for sample in x:
        s0 = sample + coeff * s1 - s2
        s2 = s1
        s1 = s0

    # Final computation
    real = s1 - s2 * math.cos(w)
    imag = s2 * math.sin(w)

    return complex(real, imag)
