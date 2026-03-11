"""
C134: Signal Processing

Composing C131 FFT -- advanced signal processing toolkit covering:
- FIR filter design (windowed sinc, Parks-McClellan approximation)
- IIR filter design (Butterworth, Chebyshev Type I/II)
- Filter application (direct form, zero-phase filtering)
- Resampling (decimation, interpolation, rational resampling)
- Spectral estimation (Welch, Bartlett, Blackman-Tukey, MUSIC)
- Windowing (Tukey, flat-top, Gaussian, Bartlett-Hann)
- Convolution and correlation enhancements
- Signal generation (chirp, square, sawtooth, noise)
- Envelope detection, analytic signal
- Filter analysis (frequency response, group delay, poles/zeros)
"""

import math
import cmath
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'C131_fft'))
from fft import (
    fft, ifft, rfft, irfft, _next_power_of_2,
    window_hamming, window_hann, window_blackman,
    linear_convolution, magnitude_spectrum, frequency_bins,
)


# ============================================================
# Signal Generation
# ============================================================

def generate_sine(freq, duration, fs, amplitude=1.0, phase=0.0):
    """Generate a sine wave signal."""
    n_samples = int(duration * fs)
    return [amplitude * math.sin(2 * math.pi * freq * i / fs + phase)
            for i in range(n_samples)]


def generate_cosine(freq, duration, fs, amplitude=1.0, phase=0.0):
    """Generate a cosine wave signal."""
    n_samples = int(duration * fs)
    return [amplitude * math.cos(2 * math.pi * freq * i / fs + phase)
            for i in range(n_samples)]


def generate_chirp(f0, f1, duration, fs, method='linear'):
    """Generate a chirp signal (frequency sweep).

    method: 'linear' (linear frequency sweep) or 'exponential'
    """
    n_samples = int(duration * fs)
    T = duration
    signal = []
    for i in range(n_samples):
        t = i / fs
        if method == 'linear':
            phase = 2 * math.pi * (f0 * t + (f1 - f0) * t * t / (2 * T))
        elif method == 'exponential':
            if f0 <= 0:
                raise ValueError("f0 must be positive for exponential chirp")
            k = (f1 / f0) ** (1.0 / T)
            phase = 2 * math.pi * f0 * (k ** t - 1) / math.log(k)
        else:
            raise ValueError(f"Unknown chirp method: {method}")
        signal.append(math.sin(phase))
    return signal


def generate_square(freq, duration, fs, duty=0.5):
    """Generate a square wave with given duty cycle."""
    n_samples = int(duration * fs)
    period = fs / freq
    signal = []
    for i in range(n_samples):
        pos = (i % period) / period
        signal.append(1.0 if pos < duty else -1.0)
    return signal


def generate_sawtooth(freq, duration, fs):
    """Generate a sawtooth wave."""
    n_samples = int(duration * fs)
    period = fs / freq
    signal = []
    for i in range(n_samples):
        pos = (i % period) / period
        signal.append(2.0 * pos - 1.0)
    return signal


def generate_triangle(freq, duration, fs):
    """Generate a triangle wave."""
    n_samples = int(duration * fs)
    period = fs / freq
    signal = []
    for i in range(n_samples):
        pos = (i % period) / period
        if pos < 0.5:
            signal.append(4.0 * pos - 1.0)
        else:
            signal.append(3.0 - 4.0 * pos)
    return signal


def generate_white_noise(n_samples, seed=None):
    """Generate white noise using LCG PRNG."""
    if seed is None:
        seed = 42
    state = seed
    signal = []
    for _ in range(n_samples):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        signal.append((state / 0x7FFFFFFF) * 2.0 - 1.0)
    return signal


def generate_impulse(n_samples, delay=0, amplitude=1.0):
    """Generate an impulse (delta) signal."""
    signal = [0.0] * n_samples
    if 0 <= delay < n_samples:
        signal[delay] = amplitude
    return signal


def generate_step(n_samples, delay=0, amplitude=1.0):
    """Generate a step (Heaviside) signal."""
    signal = [0.0] * n_samples
    for i in range(delay, n_samples):
        signal[i] = amplitude
    return signal


# ============================================================
# Additional Window Functions
# ============================================================

def window_tukey(n, alpha=0.5):
    """Tukey (tapered cosine) window. alpha=0 is rectangular, alpha=1 is Hann."""
    if n <= 1:
        return [1.0] * n
    w = []
    for i in range(n):
        if alpha <= 0:
            w.append(1.0)
        elif i < alpha * (n - 1) / 2:
            w.append(0.5 * (1 - math.cos(2 * math.pi * i / (alpha * (n - 1)))))
        elif i <= (n - 1) * (1 - alpha / 2):
            w.append(1.0)
        else:
            w.append(0.5 * (1 - math.cos(2 * math.pi * (n - 1 - i) / (alpha * (n - 1)))))
    return w


def window_flattop(n):
    """Flat-top window -- good for amplitude accuracy."""
    if n <= 1:
        return [1.0] * n
    a0, a1, a2, a3, a4 = 0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368
    w = []
    for i in range(n):
        val = (a0 - a1 * math.cos(2 * math.pi * i / (n - 1))
               + a2 * math.cos(4 * math.pi * i / (n - 1))
               - a3 * math.cos(6 * math.pi * i / (n - 1))
               + a4 * math.cos(8 * math.pi * i / (n - 1)))
        w.append(val)
    return w


def window_gaussian(n, sigma=0.4):
    """Gaussian window. sigma is relative to half-width."""
    if n <= 1:
        return [1.0] * n
    w = []
    center = (n - 1) / 2.0
    for i in range(n):
        w.append(math.exp(-0.5 * ((i - center) / (sigma * center)) ** 2))
    return w


def window_bartlett_hann(n):
    """Bartlett-Hann window."""
    if n <= 1:
        return [1.0] * n
    w = []
    for i in range(n):
        fac = i / (n - 1) - 0.5
        w.append(0.62 - 0.48 * abs(fac) + 0.38 * math.cos(2 * math.pi * fac))
    return w


# ============================================================
# FIR Filter Design
# ============================================================

def _sinc(x):
    """Normalized sinc function: sin(pi*x) / (pi*x)."""
    if abs(x) < 1e-15:
        return 1.0
    return math.sin(math.pi * x) / (math.pi * x)


def firwin_lowpass(numtaps, cutoff, fs=2.0, window_fn=None):
    """Design a lowpass FIR filter using the windowed-sinc method.

    numtaps: filter length (odd recommended)
    cutoff: cutoff frequency
    fs: sampling frequency
    window_fn: window function (default: Hamming)
    """
    if window_fn is None:
        window_fn = window_hamming
    fc = cutoff / (fs / 2)  # normalized cutoff [0, 1]
    if fc <= 0 or fc >= 1:
        raise ValueError("Cutoff must be between 0 and fs/2 (exclusive)")

    M = numtaps - 1
    h = []
    for i in range(numtaps):
        n = i - M / 2
        h.append(fc * _sinc(fc * n))

    # Apply window
    w = window_fn(numtaps)
    h = [h[i] * w[i] for i in range(numtaps)]

    # Normalize gain at DC to 1
    total = sum(h)
    if abs(total) > 1e-15:
        h = [x / total for x in h]

    return h


def firwin_highpass(numtaps, cutoff, fs=2.0, window_fn=None):
    """Design a highpass FIR filter using spectral inversion."""
    if numtaps % 2 == 0:
        numtaps += 1  # highpass needs odd length
    lp = firwin_lowpass(numtaps, cutoff, fs, window_fn)
    hp = [-x for x in lp]
    hp[numtaps // 2] += 1.0
    return hp


def firwin_bandpass(numtaps, low, high, fs=2.0, window_fn=None):
    """Design a bandpass FIR filter."""
    lp_high = firwin_lowpass(numtaps, high, fs, window_fn)
    lp_low = firwin_lowpass(numtaps, low, fs, window_fn)
    return [lp_high[i] - lp_low[i] for i in range(numtaps)]


def firwin_bandstop(numtaps, low, high, fs=2.0, window_fn=None):
    """Design a bandstop (notch) FIR filter."""
    bp = firwin_bandpass(numtaps, low, high, fs, window_fn)
    bs = [-x for x in bp]
    bs[numtaps // 2] += 1.0
    return bs


# ============================================================
# IIR Filter Design
# ============================================================

def _bilinear_transform(poles_s, zeros_s, gain_s, fs):
    """Bilinear transform from s-domain to z-domain.

    Maps s-plane poles/zeros to z-plane using s = 2*fs*(z-1)/(z+1).
    Returns (poles_z, zeros_z, gain_z).
    """
    T = 1.0 / fs
    poles_z = []
    zeros_z = []

    for p in poles_s:
        z = (1 + p * T / 2) / (1 - p * T / 2)
        poles_z.append(z)

    for z_s in zeros_s:
        z = (1 + z_s * T / 2) / (1 - z_s * T / 2)
        zeros_z.append(z)

    # Add zeros at z = -1 for each excess pole
    n_extra = len(poles_s) - len(zeros_s)
    for _ in range(n_extra):
        zeros_z.append(-1.0 + 0j)

    # Compute gain adjustment
    gain_z = gain_s
    for p in poles_s:
        gain_z *= abs(1 - p * T / 2)
    for z_s in zeros_s:
        gain_z /= abs(1 - z_s * T / 2)
    # Account for extra zeros at -1
    # gain_z doesn't need further adjustment for extra zeros at nyquist

    return poles_z, zeros_z, gain_z


def _zpk_to_ba(zeros, poles, gain):
    """Convert zero-pole-gain form to transfer function coefficients (b, a).

    H(z) = gain * prod(z - zeros) / prod(z - poles)
    Returns (b, a) where b = numerator, a = denominator coefficients.
    """
    # Build polynomial from roots
    def poly_from_roots(roots):
        coeffs = [1.0 + 0j]
        for r in roots:
            new_coeffs = [0.0 + 0j] * (len(coeffs) + 1)
            for i, c in enumerate(coeffs):
                new_coeffs[i] += c
                new_coeffs[i + 1] -= c * r
            coeffs = new_coeffs
        return [c.real for c in coeffs]

    b = poly_from_roots(zeros)
    a = poly_from_roots(poles)

    b = [gain * x for x in b]

    # Normalize so a[0] = 1
    if abs(a[0]) > 1e-15:
        norm = a[0]
        a = [x / norm for x in a]
        b = [x / norm for x in b]

    return b, a


def butter_lowpass(order, cutoff, fs):
    """Design a Butterworth lowpass IIR filter.

    Returns (b, a) transfer function coefficients.
    """
    # Analog prototype: Butterworth poles on unit circle in left half s-plane
    wc = 2 * math.pi * cutoff

    # Pre-warp for bilinear transform
    wc_warped = 2 * fs * math.tan(math.pi * cutoff / fs)

    # Analog Butterworth poles
    poles_s = []
    for k in range(order):
        theta = math.pi * (2 * k + order + 1) / (2 * order)
        p = wc_warped * complex(math.cos(theta), math.sin(theta))
        poles_s.append(p)

    zeros_s = []  # Butterworth has no finite zeros
    gain_s = wc_warped ** order

    # Bilinear transform
    poles_z, zeros_z, gain_z = _bilinear_transform(poles_s, zeros_s, gain_s, fs)

    # Normalize gain at DC (z=1)
    num_at_dc = gain_z
    for z in zeros_z:
        num_at_dc *= abs(1.0 - z)
    den_at_dc = 1.0
    for p in poles_z:
        den_at_dc *= abs(1.0 - p)
    if abs(den_at_dc) > 1e-15:
        gain_z = gain_z * den_at_dc / num_at_dc if abs(num_at_dc) > 1e-15 else gain_z

    return _zpk_to_ba(zeros_z, poles_z, gain_z)


def butter_highpass(order, cutoff, fs):
    """Design a Butterworth highpass IIR filter.

    Returns (b, a) transfer function coefficients.
    """
    # Pre-warp
    wc_warped = 2 * fs * math.tan(math.pi * cutoff / fs)

    # Start with lowpass prototype normalized to wc=1
    poles_proto = []
    for k in range(order):
        theta = math.pi * (2 * k + order + 1) / (2 * order)
        poles_proto.append(complex(math.cos(theta), math.sin(theta)))

    # LP to HP transform: s -> wc^2 / s
    poles_s = [wc_warped / p for p in poles_proto]
    zeros_s = [0.0 + 0j] * order  # HP has zeros at origin
    gain_s = 1.0
    for p in poles_proto:
        gain_s *= (-p).real if (-p).imag == 0 else abs(-p)
    # Adjust gain: each pole contributes 1/p factor
    gain_s = 1.0

    # Bilinear transform
    poles_z, zeros_z, gain_z = _bilinear_transform(poles_s, zeros_s, gain_s, fs)

    # Normalize gain at Nyquist (z=-1) for highpass
    num_at_ny = gain_z
    for z in zeros_z:
        num_at_ny *= abs(-1.0 - z)
    den_at_ny = 1.0
    for p in poles_z:
        den_at_ny *= abs(-1.0 - p)
    if abs(num_at_ny) > 1e-15:
        gain_z = gain_z * den_at_ny / num_at_ny

    return _zpk_to_ba(zeros_z, poles_z, gain_z)


def cheby1_lowpass(order, ripple_db, cutoff, fs):
    """Design a Chebyshev Type I lowpass IIR filter.

    ripple_db: maximum ripple in passband (dB)
    Returns (b, a) transfer function coefficients.
    """
    eps = math.sqrt(10 ** (ripple_db / 10) - 1)

    # Pre-warp
    wc_warped = 2 * fs * math.tan(math.pi * cutoff / fs)

    # Chebyshev Type I poles
    poles_s = []
    for k in range(order):
        theta = math.pi * (2 * k + 1) / (2 * order)
        sigma = -wc_warped * math.sinh(math.asinh(1 / eps) / order) * math.sin(theta)
        omega = wc_warped * math.cosh(math.asinh(1 / eps) / order) * math.cos(theta)
        poles_s.append(complex(sigma, omega))

    zeros_s = []

    # Gain
    gain_s = wc_warped ** order
    for p in poles_s:
        gain_s = abs(gain_s)
    gain_product = 1.0
    for p in poles_s:
        gain_product *= abs(p)
    gain_s = gain_product
    if order % 2 == 0:
        gain_s /= math.sqrt(1 + eps ** 2)

    poles_z, zeros_z, gain_z = _bilinear_transform(poles_s, zeros_s, gain_s, fs)

    # Normalize at DC
    num_dc = gain_z
    for z in zeros_z:
        num_dc *= abs(1.0 - z)
    den_dc = 1.0
    for p in poles_z:
        den_dc *= abs(1.0 - p)
    if abs(num_dc) > 1e-15:
        gain_z = gain_z * den_dc / num_dc

    return _zpk_to_ba(zeros_z, poles_z, gain_z)


# ============================================================
# Filter Application
# ============================================================

def lfilter(b, a, x):
    """Apply IIR/FIR filter using direct form II transposed.

    b: numerator coefficients
    a: denominator coefficients (a[0] should be 1)
    x: input signal
    Returns filtered signal.
    """
    n = len(x)
    nb = len(b)
    na = len(a)

    # Normalize
    if abs(a[0] - 1.0) > 1e-15:
        norm = a[0]
        b = [bi / norm for bi in b]
        a = [ai / norm for ai in a]

    # Direct form II transposed
    order = max(nb, na)
    d = [0.0] * order  # delay line
    y = []

    for i in range(n):
        xi = x[i]
        yi = b[0] * xi + d[0]
        y.append(yi)

        for j in range(1, order - 1):
            bj = b[j] if j < nb else 0.0
            aj = a[j] if j < na else 0.0
            d[j - 1] = bj * xi - aj * yi + d[j]

        if order > 1:
            bj = b[order - 1] if order - 1 < nb else 0.0
            aj = a[order - 1] if order - 1 < na else 0.0
            d[order - 2] = bj * xi - aj * yi

    return y


def filtfilt(b, a, x):
    """Zero-phase filtering: apply filter forward, then backward.

    Eliminates phase distortion by running the filter twice.
    """
    # Forward pass
    y_fwd = lfilter(b, a, x)
    # Reverse
    y_fwd_rev = list(reversed(y_fwd))
    # Backward pass
    y_bwd = lfilter(b, a, y_fwd_rev)
    # Reverse again
    return list(reversed(y_bwd))


def fir_filter(h, x):
    """Apply FIR filter h to signal x using convolution."""
    n = len(x)
    m = len(h)
    y = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(m):
            if i - j >= 0:
                s += h[j] * x[i - j]
        y[i] = s
    return y


def fft_convolve(a, b):
    """Fast convolution using FFT (equivalent to linear convolution)."""
    n = len(a) + len(b) - 1
    size = _next_power_of_2(n)

    a_padded = list(a) + [0.0] * (size - len(a))
    b_padded = list(b) + [0.0] * (size - len(b))

    A = fft(a_padded)
    B = fft(b_padded)
    C = [A[i] * B[i] for i in range(size)]
    c = ifft(C)

    return [c[i].real for i in range(n)]


def overlap_add(h, x, block_size=None):
    """Overlap-add method for long signal convolution with short filter."""
    M = len(h)
    if block_size is None:
        block_size = max(M, 256)

    N = block_size + M - 1
    fft_size = _next_power_of_2(N)

    # FFT of filter (padded)
    h_padded = list(h) + [0.0] * (fft_size - M)
    H = fft(h_padded)

    out_len = len(x) + M - 1
    y = [0.0] * out_len

    pos = 0
    while pos < len(x):
        block = x[pos:pos + block_size]
        if len(block) < block_size:
            block = list(block) + [0.0] * (block_size - len(block))

        # Pad and FFT
        block_padded = list(block) + [0.0] * (fft_size - block_size)
        X = fft(block_padded)

        # Multiply and IFFT
        Y = [X[i] * H[i] for i in range(fft_size)]
        yy = ifft(Y)

        # Overlap-add
        for i in range(min(N, out_len - pos)):
            y[pos + i] += yy[i].real

        pos += block_size

    return y


# ============================================================
# Filter Analysis
# ============================================================

def freqz(b, a=None, n_points=512, fs=None):
    """Compute frequency response of a digital filter.

    Returns (frequencies, H) where H is complex frequency response.
    """
    if a is None:
        a = [1.0]
    if fs is None:
        fs = 2 * math.pi

    freqs = []
    H = []

    for k in range(n_points):
        w = math.pi * k / n_points
        z = cmath.exp(1j * w)

        # Evaluate numerator
        num = 0.0 + 0j
        for i, bi in enumerate(b):
            num += bi * z ** (-i)

        # Evaluate denominator
        den = 0.0 + 0j
        for i, ai in enumerate(a):
            den += ai * z ** (-i)

        if abs(den) > 1e-15:
            H.append(num / den)
        else:
            H.append(complex(float('inf'), 0))

        freqs.append(w * fs / (2 * math.pi))

    return freqs, H


def group_delay(b, a=None, n_points=512):
    """Compute group delay of a digital filter.

    Returns (frequencies, delays) where delay is in samples.
    """
    if a is None:
        a = [1.0]

    freqs = []
    delays = []

    for k in range(n_points):
        w = math.pi * k / n_points
        z = cmath.exp(1j * w)

        # Numerator and its derivative
        num = 0.0 + 0j
        num_d = 0.0 + 0j
        for i, bi in enumerate(b):
            num += bi * z ** (-i)
            num_d += -i * bi * z ** (-i)

        # Denominator and its derivative
        den = 0.0 + 0j
        den_d = 0.0 + 0j
        for i, ai in enumerate(a):
            den += ai * z ** (-i)
            den_d += -i * ai * z ** (-i)

        if abs(num) > 1e-15 and abs(den) > 1e-15:
            # Group delay = -d/dw arg(H) = Re{z * H'/H}
            # where H' is dH/dz
            # tau = Re{(num_d/num - den_d/den) * z / (-1j)}
            gd_num = (num_d * den - num * den_d) / (den * den)
            tau = -(gd_num / (num / den) * z / (-1j)).real
            delays.append(tau)
        else:
            delays.append(0.0)

        freqs.append(w)

    return freqs, delays


# ============================================================
# Spectral Estimation
# ============================================================

def welch(x, nperseg=256, noverlap=None, fs=1.0, window_fn=None):
    """Welch's method for power spectral density estimation.

    Divides signal into overlapping segments, windows each,
    computes periodogram, and averages.

    Returns (frequencies, psd).
    """
    if window_fn is None:
        window_fn = window_hann
    if noverlap is None:
        noverlap = nperseg // 2

    n = len(x)
    w = window_fn(nperseg)

    # Window normalization
    win_sum_sq = sum(wi * wi for wi in w)

    step = nperseg - noverlap
    n_segments = 0

    nfft = nperseg
    psd = [0.0] * (nfft // 2 + 1)

    pos = 0
    while pos + nperseg <= n:
        segment = x[pos:pos + nperseg]
        # Apply window
        windowed = [segment[i] * w[i] for i in range(nperseg)]

        # FFT
        X = fft(windowed + [0.0] * (nfft - nperseg) if nfft > nperseg else windowed)

        # Periodogram (one-sided)
        for k in range(nfft // 2 + 1):
            psd[k] += abs(X[k]) ** 2

        n_segments += 1
        pos += step

    if n_segments == 0:
        return list(range(nfft // 2 + 1)), psd

    # Average and normalize
    scale = 1.0 / (fs * win_sum_sq * n_segments)
    psd = [p * scale for p in psd]
    # Double one-sided (except DC and Nyquist)
    for k in range(1, nfft // 2):
        psd[k] *= 2

    freqs = [k * fs / nfft for k in range(nfft // 2 + 1)]
    return freqs, psd


def bartlett_method(x, nperseg=256, fs=1.0):
    """Bartlett's method: non-overlapping segment averaging (Welch with no overlap, rectangular window)."""
    def window_rect(n):
        return [1.0] * n
    return welch(x, nperseg=nperseg, noverlap=0, fs=fs, window_fn=window_rect)


def blackman_tukey(x, max_lag=None, fs=1.0, window_fn=None):
    """Blackman-Tukey spectral estimation.

    Estimates PSD via windowed autocorrelation and FFT.
    """
    if window_fn is None:
        window_fn = window_hamming

    n = len(x)
    if max_lag is None:
        max_lag = min(n - 1, n // 2)

    # Compute autocorrelation
    rxx = []
    mean_x = sum(x) / n
    x_centered = [xi - mean_x for xi in x]
    for lag in range(max_lag + 1):
        s = 0.0
        for i in range(n - lag):
            s += x_centered[i] * x_centered[i + lag]
        rxx.append(s / n)

    # Window the autocorrelation
    w = window_fn(2 * max_lag + 1)
    # Use only the second half of the window for positive lags
    w_half = w[max_lag:]

    # Build full (symmetric) windowed autocorrelation for FFT
    nfft = _next_power_of_2(2 * max_lag + 1)
    r_windowed = [0.0] * nfft
    r_windowed[0] = rxx[0] * w_half[0]
    for lag in range(1, max_lag + 1):
        r_windowed[lag] = rxx[lag] * w_half[lag]
        r_windowed[nfft - lag] = rxx[lag] * w_half[lag]

    # FFT to get PSD
    R = fft(r_windowed)

    n_out = nfft // 2 + 1
    psd = [R[k].real / fs for k in range(n_out)]
    # Ensure non-negative
    psd = [max(0.0, p) for p in psd]

    freqs = [k * fs / nfft for k in range(n_out)]
    return freqs, psd


def periodogram(x, fs=1.0, window_fn=None):
    """Simple periodogram PSD estimate.

    Returns (frequencies, psd).
    """
    n = len(x)
    if window_fn is not None:
        w = window_fn(n)
        x_w = [x[i] * w[i] for i in range(n)]
        win_sum_sq = sum(wi * wi for wi in w)
    else:
        x_w = list(x)
        win_sum_sq = float(n)

    X = fft(x_w)
    n_out = n // 2 + 1

    psd = [abs(X[k]) ** 2 / (fs * win_sum_sq) for k in range(n_out)]
    # One-sided doubling
    for k in range(1, n // 2):
        psd[k] *= 2

    freqs = [k * fs / n for k in range(n_out)]
    return freqs, psd


# ============================================================
# Resampling
# ============================================================

def decimate(x, factor, filter_order=8):
    """Decimate signal by integer factor with anti-aliasing filter.

    Applies lowpass filter then downsamples.
    """
    if factor < 1:
        raise ValueError("Decimation factor must be >= 1")
    if factor == 1:
        return list(x)

    # Anti-aliasing lowpass filter
    cutoff = 0.8 / factor  # slightly below Nyquist of decimated signal
    h = firwin_lowpass(filter_order * factor + 1, cutoff, fs=2.0)

    # Apply filter
    filtered = fir_filter(h, x)

    # Compensate for filter delay
    delay = len(h) // 2

    # Downsample
    result = []
    for i in range(0, len(x), factor):
        idx = i + delay
        if 0 <= idx < len(filtered):
            result.append(filtered[idx])
        elif i < len(x):
            result.append(filtered[min(i, len(filtered) - 1)])

    return result


def interpolate(x, factor, filter_order=4):
    """Interpolate signal by integer factor.

    Upsamples by inserting zeros, then applies lowpass filter.
    """
    if factor < 1:
        raise ValueError("Interpolation factor must be >= 1")
    if factor == 1:
        return list(x)

    # Upsample: insert zeros
    upsampled = [0.0] * (len(x) * factor)
    for i in range(len(x)):
        upsampled[i * factor] = x[i] * factor  # scale to maintain amplitude

    # Lowpass filter to remove imaging
    cutoff = 1.0 / factor
    ntaps = filter_order * factor * 2 + 1
    h = firwin_lowpass(ntaps, cutoff, fs=2.0)

    # Apply filter
    filtered = fir_filter(h, upsampled)

    # Compensate for delay
    delay = len(h) // 2
    result = filtered[delay:delay + len(x) * factor]
    if len(result) < len(x) * factor:
        result.extend([0.0] * (len(x) * factor - len(result)))

    return result


def resample(x, num):
    """Resample signal to num samples using FFT method."""
    n = len(x)
    if n == 0:
        return []
    if num == n:
        return list(x)

    X = fft(x)

    if num > n:
        # Zero-pad in frequency domain
        Y = [0.0 + 0j] * num
        half = n // 2
        for i in range(half):
            Y[i] = X[i]
        for i in range(half):
            Y[num - half + i] = X[n - half + i]
        if n % 2 == 0:
            # Split Nyquist bin
            Y[half] = X[half] / 2
            Y[num - half] = X[half] / 2
    else:
        # Truncate in frequency domain
        Y = [0.0 + 0j] * num
        half = num // 2
        for i in range(half):
            Y[i] = X[i]
        for i in range(half):
            Y[num - half + i] = X[n - half + i]
        if num % 2 == 0:
            Y[half] = X[half] + X[n - half]

    y = ifft(Y)
    scale = num / n
    return [yi.real * scale for yi in y]


def resample_poly(x, up, down, filter_order=4):
    """Polyphase resampling: upsample by up, then downsample by down."""
    if up == down:
        return list(x)

    # Upsample
    upsampled = interpolate(x, up, filter_order)

    # Downsample (already filtered by interpolation)
    result = []
    for i in range(0, len(upsampled), down):
        result.append(upsampled[i])

    return result


# ============================================================
# Analytic Signal & Envelope
# ============================================================

def analytic_signal(x):
    """Compute the analytic signal using the Hilbert transform.

    Returns complex signal where real part is original, imaginary is Hilbert transform.
    """
    n = len(x)
    if n == 0:
        return []

    X = fft(x)

    # Double positive frequencies, zero negative frequencies
    H = [0.0 + 0j] * n
    H[0] = X[0]  # DC
    for k in range(1, (n + 1) // 2):
        H[k] = 2 * X[k]
    if n % 2 == 0:
        H[n // 2] = X[n // 2]  # Nyquist

    h = ifft(H)
    return h


def envelope_detect(x):
    """Compute the amplitude envelope of a signal."""
    z = analytic_signal(x)
    return [abs(zi) for zi in z]


def instantaneous_phase(x):
    """Compute the instantaneous phase of a signal."""
    z = analytic_signal(x)
    return [cmath.phase(zi) for zi in z]


def instantaneous_freq(x, fs=1.0):
    """Compute the instantaneous frequency of a signal."""
    phase = instantaneous_phase(x)
    n = len(phase)
    freq = [0.0] * n

    for i in range(1, n):
        dp = phase[i] - phase[i - 1]
        # Unwrap
        while dp > math.pi:
            dp -= 2 * math.pi
        while dp < -math.pi:
            dp += 2 * math.pi
        freq[i] = dp * fs / (2 * math.pi)

    freq[0] = freq[1] if n > 1 else 0.0
    return freq


# ============================================================
# Correlation and Coherence
# ============================================================

def xcorr(x, y, maxlag=None):
    """Cross-correlation with lag output.

    Returns (lags, correlation).
    """
    nx = len(x)
    ny = len(y)
    if maxlag is None:
        maxlag = max(nx, ny) - 1

    # Use FFT-based correlation
    n = nx + ny - 1
    size = _next_power_of_2(n)

    X = fft(list(x) + [0.0] * (size - nx))
    Y = fft(list(y) + [0.0] * (size - ny))

    # Cross-correlation = IFFT(X * conj(Y))
    R = [X[i] * Y[i].conjugate() for i in range(size)]
    r = ifft(R)

    # Extract lags
    lags = list(range(-maxlag, maxlag + 1))
    corr = []
    for lag in lags:
        if lag >= 0:
            idx = lag
        else:
            idx = size + lag
        if 0 <= idx < size:
            corr.append(r[idx].real)
        else:
            corr.append(0.0)

    return lags, corr


def coherence(x, y, nperseg=256, noverlap=None, fs=1.0, window_fn=None):
    """Magnitude squared coherence between two signals.

    Returns (frequencies, Cxy) where Cxy is in [0, 1].
    """
    if window_fn is None:
        window_fn = window_hann
    if noverlap is None:
        noverlap = nperseg // 2

    w = window_fn(nperseg)
    step = nperseg - noverlap
    n = min(len(x), len(y))

    nfft = nperseg
    n_out = nfft // 2 + 1
    Pxx = [0.0] * n_out
    Pyy = [0.0] * n_out
    Pxy = [0.0 + 0j] * n_out
    n_segments = 0

    pos = 0
    while pos + nperseg <= n:
        sx = [x[pos + i] * w[i] for i in range(nperseg)]
        sy = [y[pos + i] * w[i] for i in range(nperseg)]

        X = fft(sx)
        Y = fft(sy)

        for k in range(n_out):
            Pxx[k] += abs(X[k]) ** 2
            Pyy[k] += abs(Y[k]) ** 2
            Pxy[k] += X[k] * Y[k].conjugate()

        n_segments += 1
        pos += step

    if n_segments == 0:
        freqs = [k * fs / nfft for k in range(n_out)]
        return freqs, [0.0] * n_out

    # Coherence = |Pxy|^2 / (Pxx * Pyy)
    Cxy = []
    for k in range(n_out):
        denom = Pxx[k] * Pyy[k]
        if denom > 1e-30:
            Cxy.append(abs(Pxy[k]) ** 2 / denom)
        else:
            Cxy.append(0.0)

    freqs = [k * fs / nfft for k in range(n_out)]
    return freqs, Cxy


# ============================================================
# Signal Metrics
# ============================================================

def rms(x):
    """Root mean square of a signal."""
    if len(x) == 0:
        return 0.0
    return math.sqrt(sum(xi * xi for xi in x) / len(x))


def snr(signal, noise):
    """Signal-to-noise ratio in dB."""
    sig_power = sum(s * s for s in signal)
    noise_power = sum(n * n for n in noise)
    if noise_power < 1e-30:
        return float('inf')
    return 10 * math.log10(sig_power / noise_power)


def thd(x, fs=1.0, fundamental_freq=None):
    """Total harmonic distortion.

    Returns THD as a ratio (not dB).
    """
    n = len(x)
    X = fft(x)
    mags = [abs(X[k]) for k in range(n // 2)]

    if fundamental_freq is not None:
        fund_bin = int(round(fundamental_freq * n / fs))
    else:
        # Find peak
        fund_bin = 1
        for k in range(1, n // 2):
            if mags[k] > mags[fund_bin]:
                fund_bin = k

    fund_mag = mags[fund_bin]
    if fund_mag < 1e-30:
        return 0.0

    # Sum harmonics
    harm_power = 0.0
    harm = 2
    while fund_bin * harm < n // 2:
        harm_power += mags[fund_bin * harm] ** 2
        harm += 1

    return math.sqrt(harm_power) / fund_mag


def peak_to_peak(x):
    """Peak-to-peak amplitude."""
    if len(x) == 0:
        return 0.0
    return max(x) - min(x)


def crest_factor(x):
    """Crest factor: peak / RMS."""
    r = rms(x)
    if r < 1e-30:
        return 0.0
    return max(abs(xi) for xi in x) / r


def zero_crossings(x):
    """Count zero crossings in a signal."""
    count = 0
    for i in range(1, len(x)):
        if x[i - 1] * x[i] < 0:
            count += 1
    return count


# ============================================================
# Median Filter
# ============================================================

def medfilt(x, kernel_size=3):
    """Apply median filter to a 1D signal."""
    if kernel_size % 2 == 0:
        kernel_size += 1

    n = len(x)
    half = kernel_size // 2
    result = []

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window = sorted(x[start:end])
        result.append(window[len(window) // 2])

    return result


# ============================================================
# Moving Average and Smoothing
# ============================================================

def moving_average(x, window_size):
    """Simple moving average filter."""
    if window_size <= 0:
        return list(x)
    n = len(x)
    result = []
    for i in range(n):
        start = max(0, i - window_size + 1)
        result.append(sum(x[start:i + 1]) / (i - start + 1))
    return result


def exponential_moving_average(x, alpha):
    """Exponential moving average. alpha in (0, 1], higher = more responsive."""
    if len(x) == 0:
        return []
    result = [x[0]]
    for i in range(1, len(x)):
        result.append(alpha * x[i] + (1 - alpha) * result[-1])
    return result


def savgol_filter(x, window_size, poly_order):
    """Savitzky-Golay smoothing filter.

    Fits polynomial of given order to each window and evaluates at center.
    """
    if window_size % 2 == 0:
        window_size += 1

    half = window_size // 2
    n = len(x)
    result = [0.0] * n

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        actual_half_left = i - start
        actual_half_right = end - 1 - i

        # Local points
        pts = []
        for j in range(start, end):
            pts.append((j - i, x[j]))

        m = len(pts)
        order = min(poly_order, m - 1)

        # Fit polynomial using normal equations
        # Build Vandermonde-like system
        A = [[0.0] * (order + 1) for _ in range(order + 1)]
        b_vec = [0.0] * (order + 1)

        for t_val, y_val in pts:
            for r in range(order + 1):
                for c in range(order + 1):
                    A[r][c] += t_val ** (r + c)
                b_vec[r] += y_val * t_val ** r

        # Solve using Gaussian elimination
        coeffs = _solve_linear(A, b_vec)

        # Evaluate at center (t=0) -> just coeffs[0]
        result[i] = coeffs[0]

    return result


def _solve_linear(A, b):
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    n = len(b)
    # Augmented matrix
    M = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        # Pivot
        max_row = col
        for row in range(col + 1, n):
            if abs(M[row][col]) > abs(M[max_row][col]):
                max_row = row
        M[col], M[max_row] = M[max_row], M[col]

        if abs(M[col][col]) < 1e-15:
            continue

        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for j in range(col, n + 1):
                M[row][j] -= factor * M[col][j]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(M[i][i]) < 1e-15:
            x[i] = 0.0
            continue
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s / M[i][i]

    return x


# ============================================================
# Differentiation and Integration
# ============================================================

def diff(x):
    """First difference (discrete derivative)."""
    return [x[i + 1] - x[i] for i in range(len(x) - 1)]


def cumsum(x):
    """Cumulative sum (discrete integration)."""
    result = []
    s = 0.0
    for xi in x:
        s += xi
        result.append(s)
    return result


def unwrap(phase, discont=math.pi):
    """Unwrap phase angles to avoid discontinuities."""
    if len(phase) <= 1:
        return list(phase)

    result = [phase[0]]
    for i in range(1, len(phase)):
        d = phase[i] - phase[i - 1]
        while d > discont:
            d -= 2 * math.pi
        while d < -discont:
            d += 2 * math.pi
        result.append(result[-1] + d)

    return result


# ============================================================
# MUSIC Algorithm (Spectral Estimation)
# ============================================================

def music(x, n_signals, n_fft=256, fs=1.0):
    """MUltiple SIgnal Classification (MUSIC) algorithm.

    Estimates frequencies of sinusoidal components in noise.
    Uses eigendecomposition of autocorrelation matrix.

    x: input signal
    n_signals: number of sinusoidal components
    n_fft: number of frequency evaluation points
    fs: sampling frequency

    Returns (frequencies, pseudospectrum).
    """
    n = len(x)
    M = min(n // 2, max(2 * n_signals + 1, 16))  # correlation matrix size

    # Build autocorrelation matrix
    R = [[0.0 + 0j] * M for _ in range(M)]
    for i in range(M):
        for j in range(M):
            s = 0.0 + 0j
            count = 0
            for k in range(n - abs(i - j)):
                if k + max(i, j) < n:
                    s += complex(x[k + i]) * complex(x[k + j])
                    count += 1
            if count > 0:
                R[i][j] = s / count

    # Eigendecomposition via power iteration (simplified)
    eigenvalues, eigenvectors = _eigen_decompose(R, M)

    # Sort by eigenvalue magnitude (descending)
    indices = sorted(range(M), key=lambda i: -abs(eigenvalues[i]))

    # Noise subspace = eigenvectors corresponding to smallest eigenvalues
    noise_vecs = []
    for i in range(n_signals, M):
        noise_vecs.append([eigenvectors[j][indices[i]] for j in range(M)])

    # Compute pseudospectrum
    freqs = []
    pspectrum = []

    for k in range(n_fft):
        f = k * fs / (2 * n_fft)
        freqs.append(f)

        # Steering vector
        a = [cmath.exp(2j * cmath.pi * f * m / fs) for m in range(M)]

        # Pseudospectrum = 1 / (a^H * E_n * E_n^H * a)
        denom = 0.0
        for nvec in noise_vecs:
            # a^H * nvec
            proj = sum(a[m].conjugate() * nvec[m] for m in range(M))
            denom += abs(proj) ** 2

        if denom > 1e-30:
            pspectrum.append(1.0 / denom)
        else:
            pspectrum.append(1e15)

    return freqs, pspectrum


def _eigen_decompose(A, n):
    """Simple eigendecomposition using QR iteration.

    Returns (eigenvalues, eigenvectors_as_columns).
    """
    # Start with A, iterate QR decomposition
    # For real symmetric case, this converges to diagonal

    # Work with real part if matrix is effectively real
    is_real = all(abs(A[i][j].imag) < 1e-10 for i in range(n) for j in range(n))

    if is_real:
        return _eigen_symmetric_real(
            [[A[i][j].real for j in range(n)] for i in range(n)], n
        )

    # For complex matrices, use simplified approach
    # Power iteration to find dominant eigenvectors
    eigenvalues = [0.0 + 0j] * n
    eigenvectors = [[0.0 + 0j] * n for _ in range(n)]

    # Initialize eigenvectors to identity
    for i in range(n):
        eigenvectors[i][i] = 1.0

    # Simple QR iteration (limited steps)
    Ak = [row[:] for row in A]
    Q_total = [[1.0 + 0j if i == j else 0.0 + 0j for j in range(n)] for i in range(n)]

    for _ in range(50):
        Q, R = _qr_decompose_complex(Ak, n)
        # A_next = R * Q
        Ak = _mat_mul_complex(R, Q, n)
        Q_total = _mat_mul_complex(Q_total, Q, n)

    for i in range(n):
        eigenvalues[i] = Ak[i][i]

    return eigenvalues, Q_total


def _eigen_symmetric_real(A, n):
    """Eigendecomposition for real symmetric matrix via Jacobi iteration."""
    # Copy
    S = [row[:] for row in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(100 * n):
        # Find largest off-diagonal element
        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(S[i][j]) > max_val:
                    max_val = abs(S[i][j])
                    p, q = i, j

        if max_val < 1e-12:
            break

        # Compute rotation
        if abs(S[p][p] - S[q][q]) < 1e-15:
            theta = math.pi / 4
        else:
            theta = 0.5 * math.atan2(2 * S[p][q], S[p][p] - S[q][q])

        c = math.cos(theta)
        s = math.sin(theta)

        # Apply rotation
        for i in range(n):
            sip = S[i][p]
            siq = S[i][q]
            S[i][p] = c * sip + s * siq
            S[i][q] = -s * sip + c * siq

        for j in range(n):
            spj = S[p][j]
            sqj = S[q][j]
            S[p][j] = c * spj + s * sqj
            S[q][j] = -s * spj + c * sqj

        # Update eigenvectors
        for i in range(n):
            vip = V[i][p]
            viq = V[i][q]
            V[i][p] = c * vip + s * viq
            V[i][q] = -s * vip + c * viq

    eigenvalues = [S[i][i] for i in range(n)]
    # Return as complex for API consistency
    return [complex(e) for e in eigenvalues], [[complex(V[i][j]) for j in range(n)] for i in range(n)]


def _qr_decompose_complex(A, n):
    """QR decomposition using Gram-Schmidt for complex matrices."""
    Q = [[0.0 + 0j] * n for _ in range(n)]
    R = [[0.0 + 0j] * n for _ in range(n)]

    for j in range(n):
        # v = column j of A
        v = [A[i][j] for i in range(n)]

        for k in range(j):
            # R[k][j] = Q[:,k]^H * A[:,j]
            R[k][j] = sum(Q[i][k].conjugate() * A[i][j] for i in range(n))
            for i in range(n):
                v[i] -= R[k][j] * Q[i][k]

        # Normalize
        norm = math.sqrt(sum(abs(vi) ** 2 for vi in v))
        R[j][j] = norm
        if norm > 1e-15:
            for i in range(n):
                Q[i][j] = v[i] / norm
        else:
            for i in range(n):
                Q[i][j] = 0.0

    return Q, R


def _mat_mul_complex(A, B, n):
    """Multiply two n x n complex matrices."""
    C = [[0.0 + 0j] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0 + 0j
            for k in range(n):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


# ============================================================
# Utility
# ============================================================

def db(x):
    """Convert amplitude to decibels."""
    if isinstance(x, list):
        return [20 * math.log10(max(abs(xi), 1e-30)) for xi in x]
    return 20 * math.log10(max(abs(x), 1e-30))


def db_power(x):
    """Convert power to decibels."""
    if isinstance(x, list):
        return [10 * math.log10(max(xi, 1e-30)) for xi in x]
    return 10 * math.log10(max(x, 1e-30))


def normalize(x):
    """Normalize signal to [-1, 1]."""
    if len(x) == 0:
        return []
    peak = max(abs(xi) for xi in x)
    if peak < 1e-30:
        return list(x)
    return [xi / peak for xi in x]


def zero_pad_to(x, n):
    """Zero-pad signal to length n."""
    if len(x) >= n:
        return list(x[:n])
    return list(x) + [0.0] * (n - len(x))


def detrend(x, order=1):
    """Remove polynomial trend from signal.

    order=0: remove mean, order=1: remove linear trend.
    """
    n = len(x)
    if n == 0:
        return []

    if order == 0:
        mean = sum(x) / n
        return [xi - mean for xi in x]

    # Linear detrend: fit y = a + b*t
    t = list(range(n))
    t_mean = sum(t) / n
    x_mean = sum(x) / n

    num = sum((t[i] - t_mean) * (x[i] - x_mean) for i in range(n))
    den = sum((t[i] - t_mean) ** 2 for i in range(n))

    if abs(den) < 1e-15:
        return [xi - x_mean for xi in x]

    b = num / den
    a = x_mean - b * t_mean

    return [x[i] - (a + b * i) for i in range(n)]
