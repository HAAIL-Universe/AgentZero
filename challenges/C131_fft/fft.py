"""
C131: Fast Fourier Transform

Standalone numerical algorithm implementation covering:
- Cooley-Tukey radix-2 FFT (power-of-2 sizes)
- Bluestein's algorithm (arbitrary-size FFT)
- Inverse FFT
- Real-valued FFT (optimized for real inputs)
- Convolution (linear and circular)
- Polynomial multiplication
- Power spectral density
- Short-Time Fourier Transform (STFT)
- Windowing functions (Hamming, Hann, Blackman, Kaiser)
- Zero-padding and frequency resolution
- Cross-correlation
- Hilbert transform
- Cepstrum analysis
"""

import math
import cmath


# ============================================================
# Core FFT Algorithms
# ============================================================

def _bit_reverse(x, n_bits):
    """Reverse the bits of x using n_bits bits."""
    result = 0
    for _ in range(n_bits):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result


def _next_power_of_2(n):
    """Return smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def fft(x):
    """
    Compute the DFT of sequence x using Cooley-Tukey radix-2 FFT.
    If len(x) is not a power of 2, uses Bluestein's algorithm.
    Returns list of complex numbers.
    """
    n = len(x)
    if n == 0:
        return []
    if n == 1:
        return [complex(x[0])]

    # Check if power of 2
    if n & (n - 1) == 0:
        return _fft_radix2(x)
    else:
        return _fft_bluestein(x)


def _fft_radix2(x):
    """Cooley-Tukey radix-2 iterative FFT. n must be power of 2."""
    n = len(x)
    n_bits = n.bit_length() - 1

    # Bit-reversal permutation
    result = [complex(0)] * n
    for i in range(n):
        result[_bit_reverse(i, n_bits)] = complex(x[i])

    # Butterfly operations
    length = 2
    while length <= n:
        half = length // 2
        w_n = cmath.exp(-2j * cmath.pi / length)
        for start in range(0, n, length):
            w = 1.0 + 0j
            for j in range(half):
                u = result[start + j]
                t = w * result[start + j + half]
                result[start + j] = u + t
                result[start + j + half] = u - t
                w *= w_n
        length *= 2

    return result


def _fft_bluestein(x):
    """Bluestein's algorithm for arbitrary-size FFT."""
    n = len(x)
    m = _next_power_of_2(2 * n - 1)

    # Chirp sequence: w_k = exp(-j*pi*k^2/n)
    chirp = [cmath.exp(-1j * cmath.pi * k * k / n) for k in range(n)]
    chirp_conj = [c.conjugate() for c in chirp]

    # Build sequences of length m
    a = [complex(0)] * m
    b = [complex(0)] * m

    for k in range(n):
        a[k] = complex(x[k]) * chirp[k]

    b[0] = chirp_conj[0]
    for k in range(1, n):
        b[k] = chirp_conj[k]
        b[m - k] = chirp_conj[k]

    # Convolution via radix-2 FFT
    fa = _fft_radix2(a)
    fb = _fft_radix2(b)
    fc = [fa[i] * fb[i] for i in range(m)]

    # Inverse FFT of product
    fc_inv = _ifft_radix2(fc)

    # Extract result
    result = [chirp[k] * fc_inv[k] for k in range(n)]
    return result


def ifft(X):
    """Compute the inverse DFT."""
    n = len(X)
    if n == 0:
        return []
    if n == 1:
        return [complex(X[0])]

    if n & (n - 1) == 0:
        return _ifft_radix2(X)
    else:
        return _ifft_bluestein(X)


def _ifft_radix2(X):
    """Inverse FFT using conjugate trick. n must be power of 2."""
    n = len(X)
    # Conjugate, FFT, conjugate, divide by n
    conj = [complex(v).conjugate() for v in X]
    result = _fft_radix2(conj)
    return [v.conjugate() / n for v in result]


def _ifft_bluestein(X):
    """Inverse FFT for arbitrary sizes."""
    n = len(X)
    conj = [complex(v).conjugate() for v in X]
    result = _fft_bluestein(conj)
    return [v.conjugate() / n for v in result]


# ============================================================
# Real-Valued FFT
# ============================================================

def rfft(x):
    """
    FFT optimized for real-valued input.
    Returns n//2 + 1 complex values (positive frequencies only).
    """
    n = len(x)
    if n == 0:
        return []
    full = fft(x)
    return full[:n // 2 + 1]


def irfft(X, n=None):
    """
    Inverse of rfft. Reconstructs real signal from positive frequencies.
    n is the output length (default: 2*(len(X)-1)).
    """
    if n is None:
        n = 2 * (len(X) - 1)

    # Reconstruct full spectrum using conjugate symmetry
    full = list(X)
    for k in range(len(X), n):
        full.append(X[n - k].conjugate())

    result = ifft(full)
    return [v.real for v in result]


# ============================================================
# Windowing Functions
# ============================================================

def window_rectangular(n):
    """Rectangular window (no windowing)."""
    return [1.0] * n


def window_hamming(n):
    """Hamming window."""
    if n <= 1:
        return [1.0] * n
    return [0.54 - 0.46 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)]


def window_hann(n):
    """Hann window."""
    if n <= 1:
        return [1.0] * n
    return [0.5 * (1 - math.cos(2 * math.pi * i / (n - 1))) for i in range(n)]


def window_blackman(n):
    """Blackman window."""
    if n <= 1:
        return [1.0] * n
    return [
        0.42 - 0.5 * math.cos(2 * math.pi * i / (n - 1))
        + 0.08 * math.cos(4 * math.pi * i / (n - 1))
        for i in range(n)
    ]


def window_kaiser(n, beta=8.6):
    """Kaiser window with shape parameter beta."""
    if n <= 1:
        return [1.0] * n

    def bessel_i0(x):
        """Modified Bessel function of the first kind, order 0."""
        val = 1.0
        term = 1.0
        for k in range(1, 50):
            term *= (x / (2 * k)) ** 2
            val += term
            if term < 1e-15 * val:
                break
        return val

    denom = bessel_i0(beta)
    result = []
    for i in range(n):
        alpha = 2.0 * i / (n - 1) - 1.0
        result.append(bessel_i0(beta * math.sqrt(max(0, 1 - alpha * alpha))) / denom)
    return result


def apply_window(x, window_fn):
    """Apply a window function to signal x."""
    w = window_fn(len(x))
    return [x[i] * w[i] for i in range(len(x))]


# ============================================================
# Convolution & Correlation
# ============================================================

def circular_convolution(a, b):
    """Circular convolution of two sequences of the same length."""
    n = len(a)
    if n == 0:
        return []
    fa = fft(a)
    fb = fft(b)
    fc = [fa[i] * fb[i] for i in range(n)]
    result = ifft(fc)
    return [v.real for v in result]


def linear_convolution(a, b):
    """Linear convolution via FFT (zero-padded)."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return []
    n = _next_power_of_2(na + nb - 1)

    # Zero-pad
    a_pad = list(a) + [0] * (n - na)
    b_pad = list(b) + [0] * (n - nb)

    fa = fft(a_pad)
    fb = fft(b_pad)
    fc = [fa[i] * fb[i] for i in range(n)]
    result = ifft(fc)
    return [v.real for v in result[:na + nb - 1]]


def cross_correlation(a, b):
    """Cross-correlation of a and b via FFT."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return []
    n = _next_power_of_2(na + nb - 1)

    a_pad = list(a) + [0] * (n - na)
    b_pad = list(b) + [0] * (n - nb)

    fa = fft(a_pad)
    fb = fft(b_pad)
    # Correlation: conj(fb) * fa
    fc = [fa[i] * fb[i].conjugate() for i in range(n)]
    result = ifft(fc)
    return [v.real for v in result[:na + nb - 1]]


def autocorrelation(x):
    """Autocorrelation of x via FFT."""
    return cross_correlation(x, x)


# ============================================================
# Polynomial Multiplication
# ============================================================

def poly_multiply(a, b):
    """
    Multiply two polynomials represented as coefficient lists.
    a[i] is the coefficient of x^i.
    Returns coefficient list of the product.
    """
    if not a or not b:
        return []
    result = linear_convolution(a, b)
    # Round near-integer coefficients
    return [round(v, 10) for v in result]


# ============================================================
# Spectral Analysis
# ============================================================

def magnitude_spectrum(x):
    """Compute magnitude spectrum |X[k]|."""
    X = fft(x)
    return [abs(v) for v in X]


def phase_spectrum(x):
    """Compute phase spectrum arg(X[k])."""
    X = fft(x)
    return [cmath.phase(v) for v in X]


def power_spectral_density(x, fs=1.0, window_fn=None):
    """
    Compute power spectral density of x.
    fs: sampling frequency.
    window_fn: windowing function (default: rectangular).
    Returns (frequencies, psd) tuple.
    """
    n = len(x)
    if n == 0:
        return [], []

    if window_fn is not None:
        x = apply_window(x, window_fn)

    X = fft(x)
    psd = [(abs(v) ** 2) / n for v in X]

    freqs = [k * fs / n for k in range(n)]
    return freqs, psd


def frequency_bins(n, fs=1.0):
    """Return frequency values for each FFT bin."""
    return [k * fs / n for k in range(n)]


# ============================================================
# Short-Time Fourier Transform (STFT)
# ============================================================

def stft(x, window_size, hop_size=None, window_fn=None):
    """
    Short-Time Fourier Transform.
    x: input signal
    window_size: size of each analysis window
    hop_size: step between windows (default: window_size // 2)
    window_fn: windowing function (default: Hann)
    Returns list of FFT frames (each frame is list of complex).
    """
    n = len(x)
    if window_fn is None:
        window_fn = window_hann
    if hop_size is None:
        hop_size = window_size // 2

    window = window_fn(window_size)
    frames = []

    pos = 0
    while pos + window_size <= n:
        segment = [x[pos + i] * window[i] for i in range(window_size)]
        frames.append(fft(segment))
        pos += hop_size

    return frames


def istft(frames, window_size, hop_size=None, window_fn=None):
    """
    Inverse STFT using overlap-add.
    Returns reconstructed signal.
    """
    if not frames:
        return []

    if window_fn is None:
        window_fn = window_hann
    if hop_size is None:
        hop_size = window_size // 2

    window = window_fn(window_size)
    n_frames = len(frames)
    output_len = window_size + (n_frames - 1) * hop_size
    output = [0.0] * output_len
    window_sum = [0.0] * output_len

    for i, frame in enumerate(frames):
        segment = ifft(frame)
        pos = i * hop_size
        for j in range(window_size):
            output[pos + j] += segment[j].real * window[j]
            window_sum[pos + j] += window[j] ** 2

    # Normalize by window sum
    result = []
    for i in range(output_len):
        if window_sum[i] > 1e-10:
            result.append(output[i] / window_sum[i])
        else:
            result.append(0.0)

    return result


# ============================================================
# Spectrogram
# ============================================================

def spectrogram(x, window_size, hop_size=None, window_fn=None):
    """
    Compute spectrogram (magnitude STFT).
    Returns (times, frequencies, magnitude_matrix).
    magnitude_matrix[frame][freq_bin].
    """
    frames = stft(x, window_size, hop_size, window_fn)
    if not frames:
        return [], [], []

    if hop_size is None:
        hop_size = window_size // 2

    times = [i * hop_size for i in range(len(frames))]
    freqs = list(range(window_size))
    magnitudes = [[abs(v) for v in frame] for frame in frames]

    return times, freqs, magnitudes


# ============================================================
# Advanced Transforms
# ============================================================

def hilbert_transform(x):
    """
    Compute the analytic signal using the Hilbert transform.
    Returns list of complex numbers (analytic signal).
    """
    n = len(x)
    if n == 0:
        return []

    X = fft(x)

    # Build the multiplier: h[0]=1, h[n/2]=1, h[1..n/2-1]=2, h[n/2+1..]=0
    h = [0.0] * n
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        for k in range(1, n // 2):
            h[k] = 2.0
    else:
        for k in range(1, (n + 1) // 2):
            h[k] = 2.0

    Xa = [X[k] * h[k] for k in range(n)]
    return ifft(Xa)


def envelope(x):
    """Compute the envelope (instantaneous amplitude) of x."""
    analytic = hilbert_transform(x)
    return [abs(v) for v in analytic]


def instantaneous_frequency(x, fs=1.0):
    """Compute instantaneous frequency of x."""
    analytic = hilbert_transform(x)
    n = len(analytic)
    if n < 2:
        return []

    phases = [cmath.phase(v) for v in analytic]
    freq = []
    for i in range(1, n):
        dp = phases[i] - phases[i - 1]
        # Unwrap
        while dp > math.pi:
            dp -= 2 * math.pi
        while dp < -math.pi:
            dp += 2 * math.pi
        freq.append(dp * fs / (2 * math.pi))

    return freq


def cepstrum(x):
    """
    Compute the real cepstrum of x.
    cepstrum = IFFT(log(|FFT(x)|))
    """
    n = len(x)
    if n == 0:
        return []

    X = fft(x)
    log_mag = [math.log(max(abs(v), 1e-20)) for v in X]
    result = ifft(log_mag)
    return [v.real for v in result]


# ============================================================
# Utility Functions
# ============================================================

def zero_pad(x, target_length):
    """Zero-pad signal x to target_length."""
    if len(x) >= target_length:
        return list(x)
    return list(x) + [0] * (target_length - len(x))


def fftshift(X):
    """Shift zero-frequency component to center."""
    n = len(X)
    mid = n // 2
    return X[mid:] + X[:mid]


def ifftshift(X):
    """Inverse of fftshift."""
    n = len(X)
    mid = (n + 1) // 2
    return X[mid:] + X[:mid]


def dft_naive(x):
    """Naive O(n^2) DFT for testing/comparison."""
    n = len(x)
    result = []
    for k in range(n):
        s = 0j
        for j in range(n):
            s += complex(x[j]) * cmath.exp(-2j * cmath.pi * j * k / n)
        result.append(s)
    return result


def frequency_resolution(n, fs=1.0):
    """Return the frequency resolution (Hz per bin)."""
    return fs / n


def dominant_frequency(x, fs=1.0):
    """Find the dominant frequency in a signal."""
    n = len(x)
    if n == 0:
        return 0.0
    X = fft(x)
    # Only look at positive frequencies
    mags = [abs(X[k]) for k in range(1, n // 2 + 1)]
    if not mags:
        return 0.0
    peak_bin = mags.index(max(mags)) + 1
    return peak_bin * fs / n


def band_pass_filter(x, low_freq, high_freq, fs=1.0):
    """Apply an ideal band-pass filter in the frequency domain."""
    n = len(x)
    if n == 0:
        return []

    X = fft(x)
    filtered = list(X)

    for k in range(n):
        freq = k * fs / n
        # Map to [-fs/2, fs/2] range
        if freq > fs / 2:
            freq = fs - freq
        if freq < low_freq or freq > high_freq:
            filtered[k] = 0j

    result = ifft(filtered)
    return [v.real for v in result]


def low_pass_filter(x, cutoff_freq, fs=1.0):
    """Apply ideal low-pass filter."""
    return band_pass_filter(x, 0, cutoff_freq, fs)


def high_pass_filter(x, cutoff_freq, fs=1.0):
    """Apply ideal high-pass filter."""
    return band_pass_filter(x, cutoff_freq, fs / 2, fs)


# ============================================================
# DCT (Discrete Cosine Transform)
# ============================================================

def dct(x):
    """
    Compute DCT-II (the standard DCT used in JPEG, MP3, etc).
    DCT-II: X[k] = sum_n x[n] * cos(pi/N * (n + 0.5) * k)
    """
    n = len(x)
    if n == 0:
        return []

    result = []
    for k in range(n):
        s = 0.0
        for j in range(n):
            s += x[j] * math.cos(math.pi / n * (j + 0.5) * k)
        result.append(s)
    return result


def idct(X):
    """
    Compute inverse DCT-II (DCT-III with normalization).
    """
    n = len(X)
    if n == 0:
        return []

    result = []
    for j in range(n):
        s = X[0] / 2.0
        for k in range(1, n):
            s += X[k] * math.cos(math.pi / n * k * (j + 0.5))
        result.append(2.0 * s / n)
    return result


# ============================================================
# Goertzel Algorithm (single-frequency DFT)
# ============================================================

def goertzel(x, target_freq, fs=1.0):
    """
    Goertzel algorithm: compute DFT at a single frequency.
    More efficient than full FFT when only one frequency is needed.
    Returns complex value.
    """
    n = len(x)
    if n == 0:
        return 0j

    k = target_freq * n / fs
    w = 2 * math.pi * k / n
    coeff = 2 * math.cos(w)

    s0 = 0.0
    s1 = 0.0
    s2 = 0.0

    for sample in x:
        s0 = sample + coeff * s1 - s2
        s2 = s1
        s1 = s0

    # Final computation
    real = s1 - s2 * math.cos(w)
    imag = -s2 * math.sin(w)
    return complex(real, imag)


# ============================================================
# Chirp Z-Transform
# ============================================================

def czt(x, m=None, w=None, a=None):
    """
    Chirp Z-Transform.
    Evaluates the Z-transform at m points along a spiral in the z-plane.
    x: input sequence
    m: number of output points (default: len(x))
    w: ratio between points on spiral (default: exp(-2j*pi/m))
    a: starting point on spiral (default: 1)
    """
    n = len(x)
    if n == 0:
        return []
    if m is None:
        m = n
    if w is None:
        w = cmath.exp(-2j * cmath.pi / m)
    if a is None:
        a = 1.0 + 0j

    # Build sequences for Bluestein-like computation
    L = _next_power_of_2(n + m - 1)

    # Chirp: w^(k^2/2)
    chirp = [w ** (k * k / 2.0) for k in range(max(n, m))]

    # Build y: x[k] * a^(-k) * chirp[k]
    y = [complex(0)] * L
    for k in range(n):
        y[k] = complex(x[k]) * (a ** (-k)) * chirp[k]

    # Build h: conj(chirp[k]) for k in range(m), plus wrap-around
    h = [complex(0)] * L
    for k in range(m):
        h[k] = chirp[k].conjugate()
    for k in range(1, n):
        h[L - k] = chirp[k].conjugate()

    # Convolve
    Y = _fft_radix2(y)
    H = _fft_radix2(h)
    G = [Y[i] * H[i] for i in range(L)]
    g = _ifft_radix2(G)

    # Extract result
    result = [chirp[k] * g[k] for k in range(m)]
    return result
