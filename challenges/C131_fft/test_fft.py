"""Tests for C131: Fast Fourier Transform"""

import math
import cmath
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fft import (
    fft, ifft, rfft, irfft,
    _next_power_of_2, _bit_reverse,
    dft_naive, fftshift, ifftshift, zero_pad,
    window_rectangular, window_hamming, window_hann,
    window_blackman, window_kaiser, apply_window,
    circular_convolution, linear_convolution,
    cross_correlation, autocorrelation,
    poly_multiply,
    magnitude_spectrum, phase_spectrum,
    power_spectral_density, frequency_bins,
    stft, istft, spectrogram,
    hilbert_transform, envelope, instantaneous_frequency,
    cepstrum,
    dominant_frequency, frequency_resolution,
    band_pass_filter, low_pass_filter, high_pass_filter,
    dct, idct, goertzel, czt,
)


def approx_eq(a, b, tol=1e-8):
    """Check if two values are approximately equal."""
    if isinstance(a, complex) or isinstance(b, complex):
        return abs(complex(a) - complex(b)) < tol
    return abs(a - b) < tol


def approx_list(a, b, tol=1e-8):
    """Check if two lists are element-wise approximately equal."""
    if len(a) != len(b):
        return False
    return all(approx_eq(x, y, tol) for x, y in zip(a, b))


# ============================================================
# Utility Tests
# ============================================================

class TestUtilities:
    def test_next_power_of_2(self):
        assert _next_power_of_2(1) == 1
        assert _next_power_of_2(2) == 2
        assert _next_power_of_2(3) == 4
        assert _next_power_of_2(5) == 8
        assert _next_power_of_2(17) == 32

    def test_bit_reverse(self):
        assert _bit_reverse(0, 3) == 0
        assert _bit_reverse(1, 3) == 4
        assert _bit_reverse(6, 3) == 3

    def test_zero_pad(self):
        assert zero_pad([1, 2, 3], 5) == [1, 2, 3, 0, 0]
        assert zero_pad([1, 2, 3], 3) == [1, 2, 3]
        assert zero_pad([1, 2, 3], 2) == [1, 2, 3]

    def test_fftshift(self):
        x = [0, 1, 2, 3, 4, 5, 6, 7]
        shifted = fftshift(x)
        assert shifted == [4, 5, 6, 7, 0, 1, 2, 3]

    def test_ifftshift(self):
        x = [4, 5, 6, 7, 0, 1, 2, 3]
        unshifted = ifftshift(x)
        assert unshifted == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_fftshift_ifftshift_roundtrip(self):
        x = list(range(8))
        assert ifftshift(fftshift(x)) == x

    def test_frequency_resolution(self):
        assert approx_eq(frequency_resolution(100, 1000.0), 10.0)
        assert approx_eq(frequency_resolution(256, 44100.0), 44100.0 / 256)


# ============================================================
# Core FFT Tests
# ============================================================

class TestFFT:
    def test_empty(self):
        assert fft([]) == []
        assert ifft([]) == []

    def test_single_element(self):
        result = fft([5.0])
        assert approx_eq(result[0], 5.0)

    def test_two_elements(self):
        result = fft([1, 1])
        assert approx_eq(result[0], 2)
        assert approx_eq(result[1], 0)

    def test_four_elements(self):
        x = [1, 0, -1, 0]
        result = fft(x)
        assert approx_eq(result[0], 0)
        assert approx_eq(result[1], 2)
        assert approx_eq(result[2], 0)
        assert approx_eq(result[3], 2, 1e-6)  # conj of result[1] since real input

    def test_power_of_2_matches_naive(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        result = fft(x)
        expected = dft_naive(x)
        assert approx_list(result, expected, 1e-6)

    def test_dc_component(self):
        x = [3, 3, 3, 3]
        result = fft(x)
        assert approx_eq(result[0], 12)
        for k in range(1, 4):
            assert approx_eq(result[k], 0, 1e-6)

    def test_pure_cosine(self):
        n = 64
        x = [math.cos(2 * math.pi * 5 * i / n) for i in range(n)]
        result = fft(x)
        mags = [abs(v) for v in result]
        # Peak at bin 5 and bin n-5
        assert mags[5] > n / 2 - 1
        assert mags[n - 5] > n / 2 - 1

    def test_pure_sine(self):
        n = 64
        x = [math.sin(2 * math.pi * 3 * i / n) for i in range(n)]
        result = fft(x)
        mags = [abs(v) for v in result]
        assert mags[3] > n / 2 - 1

    def test_linearity(self):
        x1 = [1, 2, 3, 4]
        x2 = [4, 3, 2, 1]
        a, b = 2.5, -1.5
        combined = [a * x1[i] + b * x2[i] for i in range(4)]
        f1 = fft(x1)
        f2 = fft(x2)
        fc = fft(combined)
        for k in range(4):
            expected = a * f1[k] + b * f2[k]
            assert approx_eq(fc[k], expected, 1e-6)

    def test_parseval_theorem(self):
        x = [1, 3, -2, 4, 0, -1, 2, 5]
        n = len(x)
        time_energy = sum(abs(v) ** 2 for v in x)
        X = fft(x)
        freq_energy = sum(abs(v) ** 2 for v in X) / n
        assert approx_eq(time_energy, freq_energy, 1e-6)

    def test_large_power_of_2(self):
        n = 256
        x = [math.sin(2 * math.pi * 10 * i / n) for i in range(n)]
        result = fft(x)
        mags = [abs(v) for v in result]
        assert mags[10] > n / 2 - 1

    def test_impulse(self):
        x = [1, 0, 0, 0, 0, 0, 0, 0]
        result = fft(x)
        for v in result:
            assert approx_eq(v, 1, 1e-6)

    def test_delayed_impulse(self):
        n = 8
        x = [0, 1, 0, 0, 0, 0, 0, 0]
        result = fft(x)
        for k in range(n):
            expected = cmath.exp(-2j * cmath.pi * k / n)
            assert approx_eq(result[k], expected, 1e-6)


# ============================================================
# Inverse FFT Tests
# ============================================================

class TestIFFT:
    def test_roundtrip_power_of_2(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        result = ifft(fft(x))
        for i, v in enumerate(result):
            assert approx_eq(v.real, x[i], 1e-6)

    def test_roundtrip_complex(self):
        x = [1 + 2j, 3 - 1j, -2 + 0.5j, 4]
        result = ifft(fft(x))
        for i, v in enumerate(result):
            assert approx_eq(v, complex(x[i]), 1e-6)

    def test_roundtrip_large(self):
        n = 128
        x = [math.sin(i * 0.1) + 0.5 * math.cos(i * 0.3) for i in range(n)]
        result = ifft(fft(x))
        for i in range(n):
            assert approx_eq(result[i].real, x[i], 1e-6)


# ============================================================
# Bluestein (Arbitrary Size) Tests
# ============================================================

class TestBluestein:
    def test_prime_length(self):
        x = [1, 2, 3, 4, 5]  # length 5 (prime)
        result = fft(x)
        expected = dft_naive(x)
        assert approx_list(result, expected, 1e-6)

    def test_non_power_of_2(self):
        x = [1, 2, 3, 4, 5, 6]  # length 6
        result = fft(x)
        expected = dft_naive(x)
        assert approx_list(result, expected, 1e-6)

    def test_length_7(self):
        x = [math.sin(i) for i in range(7)]
        result = fft(x)
        expected = dft_naive(x)
        assert approx_list(result, expected, 1e-6)

    def test_roundtrip_arbitrary_size(self):
        x = [1, 2, 3, 4, 5, 6, 7]
        result = ifft(fft(x))
        for i, v in enumerate(result):
            assert approx_eq(v.real, x[i], 1e-6)

    def test_length_3(self):
        x = [1, 2, 3]
        result = fft(x)
        expected = dft_naive(x)
        assert approx_list(result, expected, 1e-6)

    def test_length_12(self):
        x = list(range(12))
        result = fft(x)
        expected = dft_naive(x)
        assert approx_list(result, expected, 1e-5)


# ============================================================
# Real FFT Tests
# ============================================================

class TestRealFFT:
    def test_rfft_length(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        result = rfft(x)
        assert len(result) == 5  # n//2 + 1

    def test_rfft_matches_fft(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        full = fft(x)
        half = rfft(x)
        for i in range(len(half)):
            assert approx_eq(half[i], full[i], 1e-6)

    def test_irfft_roundtrip(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        X = rfft(x)
        result = irfft(X)
        for i in range(len(x)):
            assert approx_eq(result[i], x[i], 1e-6)


# ============================================================
# Window Function Tests
# ============================================================

class TestWindows:
    def test_rectangular(self):
        w = window_rectangular(4)
        assert w == [1.0, 1.0, 1.0, 1.0]

    def test_hamming_endpoints(self):
        w = window_hamming(5)
        assert approx_eq(w[0], 0.08)
        assert approx_eq(w[4], 0.08)
        assert approx_eq(w[2], 1.0)

    def test_hann_endpoints(self):
        w = window_hann(5)
        assert approx_eq(w[0], 0.0)
        assert approx_eq(w[4], 0.0)
        assert approx_eq(w[2], 1.0)

    def test_blackman_endpoints(self):
        w = window_blackman(5)
        assert approx_eq(w[0], 0.0, 1e-6)
        assert approx_eq(w[4], 0.0, 1e-6)

    def test_kaiser_shape(self):
        w = window_kaiser(8, beta=5.0)
        assert len(w) == 8
        # Kaiser is symmetric
        for i in range(4):
            assert approx_eq(w[i], w[7 - i], 1e-6)

    def test_window_symmetry(self):
        for wfn in [window_hamming, window_hann, window_blackman]:
            w = wfn(8)
            for i in range(4):
                assert approx_eq(w[i], w[7 - i], 1e-6)

    def test_apply_window(self):
        x = [1, 2, 3, 4]
        w = apply_window(x, window_rectangular)
        assert w == [1, 2, 3, 4]

    def test_single_element_windows(self):
        for wfn in [window_hamming, window_hann, window_blackman, window_kaiser]:
            w = wfn(1)
            assert len(w) == 1
            assert approx_eq(w[0], 1.0)


# ============================================================
# Convolution Tests
# ============================================================

class TestConvolution:
    def test_circular_identity(self):
        a = [1, 0, 0, 0]
        b = [5, 3, 2, 1]
        result = circular_convolution(a, b)
        for i in range(4):
            assert approx_eq(result[i], b[i], 1e-6)

    def test_linear_convolution(self):
        a = [1, 2, 3]
        b = [4, 5]
        result = linear_convolution(a, b)
        expected = [4, 13, 22, 15]
        assert len(result) == 4
        for i in range(4):
            assert approx_eq(result[i], expected[i], 1e-6)

    def test_linear_convolution_commutative(self):
        a = [1, 2, 3, 4]
        b = [5, 6, 7]
        r1 = linear_convolution(a, b)
        r2 = linear_convolution(b, a)
        assert approx_list(r1, r2, 1e-6)

    def test_linear_convolution_length(self):
        a = [1, 2, 3, 4, 5]
        b = [1, 2, 3]
        result = linear_convolution(a, b)
        assert len(result) == 7  # na + nb - 1

    def test_empty_convolution(self):
        assert linear_convolution([], [1, 2]) == []
        assert circular_convolution([], []) == []


# ============================================================
# Correlation Tests
# ============================================================

class TestCorrelation:
    def test_autocorrelation_peak_at_zero(self):
        x = [1, 2, 3, 4, 5]
        ac = autocorrelation(x)
        # Peak at zero lag
        assert ac[0] >= max(ac[1:])

    def test_cross_correlation_shifted(self):
        a = [0, 0, 1, 0, 0, 0, 0, 0]
        b = [0, 0, 0, 0, 1, 0, 0, 0]
        cc = cross_correlation(a, b)
        # Peak indicates the shift
        peak_idx = cc.index(max(cc))
        # a has impulse at 2, b at 4, so shift is -2 (or equivalently at the end)
        assert approx_eq(max(cc), 1.0, 1e-6)

    def test_cross_correlation_length(self):
        a = [1, 2, 3]
        b = [4, 5, 6, 7]
        cc = cross_correlation(a, b)
        assert len(cc) == 6  # na + nb - 1


# ============================================================
# Polynomial Multiplication Tests
# ============================================================

class TestPolyMultiply:
    def test_simple(self):
        # (1 + 2x)(3 + 4x) = 3 + 10x + 8x^2
        a = [1, 2]
        b = [3, 4]
        result = poly_multiply(a, b)
        expected = [3, 10, 8]
        assert approx_list(result, expected, 1e-6)

    def test_squares(self):
        # (1 + x + x^2)^2 = 1 + 2x + 3x^2 + 2x^3 + x^4
        a = [1, 1, 1]
        result = poly_multiply(a, a)
        expected = [1, 2, 3, 2, 1]
        assert approx_list(result, expected, 1e-6)

    def test_constant(self):
        a = [5]
        b = [3]
        result = poly_multiply(a, b)
        assert approx_list(result, [15], 1e-6)

    def test_empty(self):
        assert poly_multiply([], [1, 2]) == []


# ============================================================
# Spectral Analysis Tests
# ============================================================

class TestSpectralAnalysis:
    def test_magnitude_spectrum(self):
        x = [1, 0, 0, 0]
        mags = magnitude_spectrum(x)
        for m in mags:
            assert approx_eq(m, 1.0, 1e-6)

    def test_phase_spectrum_real(self):
        x = [1, 1, 1, 1]
        phases = phase_spectrum(x)
        assert approx_eq(phases[0], 0, 1e-6)

    def test_psd_shape(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        freqs, psd = power_spectral_density(x, fs=100.0)
        assert len(freqs) == 8
        assert len(psd) == 8
        assert all(p >= 0 for p in psd)

    def test_psd_with_window(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        freqs, psd = power_spectral_density(x, fs=100.0, window_fn=window_hann)
        assert len(psd) == 8

    def test_frequency_bins(self):
        bins = frequency_bins(8, fs=100.0)
        assert approx_eq(bins[0], 0)
        assert approx_eq(bins[1], 12.5)

    def test_dominant_frequency(self):
        n = 128
        fs = 1000.0
        f0 = 50.0
        x = [math.sin(2 * math.pi * f0 * i / fs) for i in range(n)]
        df = dominant_frequency(x, fs)
        assert abs(df - f0) < fs / n  # within one bin


# ============================================================
# STFT Tests
# ============================================================

class TestSTFT:
    def test_stft_shape(self):
        n = 64
        x = [math.sin(2 * math.pi * 5 * i / n) for i in range(n)]
        frames = stft(x, window_size=16, hop_size=8)
        assert len(frames) == 7  # (64-16)/8 + 1

    def test_stft_frame_size(self):
        x = list(range(32))
        frames = stft(x, window_size=8, hop_size=4)
        for frame in frames:
            assert len(frame) == 8

    def test_istft_roundtrip(self):
        n = 64
        x = [math.sin(2 * math.pi * 3 * i / n) + 0.5 * math.cos(i * 0.7)
             for i in range(n)]
        ws = 16
        hop = 8
        frames = stft(x, window_size=ws, hop_size=hop)
        reconstructed = istft(frames, window_size=ws, hop_size=hop)
        # Check overlap region
        for i in range(ws // 2, n - ws // 2):
            assert approx_eq(reconstructed[i], x[i], 1e-3)

    def test_spectrogram_shape(self):
        x = list(range(64))
        times, freqs, mags = spectrogram(x, window_size=16, hop_size=8)
        assert len(times) == len(mags)
        assert len(freqs) == 16
        for row in mags:
            assert len(row) == 16


# ============================================================
# Hilbert Transform Tests
# ============================================================

class TestHilbert:
    def test_analytic_signal_real_part(self):
        n = 64
        x = [math.cos(2 * math.pi * 3 * i / n) for i in range(n)]
        analytic = hilbert_transform(x)
        # Real part should match original
        for i in range(n):
            assert approx_eq(analytic[i].real, x[i], 1e-6)

    def test_envelope_of_cosine(self):
        n = 128
        x = [math.cos(2 * math.pi * 5 * i / n) for i in range(n)]
        env = envelope(x)
        # Envelope of pure cosine should be approximately 1
        for i in range(10, n - 10):  # Avoid edges
            assert approx_eq(env[i], 1.0, 0.1)

    def test_instantaneous_frequency(self):
        n = 128
        fs = 1000.0
        f0 = 50.0
        x = [math.cos(2 * math.pi * f0 * i / fs) for i in range(n)]
        ifreq = instantaneous_frequency(x, fs)
        # Should be approximately f0 in the middle
        mid = len(ifreq) // 2
        assert abs(ifreq[mid] - f0) < 5.0

    def test_hilbert_empty(self):
        assert hilbert_transform([]) == []


# ============================================================
# Cepstrum Tests
# ============================================================

class TestCepstrum:
    def test_cepstrum_length(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        c = cepstrum(x)
        assert len(c) == 8

    def test_cepstrum_empty(self):
        assert cepstrum([]) == []


# ============================================================
# Filter Tests
# ============================================================

class TestFilters:
    def test_low_pass_removes_high_freq(self):
        n = 128
        fs = 1000.0
        # Signal with 50Hz and 300Hz components
        x = [math.sin(2 * math.pi * 50 * i / fs) +
             math.sin(2 * math.pi * 300 * i / fs) for i in range(n)]
        filtered = low_pass_filter(x, 100.0, fs)
        # FFT of filtered should have no 300Hz component
        X = fft(filtered)
        bin_300 = round(300 * n / fs)
        bin_50 = round(50 * n / fs)
        assert abs(X[bin_300]) < 1.0
        assert abs(X[bin_50]) > n / 4

    def test_high_pass_removes_low_freq(self):
        n = 128
        fs = 1000.0
        x = [math.sin(2 * math.pi * 50 * i / fs) +
             math.sin(2 * math.pi * 300 * i / fs) for i in range(n)]
        filtered = high_pass_filter(x, 200.0, fs)
        X = fft(filtered)
        bin_50 = round(50 * n / fs)
        bin_300 = round(300 * n / fs)
        assert abs(X[bin_50]) < 1.0
        assert abs(X[bin_300]) > n / 4

    def test_band_pass(self):
        n = 128
        fs = 1000.0
        x = [math.sin(2 * math.pi * 50 * i / fs) +
             math.sin(2 * math.pi * 150 * i / fs) +
             math.sin(2 * math.pi * 350 * i / fs) for i in range(n)]
        filtered = band_pass_filter(x, 100.0, 200.0, fs)
        X = fft(filtered)
        bin_50 = round(50 * n / fs)
        bin_150 = round(150 * n / fs)
        bin_350 = round(350 * n / fs)
        assert abs(X[bin_50]) < 1.0
        assert abs(X[bin_150]) > n / 4
        assert abs(X[bin_350]) < 1.0


# ============================================================
# DCT Tests
# ============================================================

class TestDCT:
    def test_dct_roundtrip(self):
        x = [1.0, 2.0, 3.0, 4.0]
        X = dct(x)
        result = idct(X)
        assert approx_list(result, x, 1e-6)

    def test_dct_dc(self):
        x = [3.0, 3.0, 3.0, 3.0]
        X = dct(x)
        assert approx_eq(X[0], 12.0, 1e-6)

    def test_dct_energy_compaction(self):
        # Smooth signal should have energy concentrated in low coefficients
        n = 16
        x = [math.cos(math.pi * i / n) for i in range(n)]
        X = dct(x)
        low_energy = sum(v ** 2 for v in X[:4])
        total_energy = sum(v ** 2 for v in X)
        assert low_energy > 0.9 * total_energy

    def test_dct_empty(self):
        assert dct([]) == []
        assert idct([]) == []

    def test_dct_single(self):
        X = dct([5.0])
        assert approx_eq(X[0], 5.0)
        result = idct(X)
        assert approx_eq(result[0], 5.0)

    def test_dct_longer(self):
        x = [float(i) for i in range(8)]
        X = dct(x)
        result = idct(X)
        assert approx_list(result, x, 1e-6)


# ============================================================
# Goertzel Algorithm Tests
# ============================================================

class TestGoertzel:
    def test_goertzel_matches_fft(self):
        n = 64
        fs = 1000.0
        # Use freq that lands exactly on a bin: k=8, f=8*1000/64=125
        f0 = 125.0
        x = [math.sin(2 * math.pi * f0 * i / fs) for i in range(n)]
        g = goertzel(x, f0, fs)
        X = fft(x)
        bin_f = round(f0 * n / fs)
        assert approx_eq(abs(g), abs(X[bin_f]), 1e-4)

    def test_goertzel_dc(self):
        x = [3.0, 3.0, 3.0, 3.0]
        g = goertzel(x, 0.0, 1.0)
        assert approx_eq(abs(g), 12.0, 1e-6)

    def test_goertzel_empty(self):
        g = goertzel([], 100.0)
        assert g == 0j

    def test_goertzel_single_freq(self):
        n = 128
        fs = 1000.0
        f0 = 200.0
        x = [math.cos(2 * math.pi * f0 * i / fs) for i in range(n)]
        g = goertzel(x, f0, fs)
        assert abs(g) > n / 2 - 5


# ============================================================
# Chirp Z-Transform Tests
# ============================================================

class TestCZT:
    def test_czt_matches_fft(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        result = czt(x)
        expected = fft(x)
        assert approx_list(result, expected, 1e-4)

    def test_czt_fewer_points(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        m = 4
        result = czt(x, m=m)
        # CZT with m=4 evaluates Z-transform at exp(-2j*pi*k/4), k=0..3
        # Verify against naive computation
        n = len(x)
        w = cmath.exp(-2j * cmath.pi / m)
        for k in range(m):
            expected = sum(x[j] * w ** (j * k) for j in range(n))
            assert approx_eq(result[k], expected, 1e-4)

    def test_czt_more_points(self):
        x = [1, 2, 3, 4]
        result = czt(x, m=8)
        assert len(result) == 8

    def test_czt_empty(self):
        assert czt([]) == []


# ============================================================
# Integration / Stress Tests
# ============================================================

class TestIntegration:
    def test_convolution_theorem(self):
        """Multiplication in freq domain = convolution in time domain."""
        a = [1, 2, 3, 4, 0, 0, 0, 0]
        b = [5, 6, 7, 8, 0, 0, 0, 0]
        # Circular convolution
        circ = circular_convolution(a, b)
        # Pointwise multiplication in freq domain
        fa = fft(a)
        fb = fft(b)
        prod = [fa[i] * fb[i] for i in range(8)]
        result = ifft(prod)
        for i in range(8):
            assert approx_eq(circ[i], result[i].real, 1e-6)

    def test_fft_symmetry_real_input(self):
        """For real input, X[k] = conj(X[n-k])."""
        x = [1, 3, 5, 7, 2, 4, 6, 8]
        X = fft(x)
        n = len(x)
        for k in range(1, n // 2):
            assert approx_eq(X[k], X[n - k].conjugate(), 1e-6)

    def test_time_shift_property(self):
        """Shifting by d in time multiplies by exp(-2j*pi*k*d/n) in freq."""
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        d = 2
        n = len(x)
        # Shift x by d (circular)
        shifted = x[d:] + x[:d]
        X = fft(x)
        Xs = fft(shifted)
        for k in range(n):
            # Left circular shift by d: X_s[k] = X[k] * exp(+2j*pi*k*d/n)
            expected = X[k] * cmath.exp(2j * cmath.pi * k * d / n)
            assert approx_eq(Xs[k], expected, 1e-6)

    def test_frequency_domain_filtering_roundtrip(self):
        """Filter in freq domain and verify recovery."""
        n = 64
        fs = 1000.0
        f1, f2 = 100.0, 400.0
        x = [math.sin(2 * math.pi * f1 * i / fs) +
             math.sin(2 * math.pi * f2 * i / fs) for i in range(n)]
        # Remove f2
        filtered = low_pass_filter(x, 200.0, fs)
        # Dominant freq should be f1
        df = dominant_frequency(filtered, fs)
        assert abs(df - f1) < fs / n * 2

    def test_multi_component_signal(self):
        """Detect multiple frequencies in a signal."""
        n = 256
        fs = 1000.0
        freqs_in = [50.0, 120.0, 200.0]
        x = [sum(math.sin(2 * math.pi * f * i / fs) for f in freqs_in)
             for i in range(n)]
        X = fft(x)
        mags = [abs(v) for v in X[:n // 2]]
        # Find peaks
        peaks = []
        for i in range(1, len(mags) - 1):
            if mags[i] > mags[i - 1] and mags[i] > mags[i + 1] and mags[i] > n / 4:
                peaks.append(i * fs / n)
        for f in freqs_in:
            assert any(abs(p - f) < fs / n for p in peaks)

    def test_zero_padding_improves_resolution(self):
        """Zero-padding gives finer frequency grid."""
        n = 16
        x = [math.cos(2 * math.pi * 3 * i / n) for i in range(n)]
        X1 = fft(x)
        x_padded = x + [0] * 48  # 4x padding
        X2 = fft(x_padded)
        assert len(X2) == 64
        assert len(X1) == 16

    def test_stft_spectrogram_pipeline(self):
        """Full analysis pipeline: generate, STFT, spectrogram."""
        n = 128
        fs = 1000.0
        # Chirp signal (frequency increases over time)
        x = [math.sin(2 * math.pi * (50 + 200 * i / n) * i / fs)
             for i in range(n)]
        times, freqs, mags = spectrogram(x, window_size=32, hop_size=16)
        assert len(times) > 0
        assert len(mags) == len(times)

    def test_envelope_of_am_signal(self):
        """Envelope detection of amplitude-modulated signal."""
        n = 256
        fs = 10000.0
        f_carrier = 1000.0
        f_mod = 50.0
        # AM signal: (1 + 0.5*cos(2*pi*f_mod*t)) * cos(2*pi*f_carrier*t)
        x = [(1 + 0.5 * math.cos(2 * math.pi * f_mod * i / fs)) *
             math.cos(2 * math.pi * f_carrier * i / fs) for i in range(n)]
        env = envelope(x)
        # Envelope should follow the modulating signal
        assert len(env) == n
        # Check envelope is positive
        assert all(e >= -0.1 for e in env)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
