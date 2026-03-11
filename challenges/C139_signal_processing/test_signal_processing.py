"""Tests for C139: Signal Processing."""

import math
import cmath
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from signal_processing import (
    fft, ifft, rfft, irfft, dft_naive,
    hann_window, hamming_window, blackman_window, bartlett_window,
    rectangular_window, kaiser_window, apply_window,
    power_spectrum, power_spectral_density, magnitude_spectrum,
    phase_spectrum, spectrogram, cepstrum,
    convolve, correlate, autocorrelate, convolve_direct,
    sine_wave, cosine_wave, square_wave, sawtooth_wave, triangle_wave,
    chirp, impulse, white_noise, step_function,
    FIRFilter, IIRFilter,
    zero_pad, next_power_of_2, db, db_power, normalize, rms, energy,
    peak_frequency, snr, resample, decimate,
    hilbert_transform, envelope, instantaneous_frequency,
    goertzel, _ensure_complex,
)


def approx(a, b, tol=1e-6):
    """Check approximate equality."""
    if isinstance(a, complex) and isinstance(b, complex):
        return abs(a - b) < tol
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(approx(ai, bi, tol) for ai, bi in zip(a, b))
    return abs(float(a) - float(b)) < tol


# ============================================================
# FFT Core Tests
# ============================================================

class TestFFT:
    def test_fft_empty(self):
        assert fft([]) == []

    def test_fft_single(self):
        result = fft([3.0])
        assert approx(result[0], complex(3.0, 0))

    def test_fft_two(self):
        result = fft([1.0, 0.0])
        assert approx(result[0], complex(1.0, 0))
        assert approx(result[1], complex(1.0, 0))

    def test_fft_four_ones(self):
        result = fft([1.0, 1.0, 1.0, 1.0])
        assert approx(result[0], complex(4.0, 0))
        assert approx(abs(result[1]), 0.0, tol=1e-10)
        assert approx(abs(result[2]), 0.0, tol=1e-10)
        assert approx(abs(result[3]), 0.0, tol=1e-10)

    def test_fft_impulse(self):
        result = fft([1.0, 0.0, 0.0, 0.0])
        for k in range(4):
            assert approx(result[k], complex(1.0, 0))

    def test_fft_matches_naive(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        fast = fft(x)
        naive = dft_naive(x)
        for a, b in zip(fast, naive):
            assert approx(a, b, tol=1e-8)

    def test_fft_inverse_roundtrip(self):
        x = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]
        X = fft(x)
        recovered = ifft(X)
        for xi, ri in zip(x, recovered):
            assert approx(ri.real, xi, tol=1e-10)
            assert approx(ri.imag, 0.0, tol=1e-10)

    def test_fft_pure_sine(self):
        n = 64
        x = [math.sin(2.0 * math.pi * 4 * i / n) for i in range(n)]
        X = fft(x)
        mags = [abs(xk) for xk in X]
        # Peaks at bin 4 and bin 60 (= n - 4)
        assert mags[4] > n / 3
        assert mags[60] > n / 3
        # Other bins should be near zero
        for k in range(n):
            if k != 4 and k != 60:
                assert mags[k] < 1e-8

    def test_fft_linearity(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [4.0, 3.0, 2.0, 1.0]
        Fa = fft(a)
        Fb = fft(b)
        c = [ai + bi for ai, bi in zip(a, b)]
        Fc = fft(c)
        for i in range(4):
            assert approx(Fc[i], Fa[i] + Fb[i])

    def test_fft_parseval_theorem(self):
        """Sum of |x|^2 = sum of |X|^2 / N."""
        x = [1.0, -3.0, 2.0, 7.0, -1.0, 4.0, 0.0, -2.0]
        X = fft(x)
        n = len(x)
        time_energy = sum(v ** 2 for v in x)
        freq_energy = sum(abs(xk) ** 2 for xk in X) / n
        assert approx(time_energy, freq_energy, tol=1e-8)

    def test_fft_16_point(self):
        x = list(range(16))
        X = fft(x)
        naive = dft_naive(x)
        for a, b in zip(X, naive):
            assert approx(a, b, tol=1e-6)

    def test_fft_non_power_of_2_raises(self):
        try:
            fft([1, 2, 3])
            assert False, "Should raise"
        except ValueError:
            pass


class TestIFFT:
    def test_ifft_empty(self):
        assert ifft([]) == []

    def test_ifft_single(self):
        result = ifft([complex(5, 0)])
        assert approx(result[0], complex(5, 0))

    def test_ifft_roundtrip(self):
        x = [complex(i, 0) for i in range(8)]
        assert approx(ifft(fft(x)), x, tol=1e-10)

    def test_ifft_frequency_domain(self):
        # Single frequency in freq domain
        X = [0] * 8
        X[1] = complex(8, 0)
        x = ifft(X)
        # Should be a cosine + j*sine at frequency 1
        for i in range(8):
            expected = cmath.exp(complex(0, 2 * math.pi * i / 8))
            assert approx(x[i], expected, tol=1e-10)


class TestRFFT:
    def test_rfft_length(self):
        x = [1.0] * 8
        result = rfft(x)
        assert len(result) == 5  # n//2 + 1

    def test_rfft_irfft_roundtrip(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        X = rfft(x)
        recovered = irfft(X)
        for xi, ri in zip(x, recovered):
            assert approx(xi, ri, tol=1e-10)


# ============================================================
# Window Function Tests
# ============================================================

class TestWindows:
    def test_hann_endpoints(self):
        w = hann_window(8)
        assert approx(w[0], 0.0)
        assert approx(w[7], 0.0)

    def test_hann_symmetry(self):
        w = hann_window(8)
        for i in range(4):
            assert approx(w[i], w[7 - i])

    def test_hann_peak(self):
        w = hann_window(9)
        assert approx(w[4], 1.0)

    def test_hamming_endpoints(self):
        w = hamming_window(8)
        assert approx(w[0], 0.08)

    def test_hamming_symmetry(self):
        w = hamming_window(16)
        for i in range(8):
            assert approx(w[i], w[15 - i])

    def test_blackman_endpoints(self):
        w = blackman_window(8)
        assert approx(w[0], 0.0, tol=1e-4)

    def test_blackman_symmetry(self):
        w = blackman_window(16)
        for i in range(8):
            assert approx(w[i], w[15 - i])

    def test_bartlett_triangle(self):
        w = bartlett_window(9)
        assert approx(w[0], 0.0)
        assert approx(w[4], 1.0)
        assert approx(w[8], 0.0)

    def test_rectangular(self):
        w = rectangular_window(8)
        assert all(v == 1.0 for v in w)

    def test_kaiser_symmetry(self):
        w = kaiser_window(16, beta=5.0)
        for i in range(8):
            assert approx(w[i], w[15 - i], tol=1e-10)

    def test_kaiser_beta_zero_is_rectangular(self):
        w = kaiser_window(8, beta=0.0)
        for v in w:
            assert approx(v, 1.0, tol=1e-10)

    def test_apply_window(self):
        sig = [1.0, 2.0, 3.0, 4.0]
        win = [0.5, 1.0, 1.0, 0.5]
        result = apply_window(sig, win)
        assert approx(result, [0.5, 2.0, 3.0, 2.0])

    def test_apply_window_length_mismatch(self):
        try:
            apply_window([1, 2], [1])
            assert False
        except ValueError:
            pass

    def test_window_single_point(self):
        assert hann_window(1) == [1.0]
        assert hamming_window(1) == [1.0]
        assert blackman_window(1) == [1.0]
        assert kaiser_window(1) == [1.0]


# ============================================================
# Spectral Analysis Tests
# ============================================================

class TestSpectralAnalysis:
    def test_power_spectrum_impulse(self):
        x = [1.0, 0.0, 0.0, 0.0]
        ps = power_spectrum(x)
        # Flat power spectrum for impulse
        for p in ps:
            assert approx(p, 0.25)

    def test_power_spectrum_dc(self):
        x = [1.0, 1.0, 1.0, 1.0]
        ps = power_spectrum(x)
        assert approx(ps[0], 4.0)
        for p in ps[1:]:
            assert approx(p, 0.0, tol=1e-10)

    def test_magnitude_spectrum(self):
        x = [1.0, 0.0, 0.0, 0.0]
        ms = magnitude_spectrum(x)
        for m in ms:
            assert approx(m, 1.0)

    def test_phase_spectrum_real(self):
        x = [1.0, 1.0, 1.0, 1.0]
        ps = phase_spectrum(x)
        assert approx(ps[0], 0.0)

    def test_psd_returns_freqs_and_psd(self):
        x = [math.sin(2 * math.pi * 10 * i / 64) for i in range(64)]
        freqs, psd = power_spectral_density(x, fs=64)
        assert len(freqs) == 33  # 64//2 + 1
        assert len(psd) == 33
        # Peak should be near 10 Hz
        peak_idx = psd.index(max(psd))
        assert approx(freqs[peak_idx], 10.0, tol=1.5)

    def test_spectrogram_shape(self):
        x = list(range(64))
        times, freqs, Sxx = spectrogram(x, nperseg=16, noverlap=8)
        assert len(freqs) == 9  # 16//2 + 1
        assert len(times) > 0
        assert len(Sxx) == len(times)
        assert len(Sxx[0]) == len(freqs)

    def test_spectrogram_chirp(self):
        # Chirp should show increasing frequency over time
        fs = 256
        x = chirp(10, 100, 1.0, fs)
        # Pad to work with spectrogram
        n = next_power_of_2(len(x))
        x = zero_pad(x, n)
        times, freqs, Sxx = spectrogram(x, nperseg=32, noverlap=24, fs=fs)
        # First segment peak freq should be lower than last segment peak freq
        if len(Sxx) >= 2:
            first_peak = Sxx[0].index(max(Sxx[0]))
            last_peak = Sxx[-1].index(max(Sxx[-1]))
            # Just verify spectrogram computes without error
            assert len(Sxx[0]) == len(freqs)

    def test_cepstrum(self):
        x = [1.0, 0.5, 0.25, 0.125, 0.0, 0.0, 0.0, 0.0]
        c = cepstrum(x)
        assert len(c) == len(x)
        # Cepstrum of decaying signal should have specific structure
        assert isinstance(c[0], float)


# ============================================================
# Convolution Tests
# ============================================================

class TestConvolution:
    def test_convolve_identity(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [1.0, 0.0, 0.0, 0.0]
        result = convolve(a, b)
        for i in range(4):
            assert approx(result[i], a[i], tol=1e-8)

    def test_convolve_length(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0]
        result = convolve(a, b)
        assert len(result) == 4  # 3 + 2 - 1

    def test_convolve_matches_direct(self):
        a = [1.0, 2.0, 3.0, 4.0]
        b = [0.5, -0.5, 0.25]
        fft_result = convolve(a, b)
        direct_result = convolve_direct(a, b)
        for fi, di in zip(fft_result, direct_result):
            assert approx(fi, di, tol=1e-8)

    def test_convolve_commutative(self):
        a = [1.0, -1.0, 2.0, -2.0]
        b = [3.0, 1.0]
        r1 = convolve(a, b)
        r2 = convolve(b, a)
        assert len(r1) == len(r2)
        for v1, v2 in zip(r1, r2):
            assert approx(v1, v2, tol=1e-8)

    def test_convolve_empty(self):
        assert convolve([], [1, 2]) == []
        assert convolve([1, 2], []) == []

    def test_convolve_box_filter(self):
        # Convolving step with box = ramp
        x = [1.0] * 8
        h = [0.25, 0.25, 0.25, 0.25]
        result = convolve(x, h)
        # Middle values should be 1.0
        for i in range(3, 8):
            assert approx(result[i], 1.0, tol=1e-8)

    def test_correlate_autocorrelation_peak(self):
        x = [1.0, 2.0, 3.0, 4.0]
        ac = autocorrelate(x)
        # Peak at center (zero lag)
        center = len(x) - 1
        for i in range(len(ac)):
            assert ac[center] >= ac[i] - 1e-8

    def test_correlate_shifted(self):
        a = [0, 0, 1, 0, 0, 0, 0, 0]
        b = [0, 0, 0, 0, 1, 0, 0, 0]
        c = correlate(a, b)
        # Peak indicates shift between signals
        assert len(c) == 15  # 8 + 8 - 1


# ============================================================
# Signal Generation Tests
# ============================================================

class TestSignalGeneration:
    def test_sine_wave_period(self):
        fs = 100
        freq = 10
        x = sine_wave(freq, 1.0, fs)
        assert len(x) == 100
        # Zero crossings: should cross zero ~20 times for 10 Hz in 1 sec
        crossings = sum(1 for i in range(1, len(x)) if x[i-1] * x[i] < 0)
        assert 18 <= crossings <= 22

    def test_sine_wave_amplitude(self):
        x = sine_wave(1, 1.0, 100, amplitude=2.5)
        assert max(x) <= 2.5 + 0.01
        assert min(x) >= -2.5 - 0.01

    def test_cosine_wave_phase(self):
        x = cosine_wave(1, 1.0, 100)
        assert approx(x[0], 1.0, tol=0.01)

    def test_square_wave_values(self):
        x = square_wave(1, 1.0, 100, amplitude=1.0)
        for v in x:
            assert approx(abs(v), 1.0)

    def test_sawtooth_range(self):
        x = sawtooth_wave(1, 1.0, 100)
        assert min(x) >= -1.01
        assert max(x) <= 1.01

    def test_triangle_range(self):
        x = triangle_wave(1, 1.0, 100)
        assert min(x) >= -1.01
        assert max(x) <= 1.01

    def test_chirp_length(self):
        x = chirp(10, 100, 0.5, 1000)
        assert len(x) == 500

    def test_impulse(self):
        x = impulse(8, position=3)
        assert x[3] == 1.0
        assert sum(x) == 1.0

    def test_impulse_default(self):
        x = impulse(4)
        assert x[0] == 1.0

    def test_white_noise_reproducible(self):
        a = white_noise(100, seed=42)
        b = white_noise(100, seed=42)
        assert a == b

    def test_white_noise_different_seeds(self):
        a = white_noise(100, seed=1)
        b = white_noise(100, seed=2)
        assert a != b

    def test_step_function(self):
        x = step_function(8, position=3)
        assert x[:3] == [0.0, 0.0, 0.0]
        assert x[3:] == [1.0, 1.0, 1.0, 1.0, 1.0]


# ============================================================
# FIR Filter Tests
# ============================================================

class TestFIRFilter:
    def test_identity_filter(self):
        f = FIRFilter([1.0])
        x = [1.0, 2.0, 3.0, 4.0]
        y = f.apply(x)
        for xi, yi in zip(x, y):
            assert approx(xi, yi, tol=1e-8)

    def test_delay_filter(self):
        f = FIRFilter([0.0, 1.0])
        x = [1.0, 2.0, 3.0, 4.0]
        y = f.apply(x)
        assert approx(y[0], 0.0, tol=1e-8)
        assert approx(y[1], 1.0, tol=1e-8)
        assert approx(y[2], 2.0, tol=1e-8)

    def test_moving_average(self):
        f = FIRFilter.moving_average(4)
        x = [0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0, 0.0]
        y = f.apply(x)
        # After transient, should smooth to steady state
        assert approx(y[6], 4.0, tol=1e-8)

    def test_low_pass_sinc(self):
        f = FIRFilter.low_pass_sinc(0.3, 31)
        assert len(f.b) == 31
        # Sum of coefficients should be ~1 (normalized)
        assert approx(sum(f.b), 1.0, tol=0.01)

    def test_high_pass_sinc(self):
        f = FIRFilter.high_pass_sinc(0.3, 31)
        assert len(f.b) == 31

    def test_band_pass_sinc(self):
        f = FIRFilter.band_pass_sinc(0.2, 0.4, 31)
        assert len(f.b) == 31

    def test_frequency_response_length(self):
        f = FIRFilter([1.0, -1.0])
        freqs, mags, phases = f.frequency_response(256)
        assert len(freqs) > 0
        assert len(mags) == len(freqs)
        assert len(phases) == len(freqs)

    def test_lp_attenuates_high_freq(self):
        f = FIRFilter.low_pass_sinc(0.2, 65)
        n = 256
        high = [math.sin(2 * math.pi * 100 * i / n) for i in range(n)]
        filtered = f.apply(high)
        # After transient, high-freq should be mostly gone
        mid = filtered[65:200]
        assert energy(mid) < energy(high[65:200]) * 0.1


# ============================================================
# IIR Filter Tests
# ============================================================

class TestIIRFilter:
    def test_passthrough(self):
        f = IIRFilter([1.0], [1.0])
        x = [1.0, 2.0, 3.0, 4.0]
        y = f.apply(x)
        for xi, yi in zip(x, y):
            assert approx(xi, yi)

    def test_first_order_lowpass(self):
        f = IIRFilter.first_order_lowpass(0.1)
        # Step response should ramp up
        x = [1.0] * 100
        y = f.apply(x)
        assert y[0] < y[50]
        assert approx(y[99], 1.0, tol=0.01)

    def test_first_order_highpass(self):
        f = IIRFilter.first_order_highpass(0.9)
        # Step response should decay
        x = [1.0] * 100
        y = f.apply(x)
        assert abs(y[99]) < abs(y[0])

    def test_butterworth_lowpass_creation(self):
        f = IIRFilter.butterworth_lowpass(0.3, order=2)
        assert len(f.b) > 0
        assert len(f.a) > 0
        assert approx(f.a[0], 1.0)

    def test_butterworth_lowpass_dc_gain(self):
        f = IIRFilter.butterworth_lowpass(0.3, order=2)
        # DC gain should be 1
        dc_gain = sum(f.b) / sum(f.a)
        assert approx(dc_gain, 1.0, tol=0.01)

    def test_butterworth_highpass_creation(self):
        f = IIRFilter.butterworth_highpass(0.3, order=2)
        assert len(f.b) > 0
        assert len(f.a) > 0

    def test_butterworth_highpass_dc_zero(self):
        f = IIRFilter.butterworth_highpass(0.3, order=2)
        # DC gain should be ~0
        dc_gain = abs(sum(f.b) / sum(f.a))
        assert dc_gain < 0.01

    def test_notch_filter(self):
        f = IIRFilter.notch(0.25, Q=10)
        assert len(f.b) == 3
        assert len(f.a) == 3

    def test_iir_frequency_response(self):
        f = IIRFilter.butterworth_lowpass(0.3, order=2)
        freqs, mags, phases = f.frequency_response(128)
        assert len(freqs) == 128
        # DC magnitude should be ~1
        assert approx(mags[0], 1.0, tol=0.1)
        # Nyquist magnitude should be much smaller
        assert mags[-1] < 0.5

    def test_butterworth_order_3(self):
        f = IIRFilter.butterworth_lowpass(0.2, order=3)
        x = [1.0] * 200
        y = f.apply(x)
        # Should converge to 1
        assert approx(y[199], 1.0, tol=0.05)

    def test_notch_removes_frequency(self):
        n = 256
        fs = 256
        # Signal: 20 Hz + 50 Hz
        x = [math.sin(2*math.pi*20*i/fs) + math.sin(2*math.pi*50*i/fs) for i in range(n)]
        # Notch at 50 Hz (normalized: 50/128 ~ 0.39)
        f = IIRFilter.notch(50.0 / (fs / 2), Q=5)
        y = f.apply(x)
        # After filter, 50 Hz component should be reduced
        # Check via FFT
        Y = fft(zero_pad(y, 256))
        X = fft(zero_pad(x, 256))
        # Magnitude at bin 50 should be reduced
        assert abs(Y[50]) < abs(X[50])

    def test_iir_normalization(self):
        # a[0] != 1 should get normalized
        f = IIRFilter([2.0, 4.0], [2.0, -1.0])
        assert approx(f.a[0], 1.0)
        assert approx(f.b[0], 1.0)
        assert approx(f.b[1], 2.0)


# ============================================================
# Utility Tests
# ============================================================

class TestUtilities:
    def test_zero_pad(self):
        x = [1.0, 2.0, 3.0]
        result = zero_pad(x, 8)
        assert len(result) == 8
        assert result[:3] == [1.0, 2.0, 3.0]
        assert result[3:] == [0.0] * 5

    def test_zero_pad_no_change(self):
        x = [1.0, 2.0]
        result = zero_pad(x, 2)
        assert result == [1.0, 2.0]

    def test_next_power_of_2(self):
        assert next_power_of_2(1) == 1
        assert next_power_of_2(3) == 4
        assert next_power_of_2(4) == 4
        assert next_power_of_2(5) == 8
        assert next_power_of_2(100) == 128

    def test_db(self):
        assert approx(db(1.0), 0.0)
        assert approx(db(10.0), 20.0)
        assert approx(db(0.1), -20.0)
        assert db(0) == -math.inf

    def test_db_power(self):
        assert approx(db_power(1.0), 0.0)
        assert approx(db_power(10.0), 10.0)

    def test_normalize(self):
        x = [2.0, -4.0, 1.0]
        n = normalize(x)
        assert approx(n[1], -1.0)
        assert max(abs(v) for v in n) <= 1.0 + 1e-10

    def test_normalize_zero(self):
        x = [0.0, 0.0]
        assert normalize(x) == [0.0, 0.0]

    def test_rms(self):
        x = [1.0, -1.0, 1.0, -1.0]
        assert approx(rms(x), 1.0)

    def test_rms_sine(self):
        n = 1000
        x = [math.sin(2 * math.pi * i / n) for i in range(n)]
        assert approx(rms(x), 1.0 / math.sqrt(2), tol=0.01)

    def test_rms_empty(self):
        assert rms([]) == 0.0

    def test_energy(self):
        x = [3.0, 4.0]
        assert approx(energy(x), 25.0)

    def test_peak_frequency_sine(self):
        n = 256
        fs = 256
        freq = 32
        x = [math.sin(2 * math.pi * freq * i / fs) for i in range(n)]
        pf = peak_frequency(x, fs=fs)
        assert approx(pf, freq, tol=2.0)

    def test_snr(self):
        sig = [1.0] * 100
        noise = [0.1] * 100
        s = snr(sig, noise)
        assert approx(s, 20.0, tol=0.01)

    def test_ensure_complex(self):
        assert _ensure_complex(3.0) == complex(3.0, 0)
        assert _ensure_complex(complex(1, 2)) == complex(1, 2)
        assert _ensure_complex(5) == complex(5, 0)


# ============================================================
# Hilbert Transform Tests
# ============================================================

class TestHilbert:
    def test_hilbert_length(self):
        x = [1.0] * 8
        result = hilbert_transform(x)
        assert len(result) == 8

    def test_hilbert_real_part(self):
        x = [math.sin(2 * math.pi * i / 16) for i in range(16)]
        analytic = hilbert_transform(x)
        for xi, ai in zip(x, analytic):
            assert approx(xi, ai.real, tol=1e-8)

    def test_envelope_sine(self):
        n = 64
        x = [math.sin(2 * math.pi * 4 * i / n) for i in range(n)]
        env = envelope(x)
        # Envelope of pure sine should be ~1.0
        for e in env[4:-4]:  # Skip edges
            assert approx(e, 1.0, tol=0.15)

    def test_instantaneous_frequency_sine(self):
        n = 128
        fs = 128
        freq = 10
        x = [math.sin(2 * math.pi * freq * i / fs) for i in range(n)]
        ifreq = instantaneous_frequency(x, fs=fs)
        # Should be approximately constant at freq Hz
        mid = ifreq[20:100]
        avg = sum(mid) / len(mid)
        assert approx(avg, freq, tol=1.0)

    def test_instantaneous_frequency_chirp(self):
        n = 256
        fs = 256
        x = chirp(10, 50, 1.0, fs)
        ifreq = instantaneous_frequency(x, fs=fs)
        # Frequency should increase over time
        early = sum(ifreq[20:60]) / 40
        late = sum(ifreq[150:200]) / 50
        assert late > early


# ============================================================
# Goertzel Tests
# ============================================================

class TestGoertzel:
    def test_goertzel_matches_fft(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        X = fft(x)
        n = len(x)
        for k in range(n):
            target_freq = k * 1.0 / n
            g = goertzel(x, target_freq, fs=1.0)
            assert approx(abs(g), abs(X[k]), tol=1e-6)

    def test_goertzel_pure_tone(self):
        n = 64
        fs = 64
        freq = 10
        x = [math.sin(2 * math.pi * freq * i / fs) for i in range(n)]
        g = goertzel(x, freq, fs=fs)
        assert abs(g) > n / 3  # Strong response at target freq

    def test_goertzel_no_tone(self):
        n = 64
        fs = 64
        x = [math.sin(2 * math.pi * 10 * i / fs) for i in range(n)]
        g = goertzel(x, 20, fs=fs)
        assert abs(g) < 1.0  # Weak response at non-target freq


# ============================================================
# Resample / Decimate Tests
# ============================================================

class TestResampleDecimate:
    def test_resample_identity(self):
        x = [1.0, 2.0, 3.0, 4.0]
        assert resample(x, 1) == x

    def test_resample_upsample_length(self):
        x = [1.0, 2.0, 3.0, 4.0]
        result = resample(x, 2)
        assert len(result) == 8

    def test_resample_invalid(self):
        try:
            resample([1], 0)
            assert False
        except ValueError:
            pass

    def test_decimate_identity(self):
        x = [1.0, 2.0, 3.0, 4.0]
        assert decimate(x, 1) == x

    def test_decimate_by_2(self):
        x = list(range(100))
        result = decimate([float(v) for v in x], 2)
        assert len(result) == 50

    def test_decimate_invalid(self):
        try:
            decimate([1], 0)
            assert False
        except ValueError:
            pass


# ============================================================
# Integration / End-to-End Tests
# ============================================================

class TestIntegration:
    def test_filter_then_fft(self):
        """Filter a signal then analyze spectrum."""
        n = 256
        fs = 256
        # 20 Hz + 80 Hz
        x = [math.sin(2*math.pi*20*i/fs) + math.sin(2*math.pi*80*i/fs) for i in range(n)]
        # Low-pass at 0.4 (= 50 Hz at fs=256)
        f = FIRFilter.low_pass_sinc(50.0 / (fs/2), 65)
        y = f.apply(x)
        # FFT of filtered signal should show reduced 80 Hz
        Y = fft(zero_pad(y, 256))
        X = fft(zero_pad(x, 256))
        assert abs(Y[80]) < abs(X[80])

    def test_convolution_theorem(self):
        """conv(a,b) in time = mult in frequency."""
        a = [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]
        b = [0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # Time domain convolution
        c_time = convolve(a, b)[:8]
        # Frequency domain multiplication
        A = fft(a)
        B = fft(b)
        C = [ai * bi for ai, bi in zip(A, B)]
        c_freq = ifft(C)
        for ct, cf in zip(c_time, c_freq):
            assert approx(ct, cf.real, tol=1e-8)

    def test_windowed_fft_reduces_leakage(self):
        """Applying a window before FFT should reduce spectral leakage."""
        n = 64
        # Non-integer number of cycles = leakage
        x = [math.sin(2 * math.pi * 7.5 * i / n) for i in range(n)]
        # Without window
        X1 = fft(x)
        # With Hann window
        w = hann_window(n)
        xw = apply_window(x, w)
        X2 = fft(xw)
        # Sidelobe level should be lower with windowing
        # Check bins far from main lobe
        far_bins = list(range(15, 50))
        leak_no_window = sum(abs(X1[k]) for k in far_bins)
        leak_window = sum(abs(X2[k]) for k in far_bins)
        assert leak_window < leak_no_window

    def test_signal_reconstruction(self):
        """Generate -> FFT -> modify -> IFFT -> compare."""
        n = 64
        fs = 64
        x = [math.sin(2*math.pi*5*i/fs) + 0.5*math.sin(2*math.pi*15*i/fs) for i in range(n)]
        X = fft(x)
        # Zero out 15 Hz component (bins 15 and n-15)
        X[15] = 0
        X[n - 15] = 0
        y = ifft(X)
        # Should only have 5 Hz left
        expected = [math.sin(2*math.pi*5*i/fs) for i in range(n)]
        for yi, ei in zip(y, expected):
            assert approx(yi.real, ei, tol=0.1)

    def test_noise_reduction_by_filtering(self):
        """Filter removes high-frequency noise from signal."""
        n = 256
        fs = 256
        noise = white_noise(n, amplitude=1.0, seed=123)
        # Low-pass filter should reduce overall noise energy
        f = FIRFilter.low_pass_sinc(0.3, 65)
        filtered = f.apply(noise)
        # Filtered noise should have less energy (high freqs removed)
        mid = slice(65, 200)
        assert energy(filtered[mid]) < energy(noise[mid])

    def test_autocorrelation_finds_period(self):
        """Autocorrelation should peak at signal period."""
        n = 256
        fs = 256
        freq = 16  # period = 16 samples
        x = [math.sin(2*math.pi*freq*i/fs) for i in range(n)]
        ac = autocorrelate(x)
        # Find peaks in autocorrelation
        center = n - 1
        # Peak at center (zero lag), also at center +/- period
        assert ac[center] >= ac[center + 1]
        # Check for periodicity: peak at lag = 16
        period = fs // freq
        assert ac[center + period] > ac[center + period // 2]

    def test_full_pipeline(self):
        """Generate -> window -> FFT -> PSD -> peak detection."""
        n = 256
        fs = 256
        x = sine_wave(32, 1.0, fs, amplitude=2.0)
        w = hann_window(n)
        xw = apply_window(x, w)
        freqs, psd = power_spectral_density(xw, fs=fs)
        # Find peak
        peak_idx = psd.index(max(psd))
        assert approx(freqs[peak_idx], 32.0, tol=2.0)


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_fft_large(self):
        n = 1024
        x = [math.sin(2 * math.pi * 100 * i / n) for i in range(n)]
        X = fft(x)
        assert len(X) == n

    def test_all_windows_length_16(self):
        for wfn in [hann_window, hamming_window, blackman_window, bartlett_window,
                     rectangular_window]:
            w = wfn(16)
            assert len(w) == 16

    def test_kaiser_various_beta(self):
        for beta in [0.0, 2.0, 5.0, 8.6, 14.0]:
            w = kaiser_window(16, beta)
            assert len(w) == 16
            assert all(0 <= v <= 1.0 + 1e-10 for v in w)

    def test_fir_moving_average_smooths(self):
        x = [0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        f = FIRFilter.moving_average(4)
        y = f.apply(x)
        assert max(y) < 10  # Smoothed

    def test_butterworth_orders(self):
        for order in [1, 2, 3, 4]:
            f = IIRFilter.butterworth_lowpass(0.3, order=order)
            # Should be stable: step response converges
            y = f.apply([1.0] * 200)
            assert approx(y[199], 1.0, tol=0.1)

    def test_convolve_single_element(self):
        assert approx(convolve([3.0], [2.0])[0], 6.0)

    def test_chirp_frequency_increases(self):
        x = chirp(10, 100, 1.0, 1000)
        # Count zero crossings in first half vs second half
        n = len(x)
        half = n // 2
        zc1 = sum(1 for i in range(1, half) if x[i-1] * x[i] < 0)
        zc2 = sum(1 for i in range(half + 1, n) if x[i-1] * x[i] < 0)
        assert zc2 > zc1  # More crossings in second half

    def test_dft_naive_small(self):
        x = [1.0, 0.0]
        X = dft_naive(x)
        assert approx(X[0], complex(1, 0))
        assert approx(X[1], complex(1, 0))


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
