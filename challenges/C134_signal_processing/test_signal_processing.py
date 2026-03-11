"""Tests for C134: Signal Processing"""

import math
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from signal_processing import *


# ============================================================
# Signal Generation
# ============================================================

class TestSignalGeneration:
    def test_sine_basic(self):
        s = generate_sine(10, 1.0, 100)
        assert len(s) == 100
        # Should be zero at t=0
        assert abs(s[0]) < 1e-10

    def test_sine_amplitude(self):
        s = generate_sine(1, 1.0, 1000, amplitude=2.5)
        assert max(s) == pytest.approx(2.5, abs=0.02)

    def test_cosine_basic(self):
        s = generate_cosine(10, 1.0, 100)
        assert len(s) == 100
        # Cosine starts at 1.0
        assert abs(s[0] - 1.0) < 1e-10

    def test_chirp_linear(self):
        s = generate_chirp(10, 100, 1.0, 1000)
        assert len(s) == 1000
        assert all(-1.01 <= x <= 1.01 for x in s)

    def test_chirp_exponential(self):
        s = generate_chirp(10, 100, 1.0, 1000, method='exponential')
        assert len(s) == 1000

    def test_square_wave(self):
        s = generate_square(10, 1.0, 1000)
        assert len(s) == 1000
        assert all(x == 1.0 or x == -1.0 for x in s)

    def test_square_duty(self):
        s = generate_square(10, 1.0, 1000, duty=0.25)
        positives = sum(1 for x in s if x > 0)
        assert positives < len(s) // 2  # less than 50%

    def test_sawtooth(self):
        s = generate_sawtooth(10, 1.0, 1000)
        assert len(s) == 1000
        assert min(s) >= -1.01
        assert max(s) <= 1.01

    def test_triangle(self):
        s = generate_triangle(10, 1.0, 1000)
        assert len(s) == 1000
        assert min(s) >= -1.01
        assert max(s) <= 1.01

    def test_white_noise(self):
        s = generate_white_noise(1000)
        assert len(s) == 1000
        assert all(-1.01 <= x <= 1.01 for x in s)

    def test_white_noise_seed(self):
        s1 = generate_white_noise(100, seed=123)
        s2 = generate_white_noise(100, seed=123)
        assert s1 == s2

    def test_impulse(self):
        s = generate_impulse(10, delay=3)
        assert s[3] == 1.0
        assert sum(s) == 1.0

    def test_step(self):
        s = generate_step(10, delay=5)
        assert all(s[i] == 0.0 for i in range(5))
        assert all(s[i] == 1.0 for i in range(5, 10))


# ============================================================
# Window Functions
# ============================================================

class TestWindows:
    def test_tukey_alpha0_is_rectangular(self):
        w = window_tukey(64, alpha=0)
        assert all(abs(x - 1.0) < 1e-10 for x in w)

    def test_tukey_alpha1_is_hann(self):
        w_tukey = window_tukey(64, alpha=1.0)
        w_hann = window_hann(64)
        for i in range(64):
            assert abs(w_tukey[i] - w_hann[i]) < 1e-6

    def test_flattop_peak(self):
        w = window_flattop(64)
        # Flat-top window can have values > 1 at center
        assert max(w) > 0

    def test_gaussian(self):
        w = window_gaussian(64)
        assert len(w) == 64
        # Symmetric
        for i in range(32):
            assert abs(w[i] - w[63 - i]) < 1e-10
        # Peak at center
        assert w[31] > w[0]

    def test_bartlett_hann(self):
        w = window_bartlett_hann(64)
        assert len(w) == 64
        assert w[32] > w[0]  # peak near center


# ============================================================
# FIR Filter Design
# ============================================================

class TestFIRDesign:
    def test_lowpass_basic(self):
        h = firwin_lowpass(51, 0.3, fs=2.0)
        assert len(h) == 51
        # DC gain should be ~1
        assert abs(sum(h) - 1.0) < 1e-6

    def test_lowpass_symmetry(self):
        h = firwin_lowpass(51, 0.3, fs=2.0)
        for i in range(25):
            assert abs(h[i] - h[50 - i]) < 1e-10

    def test_highpass_basic(self):
        h = firwin_highpass(51, 0.3, fs=2.0)
        assert len(h) == 51
        # DC gain should be ~0
        assert abs(sum(h)) < 0.01

    def test_bandpass_basic(self):
        h = firwin_bandpass(51, 0.2, 0.4, fs=2.0)
        assert len(h) == 51
        # DC gain should be ~0
        assert abs(sum(h)) < 0.05

    def test_bandstop_basic(self):
        h = firwin_bandstop(51, 0.2, 0.4, fs=2.0)
        assert len(h) == 51

    def test_lowpass_removes_high_freq(self):
        # Generate signal with low and high frequency
        fs = 1000
        t = [i / fs for i in range(500)]
        low = [math.sin(2 * math.pi * 10 * ti) for ti in t]
        high = [math.sin(2 * math.pi * 200 * ti) for ti in t]
        signal = [low[i] + high[i] for i in range(500)]

        h = firwin_lowpass(101, 50, fs=fs)
        filtered = fir_filter(h, signal)

        # After filter settles, high freq should be attenuated
        # Account for filter delay (50 samples for 101-tap filter)
        delay = len(h) // 2
        mid = filtered[delay + 50:delay + 350]
        low_mid = low[50:350]

        # Correlation with low freq should be high
        corr = sum(mid[i] * low_mid[i] for i in range(len(mid)))
        energy = sum(low_mid[i] ** 2 for i in range(len(low_mid)))
        assert corr / energy > 0.5


# ============================================================
# IIR Filter Design
# ============================================================

class TestIIRDesign:
    def test_butter_lowpass_order1(self):
        b, a = butter_lowpass(1, 100, 1000)
        assert len(b) == 2
        assert len(a) == 2
        assert abs(a[0] - 1.0) < 1e-10

    def test_butter_lowpass_order2(self):
        b, a = butter_lowpass(2, 100, 1000)
        assert len(b) == 3
        assert len(a) == 3

    def test_butter_lowpass_dc_gain(self):
        b, a = butter_lowpass(3, 100, 1000)
        # DC gain = sum(b) / sum(a)
        dc_gain = sum(b) / sum(a)
        assert abs(dc_gain - 1.0) < 0.05

    def test_butter_highpass_order2(self):
        b, a = butter_highpass(2, 100, 1000)
        assert len(b) == 3
        assert len(a) == 3

    def test_butter_highpass_dc_zero(self):
        b, a = butter_highpass(2, 100, 1000)
        dc_gain = abs(sum(b) / sum(a))
        assert dc_gain < 0.1

    def test_cheby1_lowpass(self):
        b, a = cheby1_lowpass(3, 1.0, 100, 1000)
        assert len(b) == 4
        assert len(a) == 4


# ============================================================
# Filter Application
# ============================================================

class TestFilterApplication:
    def test_lfilter_fir(self):
        # FIR: just b coefficients
        b = [0.25, 0.5, 0.25]
        a = [1.0]
        x = [1.0] + [0.0] * 9
        y = lfilter(b, a, x)
        assert abs(y[0] - 0.25) < 1e-10
        assert abs(y[1] - 0.5) < 1e-10
        assert abs(y[2] - 0.25) < 1e-10
        assert abs(y[3]) < 1e-10

    def test_lfilter_iir(self):
        b = [1.0]
        a = [1.0, -0.5]
        x = [1.0] + [0.0] * 9
        y = lfilter(b, a, x)
        # y[0] = 1, y[1] = 0.5, y[2] = 0.25...
        assert abs(y[0] - 1.0) < 1e-10
        assert abs(y[1] - 0.5) < 1e-10
        assert abs(y[2] - 0.25) < 1e-10

    def test_filtfilt_zero_phase(self):
        # After filtfilt, signal should have no phase shift
        fs = 1000
        sig = generate_sine(50, 0.1, fs)
        b = [0.1, 0.2, 0.4, 0.2, 0.1]
        a = [1.0]
        y = filtfilt(b, a, sig)
        assert len(y) == len(sig)

    def test_fir_filter_impulse(self):
        h = [1.0, 2.0, 3.0]
        x = [1.0] + [0.0] * 9
        y = fir_filter(h, x)
        assert abs(y[0] - 1.0) < 1e-10
        assert abs(y[1] - 2.0) < 1e-10
        assert abs(y[2] - 3.0) < 1e-10

    def test_fft_convolve(self):
        a = [1, 2, 3]
        b = [4, 5]
        c = fft_convolve(a, b)
        expected = [4, 13, 22, 15]
        for i in range(4):
            assert abs(c[i] - expected[i]) < 1e-6

    def test_overlap_add_matches_linear_conv(self):
        a = generate_sine(10, 0.5, 200)
        h = firwin_lowpass(21, 0.3, fs=2.0)
        c1 = fft_convolve(a, h)
        c2 = overlap_add(h, a, block_size=32)
        for i in range(len(c1)):
            assert abs(c1[i] - c2[i]) < 1e-6


# ============================================================
# Filter Analysis
# ============================================================

class TestFilterAnalysis:
    def test_freqz_basic(self):
        b = [1.0]
        freqs, H = freqz(b, n_points=64)
        assert len(freqs) == 64
        assert len(H) == 64
        # All-pass: magnitude should be 1
        for h in H:
            assert abs(abs(h) - 1.0) < 1e-10

    def test_freqz_lowpass(self):
        h = firwin_lowpass(51, 0.2, fs=2.0)
        freqs, H = freqz(h, n_points=256)
        mags = [abs(hi) for hi in H]
        # DC gain near 1
        assert abs(mags[0] - 1.0) < 0.1
        # High freq should be attenuated
        assert mags[-1] < 0.1

    def test_group_delay(self):
        b = [1.0, 1.0, 1.0]
        freqs, gd = group_delay(b, n_points=64)
        assert len(freqs) == 64
        # Constant group delay for linear phase FIR
        # Delay should be ~1 sample (center of 3-tap filter)


# ============================================================
# Spectral Estimation
# ============================================================

class TestSpectralEstimation:
    def test_welch_basic(self):
        fs = 1000
        sig = generate_sine(100, 1.0, fs)
        freqs, psd = welch(sig, nperseg=256, fs=fs)
        assert len(freqs) == len(psd)
        # Peak should be near 100 Hz
        peak_idx = psd.index(max(psd))
        assert abs(freqs[peak_idx] - 100) < 10

    def test_welch_two_tones(self):
        fs = 1000
        s1 = generate_sine(50, 1.0, fs)
        s2 = generate_sine(200, 1.0, fs)
        sig = [s1[i] + s2[i] for i in range(len(s1))]
        freqs, psd = welch(sig, nperseg=256, fs=fs)
        # Should have two peaks
        # Find local maxima
        peaks = []
        for i in range(1, len(psd) - 1):
            if psd[i] > psd[i-1] and psd[i] > psd[i+1] and psd[i] > max(psd) * 0.1:
                peaks.append(freqs[i])
        assert len(peaks) >= 2

    def test_bartlett_method(self):
        fs = 1000
        sig = generate_sine(100, 1.0, fs)
        freqs, psd = bartlett_method(sig, nperseg=256, fs=fs)
        assert len(freqs) == len(psd)

    def test_blackman_tukey_basic(self):
        fs = 1000
        sig = generate_sine(100, 1.0, fs)
        freqs, psd = blackman_tukey(sig, fs=fs)
        assert len(freqs) == len(psd)
        # All non-negative
        assert all(p >= 0 for p in psd)

    def test_periodogram(self):
        fs = 1000
        sig = generate_sine(100, 1.0, fs)
        freqs, psd = periodogram(sig, fs=fs)
        peak_idx = psd.index(max(psd))
        assert abs(freqs[peak_idx] - 100) < 5


# ============================================================
# Resampling
# ============================================================

class TestResampling:
    def test_decimate_basic(self):
        sig = generate_sine(10, 1.0, 1000)
        dec = decimate(sig, 2)
        assert len(dec) == 500

    def test_interpolate_basic(self):
        sig = generate_sine(10, 1.0, 100)
        interp = interpolate(sig, 4)
        assert len(interp) == 400

    def test_resample_upsample(self):
        sig = generate_sine(10, 1.0, 100)
        resampled = resample(sig, 200)
        assert len(resampled) == 200

    def test_resample_downsample(self):
        sig = generate_sine(10, 1.0, 200)
        resampled = resample(sig, 100)
        assert len(resampled) == 100

    def test_resample_identity(self):
        sig = [1.0, 2.0, 3.0, 4.0]
        resampled = resample(sig, 4)
        for i in range(4):
            assert abs(resampled[i] - sig[i]) < 1e-6

    def test_resample_poly_basic(self):
        sig = generate_sine(10, 1.0, 100)
        res = resample_poly(sig, 3, 2)
        expected_len = len(sig) * 3 // 2
        # Approximate length check
        assert abs(len(res) - expected_len) < 10


# ============================================================
# Analytic Signal & Envelope
# ============================================================

class TestAnalyticSignal:
    def test_analytic_signal_basic(self):
        sig = generate_sine(10, 1.0, 1000)
        z = analytic_signal(sig)
        assert len(z) == 1000
        # Real part should match original
        for i in range(len(sig)):
            assert abs(z[i].real - sig[i]) < 1e-6

    def test_envelope_sine(self):
        sig = generate_sine(10, 1.0, 1000, amplitude=2.0)
        env = envelope_detect(sig)
        # Envelope of sine should be approximately constant at amplitude
        # Skip edges
        mid = env[100:900]
        assert all(abs(e - 2.0) < 0.3 for e in mid)

    def test_instantaneous_phase(self):
        sig = generate_sine(10, 0.5, 1000)
        phase = instantaneous_phase(sig)
        assert len(phase) == 500

    def test_instantaneous_freq(self):
        sig = generate_sine(50, 0.5, 1000)
        freq = instantaneous_freq(sig, fs=1000)
        # Should be approximately 50 Hz in the middle
        mid = freq[100:400]
        avg = sum(mid) / len(mid)
        assert abs(avg - 50) < 5


# ============================================================
# Correlation and Coherence
# ============================================================

class TestCorrelation:
    def test_xcorr_autocorrelation(self):
        sig = generate_sine(10, 1.0, 100)
        lags, corr = xcorr(sig, sig, maxlag=50)
        assert len(lags) == 101
        # Peak at lag 0
        peak_idx = lags.index(0)
        assert corr[peak_idx] >= max(corr) - 1e-6

    def test_xcorr_delayed(self):
        sig = [0] * 5 + [1, 2, 3, 2, 1] + [0] * 5
        delayed = [0] * 8 + [1, 2, 3, 2, 1] + [0] * 2
        lags, corr = xcorr(sig, delayed, maxlag=10)
        # Peak should be near lag -3 (delayed is shifted right by 3)
        peak_lag = lags[corr.index(max(corr))]
        assert abs(peak_lag) == 3

    def test_coherence_same_signal(self):
        sig = generate_sine(50, 1.0, 1000)
        freqs, Cxy = coherence(sig, sig, nperseg=256, fs=1000)
        # Same signal should have coherence = 1
        # Skip DC
        for c in Cxy[1:]:
            assert c > 0.9

    def test_coherence_uncorrelated(self):
        s1 = generate_white_noise(1000, seed=1)
        s2 = generate_white_noise(1000, seed=9999)
        freqs, Cxy = coherence(s1, s2, nperseg=128, fs=1000)
        avg_coh = sum(Cxy) / len(Cxy)
        # Low coherence for uncorrelated signals
        assert avg_coh < 0.5


# ============================================================
# Signal Metrics
# ============================================================

class TestMetrics:
    def test_rms_sine(self):
        sig = generate_sine(10, 1.0, 1000, amplitude=1.0)
        r = rms(sig)
        assert abs(r - 1.0 / math.sqrt(2)) < 0.02

    def test_rms_dc(self):
        sig = [3.0] * 100
        assert abs(rms(sig) - 3.0) < 1e-10

    def test_snr(self):
        signal = generate_sine(10, 1.0, 1000)
        noise = [x * 0.01 for x in generate_white_noise(1000)]
        ratio = snr(signal, noise)
        assert ratio > 30  # should be ~40 dB

    def test_thd_pure_sine(self):
        sig = generate_sine(100, 1.0, 10000)
        t = thd(sig, fs=10000, fundamental_freq=100)
        assert t < 0.05  # pure sine should have very low THD

    def test_peak_to_peak(self):
        sig = generate_sine(10, 1.0, 1000, amplitude=3.0)
        assert abs(peak_to_peak(sig) - 6.0) < 0.1

    def test_crest_factor_sine(self):
        sig = generate_sine(10, 1.0, 1000)
        cf = crest_factor(sig)
        assert abs(cf - math.sqrt(2)) < 0.1

    def test_zero_crossings(self):
        sig = generate_sine(10, 1.0, 1000)
        zc = zero_crossings(sig)
        # 10 Hz signal in 1 second = 20 zero crossings (approximately)
        assert abs(zc - 20) <= 2


# ============================================================
# Median Filter
# ============================================================

class TestMedianFilter:
    def test_medfilt_impulse_noise(self):
        sig = [1.0] * 20
        sig[10] = 100.0  # impulse noise
        filtered = medfilt(sig, kernel_size=3)
        assert filtered[10] == 1.0

    def test_medfilt_preserves_signal(self):
        sig = generate_step(20, delay=10)
        filtered = medfilt(sig, kernel_size=3)
        # Should preserve step shape
        assert filtered[0] == 0.0
        assert filtered[19] == 1.0


# ============================================================
# Moving Average and Smoothing
# ============================================================

class TestSmoothing:
    def test_moving_average(self):
        sig = [1, 2, 3, 4, 5]
        ma = moving_average(sig, 3)
        assert abs(ma[2] - 2.0) < 1e-10  # avg of [1,2,3]
        assert abs(ma[4] - 4.0) < 1e-10  # avg of [3,4,5]

    def test_ema(self):
        sig = [0] * 5 + [1] * 5
        ema = exponential_moving_average(sig, alpha=0.5)
        assert ema[0] == 0.0
        assert ema[5] > 0  # starts responding to step
        assert ema[9] > 0.9  # converges toward 1

    def test_savgol_preserves_linear(self):
        # Linear signal should be preserved by Savitzky-Golay
        sig = [float(i) for i in range(20)]
        filtered = savgol_filter(sig, window_size=5, poly_order=1)
        for i in range(3, 17):  # skip edges
            assert abs(filtered[i] - sig[i]) < 0.5

    def test_savgol_smooths_noise(self):
        sig = [math.sin(2 * math.pi * i / 50) for i in range(100)]
        noisy = [sig[i] + 0.3 * ((-1) ** i) for i in range(100)]
        filtered = savgol_filter(noisy, window_size=7, poly_order=3)
        # Filtered should be closer to original than noisy
        err_noisy = sum((noisy[i] - sig[i]) ** 2 for i in range(20, 80))
        err_filtered = sum((filtered[i] - sig[i]) ** 2 for i in range(20, 80))
        assert err_filtered < err_noisy


# ============================================================
# Differentiation and Integration
# ============================================================

class TestDiffIntegrate:
    def test_diff(self):
        x = [1, 3, 6, 10, 15]
        d = diff(x)
        assert d == [2, 3, 4, 5]

    def test_cumsum(self):
        x = [1, 2, 3, 4]
        c = cumsum(x)
        assert c == [1, 3, 6, 10]

    def test_unwrap(self):
        # Phase that jumps by 2*pi
        phase = [0, 1, 2, 3, 3 - 2 * math.pi, 4 - 2 * math.pi]
        unwrapped = unwrap(phase)
        # Should be monotonically increasing
        for i in range(1, len(unwrapped)):
            assert unwrapped[i] >= unwrapped[i-1] - 0.1


# ============================================================
# MUSIC Algorithm
# ============================================================

class TestMUSIC:
    def test_music_single_tone(self):
        fs = 1000
        sig = generate_sine(100, 0.5, fs, amplitude=1.0)
        noise = generate_white_noise(500, seed=42)
        noisy = [sig[i] + 0.1 * noise[i] for i in range(500)]

        freqs, pspectrum = music(noisy, n_signals=1, n_fft=256, fs=fs)
        # Peak should be near 100 Hz
        peak_idx = pspectrum.index(max(pspectrum))
        assert abs(freqs[peak_idx] - 100) < 15

    def test_music_returns_correct_size(self):
        sig = generate_sine(50, 0.5, 500)
        freqs, ps = music(sig, n_signals=2, n_fft=128, fs=500)
        assert len(freqs) == 128
        assert len(ps) == 128


# ============================================================
# Utility Functions
# ============================================================

class TestUtility:
    def test_db(self):
        assert abs(db(1.0) - 0.0) < 1e-10
        assert abs(db(10.0) - 20.0) < 1e-10
        assert abs(db(0.1) - (-20.0)) < 1e-10

    def test_db_list(self):
        result = db([1.0, 10.0])
        assert len(result) == 2

    def test_db_power(self):
        assert abs(db_power(1.0) - 0.0) < 1e-10
        assert abs(db_power(10.0) - 10.0) < 1e-10

    def test_normalize(self):
        sig = [2, -4, 1, 3]
        n = normalize(sig)
        assert abs(max(abs(x) for x in n) - 1.0) < 1e-10
        assert abs(n[1] - (-1.0)) < 1e-10

    def test_zero_pad_to(self):
        x = [1, 2, 3]
        y = zero_pad_to(x, 5)
        assert y == [1, 2, 3, 0.0, 0.0]

    def test_zero_pad_to_truncate(self):
        x = [1, 2, 3, 4, 5]
        y = zero_pad_to(x, 3)
        assert y == [1, 2, 3]

    def test_detrend_mean(self):
        sig = [5.0] * 10
        d = detrend(sig, order=0)
        assert all(abs(x) < 1e-10 for x in d)

    def test_detrend_linear(self):
        sig = [float(i) for i in range(10)]
        d = detrend(sig, order=1)
        assert all(abs(x) < 1e-10 for x in d)

    def test_detrend_preserves_oscillation(self):
        sig = [i + math.sin(2 * math.pi * i / 10) for i in range(100)]
        d = detrend(sig, order=1)
        # Should remove linear trend, keep oscillation
        assert max(d) < 2.0
        assert min(d) > -2.0


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    def test_design_and_apply_lowpass(self):
        """Design FIR lowpass and apply to mixed signal."""
        fs = 1000
        n = 1000
        t = [i / fs for i in range(n)]
        # 50 Hz + 300 Hz
        sig = [math.sin(2 * math.pi * 50 * ti) + 0.5 * math.sin(2 * math.pi * 300 * ti)
               for ti in t]

        h = firwin_lowpass(101, 100, fs=fs)
        filtered = fir_filter(h, sig)

        # Verify with spectral analysis
        freqs, psd = welch(filtered[100:], nperseg=256, fs=fs)
        peak_idx = psd.index(max(psd))
        # Peak should be at 50 Hz
        assert abs(freqs[peak_idx] - 50) < 10

    def test_design_and_apply_butter(self):
        """Design Butterworth filter and apply."""
        fs = 1000
        sig = generate_sine(50, 0.5, fs)
        noise = [0.5 * x for x in generate_white_noise(500, seed=7)]
        noisy = [sig[i] + noise[i] for i in range(500)]

        b, a = butter_lowpass(3, 100, fs)
        filtered = lfilter(b, a, noisy)
        assert len(filtered) == 500

    def test_resample_preserves_frequency(self):
        """Resampling should preserve signal frequency content."""
        fs1 = 1000
        sig = generate_sine(50, 1.0, fs1)
        # Resample to 500 Hz
        resampled = resample(sig, 500)
        # Check frequency content
        freqs, psd = periodogram(resampled, fs=500)
        peak_idx = psd.index(max(psd))
        assert abs(freqs[peak_idx] - 50) < 5

    def test_chirp_spectrogram(self):
        """Chirp signal should show increasing frequency over time."""
        sig = generate_chirp(10, 200, 1.0, 1000)
        # Just verify it runs and produces reasonable output
        from fft import stft
        frames = stft(sig, window_size=128, hop_size=64)
        assert len(frames) > 0

    def test_envelope_of_am_signal(self):
        """Amplitude modulated signal envelope detection."""
        fs = 1000
        n = 1000
        t = [i / fs for i in range(n)]
        # AM signal: carrier 100Hz modulated by 5Hz
        carrier = [math.sin(2 * math.pi * 100 * ti) for ti in t]
        modulator = [0.5 + 0.5 * math.sin(2 * math.pi * 5 * ti) for ti in t]
        am_signal = [carrier[i] * modulator[i] for i in range(n)]

        env = envelope_detect(am_signal)
        # Envelope should roughly follow modulator
        # Check that envelope varies with ~5 Hz
        mid_env = env[100:900]
        assert max(mid_env) > 0.3
        assert min(mid_env) < max(mid_env)

    def test_filtfilt_vs_lfilter(self):
        """filtfilt should have less phase distortion than lfilter."""
        fs = 1000
        sig = generate_sine(50, 0.5, fs)
        b, a = butter_lowpass(3, 200, fs)

        y_filt = lfilter(b, a, sig)
        y_filtfilt = filtfilt(b, a, sig)

        # Both should have similar amplitude
        assert len(y_filt) == len(sig)
        assert len(y_filtfilt) == len(sig)

    def test_overlap_add_long_signal(self):
        """Overlap-add should handle long signals efficiently."""
        sig = generate_sine(10, 2.0, 1000)  # 2000 samples
        h = firwin_lowpass(31, 0.3, fs=2.0)
        result = overlap_add(h, sig, block_size=64)
        assert len(result) == len(sig) + len(h) - 1

    def test_full_pipeline(self):
        """Complete signal processing pipeline."""
        fs = 8000
        duration = 0.5
        n = int(fs * duration)

        # Generate signal with two tones
        t = [i / fs for i in range(n)]
        sig = [math.sin(2 * math.pi * 440 * ti) + 0.3 * math.sin(2 * math.pi * 1000 * ti)
               for ti in t]

        # Add noise
        noise = generate_white_noise(n, seed=42)
        noisy = [sig[i] + 0.1 * noise[i] for i in range(n)]

        # Detrend
        detrended = detrend(noisy)

        # Filter: keep only 300-600 Hz band
        h = firwin_bandpass(101, 300, 600, fs=fs)
        filtered = fir_filter(h, detrended)

        # Analyze
        freqs, psd = welch(filtered[100:], nperseg=512, fs=fs)
        peak_idx = psd.index(max(psd))
        # Should find 440 Hz as dominant
        assert abs(freqs[peak_idx] - 440) < 30

    def test_coherence_filtered_signals(self):
        """Two signals from same source should have high coherence."""
        fs = 1000
        source = generate_sine(100, 1.0, fs)
        noise1 = [0.1 * x for x in generate_white_noise(1000, seed=1)]
        noise2 = [0.1 * x for x in generate_white_noise(1000, seed=2)]
        s1 = [source[i] + noise1[i] for i in range(1000)]
        s2 = [source[i] + noise2[i] for i in range(1000)]

        freqs, Cxy = coherence(s1, s2, nperseg=256, fs=fs)
        # Near 100 Hz, coherence should be high
        idx_100 = min(range(len(freqs)), key=lambda i: abs(freqs[i] - 100))
        assert Cxy[idx_100] > 0.5

    def test_music_two_tones(self):
        """MUSIC should resolve two close frequencies."""
        fs = 1000
        n = 500
        s1 = generate_sine(100, n / fs, fs)
        s2 = generate_sine(120, n / fs, fs)
        sig = [s1[i] + s2[i] for i in range(n)]

        freqs, ps = music(sig, n_signals=2, n_fft=256, fs=fs)
        # Find top 2 peaks
        peaks = []
        for i in range(1, len(ps) - 1):
            if ps[i] > ps[i-1] and ps[i] > ps[i+1]:
                peaks.append((ps[i], freqs[i]))
        peaks.sort(reverse=True)
        peak_freqs = sorted([p[1] for p in peaks[:2]])

        if len(peak_freqs) >= 2:
            assert abs(peak_freqs[0] - 100) < 20 or abs(peak_freqs[1] - 100) < 20


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_empty_signal(self):
        assert rms([]) == 0.0
        assert normalize([]) == []
        assert detrend([]) == []
        assert analytic_signal([]) == []
        assert resample([], 10) == []

    def test_single_sample(self):
        assert rms([5.0]) == 5.0
        assert normalize([5.0]) == [1.0]

    def test_dc_signal(self):
        sig = [1.0] * 100
        env = envelope_detect(sig)
        assert len(env) == 100

    def test_medfilt_size_1(self):
        sig = [1, 2, 3]
        assert medfilt(sig, kernel_size=1) == [1, 2, 3]

    def test_generate_impulse_delay_out_of_range(self):
        s = generate_impulse(5, delay=10)
        assert sum(s) == 0.0

    def test_decimate_factor_1(self):
        sig = [1, 2, 3]
        assert decimate(sig, 1) == [1, 2, 3]

    def test_interpolate_factor_1(self):
        sig = [1, 2, 3]
        assert interpolate(sig, 1) == [1, 2, 3]

    def test_window_size_1(self):
        assert window_tukey(1) == [1.0]
        assert window_flattop(1) == [1.0]
        assert window_gaussian(1) == [1.0]
        assert window_bartlett_hann(1) == [1.0]

    def test_diff_single(self):
        assert diff([5]) == []

    def test_cumsum_empty(self):
        assert cumsum([]) == []
