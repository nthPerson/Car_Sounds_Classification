/*
 * On-device mel-spectrogram feature extraction using CMSIS-DSP.
 *
 * Replicates the Python training pipeline:
 *   librosa.melspectrogram() -> power_to_db(ref=np.max) -> z-score normalize
 *
 * Validated by src/validate_features.py (cosine similarity >= 0.9999).
 */

#include "feature_extraction.h"
#include "config.h"
#include "mel_filterbank.h"
#include "normalization.h"
#include "hann_window.h"

#include <arm_math.h>
#include <math.h>

// ─── FFT Instance ───────────────────────────────────────────────────────

static arm_rfft_fast_instance_f32 fft_instance;

void feature_extraction_init() {
    arm_rfft_fast_init_f32(&fft_instance, N_FFT);
}

// ─── Mel-Spectrogram Computation ────────────────────────────────────────

void compute_mel_spectrogram(const int16_t* audio, int audio_len, float* output) {
    // Per-frame workspace (stack-allocated, reused)
    float frame[N_FFT];
    float fft_out[N_FFT];
    float power[N_FFT_BINS];

    // ── Phase A: Peak normalization factor ────────────────────────────
    // Find max absolute sample value for normalization
    int16_t max_abs = 0;
    for (int i = 0; i < audio_len; i++) {
        int16_t val = audio[i];
        if (val < 0) val = -val;
        if (val > max_abs) max_abs = val;
    }
    float norm_factor = (max_abs > 0) ? (1.0f / (float)max_abs) : (1.0f / 32768.0f);

    // ── Phase B: Frame-by-frame FFT + mel energy ─────────────────────
    for (int t = 0; t < N_FRAMES; t++) {
        int offset = t * HOP_LENGTH;

        // Extract frame: convert int16 -> float32 with peak normalization
        for (int i = 0; i < N_FFT; i++) {
            int idx = offset + i;
            if (idx < audio_len) {
                frame[i] = (float)audio[idx] * norm_factor;
            } else {
                frame[i] = 0.0f;
            }
        }

        // Apply Hann window
        arm_mult_f32(frame, (float*)hann_window, frame, N_FFT);

        // Compute real FFT
        // arm_rfft_fast_f32 output format:
        //   fft_out[0] = DC component (real)
        //   fft_out[1] = Nyquist component (real)
        //   fft_out[2..N_FFT-1] = interleaved real/imag for bins 1..N_FFT/2-1
        arm_rfft_fast_f32(&fft_instance, frame, fft_out, 0);

        // Compute power spectrum (magnitude squared)
        // DC bin
        power[0] = fft_out[0] * fft_out[0];
        // Nyquist bin
        power[N_FFT / 2] = fft_out[1] * fft_out[1];
        // Bins 1 to N_FFT/2 - 1 (interleaved real/imag)
        for (int k = 1; k < N_FFT / 2; k++) {
            float re = fft_out[2 * k];
            float im = fft_out[2 * k + 1];
            power[k] = re * re + im * im;
        }

        // Apply sparse mel filterbank
        int weight_offset = 0;
        for (int m = 0; m < N_MELS; m++) {
            float mel_energy = 0.0f;
            int start = mel_filter_info[m].start_bin;
            int nw = mel_filter_info[m].num_weights;

            for (int j = 0; j < nw; j++) {
                mel_energy += mel_filter_weights[weight_offset + j] * power[start + j];
            }
            weight_offset += nw;

            // Store in row-major order: mel_band * N_FRAMES + frame
            output[m * N_FRAMES + t] = mel_energy;
        }
    }

    // ── Phase C: Global max log conversion (ref=np.max, top_db=80) ───
    // Find maximum mel energy across all bands and frames
    float max_energy = 0.0f;
    for (int i = 0; i < SPECTROGRAM_SIZE; i++) {
        if (output[i] > max_energy) {
            max_energy = output[i];
        }
    }
    if (max_energy < 1e-10f) {
        max_energy = 1e-10f;
    }

    // Apply: 10 * log10(energy / max_energy)
    // Clamp to -TOP_DB minimum (matches librosa's default top_db=80)
    for (int i = 0; i < SPECTROGRAM_SIZE; i++) {
        float val = 10.0f * log10f(output[i] / max_energy + 1e-10f);
        if (val < -TOP_DB) {
            val = -TOP_DB;
        }
        output[i] = val;
    }
}

// ─── Normalization ──────────────────────────────────────────────────────

void normalize_spectrogram(float* spectrogram) {
    // Per-band z-score: (x - mean) / std
    for (int m = 0; m < N_MELS; m++) {
        float mean = mel_norm_mean[m];
        float std = mel_norm_std[m];
        for (int t = 0; t < N_FRAMES; t++) {
            int idx = m * N_FRAMES + t;
            spectrogram[idx] = (spectrogram[idx] - mean) / std;
        }
    }
}

// ─── Quantization ───────────────────────────────────────────────────────

void quantize_input(const float* spectrogram, int8_t* output,
                    float scale, int zero_point) {
    for (int i = 0; i < SPECTROGRAM_SIZE; i++) {
        int32_t q = (int32_t)roundf(spectrogram[i] / scale) + zero_point;
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        output[i] = (int8_t)q;
    }
}
