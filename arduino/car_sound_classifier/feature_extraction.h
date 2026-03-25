#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include <stdint.h>

// Initialize FFT instance. Call once in setup().
void feature_extraction_init();

// Compute log-mel spectrogram from raw int16 audio.
// Output is row-major: output[mel_band * N_FRAMES + frame].
// Shape: (N_MELS, N_FRAMES) = (40, 92) = 3680 floats.
void compute_mel_spectrogram(const int16_t* audio, int audio_len, float* output);

// Apply per-band z-score normalization in-place.
void normalize_spectrogram(float* spectrogram);

// Quantize float32 spectrogram to int8 model input.
void quantize_input(const float* spectrogram, int8_t* output,
                    float scale, int zero_point);

#endif // FEATURE_EXTRACTION_H
