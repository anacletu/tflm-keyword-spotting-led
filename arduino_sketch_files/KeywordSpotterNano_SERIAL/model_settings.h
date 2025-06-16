/*  Copyright 2025 github/anacletu
    Project: tflm-keyword-spotting-led
    Licensed under the MIT License
*/

/*  model_settings.h */
#ifndef MODEL_SETTINGS_H_
#define MODEL_SETTINGS_H_

#include <cstdint> // For int8_t

// --- Compile-time constants for dimensions ---
constexpr int g_num_mfcc_features = 13;                                                       // Number of MFCC coefficients (output of DCT)
constexpr int g_num_mfcc_frames = 100;                                                        // Number of time frames in the MFCC matrix
constexpr int g_num_channels = 1;                                                             // Input channel (for CNN)
constexpr int g_input_tensor_size = g_num_mfcc_features * g_num_mfcc_frames * g_num_channels; // 13 * 100 * 1 = 1300
constexpr int g_num_output_classes = 12;                                                      // Number of output classes
constexpr int g_sample_rate = 16000;                                                          // Audio sample rate

// --- Extern const for values that might be "configurable" or from Python ---
// Model Quantization Parameters
extern const float g_input_scale;
extern const int g_input_zero_point;
extern const float g_output_scale;
extern const int g_output_zero_point;

// Model Output Labels
extern const char *const g_category_labels[g_num_output_classes];

// Recognition Confidence Threshold
extern const int8_t g_recognition_threshold_int8;

#endif // MODEL_SETTINGS_H_
