/*  Copyright 2025 github/anacletu
    Project: tflm-keyword-spotting-led
    Licensed under the MIT License
*/

#/*  model_settings.cpp */
#include "model_settings.h"

// --- UPDATED VALUES from new Python TFLite INT8 evaluation ---
// Input details: 'quantization': (2.7671010494232178, 64)
const float g_input_scale = 2.7671010494232178f; // Added 'f', using full precision
const int g_input_zero_point = 64;

// Output details: 'quantization': (0.00390625, -128)
const float g_output_scale = 0.00390625f;
const int g_output_zero_point = -128;

// Output classes
const char *const g_category_labels[g_num_output_classes] = {
    "down", "go", "left", "no", "off", "on",
    "right", "stop", "up", "yes", "_silence_", "_unknown_"};

// Recognition Confidence Threshold
// For ~70% based on the output scale/zp:
// int8_threshold = round(0.5 / 0.00390625) + (-128) = ~0
const int8_t g_recognition_threshold_int8 = 0; // Updated to 0 based on new evaluation