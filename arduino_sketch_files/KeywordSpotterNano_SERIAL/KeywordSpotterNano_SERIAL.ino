/* Copyright 2025 github/anacletu
    Project: tflm-keyword-spotting-led
    Licensed under the MIT License
*/

#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// MODEL AND APPLICATION SETTINGS
#include "model_data.h" // C array from xxd
#include "model_settings.h"

// LED Pin Configuration
const int redLedPin = D2;             // Digital pin 2
const int greenLedPin = D3;           // Digital pin 3
const int blueLedPin = D4;            // Digital pin 4
const int statusLedPin = LED_BUILTIN; // Using the onboard LED for status

// TFLM GLOBALS
namespace
{ // Anonymous namespace for local linkage
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *model_input = nullptr;
  TfLiteTensor *model_output = nullptr;

  // Tensor Arena: Memory for input, output, and intermediate tensors.
  constexpr int kTensorArenaSize = 45 * 1024; // Based on a 33KB model
  alignas(16) uint8_t tensor_arena[kTensorArenaSize];
} // anonymous namespace

// SETUP FUNCTION
void setup()
{
  Serial.begin(115200);

  long serial_connect_start_time = millis();
  while (!Serial)
  { // Loop as long as Serial is not yet connected
    if (millis() - serial_connect_start_time > 5000)
    { // Check if 5 seconds have passed
      break;
    }
    delay(100);
  }

  // Initialize LED pins
  pinMode(redLedPin, OUTPUT);
  digitalWrite(redLedPin, LOW);
  pinMode(greenLedPin, OUTPUT);
  digitalWrite(greenLedPin, LOW);
  pinMode(blueLedPin, OUTPUT);
  digitalWrite(blueLedPin, LOW);
  pinMode(statusLedPin, OUTPUT);
  digitalWrite(statusLedPin, LOW); // Status LED off

  Serial.println("--- Keyword Spotter Nano (PC-Sent INT8 MFCCs) ---");
  Serial.println("Initializing TFLM System...");

  // Load the model from the C array (defined in model_data.cc)
  model = tflite::GetModel(keyword_spotting_model_0_87_int8_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    Serial.print("ERROR: Model schema version mismatch! Model: ");
    Serial.print(model->version());
    Serial.print(", Supported: ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (1)
      ; // Halt
  }
  Serial.println("Model loaded successfully.");

  static tflite::MicroMutableOpResolver<10> op_resolver;
  op_resolver.AddConv2D();         // For layers.Conv2D
  op_resolver.AddMaxPool2D();      // For layers.MaxPooling2D
  op_resolver.AddMean();           // GlobalAveragePooling2D often uses this
  op_resolver.AddFullyConnected(); // For layers.Dense
  op_resolver.AddSoftmax();        // For output activation
  op_resolver.AddQuantize();       // For INT8 models (input/output quantization)
  op_resolver.AddDequantize();     // For INT8 models (if any op needs float temporarily)
  op_resolver.AddReshape();        // Often used implicitly or by Flatten/GAP

  // Build an interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory for tensors
  if (interpreter->AllocateTensors() != kTfLiteOk)
  {
    Serial.println("ERROR: AllocateTensors() failed! Increase kTensorArenaSize?");
    while (1)
      ; // Halt
  }
  Serial.println("Tensors allocated.");

  // Get pointers to input and output tensors
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  // Verify input tensor details
  Serial.print("Input tensor dimensions: ");
  Serial.println(model_input->dims->size);
  bool input_params_ok = true;
  if (model_input->dims->size != 4)
  {
    Serial.println("ERROR: Input dims->size != 4");
    input_params_ok = false;
  }
  if (model_input->dims->data[0] != 1)
  {
    Serial.println("ERROR: Input batch_size != 1");
    input_params_ok = false;
  }
  if (model_input->dims->data[1] != g_num_mfcc_features)
  {
    Serial.print("ERROR: Input height != ");
    Serial.println(g_num_mfcc_features);
    input_params_ok = false;
  }
  if (model_input->dims->data[2] != g_num_mfcc_frames)
  {
    Serial.print("ERROR: Input width != ");
    Serial.println(g_num_mfcc_frames);
    input_params_ok = false;
  }
  if (model_input->dims->data[3] != g_num_channels)
  {
    Serial.print("ERROR: Input channels != ");
    Serial.println(g_num_channels);
    input_params_ok = false;
  }
  if (model_input->type != kTfLiteInt8)
  {
    Serial.println("ERROR: Input type != kTfLiteInt8");
    input_params_ok = false;
  }

  // Check quantization parameters of the model's input tensor
  if (abs(model_input->params.scale - g_input_scale) > 1e-6)
  { // Compare floats with tolerance
    Serial.print("ERROR: Input scale mismatch! Model expects: ");
    Serial.print(model_input->params.scale, 7);
    Serial.print(" You defined: ");
    Serial.print(g_input_scale, 7);
    input_params_ok = false;
  }
  if (model_input->params.zero_point != g_input_zero_point)
  {
    Serial.print("ERROR: Input zero_point mismatch! Model expects: ");
    Serial.print(model_input->params.zero_point);
    Serial.print(" You defined: ");
    Serial.print(g_input_zero_point);
    input_params_ok = false;
  }

  if (!input_params_ok)
  {
    Serial.println("Halting due to input parameter mismatch.");
    while (1)
      ; // Halt
  }
  Serial.println("Input tensor parameters look OK.");

  Serial.println("\nSetup complete. Waiting for 'START' command from PC...");
  Serial.print("Expected ");
  Serial.print(g_input_tensor_size);
  Serial.println(" bytes per inference.");
}

// LOOP FUNCTION
int8_t incoming_mfcc_buffer[g_input_tensor_size]; // g_input_tensor_size is constexpr from .h

void loop()
{
  static unsigned long last_loop_print_time = 0;

  if (millis() - last_loop_print_time > 2000)
  {
    Serial.println("Arduino loop running, waiting for 'S' command from PC...");
    last_loop_print_time = millis();
  }

  if (Serial.available() > 0)
  {
    if (Serial.read() == 'S')
    {
      digitalWrite(statusLedPin, HIGH);

      while (Serial.available())
        Serial.read(); // Clear any trailing bytes after 'S'

      // *** SEND 'R' (Ready) ACK back to Python ***
      Serial.write('R');
      Serial.flush(); // Ensure 'R' is sent
      Serial.println("DEBUG: Sent 'R' (Ready) ACK to PC. Waiting for MFCC data...");
      // The Serial.println above also helps Python see the 'R' if it's reading lines

      Serial.setTimeout(2000); // Increased timeout for readBytes (2 seconds)
      int bytes_read = Serial.readBytes((char *)incoming_mfcc_buffer, g_input_tensor_size);

      Serial.print("START sequence processed. Attempted to read data. Result: "); // Renamed from "START byte received"
      Serial.print(bytes_read);
      Serial.println(" bytes received.");

      digitalWrite(statusLedPin, LOW); // Turn off status LED after reception attempt

      if (bytes_read == g_input_tensor_size)
      {
        Serial.print("DEBUG: Full MFCC frame (");
        Serial.print(bytes_read);
        Serial.println(" bytes) received from PC.");
        unsigned long total_start_time = micros(); // Measure total time from here

        // Copy the received data from our buffer to the model's input tensor
        memcpy(model_input->data.int8, incoming_mfcc_buffer, g_input_tensor_size);
        Serial.println("MFCC data copied from local buffer to model input tensor.");

        // Run inference
        unsigned long inference_start_time = micros(); // For timing inference
        if (interpreter->Invoke() != kTfLiteOk)
        {
          Serial.println("ERROR: Invoke() failed!");
          return;
        }
        unsigned long inference_duration_us = micros() - inference_start_time;
        Serial.print("Inference successful. Duration: ");
        Serial.print(inference_duration_us);
        Serial.println(" us.");

        // Process the output tensor (which contains INT8 logits)
        int8_t max_score_val = -128;       // Initialize with smallest possible int8 value
        int predicted_category_index = -1; // Initialize to an invalid index

        for (int i = 0; i < g_num_output_classes; ++i)
        {
          int8_t current_score = model_output->data.int8[i];
          if (current_score > max_score_val)
          {
            max_score_val = current_score;
            predicted_category_index = i;
          }
        }

        // Confidence threshold
        if (max_score_val < g_recognition_threshold_int8)
        {
          Serial.print("Prediction confidence too low (score: ");
          Serial.print(max_score_val);

          float dequantized_max_prob = ((float)max_score_val - (float)g_output_zero_point) * g_output_scale; // Print the dequantized probability for clarity during debugging
          Serial.print(", approx. float prob: ");
          Serial.print(dequantized_max_prob, 4);

          Serial.println("). Skipping LED action.");
          // Send an ACK-low confidence response
          Serial.println("ACK_LOW");
          return;
        }

        // Handle LED output based on the prediction
        if (predicted_category_index != -1)
        {
          Serial.print("===> Predicted command: '");
          Serial.print(g_category_labels[predicted_category_index]);
          Serial.print("' (Raw INT8 Score: ");
          Serial.print(max_score_val);
          Serial.println(")");

          // Turn all color LEDs off before setting the new one
          digitalWrite(redLedPin, LOW);
          digitalWrite(greenLedPin, LOW);
          digitalWrite(blueLedPin, LOW);

          // LED control logic
          const char *command = g_category_labels[predicted_category_index];
          if (strcmp(command, "left") == 0)
          { // turn on the red LED for "left"
            digitalWrite(redLedPin, HIGH);
            Serial.println("Action: RED LED ON");
          }
          else if (strcmp(command, "up") == 0)
          { // turn green on for "up"
            digitalWrite(greenLedPin, HIGH);
            Serial.println("Action: GREEN LED ON");
          }
          else if (strcmp(command, "right") == 0)
          { // turn blue on for "right"
            digitalWrite(blueLedPin, HIGH);
            Serial.println("Action: BLUE LED ON");
          }
          else if (strcmp(command, "on") == 0)
          { // turn on all LEDs for "on"
            digitalWrite(redLedPin, HIGH);
            digitalWrite(greenLedPin, HIGH);
            digitalWrite(blueLedPin, HIGH);
            Serial.println("Action: ALL LEDs ON");
          }
          else if (strcmp(command, "off") == 0)
          { // turn off all LEDs for "off"
            Serial.println("Action: ALL LEDs OFF");
            digitalWrite(redLedPin, LOW);
            digitalWrite(greenLedPin, LOW);
            digitalWrite(blueLedPin, LOW);
          }
          else if (strcmp(command, "_silence_") == 0)
          { // special case for silence
            Serial.println("Action: SILENCE detected, no LED action.");
          }
          else if (strcmp(command, "_unknown_") == 0)
          { // special case for unknown
            Serial.println("Action: UNKNOWN command, no LED action.");
          }
          else
          { // Unknown Command Handling or other
            Serial.print("Unrecognized command label or other: '");
            Serial.print(command);
            Serial.println("'. No specific LED action.");
          }
          // Send an ACK-success response
          Serial.println("ACK_OK");
        }
        else
        { // Should not happen if predicted_category_index is initialized to -1 and max_score_val is high
          Serial.println("Error: Predicted category index invalid despite high score.");
          Serial.println("ACK_ERROR");
        }

        // Send an ACK-success response
        Serial.println("\nWaiting for next 'START' command from PC...");
        unsigned long total_duration_us = micros() - total_start_time;
        Serial.print("Total latency (Serial Rx + Inference + Serial Tx): ");
        Serial.print(total_duration_us);
        Serial.println(" us.");
      }
      else
      { // Bytes_read was NOT equal to g_input_tensor_size
        Serial.print("ERROR: Data reception failed. Expected ");
        Serial.print(g_input_tensor_size);
        Serial.print(" bytes, but received ");
        Serial.print(bytes_read);
        Serial.println(" bytes.");
        // Clear anything left in the buffer to avoid carrying over corrupted data
        while (Serial.available())
          Serial.read();            // Clear remaining data
        Serial.println("ACK_FAIL"); // Send failure acknowledgment
        Serial.println("\nWaiting for next 'S' command from PC...");
      }
    } // End of if (Serial.read() == 'S')
  } // End of if (Serial.available() > 0) for initial 'S'
}