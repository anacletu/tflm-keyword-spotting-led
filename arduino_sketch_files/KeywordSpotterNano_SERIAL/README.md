## MCU Deployment (Arduino Nano 33 BLE Sense)

This section describes how to deploy and run the keyword spotting model on an Arduino Nano 33 BLE Sense. The firmware is located in the `arduino_nano_firmware/KeywordSpotterNano/` directory.

### Prerequisites:

1.  **Hardware:**
    *   Arduino Nano 33 BLE Sense
    *   USB Micro-B Cable
    *   [Your LED setup: e.g., 3x LEDs (Red, Green, Blue) + appropriate resistors, or NeoPixel stick]
    *   Breadboard and Jumper Wires
2.  **Software:**
    *   Arduino IDE (version 2.x recommended)
    *   "Arduino Mbed OS Nano Boards" core installed via Boards Manager.
    *   Official TensorFlow Lite for Microcontrollers library installed (manually from the `tensorflow/tflite-micro-arduino-examples` GitHub repository as per their instructions, since it was removed from the Arduino Library Manager).

### Setup & Running:

1.  **Model Data:** The quantized TFLite model is already converted to a C array in `arduino_nano_firmware/KeywordSpotterNano/model_data.cc`.
2.  **Open Sketch:** Open the `arduino_nano_firmware/KeywordSpotterNano/KeywordSpotterNano.ino` sketch in the Arduino IDE.
3.  **Configure Board & Port:**
    *   In Arduino IDE: **Tools > Board > Arduino Mbed OS Nano Boards > Arduino Nano 33 BLE**.
    *   Select the correct **Tools > Port**.
4.  **Key Configuration in `KeywordSpotterNano.ino` (or related files):**
    *   Ensure `kTensorArenaSize` is adequately defined (e.g., `30 * 1024`).
    *   Verify `INPUT_SCALE`, `INPUT_ZERO_POINT`, `OUTPUT_SCALE`, `OUTPUT_ZERO_POINT` match your quantized model's parameters (these should be hardcoded from your Python evaluation).
    *   Update `kCategoryLabels` to match your 12 output classes.
    *   The OpResolver should include all ops used by your model (e.g., `Conv2D`, `MaxPool2D`, `Mean`, `FullyConnected`, `Softmax`, `Quantize`, `Dequantize`).
5.  **Serial Input for MFCCs (Basic Demo):**
    *   The current sketch is set up to receive pre-computed and quantized INT8 MFCC data via the Serial port from a Python script.
    *   Upload the `KeywordSpotterNano.ino` sketch to your Arduino.
    *   Run the companion Python script (e.g., `notebooks/send_mfcc_to_arduino.py` - you'd create this) to send MFCC data.
    *   Open the Arduino IDE's Serial Monitor (baud rate 115200) to see predictions.
    *   Observe the connected LED(s) for output corresponding to recognized keywords.
6.  **(Optional) Live Microphone Input:**
    *   [If you implement this, describe how to switch to it or if it's the default. Mention any modifications to `arduino_audio_provider.cpp` or `feature_provider.cpp` if you attempt on-device MFCCs.]

### Expected Output:
*   The Arduino will print recognized keywords to the Serial Monitor.
*   Connected LEDs will change based on the recognized command (e.g., "up" lights red LED, "off" turns LEDs off).
*   [Link to a demo video if you make one]