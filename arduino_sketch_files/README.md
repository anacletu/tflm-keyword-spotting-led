## MCU Deployment: Keyword Spotter on Arduino Nano 33 BLE Sense

The directory (`arduino_sketch_files/KeywordSpotterNano_SERIAL/`) contains the Arduino firmware to run the trained keyword spotting model on an Arduino Nano 33 BLE Sense.

**Current Implementation: PC-Driven MFCCs via Serial**
Due to challenges and time constraints in perfectly matching on-device MFCC (Mel-Frequency Cepstral Coefficients) extraction with the `librosa`-based features used for training the Keras model, this demonstration relies on a companion Python script to perform live audio capture (from PC microphone) and MFCC processing. The resulting quantized INT8 MFCC data is then sent to the Arduino via the USB Serial port for inference. This setup focuses on demonstrating the TFLM model execution, keyword recognition logic, and hardware (LED) interaction on the microcontroller. On-device MFCC extraction remains an area for future work.

### Prerequisites:

1.  **Hardware:**
    - Arduino Nano 33 BLE Sense (or Nano 33 BLE Sense Rev2).
    - USB Micro-B Data Cable.
    - LEDs for output:
      - 1x Red LED
      - 1x Green LED
      - 1x Blue LED
    - 3x ~220 Ohm current-limiting resistors.
    - Solderless Breadboard.
    - Jumper Wires.
2.  **Software (Arduino IDE Setup):**
    - Arduino IDE (version 2.3.x recommended).
    - **Board Core:** "Arduino Mbed OS Nano Boards" installed via Arduino IDE's Boards Manager.
    - **TensorFlow Lite Library:** A compatible TFLM library for Arduino.
      - _Note:_ Direct installation via Arduino Library Manager can be inconsistent. This project was successfully compiled using the **`Chirale_TensorFlowLite` library** (installable via "Add .ZIP Library..." from its GitHub repository: `https://github.com/spaziochirale/Chirale_TensorFlowLite`) after ensuring any previous TFLM library installations were removed. Other TFLM library distributions might also work.
3.  **Software (Python Environment for Sender Script):**
    - Python 3.9+ (this project used 3.11).
    - Required Python packages: `pyserial`, `numpy`, `sounddevice`, `librosa` (see main project `requirements.txt`).

### Files in this Sketch Directory:

- `KeywordSpotterNano_SERIAL.ino`: The main Arduino sketch.
- `model_data.h` & `model_data.cc`: Contains the INT8 quantized TFLite model as a C byte array (generated by `xxd`).
- `model_settings.h` & `model_settings.cpp`: Defines model-specific constants (input/output quantization parameters, dimensions, class labels, confidence threshold).

### Setup & Running the Demo:

1.  **Wire LEDs:**
    - Connect the Red, Green, and Blue LEDs (each with a series ~220 Ohm resistor) to Arduino digital pins D2, D3, and D4 respectively, with cathodes to GND.
2.  **Open Sketch:** Open `KeywordSpotterNano_SERIAL.ino` in the Arduino IDE.
3.  **Verify Constants in `model_settings.cpp`:**
    - Ensure `g_input_scale`, `g_input_zero_point`, `g_output_scale`, `g_output_zero_point`, `g_category_labels`, and other dimension constants precisely match the INT8 TFLite model that `model_data.cc` was generated from.
    - Check that `g_recognition_threshold_int8` is set to a sensible value (e.g., `51` for ~70% float probability with the model's output parameters).
4.  **Verify Model Array Name:** In `KeywordSpotterNano_SERIAL.ino`, ensure the `tflite::GetModel()` call uses the correct C array name from `model_data.cc` (e.g., `keyword_spotting_model_87_int8_tflite`).
5.  **Configure Board & Port in Arduino IDE:**
    - **Tools > Board > Arduino Mbed OS Nano Boards > Arduino Nano 33 BLE**.
    - Select the correct **Tools > Port** for your connected Nano.
6.  **Compile and Upload:** Click the "Upload" button in the Arduino IDE.
7.  **Check Arduino Serial Monitor (Briefly):**
    - Open the Arduino IDE's Serial Monitor (baud rate 115200).
    - You should see startup messages from the `setup()` function, ending with:
      ```
      Setup complete. Waiting for 'START' command from PC...
      Expected 1300 bytes per inference.
      ```
    - **Close the Arduino Serial Monitor** to free up the port for the Python script.
8.  **Run the Python Sender Script:**
    - Navigate to the directory containing the `send_mfcc_to_arduino.py` script (e.g., `project_root/utils/` from the main project).
    - Ensure the `SERIAL_PORT` variable in the Python script matches your Arduino's port.
    - Execute the script from your terminal (with the correct Python environment activated):
      - To send a pre-saved MFCC sample (e.g., for "on"):
        ```bash
        python send_mfcc_to_arduino.py on
        ```
      - To record live audio from your PC's microphone, process it, and send:
        ```bash
        python send_mfcc_to_arduino.py record
        ```
        (Speak the keyword when prompted by the script).

### Expected Behavior:

- **Python Console:**
  - Will show connection status, data sending progress.
  - Will then display messages received from the Arduino, including "DEBUG: Sent 'R' (Ready) ACK...", "START sequence processed...", inference duration, the predicted command, the LED action taken, and an "ACK_OK/ACK_LOW/ACK_FAIL" status.
- **Arduino LEDs:**
  - The onboard status LED (`LED_BUILTIN`) will light up during data reception and inference.
  - The connected Red, Green, or Blue LEDs will light up based on the recognized keyword if the prediction confidence is above the threshold. For example:
    - "left": Red LED (or your defined "up" state)
    - "up": Green LED (or your defined "on" state)
    - "off": All color LEDs off (or your defined "off" state)
- **Arduino Serial Monitor (if you were to reopen it after Python script finishes):** You would see the detailed log from the Arduino's perspective.
