# Utility Scripts

This directory contains utility scripts for the "Efficient Keyword Spotting for LED Control" project.

## `send_mfcc_to_arduino.py`

This Python script is used to test the deployed TensorFlow Lite Micro model on the Arduino Nano 33 BLE Sense (or compatible board).

**Purpose:**

- Allows you to send pre-processed and quantized MFCC (Mel-Frequency Cepstral Coefficients) data from your PC to the Arduino via a USB serial connection.
- Alternatively, it can record live audio from your PC's microphone, process it into MFCCs, quantize them, and then send them.
- The Arduino receives this data, performs inference using the onboard TFLM model, and controls LEDs based on the recognized keyword.
- The script also listens for and displays responses (like the predicted keyword and ACK messages) from the Arduino.

**How to Use:**

1.  **Prerequisites:**

    - Ensure your Arduino board is programmed with the corresponding `KeywordSpotterNano_SERIAL.ino` sketch (located in `arduino_sketch_files/KeywordSpotterNano_SERIAL/`).
    - Connect the Arduino to your PC via USB.
    - **Important:** Close the Arduino IDE's Serial Monitor (or any other program using the Arduino's serial port).
    - Make sure you have the necessary Python packages installed (see main project `requirements.txt`, especially `pyserial`, `numpy`, `sounddevice`, `librosa`).
    - If sending pre-saved MFCCs, ensure the `.npy` files (e.g., `sample_mfcc_on.npy`) are present in the `test_samples_for_arduino/` subdirectory (relative to this script).

2.  **Update `SERIAL_PORT`:**

    - Open `send_mfcc_to_arduino.py` and modify the `SERIAL_PORT` variable at the top to match the serial port your Arduino is connected to (e.g., `/dev/cu.usbmodemXXXXX` on macOS/Linux, `COMX` on Windows).

3.  **Run from Terminal:**

    - Navigate to this `utils/` directory in your terminal.
    - Activate your Python virtual environment.
    - Execute the script, providing an action as a command-line argument:

    - **To send a pre-saved MFCC sample (e.g., for the keyword "on"):**

      ```bash
      python send_mfcc_to_arduino.py on
      ```

      Replace `"on"` with any of the available keywords for which you have a saved `.npy` sample (e.g., "off", "up", "_silence_").

    - **To record live audio from your PC microphone, process it, and send:**
      ```bash
      python send_mfcc_to_arduino.py record
      ```
      You will be prompted to speak into the microphone.

4.  **Observe:**
    - The Python script's console output will show connection status, data sending, and the Arduino's responses (including the predicted keyword).
    - The LEDs connected to your Arduino should react based on the recognized keyword.

This script provides a way to perform end-to-end testing of the on-MCU inference pipeline using MFCC features generated on the PC.
