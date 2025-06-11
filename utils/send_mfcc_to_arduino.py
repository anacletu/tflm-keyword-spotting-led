import serial
import time
import numpy as np
import os
import argparse
import sounddevice as sd # For recording
import librosa           # For MFCC calculation

# --- Configuration Parameters ---
SERIAL_PORT = '/dev/cu.usbmodem2201'  # Replace with your Arduino's serial port
BAUD_RATE = 115200

# Model Input Quantization Parameters (from TFLite input_details)
MODEL_INPUT_SCALE = 2.7671010494232178 
MODEL_INPUT_ZERO_POINT = 64             

# Model Input Dimensions & Audio Parameters
N_MFCC = 13
TARGET_FRAMES = 100 # Based on MFCC params (1s audio, 16kHz, 512 FFT, 160 Hop)
NUM_CHANNELS = 1
EXPECTED_INPUT_SIZE_BYTES = N_MFCC * TARGET_FRAMES * NUM_CHANNELS # 1300 bytes

AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION_SECONDS = 1.0 
AUDIO_SAMPLES_TO_RECORD = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION_SECONDS) # 16000 for 1s
AUDIO_SAMPLES_FOR_MFCC = (TARGET_FRAMES - 1) * 160 + 512 # 16352 samples

# Librosa MFCC parameters (to match the Keras model training)
N_FFT_LIBROSA = 512
HOP_LENGTH_LIBROSA = 160
N_MELS_LIBROSA = 40

# Test Samples Directory (for pre-saved MFCCs)
SAMPLES_DIR = 'test_samples_for_arduino'

# Available actions: existing keywords + "record"
AVAILABLE_ACTIONS = ["on", "off", "up", "down", "right", "left", "_silence_", "_unknown_", "record"]

# --- Helper Functions ---
def record_audio_chunk(duration_seconds, sample_rate_hz):
    print(f"\nRecording for {duration_seconds:.1f} second(s)... Speak clearly into the microphone!")
    # Record as int16, which is what PDM on Arduino provides
    audio_data_int16 = sd.rec(int(duration_seconds * sample_rate_hz), 
                               samplerate=sample_rate_hz, 
                               channels=1, 
                               dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    return audio_data_int16.flatten() # Flatten to 1D array

def calculate_mfccs_from_audio(audio_int16, sr, n_mfcc, n_mels, n_fft, hop_length, target_frames):
    print("Calculating MFCCs from recorded audio...")
    # 1. Convert int16 to float32 and normalize (like your TF data pipeline)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    audio_float32 = np.clip(audio_float32, -1.0, 1.0) # Ensure it's strictly in range

    mfccs = librosa.feature.mfcc(
        y=audio_float32,
        sr=sr,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        window='hann',
        center=True
    )
    # mfccs shape is (n_mfcc, num_frames_from_librosa)

    # Pad or truncate MFCC frames to TARGET_FRAMES (e.g., 100)
    current_frames = mfccs.shape[1]
    if current_frames < target_frames:
        padding_width = target_frames - current_frames
        mfccs_processed = np.pad(mfccs, ((0, 0), (0, padding_width)), mode='constant', constant_values=0)
    elif current_frames > target_frames:
        mfccs_processed = mfccs[:, :target_frames]
    else:
        mfccs_processed = mfccs
    
    # Add channel dimension
    mfccs_with_channel = np.expand_dims(mfccs_processed, axis=-1) # Shape (n_mfcc, target_frames, 1)
    print(f"MFCCs calculated with shape: {mfccs_with_channel.shape}")
    return mfccs_with_channel

def load_mfcc_sample_from_file(label_name, base_dir=SAMPLES_DIR):
    # (Your existing function - seems fine)
    filename = os.path.join(base_dir, f"sample_mfcc_{label_name}.npy")
    if not os.path.exists(filename):
        print(f"ERROR: Sample file '{filename}' not found.")
        return None
    print(f"Loading pre-saved MFCC sample for '{label_name}' from {filename}")
    return np.load(filename)

# --- Main Script Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record audio or send a pre-saved quantized MFCC sample to Arduino.")
    parser.add_argument(
        "action", 
        type=str, 
        choices=AVAILABLE_ACTIONS,
        help=f"The action to perform. 'record' to capture live audio, or a keyword to send its saved MFCC. Choose from: {', '.join(AVAILABLE_ACTIONS)}"
    )
    args = parser.parse_args()
    action_to_perform = args.action

    print(f"--- Python MFCC Sender for Arduino TFLM (Action: '{action_to_perform}') ---")

    mfcc_sample_float32_HWC = None # Shape (N_MFCC, TARGET_FRAMES, NUM_CHANNELS)

    if action_to_perform == "record":
        # Record audio, then calculate MFCCs
        raw_audio_int16 = record_audio_chunk(AUDIO_DURATION_SECONDS, AUDIO_SAMPLE_RATE)
        
        # If audio length is less than AUDIO_SAMPLES_FOR_MFCC (16352), pad it
        # This ensures librosa's STFT framing behaves as expected to get enough frames.
        if len(raw_audio_int16) < AUDIO_SAMPLES_FOR_MFCC:
            padding = np.zeros(AUDIO_SAMPLES_FOR_MFCC - len(raw_audio_int16), dtype=np.int16)
            raw_audio_int16_padded = np.concatenate((raw_audio_int16, padding))
        else:
            raw_audio_int16_padded = raw_audio_int16[:AUDIO_SAMPLES_FOR_MFCC] # Ensure exact length

        mfcc_sample_float32_HWC = calculate_mfccs_from_audio(
            raw_audio_int16_padded,
            sr=AUDIO_SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_mels=N_MELS_LIBROSA,
            n_fft=N_FFT_LIBROSA,
            hop_length=HOP_LENGTH_LIBROSA,
            target_frames=TARGET_FRAMES
        )
    else:
        # Load pre-saved MFCC sample
        mfcc_sample_float32_HWC = load_mfcc_sample_from_file(action_to_perform)
    
    if mfcc_sample_float32_HWC is None:
        print("Failed to obtain MFCC data. Exiting.")
        exit(1)

    if mfcc_sample_float32_HWC.shape != (N_MFCC, TARGET_FRAMES, NUM_CHANNELS):
        print(f"ERROR: MFCC sample shape is {mfcc_sample_float32_HWC.shape}, expected {(N_MFCC, TARGET_FRAMES, NUM_CHANNELS)}")
        exit(1)
    print(f"Using float32 MFCC sample (for '{action_to_perform}') with shape: {mfcc_sample_float32_HWC.shape}")

    # 2. Quantize the float32 MFCC sample to INT8
    print(f"Quantizing using scale={MODEL_INPUT_SCALE}, zero_point={MODEL_INPUT_ZERO_POINT}")
    mfcc_sample_quantized_float = (mfcc_sample_float32_HWC / MODEL_INPUT_SCALE) + MODEL_INPUT_ZERO_POINT
    mfcc_sample_quantized_int8_np = np.round(mfcc_sample_quantized_float).astype(np.int8)
    
    mfcc_bytes_to_send = mfcc_sample_quantized_int8_np.tobytes()

    if len(mfcc_bytes_to_send) != EXPECTED_INPUT_SIZE_BYTES:
        print(f"ERROR: Number of bytes to send is {len(mfcc_bytes_to_send)}, expected {EXPECTED_INPUT_SIZE_BYTES}")
        exit(1)
    print(f"MFCC sample for '{action_to_perform}' quantized to {len(mfcc_bytes_to_send)} int8 bytes.")

    # 3. Establish Serial Connection and Send
    try:
        print(f"Attempting to connect to Arduino on {SERIAL_PORT} at {BAUD_RATE} baud...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) 
        time.sleep(2) 
        print(f"Successfully connected to {SERIAL_PORT}.")
        
        if ser.in_waiting > 0:
            print(f"Clearing {ser.in_waiting} bytes: {ser.read(ser.in_waiting).decode('latin-1', errors='replace')}")

        print("\n--- Initiating Data Transfer ---")
        ser.write(b'S')
        ser.flush()
        print("Sent 'S' (Start) byte to Arduino.")

        print("Waiting for 'R' (Ready) ACK from Arduino...")
        ready_ack_received = False
        ack_timeout_start = time.time()
        while time.time() - ack_timeout_start < 3: 
            if ser.in_waiting > 0:
                response_byte = ser.read(1)
                if response_byte == b'R':
                    print("Received 'R' (Ready) ACK from Arduino.")
                    ready_ack_received = True
                    break
                else:
                    print(f"Unexpected byte while waiting for 'R': {response_byte}")
            time.sleep(0.01)
        
        if not ready_ack_received:
            print("ERROR: Did not receive 'R' (Ready) ACK from Arduino. Aborting.")
            exit(1)

        print(f"Sending {len(mfcc_bytes_to_send)} bytes for '{action_to_perform}' (chunked)...")
        CHUNK_SIZE_PY = 32 # Smaller chunks for more reliable sending
        DELAY_CHUNK_PY = 0.01 # 10ms delay
        for i in range(0, len(mfcc_bytes_to_send), CHUNK_SIZE_PY):
            chunk = mfcc_bytes_to_send[i:i + CHUNK_SIZE_PY]
            ser.write(chunk)
            # ser.flush() # Optional: flushing every chunk can be slow but robust
            time.sleep(DELAY_CHUNK_PY)
        ser.flush() # Important final flush
        print("All MFCC data bytes sent.")
        time.sleep(0.1) # Small pause for Arduino to process last bytes before it sends ACKs

        print("\nWaiting for final response from Arduino (Ctrl+C to stop)...")
        try:
            while True:
                if ser.in_waiting > 0:
                    response_line = ser.readline().decode('utf-8', errors='replace').strip()
                    if response_line:
                        print(f"Arduino: {response_line}")
                        if response_line.startswith("ACK_"): # Break after first full ACK
                            break 
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nUser interrupted listener loop.")

    except serial.SerialException as e:
        print(f"SERIAL ERROR: {e}")
        print("- Ensure Arduino is connected and correct SERIAL_PORT is selected.")
        print("- Make sure Arduino IDE's Serial Monitor is CLOSED.")
    except Exception as e_gen:
        print(f"An unexpected error occurred: {e_gen}")
        import traceback
        traceback.print_exc()
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed.")