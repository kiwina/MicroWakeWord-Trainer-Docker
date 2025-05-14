#!/usr/bin/env python
# coding: utf-8

# <div align="center">
#   <img src="https://raw.githubusercontent.com/MasterPhooey/MicroWakeWord-Trainer-Docker/refs/heads/main/mmw.png" alt="MicroWakeWord Trainer Logo" width="100" />
#   <h1>MicroWakeWord Trainer Docker</h1>
# </div>
#
# This notebook steps you through training a basic microWakeWord model. It is intended as a **starting point** for advanced users. You should use Python 3.11.
#
# **The model generated will most likely not be usable for everyday use; it may be difficult to trigger or falsely activates too frequently. You will most likely have to experiment with many different settings to obtain a decent model!**
#
# In the comment at the start of certain blocks, I note some specific settings to consider modifying.
#
# At the end of this notebook, you will be able to download a tflite file. To use this in ESPHome, you need to write a model manifest JSON file. See the [ESPHome documentation](https://esphome.io/components/micro_wake_word) for the details and the [model repo](https://github.com/esphome/micro-wake-word-models/tree/main/models/v2) for examples.

import os
import sys
import subprocess
from pathlib import Path
import yaml
import json

# Conditional import for display, works in Jupyter, no-op in pure script
try:
    from IPython.display import Audio, display, FileLink
except ImportError:
    def display(*args, **kwargs): pass
    def Audio(*args, **kwargs): pass
    def FileLink(*args, **kwargs): pass

print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}") # Should be /data in Docker

# --- Call External Data Preparation Script ---
PREPARE_DATA_SCRIPT_PATH = Path("/data/prepare_local_data.py")
DATA_DIR_INSIDE_DOCKER = Path("/data")

print(f"\n--- Running Data Preparation Script: {PREPARE_DATA_SCRIPT_PATH} ---")
if PREPARE_DATA_SCRIPT_PATH.exists():
    try:
        completed_process = subprocess.run(
            [sys.executable, str(PREPARE_DATA_SCRIPT_PATH), "--data-dir", str(DATA_DIR_INSIDE_DOCKER)],
            capture_output=True, text=True, check=True
        )
        print("Data preparation script stdout:")
        print(completed_process.stdout)
        if completed_process.stderr:
            print("Data preparation script stderr:")
            print(completed_process.stderr)
        print("Data preparation script finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Data preparation script failed with exit code {e.returncode}")
        print("Stdout:", e.stdout)
        print("Stderr:", e.stderr)
        sys.exit("Data preparation failed, cannot continue.")
    except FileNotFoundError:
        print(f"ERROR: Python executable not found at {sys.executable}")
        sys.exit("Python executable not found.")
else:
    print(f"ERROR: {PREPARE_DATA_SCRIPT_PATH} not found. Please ensure it's copied to /data by startup.sh.")
    sys.exit(f"{PREPARE_DATA_SCRIPT_PATH} not found.")
print("--- Data Preparation Finished ---\n")

# --- Notebook Steps ---

# Define target_word (user should modify this)
target_word = 'khum_puter'  # Phonetic spellings may produce better samples

# Define paths based on DATA_DIR_INSIDE_DOCKER
PIPER_SAMPLE_GENERATOR_DIR = DATA_DIR_INSIDE_DOCKER / "piper-sample-generator"
PIPER_SCRIPT_PATH = PIPER_SAMPLE_GENERATOR_DIR / "generate_samples.py"

# Updated generated samples directory structure
GENERATED_SAMPLES_BASE_DIR = DATA_DIR_INSIDE_DOCKER / "generated_samples" / target_word
TEST_SAMPLE_OUTPUT_DIR = GENERATED_SAMPLES_BASE_DIR / "test"
WW_SAMPLES_OUTPUT_DIR = GENERATED_SAMPLES_BASE_DIR / "samples"

MIT_RIRS_PATH = DATA_DIR_INSIDE_DOCKER / "mit_rirs"
FMA_16K_PATH = DATA_DIR_INSIDE_DOCKER / "fma_16k"
AUDIOSONET_16K_PATH = DATA_DIR_INSIDE_DOCKER / "audioset_16k"
AUGMENTED_FEATURES_DIR = DATA_DIR_INSIDE_DOCKER / "generated_augmented_features"
NEGATIVE_DATASETS_PATH = DATA_DIR_INSIDE_DOCKER / "negative_datasets"
TRAINED_MODELS_BASE_PATH = DATA_DIR_INSIDE_DOCKER / "trained_models"
TRAINING_CONFIG_PATH = DATA_DIR_INSIDE_DOCKER / "training_parameters.yaml"

# Create directories
TEST_SAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WW_SAMPLES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUGMENTED_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
(TRAINED_MODELS_BASE_PATH / "wakeword").mkdir(parents=True, exist_ok=True)

# Ensure piper-sample-generator scripts are callable and model exists
if not PIPER_SCRIPT_PATH.exists():
    print(f"ERROR: Piper script not found at {PIPER_SCRIPT_PATH}. Check data preparation.")
    sys.exit("Piper script missing.")
if not (PIPER_SAMPLE_GENERATOR_DIR / "models" / "en_US-libritts_r-medium.pt").exists():
    print(f"ERROR: Piper model not found in {PIPER_SAMPLE_GENERATOR_DIR / 'models'}. Check data preparation.")
    sys.exit("Piper model missing.")

if str(PIPER_SAMPLE_GENERATOR_DIR) not in sys.path:
    sys.path.append(str(PIPER_SAMPLE_GENERATOR_DIR))
    print(f"Added {PIPER_SAMPLE_GENERATOR_DIR} to sys.path")

# Generates 1 sample of the target word for manual verification.
print(f"\n--- Generating Test Sample for '{target_word}' ---")
cmd_test_sample = [sys.executable, str(PIPER_SCRIPT_PATH), target_word, "--max-samples", "1", "--batch-size", "1", "--output-dir", str(TEST_SAMPLE_OUTPUT_DIR)]
subprocess.run(cmd_test_sample, check=True)
audio_path_test = TEST_SAMPLE_OUTPUT_DIR / "0.wav"
if audio_path_test.exists():
    print(f"Playing test sample: {audio_path_test}")
    display(Audio(str(audio_path_test), autoplay=True))
else:
    print(f"Audio file not found at {audio_path_test}.")

# Generates a larger amount of wake word samples.
print(f"\n--- Generating Wake Word Samples for '{target_word}' ---")
cmd_ww_samples = [sys.executable, str(PIPER_SCRIPT_PATH), target_word, "--max-samples", "1000", "--batch-size", "100", "--output-dir", str(WW_SAMPLES_OUTPUT_DIR)]
subprocess.run(cmd_ww_samples, check=True)
print(f"Wake word samples generated in {WW_SAMPLES_OUTPUT_DIR}")

# Sets up the augmentations.
print("\n--- Setting up Augmentations ---")
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration

clips = Clips(input_directory=str(WW_SAMPLES_OUTPUT_DIR), file_pattern='*.wav', max_clip_duration_s=None, remove_silence=False, random_split_seed=10, split_count=0.1)
augmenter = Augmentation(
    augmentation_duration_s=3.2,
    augmentation_probabilities={"SevenBandParametricEQ":0.1,"TanhDistortion":0.1,"PitchShift":0.1,"BandStopFilter":0.1,"AddColorNoise":0.1,"AddBackgroundNoise":0.75,"Gain":1.0,"RIR":0.5},
    impulse_paths=[str(MIT_RIRS_PATH)], background_paths=[str(FMA_16K_PATH), str(AUDIOSONET_16K_PATH)],
    background_min_snr_db=-5, background_max_snr_db=10, min_jitter_s=0.195, max_jitter_s=0.205
)
print("Augmentation setup complete.")

# Augment a random clip and play it back
print("\n--- Augmenting and Playing Test Clip ---")
from microwakeword.audio.audio_utils import save_clip
augmented_clip_path_test = DATA_DIR_INSIDE_DOCKER / "augmented_clip_test.wav" # Save to /data root for easy access
try:
    random_clip_data = clips.get_random_clip()
    augmented_clip_data = augmenter.augment_clip(random_clip_data)
    save_clip(augmented_clip_data, str(augmented_clip_path_test))
    print(f"Playing augmented test clip: {augmented_clip_path_test}")
    display(Audio(str(augmented_clip_path_test), autoplay=True))
except Exception as e:
    print(f"Error during test augmentation: {e}.")

# Augment samples and save the training, validation, and testing sets.
print("\n--- Generating Augmented Features for Training/Validation/Testing ---")
from mmap_ninja.ragged import RaggedMmap
AUGMENTED_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
splits_config = ["training", "validation", "testing"]
for split_item in splits_config:
    out_dir_split_item = AUGMENTED_FEATURES_DIR / split_item
    out_dir_split_item.mkdir(parents=True, exist_ok=True)
    current_split_name = "train"; current_repetition = 2
    current_spectrograms = SpectrogramGeneration(clips=clips, augmenter=augmenter, slide_frames=10, step_ms=10)
    if split_item == "validation": current_split_name = "validation"; current_repetition = 1
    elif split_item == "testing":
        current_split_name = "test"; current_repetition = 1
        current_spectrograms = SpectrogramGeneration(clips=clips, augmenter=augmenter, slide_frames=1, step_ms=10)
    print(f"Generating augmented features for {current_split_name} set...")
    try:
        RaggedMmap.from_generator(
            out_dir=str(out_dir_split_item / 'wakeword_mmap'),
            sample_generator=current_spectrograms.spectrogram_generator(split=current_split_name, repeat=current_repetition),
            batch_size=100, verbose=True,
        )
        print(f"Finished generating features for {current_split_name} set.")
    except Exception as e: print(f"Error generating features for {current_split_name} set: {e}")

# Save a yaml config that controls the training process
print("\n--- Preparing Training Configuration ---")
config_train = {
    "window_step_ms": 10, "train_dir": str(TRAINED_MODELS_BASE_PATH / "wakeword"),
    "features": [
        {"features_dir":str(AUGMENTED_FEATURES_DIR),"sampling_weight":2.0,"penalty_weight":1.0,"truth":True,"truncation_strategy":"truncate_start","type":"mmap"},
        {"features_dir":str(NEGATIVE_DATASETS_PATH/"speech"),"sampling_weight":10.0,"penalty_weight":1.0,"truth":False,"truncation_strategy":"random","type":"mmap"},
        {"features_dir":str(NEGATIVE_DATASETS_PATH/"dinner_party"),"sampling_weight":10.0,"penalty_weight":1.0,"truth":False,"truncation_strategy":"random","type":"mmap"},
        {"features_dir":str(NEGATIVE_DATASETS_PATH/"no_speech"),"sampling_weight":5.0,"penalty_weight":1.0,"truth":False,"truncation_strategy":"random","type":"mmap"},
        {"features_dir":str(NEGATIVE_DATASETS_PATH/"dinner_party_eval"),"sampling_weight":0.0,"penalty_weight":1.0,"truth":False,"truncation_strategy":"split","type":"mmap"},
    ],
    "training_steps":[10000], "positive_class_weight":[1], "negative_class_weight":[20],
    "learning_rates":[0.001], "batch_size":128,
    "time_mask_max_size":[0],"time_mask_count":[0],"freq_mask_max_size":[0],"freq_mask_count":[0],
    "eval_step_interval":500, "clip_duration_ms":1500,
    "target_minimization":0.9, "minimization_metric":None, "maximization_metric":"average_viable_recall"
}
with open(TRAINING_CONFIG_PATH, "w") as file_yaml: yaml.dump(config_train, file_yaml)
print(f"Training parameters saved to {TRAINING_CONFIG_PATH}")

# Trains a model.
print("\n--- Starting Model Training ---")
cmd_train_model = [
    sys.executable, "-m", "microwakeword.model_train_eval",
    "--training_config", str(TRAINING_CONFIG_PATH), "--train", "1", "--restore_checkpoint", "1",
    "--test_tf_nonstreaming","0","--test_tflite_nonstreaming","0","--test_tflite_nonstreaming_quantized","0",
    "--test_tflite_streaming","0","--test_tflite_streaming_quantized","1","--use_weights","best_weights",
    "mixednet", "--pointwise_filters","64,64,64,64", "--repeat_in_block","1,1,1,1",
    "--mixconv_kernel_sizes","[5],[7,11],[9,15],[23]", "--residual_connection","0,0,0,0",
    "--first_conv_filters","32", "--first_conv_kernel_size","5", "--stride","3"
]
subprocess.run(cmd_train_model, check=True)
print("Model training/evaluation finished.")

# Prepare model files for download/use
print("\n--- Preparing Output Model Files ---")
source_tflite_path = TRAINED_MODELS_BASE_PATH / "wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
destination_tflite_path = DATA_DIR_INSIDE_DOCKER / "stream_state_internal_quant.tflite"
if source_tflite_path.exists():
    shutil.copy(source_tflite_path, destination_tflite_path)
    print(f"Copied TFLite model to {destination_tflite_path}")
else:
    print(f"ERROR: Trained TFLite model not found at {source_tflite_path}")

json_output_data = {
    "type":"micro", "wake_word":target_word, "author":"kiwina",
    "website":"https://github.com/kiwina/MicroWakeWord-Trainer-Docker",
    "model":"stream_state_internal_quant.tflite", "trained_languages":["en"], "version":1, # Start at 1
    "micro":{"probability_cutoff":0.97,"sliding_window_size":5,"feature_step_size":10,"tensor_arena_size":30000,"minimum_esphome_version":"2024.7.0"}
}
destination_json_path = DATA_DIR_INSIDE_DOCKER / "stream_state_internal_quant.json"
with open(destination_json_path, "w") as json_file_out: json.dump(json_output_data, json_file_out, indent=2)
print(f"Created JSON metadata at {destination_json_path}")

print("\n--- Script Finished ---")
print(f"Output files are located in: {DATA_DIR_INSIDE_DOCKER.resolve().absolute()}")
print("If running in Docker, this corresponds to the 'microwakeword-trainer-data' directory on your host machine.")

if destination_tflite_path.exists():
    print("\nTFLite Model Link (for Jupyter environments):")
    display(FileLink(str(destination_tflite_path)))
if destination_json_path.exists():
    print("\nJSON Metadata Link (for Jupyter environments):")
    display(FileLink(str(destination_json_path)))
