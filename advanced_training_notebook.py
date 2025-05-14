#!/usr/bin/env python
# coding: utf-8

# <div align="center">
#   <img src="https://raw.githubusercontent.com/MasterPhooey/MicroWakeWord-Trainer-Docker/refs/heads/main/mmw.png" alt="MicroWakeWord Trainer Logo" width="100" />
#   <h1>MicroWakeWord Trainer Docker - Advanced</h1>
# </div>
#
# This notebook steps you through training a robust microWakeWord model. It is intended as a **starting point** for users looking to create a high-performance wake word detection model. This notebook is optimized for Python 3.11.
#
# **The model generated from this notebook is designed for practical use, but achieving optimal performance will require experimentation with various settings and datasets. The provided scripts and configurations aim to give you a strong foundation to build upon.**
#
# Throughout the notebook, you will find comments suggesting specific settings to modify and experiment with to enhance your model's performance.
#
# By the end of this notebook, you will have:
# - A trained TensorFlow Lite model ready for deployment.
# - A JSON manifest file to integrate the model with ESPHome.
#
# To use the generated model in ESPHome, refer to the [ESPHome documentation](https://esphome.io/components/micro_wake_word) for integration details. You can also explore example configurations in the [model repository](https://github.com/esphome/micro-wake-word-models/tree/main/models/v2).

import os
import sys
import subprocess
from pathlib import Path
import yaml
import json

# Conditional import for display
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

target_word = 'hey_norman'  # Phonetic spellings may produce better samples

# Define paths based on DATA_DIR_INSIDE_DOCKER
PIPER_SAMPLE_GENERATOR_DIR = DATA_DIR_INSIDE_DOCKER / "piper-sample-generator"
PIPER_SCRIPT_PATH = PIPER_SAMPLE_GENERATOR_DIR / "generate_samples.py"

# Updated generated samples directory structure
GENERATED_SAMPLES_BASE_DIR_ADV = DATA_DIR_INSIDE_DOCKER / "generated_samples" / target_word
TEST_SAMPLE_OUTPUT_DIR_ADV = GENERATED_SAMPLES_BASE_DIR_ADV / "test"
WW_SAMPLES_OUTPUT_DIR_ADV = GENERATED_SAMPLES_BASE_DIR_ADV / "samples"

MIT_RIRS_PATH_ADV = DATA_DIR_INSIDE_DOCKER / "mit_rirs"
FMA_16K_PATH_ADV = DATA_DIR_INSIDE_DOCKER / "fma_16k"
AUDIOSONET_16K_PATH_ADV = DATA_DIR_INSIDE_DOCKER / "audioset_16k"
AUGMENTED_FEATURES_DIR_ADV = DATA_DIR_INSIDE_DOCKER / "generated_augmented_features_adv" # Separate for advanced
NEGATIVE_DATASETS_PATH_ADV = DATA_DIR_INSIDE_DOCKER / "negative_datasets"
TRAINED_MODELS_BASE_PATH_ADV = DATA_DIR_INSIDE_DOCKER / "trained_models_adv" # Separate for advanced
TRAINING_CONFIG_PATH_ADV = DATA_DIR_INSIDE_DOCKER / "training_parameters_adv.yaml"

# Create directories
TEST_SAMPLE_OUTPUT_DIR_ADV.mkdir(parents=True, exist_ok=True)
WW_SAMPLES_OUTPUT_DIR_ADV.mkdir(parents=True, exist_ok=True)
AUGMENTED_FEATURES_DIR_ADV.mkdir(parents=True, exist_ok=True)
(TRAINED_MODELS_BASE_PATH_ADV / "wakeword").mkdir(parents=True, exist_ok=True)

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
print(f"\n--- Generating Test Sample for '{target_word}' (Advanced) ---")
cmd_test_sample_adv = [sys.executable, str(PIPER_SCRIPT_PATH), target_word, "--max-samples", "1", "--batch-size", "1", "--output-dir", str(TEST_SAMPLE_OUTPUT_DIR_ADV)]
subprocess.run(cmd_test_sample_adv, check=True)
audio_path_test_adv = TEST_SAMPLE_OUTPUT_DIR_ADV / "0.wav"
if audio_path_test_adv.exists():
    print(f"Playing test sample: {audio_path_test_adv}")
    display(Audio(str(audio_path_test_adv), autoplay=True))
else:
    print(f"Audio file not found at {audio_path_test_adv}.")

# Generates a larger amount of wake word samples for advanced training.
print(f"\n--- Generating Wake Word Samples for '{target_word}' (Advanced: 50k) ---")
cmd_ww_samples_adv = [sys.executable, str(PIPER_SCRIPT_PATH), target_word, "--max-samples", "50000", "--batch-size", "100", "--output-dir", str(WW_SAMPLES_OUTPUT_DIR_ADV)]
subprocess.run(cmd_ww_samples_adv, check=True)
print(f"Wake word samples generated in {WW_SAMPLES_OUTPUT_DIR_ADV}")

# Augmentation Setup
print("\n--- Setting up Augmentations (Advanced) ---")
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration

clips_adv = Clips(input_directory=str(WW_SAMPLES_OUTPUT_DIR_ADV), file_pattern='*.wav', max_clip_duration_s=5, remove_silence=True, random_split_seed=10, split_count=0.1)
augmenter_adv = Augmentation(
    augmentation_duration_s=3.2,
    augmentation_probabilities={"SevenBandParametricEQ":0.1,"TanhDistortion":0.05,"PitchShift":0.15,"BandStopFilter":0.1,"AddColorNoise":0.1,"AddBackgroundNoise":0.7,"Gain":0.8,"RIR":0.7},
    impulse_paths=[str(MIT_RIRS_PATH_ADV)], background_paths=[str(FMA_16K_PATH_ADV), str(AUDIOSONET_16K_PATH_ADV)],
    background_min_snr_db=5, background_max_snr_db=10, min_jitter_s=0.2, max_jitter_s=0.3
)
print("Advanced augmentation setup complete.")

# Augment a random clip for verification
print("\n--- Augmenting and Playing Test Clip (Advanced) ---")
from microwakeword.audio.audio_utils import save_clip
augmented_clip_path_test_adv = DATA_DIR_INSIDE_DOCKER / "augmented_clip_test_adv.wav"
try:
    random_clip_data_adv = clips_adv.get_random_clip()
    augmented_clip_data_adv = augmenter_adv.augment_clip(random_clip_data_adv)
    save_clip(augmented_clip_data_adv, str(augmented_clip_path_test_adv))
    print(f"Playing augmented test clip: {augmented_clip_path_test_adv}")
    display(Audio(str(augmented_clip_path_test_adv), autoplay=True))
except Exception as e: print(f"Error test augmentation: {e}")

# Augment samples and save sets
print("\n--- Generating Augmented Features (Advanced) ---")
from mmap_ninja.ragged import RaggedMmap
AUGMENTED_FEATURES_DIR_ADV.mkdir(parents=True, exist_ok=True)
splits_conf_adv = {"training":{"name":"train","repetition":2,"slide_frames":10},"validation":{"name":"validation","repetition":1,"slide_frames":10},"testing":{"name":"test","repetition":1,"slide_frames":1}}
for split_item_adv, conf_adv in splits_conf_adv.items():
    out_dir_split_adv = AUGMENTED_FEATURES_DIR_ADV / split_item_adv; out_dir_split_adv.mkdir(parents=True, exist_ok=True)
    print(f"Processing {split_item_adv} set (Advanced)...")
    spec_gen_adv = SpectrogramGeneration(clips=clips_adv, augmenter=augmenter_adv, slide_frames=conf_adv["slide_frames"], step_ms=10)
    try:
        RaggedMmap.from_generator(
            out_dir=str(out_dir_split_adv / 'wakeword_mmap'),
            sample_generator=spec_gen_adv.spectrogram_generator(split=conf_adv["name"], repeat=conf_adv["repetition"]),
            batch_size=100, verbose=True
        )
        print(f"Completed {split_item_adv} set (Advanced).")
    except Exception as e: print(f"Error processing {split_item_adv} set: {e}")

# Training Configuration
print("\n--- Preparing Training Configuration (Advanced) ---")
config_train_adv = {
    "window_step_ms": 10, "train_dir": str(TRAINED_MODELS_BASE_PATH_ADV / "wakeword"),
    "features": [
        {"features_dir":str(AUGMENTED_FEATURES_DIR_ADV),"sampling_weight":2.0,"penalty_weight":1.0,"truth":True,"truncation_strategy":"truncate_start","type":"mmap"},
        {"features_dir":str(NEGATIVE_DATASETS_PATH_ADV/"speech"),"sampling_weight":12.0,"penalty_weight":1.0,"truth":False,"truncation_strategy":"random","type":"mmap"},
        {"features_dir":str(NEGATIVE_DATASETS_PATH_ADV/"dinner_party"),"sampling_weight":12.0,"penalty_weight":1.0,"truth":False,"truncation_strategy":"random","type":"mmap"},
        {"features_dir":str(NEGATIVE_DATASETS_PATH_ADV/"no_speech"),"sampling_weight":5.0,"penalty_weight":1.0,"truth":False,"truncation_strategy":"random","type":"mmap"},
        {"features_dir":str(NEGATIVE_DATASETS_PATH_ADV/"dinner_party_eval"),"sampling_weight":0.0,"penalty_weight":1.0,"truth":False,"truncation_strategy":"split","type":"mmap"},
    ],
    "training_steps": [40000], "positive_class_weight": [1], "negative_class_weight": [20],
    "learning_rates": [0.0005], "batch_size": 128,
    "time_mask_max_size": [5], "time_mask_count": [2], "freq_mask_max_size": [5], "freq_mask_count": [2],
    "eval_step_interval": 1000, "clip_duration_ms": 2000,
    "target_minimization": 0.85, "minimization_metric": "false_positive_rate",
    "maximization_metric": "recall"
}
with open(TRAINING_CONFIG_PATH_ADV, "w") as f_yaml_adv: yaml.dump(config_train_adv, f_yaml_adv)
print(f"Advanced training parameters saved to {TRAINING_CONFIG_PATH_ADV}")

# Model Training
print("\n--- Starting Model Training (Advanced) ---")
cmd_train_model_adv = [
    sys.executable, "-m", "microwakeword.model_train_eval",
    "--training_config", str(TRAINING_CONFIG_PATH_ADV), "--train", "1", "--restore_checkpoint", "1",
    "--test_tf_nonstreaming", "0", "--test_tflite_nonstreaming", "0", "--test_tflite_nonstreaming_quantized", "0",
    "--test_tflite_streaming", "0", "--test_tflite_streaming_quantized", "1", "--use_weights", "best_weights",
    "mixednet", "--pointwise_filters", "64,64,64,64,64", "--repeat_in_block", "1,1,1,1,1",
    "--mixconv_kernel_sizes", "[3,5],[5,7],[7,9],[9,11],[11,13]", "--residual_connection", "0,0,0,0,0",
    "--first_conv_filters", "48", "--first_conv_kernel_size", "7", "--stride", "2"
]
subprocess.run(cmd_train_model_adv, check=True)
print("Advanced model training/evaluation finished.")

# Prepare Output Files
print("\n--- Preparing Output Model Files (Advanced) ---")
source_tflite_adv = TRAINED_MODELS_BASE_PATH_ADV / "wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
dest_tflite_adv = DATA_DIR_INSIDE_DOCKER / f"{target_word}_advanced_model.tflite"
if source_tflite_adv.exists():
    shutil.copy(source_tflite_adv, dest_tflite_adv); print(f"Copied TFLite model to {dest_tflite_adv}")
else: print(f"ERROR: Trained TFLite model not found at {source_tflite_adv}")

json_out_adv = {
    "type":"micro", "wake_word":target_word, "author":"kiwina", "website":"https://github.com/kiwina/MicroWakeWord-Trainer-Docker",
    "model":f"{target_word}_advanced_model.tflite", "trained_languages":["en"], "version":1,
    "micro":{"probability_cutoff":0.9,"sliding_window_size":6,"feature_step_size":10,"tensor_arena_size":35000,"minimum_esphome_version":"2024.7.0"}
}
dest_json_adv = DATA_DIR_INSIDE_DOCKER / f"{target_word}_advanced_model.json"
with open(dest_json_adv, "w") as f_json_adv: json.dump(json_out_adv, f_json_adv, indent=2)
print(f"Created JSON metadata at {dest_json_adv}")

print("\n--- Advanced Script Finished ---")
print(f"Output files are located in: {DATA_DIR_INSIDE_DOCKER.resolve().absolute()}")
if dest_tflite_adv.exists():
    print("\nTFLite Model Link (for Jupyter environments):")
    display(FileLink(str(dest_tflite_adv)))
if dest_json_adv.exists():
    print("\nJSON Metadata Link (for Jupyter environments):")
    display(FileLink(str(dest_json_adv)))
