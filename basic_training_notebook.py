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
import platform
import requests
import zipfile
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import scipy.io.wavfile
import numpy as np
import yaml
import json

# Conditional import for display, works in Jupyter, no-op in pure script
try:
    from IPython.display import Audio, display, FileLink
except ImportError:
    def display(*args, **kwargs): pass
    def Audio(*args, **kwargs): pass
    def FileLink(*args, **kwargs): pass

# --- Data Preparation Logic (Integrated from prepare_local_data.py) ---

def detect_docker():
    """Check if running inside a Docker container"""
    if os.path.exists('/.dockerenv'):
        return True
    if os.path.exists('/proc/1/cgroup'):
        with open('/proc/1/cgroup', 'r') as f:
            if 'docker' in f.read():
                return True
    if os.environ.get('IN_DOCKER_CONTAINER', '0') == '1': # Check for a custom env var if set in Dockerfile
        return True
    return False

IN_DOCKER = detect_docker()

# Base data directory logic
if IN_DOCKER:
    BASE_DATA_DIR = Path('/data')
    print(f"Running in Docker. Using data directory: {BASE_DATA_DIR.resolve().absolute()}")
else:
    # Default for host execution: 'microwakeword-trainer-data' in the script's parent directory
    BASE_DATA_DIR = Path(__file__).resolve().parent / 'microwakeword-trainer-data'
    print(f"Running on host. Using default data directory: {BASE_DATA_DIR.resolve().absolute()}")

BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Configuration for data preparation
PIPER_REPO_URL_MAC = "https://github.com/kahrendt/piper-sample-generator" # Kiwina fork might be preferred
PIPER_REPO_BRANCH_MAC = "mps-support"
PIPER_REPO_URL_OTHER = "https://github.com/rhasspy/piper-sample-generator" # Kiwina fork might be preferred
PIPER_REPO_DIR = BASE_DATA_DIR / "piper-sample-generator"

PIPER_MODEL_FILENAME = "en_US-libritts_r-medium.pt"
PIPER_MODEL_URL = f"https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/{PIPER_MODEL_FILENAME}"
PIPER_MODEL_DIR = PIPER_REPO_DIR / "models"
PIPER_MODEL_FILE = PIPER_MODEL_DIR / PIPER_MODEL_FILENAME

MIT_RIR_OUTPUT_DIR = BASE_DATA_DIR / "mit_rirs"
AUDIOSONET_BASE_DIR = BASE_DATA_DIR / "audioset"
AUDIOSONET_OUTPUT_WAV_DIR = BASE_DATA_DIR / "audioset_16k"
FMA_BASE_DIR = BASE_DATA_DIR / "fma"
FMA_OUTPUT_WAV_DIR = BASE_DATA_DIR / "fma_16k"
FMA_ZIP_FNAME = "fma_xs.zip" # Using xs for basic notebook
FMA_ZIP_LINK = f"https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/{FMA_ZIP_FNAME}"

NEGATIVE_FEATURES_OUTPUT_DIR = BASE_DATA_DIR / "negative_datasets"
NEGATIVE_FEATURES_LINK_ROOT = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
NEGATIVE_FEATURES_FILENAMES = ['dinner_party.zip', 'dinner_party_eval.zip', 'no_speech.zip', 'speech.zip']

# Helper Functions for Data Preparation
def run_command(command_list, description, cwd=None):
    print(f"Executing: {description} -> {' '.join(command_list)}")
    try:
        process = subprocess.run(command_list, check=True, capture_output=True, text=True, cwd=cwd)
        print(f"Success: {description}. Output (last 200 chars): {process.stdout[-200:]}")
        if process.stderr:
            print(f"Stderr: {process.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed {description}.")
        print(f"Command: {e.cmd}")
        print(f"Return code: {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Stderr: {e.stderr}")
        return False

def download_file_simple(url, output_path, description="file"):
    output_path = Path(output_path)
    if output_path.exists():
        print(f"{description} already exists at {output_path}. Skipping download.")
        return True
    print(f"Downloading {description} from {url} to {output_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(output_path, "wb") as f, tqdm(
            desc=f"Downloading {output_path.name}", total=total_size, unit="B",
            unit_scale=True, unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        print(f"Successfully downloaded {description} to {output_path}.")
        return True
    except Exception as e:
        print(f"Error downloading {description} from {url}: {e}")
        if output_path.exists(): os.remove(output_path)
        return False

def extract_zip_robust(zip_path, extract_to, expected_content_name=None):
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    final_expected_path = extract_to / expected_content_name if expected_content_name else extract_to
    
    if expected_content_name and final_expected_path.is_dir():
        print(f"Content '{expected_content_name}' already found at {final_expected_path}. Skipping extraction.")
        return True
    elif not expected_content_name and any(extract_to.iterdir()): # If no specific content, check if dir is not empty
        # This is a weaker check for general zip files.
        # For more robustness, one might check for a specific file from the zip.
        # print(f"Directory {extract_to} is not empty. Assuming already extracted. Skipping extraction.")
        # return True
        pass


    if not zip_path.exists():
        print(f"ERROR: Cannot extract, ZIP file not found: {zip_path}")
        return False

    print(f"Attempting to extract {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Successfully extracted {zip_path} to {extract_to}.")
        if expected_content_name and not final_expected_path.is_dir():
             print(f"Warning: Extraction of {zip_path.name} was called, but target directory {final_expected_path} was not created as expected.")
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False

def extract_tar_robust(tar_path, extract_to, expected_content_name=None):
    tar_path = Path(tar_path)
    extract_to = Path(extract_to)
    final_expected_path = extract_to / expected_content_name if expected_content_name else extract_to

    if expected_content_name and final_expected_path.is_dir():
        print(f"Content '{expected_content_name}' already found at {final_expected_path}. Skipping extraction.")
        return True
    
    if not tar_path.exists():
        print(f"ERROR: Cannot extract, TAR file not found: {tar_path}")
        return False

    print(f"Attempting to extract {tar_path} to {extract_to}...")
    # Using subprocess for tar as it's generally more robust for various tar formats
    if not run_command(["tar", "-xf", str(tar_path), "-C", str(extract_to)], f"Extracting {tar_path.name}"):
        return False
    
    print(f"Successfully extracted {tar_path} to {extract_to}.")
    if expected_content_name and not final_expected_path.is_dir():
            print(f"Warning: Extraction of {tar_path.name} was called, but target directory {final_expected_path} was not created as expected.")
    return True


# Main Data Preparation Steps
def prepare_piper_sample_generator():
    print("\n--- Preparing Piper Sample Generator ---")
    if not PIPER_REPO_DIR.is_dir():
        print(f"Cloning Piper Sample Generator repository to {PIPER_REPO_DIR}...")
        # Using the Kiwina fork as per plan adjustment
        clone_url = "https://github.com/kiwina/piper-sample-generator.git"
        if not run_command(["git", "clone", clone_url, str(PIPER_REPO_DIR)], "Cloning Piper Sample Generator"):
            return False
    else:
        print(f"Piper Sample Generator repository already exists at {PIPER_REPO_DIR}.")
    
    PIPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not PIPER_MODEL_FILE.exists():
        if not download_file_simple(PIPER_MODEL_URL, PIPER_MODEL_FILE, "Piper TTS Model"):
            return False
    else:
        print(f"Piper TTS model already exists at {PIPER_MODEL_FILE}.")
    
    # Ensure piper-sample-generator is in sys.path
    if str(PIPER_REPO_DIR) not in sys.path:
        sys.path.append(str(PIPER_REPO_DIR))
        print(f"Added {PIPER_REPO_DIR} to sys.path")
    return True

def prepare_mit_rir_dataset():
    print("\n--- Preparing MIT RIR Dataset ---")
    MIT_RIR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    if list(MIT_RIR_OUTPUT_DIR.glob("*.wav")): # Simple check
        print(f"MIT RIR dataset appears to be already processed in {MIT_RIR_OUTPUT_DIR}. Skipping.")
        return True

    print(f"Downloading and processing MIT RIR dataset to {MIT_RIR_OUTPUT_DIR}...")
    try:
        from datasets import load_dataset # Import here as it's a heavy dependency
        rir_dataset = load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True, trust_remote_code=True)
        for row in tqdm(rir_dataset, desc="Processing MIT RIR"):
            name = Path(row["audio"]["path"]).name
            file_path = MIT_RIR_OUTPUT_DIR / name
            if not file_path.exists():
                 scipy.io.wavfile.write(
                    file_path,
                    16000,
                    (np.array(row["audio"]["array"]) * 32767).astype(np.int16)
                )
        print(f"Finished processing MIT RIR dataset to {MIT_RIR_OUTPUT_DIR}.")
        return True
    except Exception as e:
        print(f"Error during MIT RIR dataset processing: {e}")
        return False

def prepare_audioset_dataset():
    print("\n--- Preparing Audioset Dataset (Basic - 1 part) ---")
    AUDIOSONET_BASE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIOSONET_OUTPUT_WAV_DIR.mkdir(parents=True, exist_ok=True)

    # For basic notebook, only download and process one part
    fname = "bal_train00.tar" # Using part 00 for simplicity
    tar_path = AUDIOSONET_BASE_DIR / fname
    link = f"https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/{fname}"
    
    if not download_file_simple(link, tar_path, f"Audioset part {fname}"):
        return False
    
    # Expected content dir after extraction, e.g., 'audio/' or specific subfolder from tar
    # For audioset, it extracts into subdirs like 'bal_train00/'. We'll check for 'audio' inside that.
    expected_extracted_subdir = AUDIOSONET_BASE_DIR / "audio" # A common pattern in these tars
    if not extract_tar_robust(tar_path, AUDIOSONET_BASE_DIR, "audio"): # Check for 'audio' dir
         # If 'audio' dir doesn't exist at root of AUDIOSONET_BASE_DIR, it might be inside a tar-named folder
        if not (AUDIOSONET_BASE_DIR / fname.replace(".tar","") / "audio").is_dir():
            print(f"Audioset extraction for {fname} might have failed or structure is unexpected.")
            # return False # Potentially allow to continue if some files are found

    print(f"Converting Audioset FLAC files from {AUDIOSONET_BASE_DIR} to {AUDIOSONET_OUTPUT_WAV_DIR}...")
    try:
        from datasets import Dataset, Audio # Import here
        audioset_flac_files = list(AUDIOSONET_BASE_DIR.glob("**/*.flac"))
        if not audioset_flac_files:
            print(f"No FLAC files found in {AUDIOSONET_BASE_DIR} or subdirectories. Check extraction.")
            return True # Not a failure if no files to convert, but data might be missing

        files_to_convert = [f for f in audioset_flac_files if not (AUDIOSONET_OUTPUT_WAV_DIR / (f.stem + ".wav")).exists()]
        if not files_to_convert:
            print("All found Audioset FLAC files seem to be already converted.")
            return True

        print(f"Starting conversion of {len(files_to_convert)} Audioset FLAC files...")
        temp_dataset = Dataset.from_dict({"audio": [str(p) for p in files_to_convert]}).cast_column("audio", Audio(sampling_rate=16000))
        for row in tqdm(temp_dataset, desc="Converting Audioset FLAC to WAV"):
            original_path_str = row["audio"]["path"]
            file_stem = Path(original_path_str).stem 
            wav_output_path = AUDIOSONET_OUTPUT_WAV_DIR / (file_stem + ".wav")
            try:
                audio_array = np.array(row["audio"]["array"])
                if audio_array.ndim == 0 or audio_array.size == 0: raise ValueError("Empty audio data")
                scipy.io.wavfile.write(wav_output_path, 16000, (audio_array * 32767).astype(np.int16))
            except Exception as e_inner:
                print(f"Error converting file {original_path_str} (stem: {file_stem}): {e_inner}")
        print("Audioset FLAC to WAV conversion attempt complete.")
        return True
    except Exception as e:
        print(f"Error during Audioset dataset processing: {e}")
        return False

def prepare_fma_dataset():
    print("\n--- Preparing FMA Dataset (XSmall) ---")
    FMA_BASE_DIR.mkdir(parents=True, exist_ok=True)
    FMA_OUTPUT_WAV_DIR.mkdir(parents=True, exist_ok=True)
    fma_zip_path = FMA_BASE_DIR / FMA_ZIP_FNAME
    
    if not download_file_simple(FMA_ZIP_LINK, fma_zip_path, "FMA XSmall dataset"):
        return False
    
    if not extract_zip_robust(fma_zip_path, FMA_BASE_DIR, "fma_small"):
        return False
        
    fma_extracted_mp3_dir = FMA_BASE_DIR / "fma_small"
    print(f"Converting FMA MP3 files from {fma_extracted_mp3_dir} to {FMA_OUTPUT_WAV_DIR}...")
    try:
        from datasets import Dataset, Audio # Import here
        fma_mp3_files = list(fma_extracted_mp3_dir.glob("**/*.mp3"))
        if not fma_mp3_files:
            print(f"No MP3 files found in {fma_extracted_mp3_dir}. Check extraction.")
            return True

        files_to_convert = [f for f in fma_mp3_files if not (FMA_OUTPUT_WAV_DIR / (f.stem + ".wav")).exists()]
        if not files_to_convert:
            print("All found FMA MP3 files seem to be already converted.")
            return True
            
        print(f"Starting conversion of {len(files_to_convert)} FMA MP3 files...")
        temp_dataset = Dataset.from_dict({"audio": [str(p) for p in files_to_convert]}).cast_column("audio", Audio(sampling_rate=16000))
        for row in tqdm(temp_dataset, desc="Converting FMA MP3 to WAV"):
            original_path_str = row["audio"]["path"]
            file_stem = Path(original_path_str).stem
            wav_output_path = FMA_OUTPUT_WAV_DIR / (file_stem + ".wav")
            try:
                audio_array = np.array(row["audio"]["array"])
                if audio_array.ndim == 0 or audio_array.size == 0: raise ValueError("Empty audio data")
                scipy.io.wavfile.write(wav_output_path, 16000, (audio_array * 32767).astype(np.int16))
            except Exception as e_inner:
                print(f"Error converting file {original_path_str} (stem: {file_stem}): {e_inner}")
        print("FMA MP3 to WAV conversion attempt complete.")
        return True
    except Exception as e:
        print(f"Error during FMA dataset processing: {e}")
        return False

def prepare_negative_feature_datasets():
    print("\n--- Preparing Negative Spectrogram Features ---")
    NEGATIVE_FEATURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_extracted = True
    for fname in NEGATIVE_FEATURES_FILENAMES:
        link = NEGATIVE_FEATURES_LINK_ROOT + fname
        zip_path = NEGATIVE_FEATURES_OUTPUT_DIR / fname
        extracted_content_dir_name = fname.replace('.zip', '')
        if not download_file_simple(link, zip_path, f"Negative feature set {fname}"):
            all_extracted = False
            continue
        if not extract_zip_robust(zip_path, NEGATIVE_FEATURES_OUTPUT_DIR, extracted_content_dir_name):
            all_extracted = False
    return all_extracted

def run_all_data_preparation():
    print("Starting all data preparation steps...")
    overall_success = True
    if not prepare_piper_sample_generator(): overall_success = False
    if overall_success and not prepare_mit_rir_dataset(): overall_success = False
    if overall_success and not prepare_audioset_dataset(): 
        print("Audioset preparation reported issues. Basic training might still proceed with partial data.")
        # Not setting overall_success to False for basic notebook, as it's optional/partial
    if overall_success and not prepare_fma_dataset(): 
        print("FMA preparation reported issues. Basic training might still proceed with partial data.")
        # Not setting overall_success to False for basic notebook
    if overall_success and not prepare_negative_feature_datasets(): overall_success = False
    
    if overall_success:
        print("\nAll critical data preparation steps completed successfully or with warnings for optional datasets.")
    else:
        print("\nOne or more critical data preparation steps failed. Please review logs.")
    print("-" * 50)
    print("Data preparation process finished.")
    print("Expected data locations (inside Docker at /data, on host in microwakeword-trainer-data):")
    print(f"  Piper Repo: {PIPER_REPO_DIR}")
    print(f"  Piper Model: {PIPER_MODEL_FILE}")
    print(f"  MIT RIR WAVs: {MIT_RIR_OUTPUT_DIR}")
    print(f"  Audioset WAVs: {AUDIOSONET_OUTPUT_WAV_DIR}")
    print(f"  FMA WAVs: {FMA_OUTPUT_WAV_DIR}")
    print(f"  Negative Features: {NEGATIVE_FEATURES_OUTPUT_DIR}")
    print("-" * 50)
    return overall_success

# --- End of Data Preparation Logic ---

# Execute Data Preparation
# In a .py script, this would be under if __name__ == "__main__":
# For a notebook structure, we call it directly.
if not run_all_data_preparation():
    print("CRITICAL ERROR: Data preparation failed. Notebook execution cannot continue reliably.")
    # sys.exit(1) # In a script; in notebook, subsequent cells might fail or user can stop.
else:
    print("Data preparation complete. Proceeding with notebook steps...")


# Generates 1 sample of the target word for manual verification.
target_word = 'khum_puter'  # Phonetic spellings may produce better samples

print(f"\n--- Generating Test Sample for '{target_word}' ---")
piper_script_path = PIPER_REPO_DIR / "generate_samples.py"
output_sample_dir = BASE_DATA_DIR / "generated_samples_test"
output_sample_dir.mkdir(parents=True, exist_ok=True)

if not piper_script_path.exists():
    print(f"ERROR: Piper sample generator script not found at {piper_script_path}. Check data preparation.")
else:
    cmd_test_sample = [
        sys.executable, str(piper_script_path), target_word,
        "--max-samples", "1",
        "--batch-size", "1",
        "--output-dir", str(output_sample_dir)
    ]
    run_command(cmd_test_sample, "Generating test audio sample")
    
    audio_path = output_sample_dir / "0.wav"
    if audio_path.exists():
        print(f"Playing test sample: {audio_path}")
        display(Audio(str(audio_path), autoplay=True))
    else:
        print(f"Audio file not found at {audio_path}. Sample generation might have failed.")


# Generates a larger amount of wake word samples.
print(f"\n--- Generating Wake Word Samples for '{target_word}' ---")
output_ww_dir = BASE_DATA_DIR / "generated_samples_ww"
output_ww_dir.mkdir(parents=True, exist_ok=True)

if not piper_script_path.exists():
    print(f"ERROR: Piper sample generator script not found at {piper_script_path}. Check data preparation.")
else:
    cmd_ww_samples = [
        sys.executable, str(piper_script_path), target_word,
        "--max-samples", "1000", # As per original notebook
        "--batch-size", "100",  # As per original notebook
        "--output-dir", str(output_ww_dir)
    ]
    run_command(cmd_ww_samples, "Generating wake word samples")
    print(f"Wake word samples generated in {output_ww_dir}")


# Sets up the augmentations.
print("\n--- Setting up Augmentations ---")
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration

# Paths to pre-prepared data in BASE_DATA_DIR
generated_samples_path = output_ww_dir # From previous step
mit_rirs_path_aug = MIT_RIR_OUTPUT_DIR
fma_16k_path_aug = FMA_OUTPUT_WAV_DIR
audioset_16k_path_aug = AUDIOSONET_OUTPUT_WAV_DIR

if not generated_samples_path.is_dir():
    print(f"ERROR: Wake word samples not found at {generated_samples_path}. Please run the previous step.")
if not mit_rirs_path_aug.is_dir() or not fma_16k_path_aug.is_dir() or not audioset_16k_path_aug.is_dir():
    print("ERROR: One or more augmentation data directories (MIT RIRs, FMA, Audioset) not found. Please ensure data preparation ran successfully.")

clips = Clips(input_directory=str(generated_samples_path),
              file_pattern='*.wav',
              max_clip_duration_s=None,
              remove_silence=False,
              random_split_seed=10,
              split_count=0.1,
             )

augmenter = Augmentation(augmentation_duration_s=3.2,
                         augmentation_probabilities = {
                                "SevenBandParametricEQ": 0.1,
                                "TanhDistortion": 0.1,
                                "PitchShift": 0.1,
                                "BandStopFilter": 0.1,
                                "AddColorNoise": 0.1,
                                "AddBackgroundNoise": 0.75,
                                "Gain": 1.0,
                                "RIR": 0.5,
                            },
                         impulse_paths = [str(mit_rirs_path_aug)],
                         background_paths = [str(fma_16k_path_aug), str(audioset_16k_path_aug)],
                         background_min_snr_db = -5,
                         background_max_snr_db = 10,
                         min_jitter_s = 0.195,
                         max_jitter_s = 0.205,
                        )
print("Augmentation setup complete.")

# Augment a random clip and play it back to verify it works well
print("\n--- Augmenting and Playing Test Clip ---")
from microwakeword.audio.audio_utils import save_clip

augmented_clip_path_test = BASE_DATA_DIR / "augmented_clip_test.wav"

try:
    random_clip_data = clips.get_random_clip()
    augmented_clip_data = augmenter.augment_clip(random_clip_data)
    save_clip(augmented_clip_data, str(augmented_clip_path_test))
    print(f"Playing augmented test clip: {augmented_clip_path_test}")
    display(Audio(str(augmented_clip_path_test), autoplay=True))
except Exception as e:
    print(f"Error during test augmentation: {e}. Check if previous steps ran successfully and data paths are correct.")


# Augment samples and save the training, validation, and testing sets.
print("\n--- Generating Augmented Features for Training/Validation/Testing ---")
from mmap_ninja.ragged import RaggedMmap

output_dir_augmented_features = BASE_DATA_DIR / 'generated_augmented_features'
output_dir_augmented_features.mkdir(parents=True, exist_ok=True)

splits_config = ["training", "validation", "testing"]
for split_item in splits_config:
  out_dir_split_item = output_dir_augmented_features / split_item
  out_dir_split_item.mkdir(parents=True, exist_ok=True)

  current_split_name = "train"
  current_repetition = 2

  current_spectrograms = SpectrogramGeneration(clips=clips,
                                     augmenter=augmenter,
                                     slide_frames=10,
                                     step_ms=10,
                                     )
  if split_item == "validation":
    current_split_name = "validation"
    current_repetition = 1
  elif split_item == "testing":
    current_split_name = "test"
    current_repetition = 1
    current_spectrograms = SpectrogramGeneration(clips=clips,
                                     augmenter=augmenter,
                                     slide_frames=1,
                                     step_ms=10,
                                     )

  print(f"Generating augmented features for {current_split_name} set...")
  try:
    RaggedMmap.from_generator(
        out_dir=str(out_dir_split_item / 'wakeword_mmap'),
        sample_generator=current_spectrograms.spectrogram_generator(split=current_split_name, repeat=current_repetition),
        batch_size=100,
        verbose=True,
    )
    print(f"Finished generating features for {current_split_name} set.")
  except Exception as e:
    print(f"Error generating features for {current_split_name} set: {e}")


# Save a yaml config that controls the training process
print("\n--- Preparing Training Configuration ---")
config_train = {}
config_train["window_step_ms"] = 10
config_train["train_dir"] = str(BASE_DATA_DIR / "trained_models/wakeword")
(BASE_DATA_DIR / "trained_models/wakeword").mkdir(parents=True, exist_ok=True)

config_train["features"] = [
    {
        "features_dir": str(output_dir_augmented_features), # Path inside Docker
        "sampling_weight": 2.0, "penalty_weight": 1.0, "truth": True,
        "truncation_strategy": "truncate_start", "type": "mmap",
    },
    {
        "features_dir": str(NEGATIVE_FEATURES_OUTPUT_DIR / "speech"),
        "sampling_weight": 10.0, "penalty_weight": 1.0, "truth": False,
        "truncation_strategy": "random", "type": "mmap",
    },
    {
        "features_dir": str(NEGATIVE_FEATURES_OUTPUT_DIR / "dinner_party"),
        "sampling_weight": 10.0, "penalty_weight": 1.0, "truth": False,
        "truncation_strategy": "random", "type": "mmap",
    },
    {
        "features_dir": str(NEGATIVE_FEATURES_OUTPUT_DIR / "no_speech"),
        "sampling_weight": 5.0, "penalty_weight": 1.0, "truth": False,
        "truncation_strategy": "random", "type": "mmap",
    },
    { 
        "features_dir": str(NEGATIVE_FEATURES_OUTPUT_DIR / "dinner_party_eval"),
        "sampling_weight": 0.0, "penalty_weight": 1.0, "truth": False,
        "truncation_strategy": "split", "type": "mmap",
    },
]
config_train["training_steps"] = [10000]
config_train["positive_class_weight"] = [1]
config_train["negative_class_weight"] = [20]
config_train["learning_rates"] = [0.001]
config_train["batch_size"] = 128
config_train["time_mask_max_size"] = [0]
config_train["time_mask_count"] = [0]
config_train["freq_mask_max_size"] = [0]
config_train["freq_mask_count"] = [0]
config_train["eval_step_interval"] = 500
config_train["clip_duration_ms"] = 1500
config_train["target_minimization"] = 0.9
config_train["minimization_metric"] = None
config_train["maximization_metric"] = "average_viable_recall"

training_params_path_yaml = BASE_DATA_DIR / "training_parameters.yaml"
with open(training_params_path_yaml, "w") as file_yaml:
    yaml.dump(config_train, file_yaml)
print(f"Training parameters saved to {training_params_path_yaml}")


# Trains a model.
print("\n--- Starting Model Training ---")
# LD_LIBRARY_PATH might be needed if base TF image doesn't set it up for all custom ops.
# Usually, for official TF images, it's handled.
# os.environ['LD_LIBRARY_PATH'] = "/usr/lib/x86_64-linux-gnu:" + os.environ.get('LD_LIBRARY_PATH', '')

cmd_train_model = [
    sys.executable, "-m", "microwakeword.model_train_eval",
    "--training_config", str(training_params_path_yaml),
    "--train", "1",
    "--restore_checkpoint", "1",
    "--test_tf_nonstreaming", "0",
    "--test_tflite_nonstreaming", "0",
    "--test_tflite_nonstreaming_quantized", "0",
    "--test_tflite_streaming", "0",
    "--test_tflite_streaming_quantized", "1",
    "--use_weights", "best_weights",
    "mixednet",
    "--pointwise_filters", "64,64,64,64",
    "--repeat_in_block", "1, 1, 1, 1",
    "--mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]",
    "--residual_connection", "0,0,0,0",
    "--first_conv_filters", "32",
    "--first_conv_kernel_size", "5",
    "--stride", "3"
]
run_command(cmd_train_model, "Training model")
print("Model training/evaluation finished.")


# Prepare model files for download/use
print("\n--- Preparing Output Model Files ---")
source_tflite_model_path = BASE_DATA_DIR / "trained_models/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
destination_tflite_model_path = BASE_DATA_DIR / "stream_state_internal_quant.tflite"

if source_tflite_model_path.exists():
    shutil.copy(source_tflite_model_path, destination_tflite_model__path)
    print(f"Copied TFLite model to {destination_tflite_model_path}")
else:
    print(f"ERROR: Trained TFLite model not found at {source_tflite_model_path}")

json_output_data = {
    "type": "micro",
    "wake_word": target_word,
    "author": "kiwina", # Updated author
    "website": "https://github.com/kiwina/MicroWakeWord-Trainer-Docker",
    "model": "stream_state_internal_quant.tflite",
    "trained_languages": ["en"],
    "version": 2, # User can increment this
    "micro": {
        "probability_cutoff": 0.97, # User should adjust based on testing
        "sliding_window_size": 5,
        "feature_step_size": 10,
        "tensor_arena_size": 30000, # User should adjust based on model needs
        "minimum_esphome_version": "2024.7.0"
    }
}
destination_json_meta_path = BASE_DATA_DIR / "stream_state_internal_quant.json"
with open(destination_json_meta_path, "w") as json_file_out:
    json.dump(json_output_data, json_file_out, indent=2)
print(f"Created JSON metadata at {destination_json_meta_path}")

print("\n--- Script Finished ---")
print(f"Output files are located in: {BASE_DATA_DIR.resolve().absolute()}")
print("If running in Docker, this corresponds to the 'microwakeword-trainer-data' directory on your host machine.")

if destination_tflite_model_path.exists():
    print("\nTFLite Model:")
    display(FileLink(str(destination_tflite_model_path))) # For Jupyter environments
if destination_json_meta_path.exists():
    print("\nJSON Metadata:")
    display(FileLink(str(destination_json_meta_path))) # For Jupyter environments
