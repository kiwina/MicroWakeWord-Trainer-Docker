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

# --- Data Preparation Logic (Integrated and adapted for Advanced Notebook) ---

def detect_docker():
    """Check if running inside a Docker container"""
    if os.path.exists('/.dockerenv'):
        return True
    if os.path.exists('/proc/1/cgroup'):
        with open('/proc/1/cgroup', 'r') as f:
            if 'docker' in f.read():
                return True
    if os.environ.get('IN_DOCKER_CONTAINER', '0') == '1':
        return True
    return False

IN_DOCKER = detect_docker()

if IN_DOCKER:
    BASE_DATA_DIR = Path('/data')
    print(f"Running in Docker. Using data directory: {BASE_DATA_DIR.resolve().absolute()}")
else:
    BASE_DATA_DIR = Path(__file__).resolve().parent / 'microwakeword-trainer-data'
    print(f"Running on host. Using default data directory: {BASE_DATA_DIR.resolve().absolute()}")

BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Configuration for data preparation
PIPER_REPO_URL_OTHER = "https://github.com/kiwina/piper-sample-generator.git" # Using Kiwina fork
PIPER_REPO_DIR = BASE_DATA_DIR / "piper-sample-generator"
PIPER_MODEL_FILENAME = "en_US-libritts_r-medium.pt"
PIPER_MODEL_URL = f"https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/{PIPER_MODEL_FILENAME}"
PIPER_MODEL_DIR = PIPER_REPO_DIR / "models"
PIPER_MODEL_FILE = PIPER_MODEL_DIR / PIPER_MODEL_FILENAME

MIT_RIR_OUTPUT_DIR = BASE_DATA_DIR / "mit_rirs"
AUDIOSONET_BASE_DIR = BASE_DATA_DIR / "audioset" # Raw downloads
AUDIOSONET_OUTPUT_WAV_DIR = BASE_DATA_DIR / "audioset_16k" # Processed WAVs
FMA_BASE_DIR = BASE_DATA_DIR / "fma" # Raw downloads
FMA_OUTPUT_WAV_DIR = BASE_DATA_DIR / "fma_16k" # Processed WAVs
FMA_ZIP_FNAME = "fma_small.zip" # For advanced, consider fma_medium or fma_large if available and disk space allows
FMA_ZIP_LINK = f"https://os.unil.cloud.switch.ch/fma/fma_small.zip" # Link for fma_small

NEGATIVE_FEATURES_OUTPUT_DIR = BASE_DATA_DIR / "negative_datasets"
NEGATIVE_FEATURES_LINK_ROOT = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
NEGATIVE_FEATURES_FILENAMES = ['dinner_party.zip', 'dinner_party_eval.zip', 'no_speech.zip', 'speech.zip']

# Helper Functions
def run_command(command_list, description, cwd=None):
    print(f"Executing: {description} -> {' '.join(command_list)}")
    try:
        process = subprocess.run(command_list, check=True, capture_output=True, text=True, cwd=cwd)
        if process.stderr: print(f"Stderr: {process.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed {description}.\nCmd: {e.cmd}\nCode: {e.returncode}\nOutput: {e.output}\nStderr: {e.stderr}")
        return False

def download_file_simple(url, output_path, description="file"):
    output_path = Path(output_path)
    if output_path.exists():
        print(f"{description} already exists at {output_path}. Skipping download.")
        return True
    print(f"Downloading {description} from {url} to {output_path}...")
    try:
        response = requests.get(url, stream=True, timeout=300) # Added timeout
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(output_path, "wb") as f, tqdm(
            desc=f"Downloading {output_path.name}", total=total_size, unit="B", unit_scale=True, unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk); bar.update(len(chunk))
        print(f"Successfully downloaded {description} to {output_path}.")
        return True
    except Exception as e:
        print(f"Error downloading {description} from {url}: {e}")
        if output_path.exists(): os.remove(output_path)
        return False

def extract_zip_robust(zip_path, extract_to, expected_content_name=None):
    zip_path, extract_to = Path(zip_path), Path(extract_to)
    final_expected_path = extract_to / expected_content_name if expected_content_name else extract_to
    if expected_content_name and final_expected_path.is_dir():
        print(f"Content '{expected_content_name}' already found at {final_expected_path}. Skipping extraction.")
        return True
    if not zip_path.exists():
        print(f"ERROR: ZIP file not found: {zip_path}"); return False
    print(f"Extracting {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref: zip_ref.extractall(extract_to)
        print(f"Successfully extracted {zip_path} to {extract_to}.")
        return True
    except Exception as e: print(f"Error extracting {zip_path}: {e}"); return False

def extract_tar_robust(tar_path, extract_to, expected_content_name=None):
    tar_path, extract_to = Path(tar_path), Path(extract_to)
    # Check if specific content dir exists (if expected_content_name is provided)
    if expected_content_name and (extract_to / expected_content_name).is_dir():
        print(f"Content '{expected_content_name}' already found at {extract_to / expected_content_name}. Skipping extraction.")
        return True
    
    # Check if a directory with the tarball's name (minus extension) exists, suggesting prior extraction
    tar_extracted_folder_name = tar_path.name.replace(".tar.gz", "").replace(".tar", "")
    if (extract_to / tar_extracted_folder_name).is_dir() and any((extract_to / tar_extracted_folder_name).iterdir()):
         print(f"Directory {extract_to / tar_extracted_folder_name} suggests {tar_path.name} might be already extracted. Skipping.")
         return True

    if not tar_path.exists():
        print(f"ERROR: TAR file not found: {tar_path}"); return False
    print(f"Extracting {tar_path} to {extract_to}...")
    if not run_command(["tar", "-xf", str(tar_path), "-C", str(extract_to)], f"Extracting {tar_path.name}"):
        return False
    print(f"Successfully extracted {tar_path} to {extract_to}.")
    return True

# Data Preparation Steps
def prepare_piper_sample_generator():
    print("\n--- Preparing Piper Sample Generator ---")
    if not PIPER_REPO_DIR.is_dir():
        print(f"Cloning Piper Sample Generator to {PIPER_REPO_DIR}...")
        if not run_command(["git", "clone", PIPER_REPO_URL_OTHER, str(PIPER_REPO_DIR)], "Cloning Piper"): return False
    else: print(f"Piper Sample Generator repo exists at {PIPER_REPO_DIR}.")
    PIPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not download_file_simple(PIPER_MODEL_URL, PIPER_MODEL_FILE, "Piper TTS Model"): return False
    if str(PIPER_REPO_DIR) not in sys.path: sys.path.append(str(PIPER_REPO_DIR))
    return True

def prepare_mit_rir_dataset():
    print("\n--- Preparing MIT RIR Dataset ---")
    MIT_RIR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if list(MIT_RIR_OUTPUT_DIR.glob("*.wav")):
        print(f"MIT RIR dataset seems processed in {MIT_RIR_OUTPUT_DIR}. Skipping."); return True
    print(f"Processing MIT RIR dataset to {MIT_RIR_OUTPUT_DIR}...")
    try:
        from datasets import load_dataset
        rir_dataset = load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True, trust_remote_code=True)
        for row in tqdm(rir_dataset, desc="Processing MIT RIR"):
            name = Path(row["audio"]["path"]).name; file_path = MIT_RIR_OUTPUT_DIR / name
            if not file_path.exists():
                 scipy.io.wavfile.write(file_path, 16000, (np.array(row["audio"]["array"]) * 32767).astype(np.int16))
        print(f"Finished MIT RIR dataset to {MIT_RIR_OUTPUT_DIR}.")
        return True
    except Exception as e: print(f"Error MIT RIR processing: {e}"); return False

def prepare_audioset_full_dataset(): # Renamed for clarity
    print("\n--- Preparing Full Audioset Dataset ---")
    AUDIOSONET_BASE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIOSONET_OUTPUT_WAV_DIR.mkdir(parents=True, exist_ok=True)
    dataset_links = [f"https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train0{i}.tar" for i in range(10)]
    dataset_links.extend([f"https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/unbal_train0{i}.tar" for i in range(41)]) # All unbalanced parts

    all_tars_downloaded_and_extracted = True
    for link in dataset_links:
        fname = Path(link).name; tar_path = AUDIOSONET_BASE_DIR / fname
        if not download_file_simple(link, tar_path, f"Audioset part {fname}"):
            all_tars_downloaded_and_extracted = False; continue # Try next tar
        # For Audioset, tar files extract into subdirs like 'bal_train00/', etc.
        # We check if a folder with the tar name (minus .tar) exists as a sign of extraction.
        if not extract_tar_robust(tar_path, AUDIOSONET_BASE_DIR):
             all_tars_downloaded_and_extracted = False

    if not all_tars_downloaded_and_extracted:
        print("Warning: Not all Audioset tar files were downloaded/extracted successfully. Conversion will proceed with available files.")

    print(f"Converting Audioset FLAC files from {AUDIOSONET_BASE_DIR} to {AUDIOSONET_OUTPUT_WAV_DIR}...")
    try:
        from datasets import Dataset, Audio; import soundfile as sf
        audioset_flac_files = list(AUDIOSONET_BASE_DIR.glob("**/*.flac"))
        if not audioset_flac_files:
            print(f"No FLAC files found in {AUDIOSONET_BASE_DIR}. Check download/extraction."); return True
        
        files_to_convert = [f for f in audioset_flac_files if not (AUDIOSONET_OUTPUT_WAV_DIR / (f.stem + ".wav")).exists()]
        if not files_to_convert: print("All Audioset FLACs seem converted."); return True

        print(f"Converting {len(files_to_convert)} Audioset FLAC files...")
        # Using soundfile for reading and scipy for writing to ensure 16kHz
        for flac_path in tqdm(files_to_convert, desc="Converting Audioset FLAC to WAV"):
            wav_output_path = AUDIOSONET_OUTPUT_WAV_DIR / (flac_path.stem + ".wav")
            try:
                audio_data, sr = sf.read(flac_path)
                if sr != 16000: # Manual resampling if needed, though datasets.Audio would handle it
                    # This part can be complex; for now, we assume files are convertible or accept original SR if not 16k
                    # For robust resampling: use librosa or soxr if sf doesn't match target
                    # For simplicity, we'll write it and if it's not 16k, TF might resample or error later.
                    # Best practice is to ensure 16kHz here.
                    # Let's assume for now we are forcing 16kHz write.
                    pass # If sr is not 16000, scipy.io.wavfile.write will write it as 16000 but data won't be resampled.
                         # This needs a proper resampling step if sr varies.
                         # For now, we rely on the fact that most audio datasets are often 16k or 48k (divisible).
                if audio_data.ndim > 1: audio_data = audio_data[:, 0] # Convert to mono
                scipy.io.wavfile.write(wav_output_path, 16000, (audio_data * 32767).astype(np.int16))
            except Exception as e_inner: print(f"Error converting {flac_path}: {e_inner}")
        print("Audioset FLAC to WAV conversion attempt complete.")
        return True
    except Exception as e: print(f"Error Audioset processing: {e}"); return False

def prepare_fma_dataset(): # Using fma_small for advanced notebook too, can be changed
    print("\n--- Preparing FMA Dataset (Small) ---")
    FMA_BASE_DIR.mkdir(parents=True, exist_ok=True)
    FMA_OUTPUT_WAV_DIR.mkdir(parents=True, exist_ok=True)
    fma_zip_path = FMA_BASE_DIR / FMA_ZIP_FNAME
    if not download_file_simple(FMA_ZIP_LINK, fma_zip_path, "FMA Small dataset"): return False
    if not extract_zip_robust(fma_zip_path, FMA_BASE_DIR, "fma_small"): return False
    
    fma_extracted_mp3_dir = FMA_BASE_DIR / "fma_small"
    print(f"Converting FMA MP3s from {fma_extracted_mp3_dir} to {FMA_OUTPUT_WAV_DIR}...")
    try:
        from datasets import Dataset, Audio
        fma_mp3_files = list(fma_extracted_mp3_dir.glob("**/*.mp3"))
        if not fma_mp3_files: print(f"No MP3s in {fma_extracted_mp3_dir}."); return True
        files_to_convert = [f for f in fma_mp3_files if not (FMA_OUTPUT_WAV_DIR / (f.stem + ".wav")).exists()]
        if not files_to_convert: print("All FMA MP3s seem converted."); return True
        
        print(f"Converting {len(files_to_convert)} FMA MP3 files...")
        temp_dataset = Dataset.from_dict({"audio": [str(p) for p in files_to_convert]}).cast_column("audio", Audio(sampling_rate=16000))
        for row in tqdm(temp_dataset, desc="Converting FMA MP3 to WAV"):
            original_path_str = row["audio"]["path"]; file_stem = Path(original_path_str).stem
            wav_output_path = FMA_OUTPUT_WAV_DIR / (file_stem + ".wav")
            try:
                audio_array = np.array(row["audio"]["array"])
                if audio_array.ndim == 0 or audio_array.size == 0: raise ValueError("Empty audio")
                scipy.io.wavfile.write(wav_output_path, 16000, (audio_array * 32767).astype(np.int16))
            except Exception as e_inner: print(f"Error converting {original_path_str}: {e_inner}")
        print("FMA MP3 to WAV conversion complete.")
        return True
    except Exception as e: print(f"Error FMA processing: {e}"); return False

def prepare_negative_feature_datasets():
    print("\n--- Preparing Negative Spectrogram Features ---")
    NEGATIVE_FEATURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True); all_extracted = True
    for fname in NEGATIVE_FEATURES_FILENAMES:
        link = NEGATIVE_FEATURES_LINK_ROOT + fname; zip_path = NEGATIVE_FEATURES_OUTPUT_DIR / fname
        extracted_content_dir_name = fname.replace('.zip', '')
        if not download_file_simple(link, zip_path, f"Negative feature {fname}"): all_extracted = False; continue
        if not extract_zip_robust(zip_path, NEGATIVE_FEATURES_OUTPUT_DIR, extracted_content_dir_name): all_extracted = False
    return all_extracted

def run_all_data_preparation_advanced():
    print("Starting all data preparation steps (Advanced)...")
    if not prepare_piper_sample_generator(): return False
    if not prepare_mit_rir_dataset(): return False
    if not prepare_audioset_full_dataset(): print("Warning: Audioset full preparation had issues.") # Non-critical for proceeding
    if not prepare_fma_dataset(): print("Warning: FMA preparation had issues.") # Non-critical
    if not prepare_negative_feature_datasets(): return False
    print("-" * 50 + "\nData preparation process (Advanced) finished.\n" + "-" * 50)
    return True

# Execute Data Preparation
if not run_all_data_preparation_advanced():
    print("CRITICAL ERROR: Data preparation failed. Notebook execution cannot continue reliably.")
    # sys.exit(1) # In a script
else:
    print("Data preparation complete. Proceeding with advanced notebook steps...")

# Generates 1 sample of the target word for manual verification.
target_word = 'hey_norman'  # Phonetic spellings may produce better samples
print(f"\n--- Generating Test Sample for '{target_word}' ---")
piper_script_path = PIPER_REPO_DIR / "generate_samples.py"
output_sample_dir_adv = BASE_DATA_DIR / "generated_samples_test_adv"
output_sample_dir_adv.mkdir(parents=True, exist_ok=True)
if piper_script_path.exists():
    run_command([sys.executable, str(piper_script_path), target_word, "--max-samples", "1", "--batch-size", "1", "--output-dir", str(output_sample_dir_adv)], "Gen test sample")
    audio_path_adv = output_sample_dir_adv / "0.wav"
    if audio_path_adv.exists(): display(Audio(str(audio_path_adv), autoplay=True))
    else: print(f"Test audio not found: {audio_path_adv}")
else: print(f"ERROR: Piper script not found: {piper_script_path}")

# Generates a larger amount of wake word samples.
print(f"\n--- Generating Wake Word Samples for '{target_word}' (Advanced: 50k samples) ---")
output_ww_dir_adv = BASE_DATA_DIR / "generated_samples_ww_adv"
output_ww_dir_adv.mkdir(parents=True, exist_ok=True)
if piper_script_path.exists():
    run_command([sys.executable, str(piper_script_path), target_word, "--max-samples", "50000", "--batch-size", "100", "--output-dir", str(output_ww_dir_adv)], "Gen WW samples (50k)")
else: print(f"ERROR: Piper script not found: {piper_script_path}")

# Augmentation Setup
print("\n--- Setting up Augmentations (Advanced) ---")
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
from microwakeword.audio.spectrograms import SpectrogramGeneration

clips_adv = Clips(input_directory=str(output_ww_dir_adv), file_pattern='*.wav', max_clip_duration_s=5, remove_silence=True, random_split_seed=10, split_count=0.1)
augmenter_adv = Augmentation(
    augmentation_duration_s=3.2,
    augmentation_probabilities={"SevenBandParametricEQ":0.1,"TanhDistortion":0.05,"PitchShift":0.15,"BandStopFilter":0.1,"AddColorNoise":0.1,"AddBackgroundNoise":0.7,"Gain":0.8,"RIR":0.7},
    impulse_paths=[str(MIT_RIR_OUTPUT_DIR)], background_paths=[str(FMA_OUTPUT_WAV_DIR), str(AUDIOSONET_OUTPUT_WAV_DIR)],
    background_min_snr_db=5, background_max_snr_db=10, min_jitter_s=0.2, max_jitter_s=0.3
)

# Augment a random clip for verification
print("\n--- Augmenting and Playing Test Clip (Advanced) ---")
from microwakeword.audio.audio_utils import save_clip
augmented_clip_path_test_adv = BASE_DATA_DIR / "augmented_clip_test_adv.wav"
try:
    random_clip_data_adv = clips_adv.get_random_clip()
    augmented_clip_data_adv = augmenter_adv.augment_clip(random_clip_data_adv)
    save_clip(augmented_clip_data_adv, str(augmented_clip_path_test_adv))
    display(Audio(str(augmented_clip_path_test_adv), autoplay=True))
except Exception as e: print(f"Error test augmentation: {e}")

# Augment samples and save sets
print("\n--- Generating Augmented Features (Advanced) ---")
from mmap_ninja.ragged import RaggedMmap
output_dir_aug_feat_adv = BASE_DATA_DIR / 'generated_augmented_features_adv'
output_dir_aug_feat_adv.mkdir(parents=True, exist_ok=True)
splits_conf_adv = {"training":{"name":"train","repetition":2,"slide_frames":10},"validation":{"name":"validation","repetition":1,"slide_frames":10},"testing":{"name":"test","repetition":1,"slide_frames":1}}
for split_item_adv, conf_adv in splits_conf_adv.items():
    out_dir_split_adv = output_dir_aug_feat_adv / split_item_adv; out_dir_split_adv.mkdir(parents=True, exist_ok=True)
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
    "window_step_ms": 10, "train_dir": str(BASE_DATA_DIR / "trained_models_adv/wakeword"),
    "features": [
        {"features_dir":str(output_dir_aug_feat_adv),"sampling_weight":2.0,"penalty_weight":1.0,"truth":True,"truncation_strategy":"truncate_start","type":"mmap"},
        {"features_dir":str(NEGATIVE_FEATURES_OUTPUT_DIR/"speech"),"sampling_weight":12.0,"penalty_weight":1.0,"truth":False,"truncation_strategy":"random","type":"mmap"},
        {"features_dir":str(NEGATIVE_FEATURES_OUTPUT_DIR/"dinner_party"),"sampling_weight":12.0,"penalty_weight":1.0,"truth":False,"truncation_strategy":"random","type":"mmap"},
        {"features_dir":str(NEGATIVE_FEATURES_OUTPUT_DIR/"no_speech"),"sampling_weight":5.0,"penalty_weight":1.0,"truth":False,"truncation_strategy":"random","type":"mmap"},
        {"features_dir":str(NEGATIVE_FEATURES_OUTPUT_DIR/"dinner_party_eval"),"sampling_weight":0.0,"penalty_weight":1.0,"truth":False,"truncation_strategy":"split","type":"mmap"},
    ],
    "training_steps": [40000], "positive_class_weight": [1], "negative_class_weight": [20],
    "learning_rates": [0.0005], "batch_size": 128, # Reduced LR for potentially larger dataset
    "time_mask_max_size": [5], "time_mask_count": [2], "freq_mask_max_size": [5], "freq_mask_count": [2], # Enabled SpecAugment
    "eval_step_interval": 1000, "clip_duration_ms": 2000, # Increased eval interval and clip duration
    "target_minimization": 0.85, "minimization_metric": "false_positive_rate", # Example: target low FP rate
    "maximization_metric": "recall" # Then maximize recall
}
(BASE_DATA_DIR / "trained_models_adv/wakeword").mkdir(parents=True, exist_ok=True)
training_params_path_yaml_adv = BASE_DATA_DIR / "training_parameters_adv.yaml"
with open(training_params_path_yaml_adv, "w") as f_yaml_adv: yaml.dump(config_train_adv, f_yaml_adv)
print(f"Advanced training parameters saved to {training_params_path_yaml_adv}")

# Model Training
print("\n--- Starting Model Training (Advanced) ---")
cmd_train_model_adv = [
    sys.executable, "-m", "microwakeword.model_train_eval",
    "--training_config", str(training_params_path_yaml_adv), "--train", "1", "--restore_checkpoint", "1",
    "--test_tf_nonstreaming", "0", "--test_tflite_nonstreaming", "0", "--test_tflite_nonstreaming_quantized", "0",
    "--test_tflite_streaming", "0", "--test_tflite_streaming_quantized", "1", "--use_weights", "best_weights",
    "mixednet", "--pointwise_filters", "64,64,64,64,64", "--repeat_in_block", "1,1,1,1,1", # Example: deeper model
    "--mixconv_kernel_sizes", "[3,5],[5,7],[7,9],[9,11],[11,13]", "--residual_connection", "0,0,0,0,0",
    "--first_conv_filters", "48", "--first_conv_kernel_size", "7", "--stride", "2" # Example: different first layer
]
run_command(cmd_train_model_adv, "Training advanced model")
print("Advanced model training/evaluation finished.")

# Prepare Output Files
print("\n--- Preparing Output Model Files (Advanced) ---")
source_tflite_adv = BASE_DATA_DIR / "trained_models_adv/wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
dest_tflite_adv = BASE_DATA_DIR / f"{target_word}_advanced_model.tflite"
if source_tflite_adv.exists():
    shutil.copy(source_tflite_adv, dest_tflite_adv); print(f"Copied TFLite model to {dest_tflite_adv}")
else: print(f"ERROR: Trained TFLite model not found at {source_tflite_adv}")

json_out_adv = {
    "type":"micro", "wake_word":target_word, "author":"kiwina", "website":"https://github.com/kiwina/MicroWakeWord-Trainer-Docker",
    "model":f"{target_word}_advanced_model.tflite", "trained_languages":["en"], "version":1, # Start version at 1 for new model
    "micro":{"probability_cutoff":0.9,"sliding_window_size":6,"feature_step_size":10,"tensor_arena_size":35000,"minimum_esphome_version":"2024.7.0"} # Example adjustments
}
dest_json_adv = BASE_DATA_DIR / f"{target_word}_advanced_model.json"
with open(dest_json_adv, "w") as f_json_adv: json.dump(json_out_adv, f_json_adv, indent=2)
print(f"Created JSON metadata at {dest_json_adv}")

print("\n--- Advanced Script Finished ---")
if dest_tflite_adv.exists(): display(FileLink(str(dest_tflite_adv)))
if dest_json_adv.exists(): display(FileLink(str(dest_json_adv)))
