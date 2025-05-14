#!/usr/bin/env python
# coding: utf-8

# # Advanced MicroWakeWord Training Notebook (Docker Version)
#
# This notebook provides an advanced training workflow for MicroWakeWord models, with more samples, deeper models, and additional tuning options compared to the basic notebook.
#
# Python 3.11.11 compatible.

# In[ ]:


# CELL 1: Initial Setup, Dependency Checks, Data Preparation Call, and Path Definitions (Advanced Notebook)
import os
import sys
import subprocess
import importlib.util
from pathlib import Path

# Check for required dependencies
def check_dependency(package_name, min_version=None):
    """Check if a package is installed and optionally verify its version."""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return False, f"{package_name} is not installed"

        if min_version:
            pkg = importlib.import_module(package_name)
            version = getattr(pkg, '__version__', '0.0.0')
            if version < min_version:
                return False, f"{package_name} version {version} is installed, but version {min_version} or higher is required"

        return True, f"{package_name} is installed"
    except Exception as e:
        return False, f"Error checking {package_name}: {str(e)}"

# List of required dependencies based on requirements.txt
required_dependencies = [
    "torch",
    "torchaudio",
    "torchvision",
    "audiomentations",
    "audioread",
    "librosa",
    "soundfile",
    "soxr",
    "webrtcvad",
    "datasets",
    "dill",
    "filelock",
    "fsspec",
    "huggingface_hub",
    "mmap_ninja",
    "multiprocess",
    "pandas",
    "pooch",
    "pyarrow",
    "xxhash",
    "microwakeword"  # This should be installed from the local project via Dockerfile
]

print("Checking required dependencies...")
missing_dependencies = []
for dep in required_dependencies:
    is_installed, message = check_dependency(dep)
    print(f"  {message}")
    if not is_installed:
        missing_dependencies.append(dep)

if missing_dependencies:
    print(f"\nWARNING: The following dependencies are missing: {', '.join(missing_dependencies)}")
    print("Some notebook cells may fail. Please ensure all dependencies are installed.")
    print("If running in Docker, these should be installed automatically via the Dockerfile.")
else:
    print("\nAll required dependencies are installed.")

# Conditional import for display, works in Jupyter, no-op in pure script
try:
    from IPython.display import Audio, display, FileLink
except ImportError:
    def display(*args, **kwargs): pass
    def Audio(*args, **kwargs): pass
    def FileLink(*args, **kwargs): pass

print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}") # Should be /data in Docker

# In[ ]:


# CELL 2: Initial Setup and Data Preparation (Advanced Notebook)
# This cell runs the main data preparation script.
# It ensures all necessary repositories and datasets are downloaded and processed into /data.
import os
import sys
import subprocess
from pathlib import Path

print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}") # Should be /data in Docker

PREPARE_DATA_SCRIPT_PATH = Path("/data/prepare_local_data.py")
DATA_DIR_INSIDE_DOCKER = Path("/data") # Consistent with prepare_local_data.py's default in Docker

print(f"\n--- Running Data Preparation Script: {PREPARE_DATA_SCRIPT_PATH} (Advanced Notebook) ---")
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
        sys.exit(f"Data preparation failed (script error), cannot continue. Check logs for {PREPARE_DATA_SCRIPT_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Python executable not found at {sys.executable}")
        sys.exit("Python executable not found for data prep script.")
    except Exception as e:
        print(f"An unexpected error occurred while running prepare_local_data.py: {e}")
        sys.exit(f"Unexpected error running data prep script: {e}")
else:
    print(f"ERROR: {PREPARE_DATA_SCRIPT_PATH} not found. Please ensure it's copied to /data by startup.sh.")
    sys.exit(f"{PREPARE_DATA_SCRIPT_PATH} not found, critical for data setup.")
print("--- Data Preparation Finished ---\n")

# Define target_word early as it's used in path definitions
target_word = 'hey_norman'  # Phonetic spellings may produce better samples. User should change this.
print(f"Target wake word set to: {target_word}")

# Define base paths that subsequent cells will use for the ADVANCED notebook
PIPER_SAMPLE_GENERATOR_DIR = DATA_DIR_INSIDE_DOCKER / "piper-sample-generator" # Shared
PIPER_SCRIPT_PATH = PIPER_SAMPLE_GENERATOR_DIR / "generate_samples.py"

# Updated generated samples directory structure as per feedback
GENERATED_SAMPLES_BASE_DIR_ADV = DATA_DIR_INSIDE_DOCKER / "generated_samples" / target_word # Shared base for target word
TEST_SAMPLE_OUTPUT_DIR_ADV = GENERATED_SAMPLES_BASE_DIR_ADV / "test_adv" # Suffix for advanced test samples
WW_SAMPLES_OUTPUT_DIR_ADV = GENERATED_SAMPLES_BASE_DIR_ADV / "samples_adv" # Suffix for advanced WW samples

MIT_RIRS_PATH = DATA_DIR_INSIDE_DOCKER / "mit_rirs" # Shared
FMA_16K_PATH = DATA_DIR_INSIDE_DOCKER / "fma_16k" # Shared
AUDIOSONET_16K_PATH = DATA_DIR_INSIDE_DOCKER / "audioset_16k" # Shared
NEGATIVE_DATASETS_PATH = DATA_DIR_INSIDE_DOCKER / "negative_datasets" # Shared

# Specific paths for advanced notebook outputs
AUGMENTED_FEATURES_DIR_ADV = DATA_DIR_INSIDE_DOCKER / "generated_augmented_features_adv"
TRAINED_MODELS_BASE_PATH_ADV = DATA_DIR_INSIDE_DOCKER / "trained_models_adv"
TRAINING_CONFIG_PATH_ADV = DATA_DIR_INSIDE_DOCKER / "training_parameters_adv.yaml"

# Ensure key directories exist
TEST_SAMPLE_OUTPUT_DIR_ADV.mkdir(parents=True, exist_ok=True)
WW_SAMPLES_OUTPUT_DIR_ADV.mkdir(parents=True, exist_ok=True)
AUGMENTED_FEATURES_DIR_ADV.mkdir(parents=True, exist_ok=True)
(TRAINED_MODELS_BASE_PATH_ADV / "wakeword").mkdir(parents=True, exist_ok=True)

# Function to check if assets exist in the data directory
def check_assets_exist(directory, pattern="*"):
    """Check if assets exist in the specified directory matching the pattern."""
    path = Path(directory)
    if not path.exists():
        return False, 0
    
    files = list(path.glob(pattern))
    return len(files) > 0, len(files)

# Ensure piper-sample-generator is in sys.path if its modules are imported directly later
if str(PIPER_SAMPLE_GENERATOR_DIR) not in sys.path:
    sys.path.append(str(PIPER_SAMPLE_GENERATOR_DIR))
    print(f"Added {PIPER_SAMPLE_GENERATOR_DIR} to sys.path")

print("Initial setup cell for Advanced Notebook (cell_2.py) complete.")


# In[ ]:


# CELL 3: Generate Test Sample and Wake Word Samples (Advanced)
import os
import sys
from pathlib import Path # Ensure Path is imported

# Conditional import for display
try:
    from IPython.display import Audio, display
except ImportError:
    def display(*args, **kwargs): pass
    def Audio(*args, **kwargs): pass

# Variables from cell_2.py:
# target_word, PIPER_SCRIPT_PATH, TEST_SAMPLE_OUTPUT_DIR_ADV, WW_SAMPLES_OUTPUT_DIR_ADV

if 'PIPER_SCRIPT_PATH' not in globals() or \
   'target_word' not in globals() or \
   'TEST_SAMPLE_OUTPUT_DIR_ADV' not in globals() or \
   'WW_SAMPLES_OUTPUT_DIR_ADV' not in globals():
    print("ERROR: Essential variables not defined. Ensure cell_2.py ran correctly.")
else:
    # Create directories if they don't exist (idempotent)
    TEST_SAMPLE_OUTPUT_DIR_ADV.mkdir(parents=True, exist_ok=True)
    WW_SAMPLES_OUTPUT_DIR_ADV.mkdir(parents=True, exist_ok=True)

    if not PIPER_SCRIPT_PATH.exists():
        print(f"ERROR: Piper sample generator script not found at {PIPER_SCRIPT_PATH}. Check data preparation in cell_2.")
    else:
        # 1. Generate 1 test sample of the target word for manual verification.
        print(f"\n--- Generating Test Sample for '{target_word}' (Advanced) ---")
        
        # Check if test sample already exists (asset caching)
        assets_exist, file_count = check_assets_exist(TEST_SAMPLE_OUTPUT_DIR_ADV, "*.wav")
        if assets_exist:
            print(f"Found {file_count} existing test samples in {TEST_SAMPLE_OUTPUT_DIR_ADV}")
            audio_path_test_adv = next(TEST_SAMPLE_OUTPUT_DIR_ADV.glob("*.wav"))
            print(f"Using existing test sample: {audio_path_test_adv}")
            display(Audio(str(audio_path_test_adv), autoplay=True))
        else:
            test_sample_cmd = f"\"{sys.executable}\" \"{str(PIPER_SCRIPT_PATH)}\" \"{target_word}\" \\
            --max-samples 1 \\
            --batch-size 1 \\
            --output-dir \"{str(TEST_SAMPLE_OUTPUT_DIR_ADV)}\""
            
            print(f"Executing: {test_sample_cmd}")
            try:
                if "get_ipython" in globals():
                    get_ipython().system(test_sample_cmd)
                else:
                    print("Warning: get_ipython() not available. Cannot execute command directly. For .py, use subprocess.")
                    # import subprocess
                    # subprocess.run(test_sample_cmd, shell=True, check=True)
            except Exception as e:
                print(f"Error executing sample generation command: {e}")
                print("This might be due to missing dependencies or configuration issues.")

            audio_path_test_adv = TEST_SAMPLE_OUTPUT_DIR_ADV / "0.wav"
            if audio_path_test_adv.exists():
                print(f"Playing test sample: {audio_path_test_adv}")
                display(Audio(str(audio_path_test_adv), autoplay=True))
            else:
                print(f"Audio file not found at {audio_path_test_adv}. Sample generation might have failed.")

        # 2. Generates a larger amount of wake word samples for advanced training.
        print(f"\n--- Generating Wake Word Samples for '{target_word}' (Advanced: 50k) into {WW_SAMPLES_OUTPUT_DIR_ADV} ---")
        
        # Check if wake word samples already exist (asset caching)
        assets_exist, file_count = check_assets_exist(WW_SAMPLES_OUTPUT_DIR_ADV, "*.wav")
        if assets_exist:
            print(f"Found {file_count} existing wake word samples in {WW_SAMPLES_OUTPUT_DIR_ADV}")
            print("Skipping generation of new samples. Delete the directory if you want to regenerate.")
        else:
            # Start here when trying to improve your model.
            # See https://github.com/kiwina/piper-sample-generator for the full set of
            # parameters. In particular, experiment with noise-scales and noise-scale-ws,
            # generating negative samples similar to the wake word, and generating many more
            # wake word samples, possibly with different phonetic pronunciations.
            
            ww_samples_cmd = f"\"{sys.executable}\" \"{str(PIPER_SCRIPT_PATH)}\" \"{target_word}\" \\
            --max-samples 50000 \\
            --batch-size 100 \\
            --output-dir \"{str(WW_SAMPLES_OUTPUT_DIR_ADV)}\""

            print(f"Executing: {ww_samples_cmd}")
            try:
                if "get_ipython" in globals():
                    get_ipython().system(ww_samples_cmd)
                else:
                    print("Warning: get_ipython() not available. Cannot execute command directly. For .py, use subprocess.")
                    # import subprocess
                    # subprocess.run(ww_samples_cmd, shell=True, check=True)
            except Exception as e:
                print(f"Error executing sample generation command: {e}")
                print("This might be due to missing dependencies or configuration issues.")

            print(f"Wake word samples generation command issued for {WW_SAMPLES_OUTPUT_DIR_ADV}")


# In[ ]:


# CELL 4: Sets up the augmentations (Advanced)
# This cell assumes all necessary datasets (MIT RIR, Audioset, FMA)
# have been downloaded and processed into their respective /data/..._16k directories
# by the prepare_local_data.py script called in cell_2.py.

import os
from pathlib import Path # Ensure Path is imported
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
# SpectrogramGeneration is imported in a later cell where it's used.

# These variables should be defined in cell_2.py and available globally:
# WW_SAMPLES_OUTPUT_DIR_ADV, MIT_RIRS_PATH, FMA_16K_PATH, AUDIOSONET_16K_PATH

print("\n--- Setting up Augmentations (Advanced) ---")

# Check if essential path variables from cell_2 exist
required_paths_for_cell_4_adv = {
    "WW_SAMPLES_OUTPUT_DIR_ADV": WW_SAMPLES_OUTPUT_DIR_ADV if 'WW_SAMPLES_OUTPUT_DIR_ADV' in globals() else None,
    "MIT_RIRS_PATH": MIT_RIRS_PATH if 'MIT_RIRS_PATH' in globals() else None,
    "FMA_16K_PATH": FMA_16K_PATH if 'FMA_16K_PATH' in globals() else None,
    "AUDIOSONET_16K_PATH": AUDIOSONET_16K_PATH if 'AUDIOSONET_16K_PATH' in globals() else None
}

missing_paths_adv = [name for name, path_val in required_paths_for_cell_4_adv.items() if path_val is None or not Path(path_val).exists()]

if missing_paths_adv:
    print(f"ERROR: One or more required data paths are missing or not defined from cell_2: {', '.join(missing_paths_adv)}")
    print("Please ensure cell_2.py (data preparation) ran successfully and all paths are correct.")
    clips_adv = None # Set to None to indicate failure to subsequent cells
    augmenter_adv = None
else:
    print(f"Using wake word samples from: {WW_SAMPLES_OUTPUT_DIR_ADV}")
    print(f"Using MIT RIRs from: {MIT_RIRS_PATH}")
    print(f"Using FMA 16k from: {FMA_16K_PATH}")
    print(f"Using Audioset 16k from: {AUDIOSONET_16K_PATH}")

    clips_adv = Clips(
        input_directory=str(WW_SAMPLES_OUTPUT_DIR_ADV), # From cell_2
        file_pattern='*.wav',
        max_clip_duration_s=5, # Advanced notebook might use a specific duration
        remove_silence=True,   # Advanced notebook might enable silence removal
        random_split_seed=10,
        split_count=0.1, # Corresponds to 10% for test and 10% for validation
    )

    augmenter_adv = Augmentation(
        augmentation_duration_s=3.2,
        augmentation_probabilities={ # Example probabilities, can be tuned
            "SevenBandParametricEQ": 0.1,
            "TanhDistortion": 0.05, # Less distortion than basic
            "PitchShift": 0.15,    # More pitch variation
            "BandStopFilter": 0.1,
            "AddColorNoise": 0.1,
            "AddBackgroundNoise": 0.7, # Slightly less than basic, assuming cleaner negatives
            "Gain": 0.8, # Allow more gain variation
            "RIR": 0.7,  # More reverberation
        },
        impulse_paths=[str(MIT_RIRS_PATH)], # From cell_2
        background_paths=[str(FMA_16K_PATH), str(AUDIOSONET_16K_PATH)], # From cell_2
        background_min_snr_db=5,  # Higher min SNR for advanced
        background_max_snr_db=15, # Higher max SNR
        min_jitter_s=0.15, # Adjusted jitter
        max_jitter_s=0.25,
    )
    print("Advanced augmentation setup complete.")


# In[ ]:


# CELL 5: Augment a random clip and play it back to verify it works well (Advanced)
from pathlib import Path # Ensure Path is imported

# Conditional import for display
try:
    from IPython.display import Audio, display
except ImportError:
    def display(*args, **kwargs): pass
    def Audio(*args, **kwargs): pass

from microwakeword.audio.audio_utils import save_clip # Assuming this is installed with microwakeword

# These variables should be defined from cell_2.py and cell_4.py:
# DATA_DIR_INSIDE_DOCKER, clips_adv, augmenter_adv

print("\n--- Augmenting and Playing Test Clip (Advanced) ---")

if 'DATA_DIR_INSIDE_DOCKER' not in globals() or \
   'clips_adv' not in globals() or \
   'augmenter_adv' not in globals() or \
   clips_adv is None or \
   augmenter_adv is None:
    print("ERROR: Essential variables (DATA_DIR_INSIDE_DOCKER, clips_adv, augmenter_adv) not defined or not initialized. Ensure cell_2.py and cell_4.py ran correctly.")
else:
    augmented_clip_path_test_adv = DATA_DIR_INSIDE_DOCKER / "augmented_clip_test_adv.wav" # Save to /data root

    try:
        random_clip_data_adv = clips_adv.get_random_clip()
        if random_clip_data_adv is None or random_clip_data_adv.size == 0:
            print("ERROR: get_random_clip() returned None or empty data. Check Clips setup in cell_4 and sample generation in cell_3.")
        else:
            augmented_clip_data_adv = augmenter_adv.augment_clip(random_clip_data_adv)
            save_clip(augmented_clip_data_adv, str(augmented_clip_path_test_adv))
            print(f"Playing augmented test clip: {augmented_clip_path_test_adv}")
            display(Audio(str(augmented_clip_path_test_adv), autoplay=True))
    except Exception as e:
        print(f"Error during advanced test augmentation: {e}. Check if previous cells ran successfully and data paths are correct.")
        print("Make sure 'clips_adv' and 'augmenter_adv' objects were created successfully in cell_4.")


# In[ ]:


# CELL 6: Augment samples and save the training, validation, and testing sets (Advanced)
import os
from pathlib import Path # Ensure Path is imported
from mmap_ninja.ragged import RaggedMmap
from microwakeword.audio.spectrograms import SpectrogramGeneration # Ensure this is imported

# These variables should be defined from cell_2.py and cell_4.py:
# AUGMENTED_FEATURES_DIR_ADV, clips_adv, augmenter_adv

print("\n--- Generating Augmented Features for Training/Validation/Testing (Advanced) ---")

if 'AUGMENTED_FEATURES_DIR_ADV' not in globals() or \
   'clips_adv' not in globals() or \
   'augmenter_adv' not in globals() or \
   clips_adv is None or \
   augmenter_adv is None:
    print("ERROR: Essential variables (AUGMENTED_FEATURES_DIR_ADV, clips_adv, augmenter_adv) not defined or not initialized. Ensure cell_2.py and cell_4.py ran correctly.")
else:
    AUGMENTED_FEATURES_DIR_ADV.mkdir(parents=True, exist_ok=True)

    splits_config_adv = {
        "training": {"name": "train", "repetition": 2, "slide_frames": 10}, # Default for basic, can be tuned for advanced
        "validation": {"name": "validation", "repetition": 1, "slide_frames": 10},
        "testing": {"name": "test", "repetition": 1, "slide_frames": 1}, # Streaming test
    }

    for split_item, config_adv in splits_config_adv.items():
      out_dir_split_adv = AUGMENTED_FEATURES_DIR_ADV / split_item # Use variable from cell_2
      out_dir_split_adv.mkdir(parents=True, exist_ok=True)

      # Ensure clips_adv and augmenter_adv are not None (i.e., cell_4 executed successfully)
      if clips_adv is None or augmenter_adv is None:
          print(f"Skipping feature generation for {split_item} due to missing clips_adv or augmenter_adv setup from cell_4.")
          continue

      current_spectrograms_adv = SpectrogramGeneration(
          clips=clips_adv,
          augmenter=augmenter_adv,
          slide_frames=config_adv["slide_frames"],
          step_ms=10, # Can be tuned
      )

      print(f"Generating augmented features for {config_adv['name']} set into {out_dir_split_adv} (Advanced)...")
      try:
        RaggedMmap.from_generator(
            out_dir=str(out_dir_split_adv / 'wakeword_mmap'), # Ensure path is string
            sample_generator=current_spectrograms_adv.spectrogram_generator(split=config_adv["name"], repeat=config_adv["repetition"]),
            batch_size=100, # Can be tuned
            verbose=True,
        )
        print(f"Finished generating features for {config_adv['name']} set (Advanced).")
      except Exception as e:
        print(f"Error generating features for {config_adv['name']} set (Advanced): {e}")


# In[ ]:


# CELL 7: Save a yaml config that controls the training process (Advanced)
# These hyperparamters can make a huge different in model quality.
# Experiment with sampling and penalty weights and increasing the number of
# training steps.
import yaml
import os
from pathlib import Path # Ensure Path is imported

# These variables should be defined in cell_2.py and available globally:
# AUGMENTED_FEATURES_DIR_ADV, NEGATIVE_DATASETS_PATH,
# TRAINED_MODELS_BASE_PATH_ADV, TRAINING_CONFIG_PATH_ADV

print("\n--- Preparing Training Configuration (Advanced) ---")

if 'AUGMENTED_FEATURES_DIR_ADV' not in globals() or \
   'NEGATIVE_DATASETS_PATH' not in globals() or \
   'TRAINED_MODELS_BASE_PATH_ADV' not in globals() or \
   'TRAINING_CONFIG_PATH_ADV' not in globals():
    print("ERROR: Essential path variables for training config are not defined. Ensure cell_2.py ran correctly.")
    # Optionally, raise an error or sys.exit()
    config_train_adv = {} # Create empty config to avoid further errors
else:
    # Ensure the base training directory exists
    (TRAINED_MODELS_BASE_PATH_ADV / "wakeword").mkdir(parents=True, exist_ok=True)

    config_train_adv = {
        "window_step_ms": 10,
        "train_dir": str(TRAINED_MODELS_BASE_PATH_ADV / "wakeword"), # Use variable
        "features": [
            {
                "features_dir": str(AUGMENTED_FEATURES_DIR_ADV), # Use variable
                "sampling_weight": 2.0, # Increased for advanced
                "penalty_weight": 1.0,
                "truth": True,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            },
            {
                "features_dir": str(NEGATIVE_DATASETS_PATH / "speech"), # Use variable
                "sampling_weight": 12.0, # Adjusted for advanced
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": str(NEGATIVE_DATASETS_PATH / "dinner_party"), # Use variable
                "sampling_weight": 12.0, # Adjusted for advanced
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": str(NEGATIVE_DATASETS_PATH / "no_speech"), # Use variable
                "sampling_weight": 5.0, # Balanced
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            { # Only used for validation and testing
                "features_dir": str(NEGATIVE_DATASETS_PATH / "dinner_party_eval"), # Use variable
                "sampling_weight": 0.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "split",
                "type": "mmap",
            },
        ],
        "training_steps": [40000],  # Increased for advanced
        "positive_class_weight": [1],
        "negative_class_weight": [20],  # Adjusted
        "learning_rates": [0.0005],  # Adjusted for advanced (potentially smaller LR for longer training)
        "batch_size": 128,
        "time_mask_max_size": [5],  # Enabled SpecAugment for advanced
        "time_mask_count": [2],
        "freq_mask_max_size": [5],
        "freq_mask_count": [2],
        "eval_step_interval": 1000,  # Evaluate less frequently for longer training
        "clip_duration_ms": 2000,  # Potentially longer clips for advanced
        "target_minimization": 0.85, # Stricter target for advanced
        "minimization_metric": "false_positive_rate", # Example: target low FP rate
        "maximization_metric": "recall" # Then maximize recall
    }

    with open(TRAINING_CONFIG_PATH_ADV, "w") as file_yaml_adv: # Use variable
        yaml.dump(config_train_adv, file_yaml_adv)
    print(f"Advanced training parameters saved to {TRAINING_CONFIG_PATH_ADV}")


# In[ ]:


# CELL 8: Trains a model (Advanced)
# This cell executes the main model training process using the configuration
# defined in the previous cell and saved to training_parameters_adv.yaml.

import os
import sys
from pathlib import Path # Ensure Path is imported

# TRAINING_CONFIG_PATH_ADV should be defined in cell_2.py and the YAML file created in cell_7.py
# Example: TRAINING_CONFIG_PATH_ADV = DATA_DIR_INSIDE_DOCKER / "training_parameters_adv.yaml"

print("\n--- Starting Model Training (Advanced) ---")

if 'TRAINING_CONFIG_PATH_ADV' not in globals() or not TRAINING_CONFIG_PATH_ADV.exists():
    print(f"ERROR: Training config path {TRAINING_CONFIG_PATH_ADV if 'TRAINING_CONFIG_PATH_ADV' in globals() else 'TRAINING_CONFIG_PATH_ADV variable not found'} not found or variable not defined. Ensure cell_2.py and cell_7.py ran correctly.")
else:
    print(f"Starting advanced model training using config: {TRAINING_CONFIG_PATH_ADV}")

    # Construct the command string for the ! operator
    # Using more advanced/deeper model parameters as an example
    training_command_adv = f"\"{sys.executable}\" -m microwakeword.model_train_eval \\
    --training_config='{str(TRAINING_CONFIG_PATH_ADV)}' \\
    --train 1 \\
    --restore_checkpoint 1 \\
    --test_tf_nonstreaming 0 \\
    --test_tflite_nonstreaming 0 \\
    --test_tflite_nonstreaming_quantized 0 \\
    --test_tflite_streaming 0 \\
    --test_tflite_streaming_quantized 1 \\
    --use_weights \"best_weights\" \\
    mixednet \\
    --pointwise_filters \"64,64,64,64,64\" \\
    --repeat_in_block  \"1,1,1,1,1\" \\
    --mixconv_kernel_sizes '[3,5],[5,7],[7,9],[9,11],[11,13]' \\
    --residual_connection \"0,0,0,0,0\" \\
    --first_conv_filters 48 \\
    --first_conv_kernel_size 7 \\
    --stride 2" # Example: different first layer, stride

    print(f"Executing advanced training command:\n{training_command_adv}")

    if "get_ipython" in globals():
        get_ipython().system(training_command_adv)
    else:
        print("Warning: get_ipython() not available. Cannot execute training command directly in this .py script.")
        print("This cell is intended to be converted back to an .ipynb cell.")
        # As a fallback for pure .py script testing:
        # import subprocess
        # subprocess.run(training_command_adv, shell=True, check=True)

print("Advanced model training/evaluation finished (or command printed if not in IPython).")


# In[ ]:


# CELL 9: Prepare Output Model Files (Advanced)
import shutil
import json
import os
from pathlib import Path # Ensure Path is imported

# Conditional import for display
try:
    from IPython.display import FileLink, display
except ImportError:
    def display(*args, **kwargs): pass
    def FileLink(*args, **kwargs): pass

# These variables should be defined in cell_2.py:
# target_word, TRAINED_MODELS_BASE_PATH_ADV, DATA_DIR_INSIDE_DOCKER

print("\n--- Preparing Output Model Files (Advanced) ---")

if 'target_word' not in globals() or \
   'TRAINED_MODELS_BASE_PATH_ADV' not in globals() or \
   'DATA_DIR_INSIDE_DOCKER' not in globals():
    print("ERROR: Essential variables (target_word, TRAINED_MODELS_BASE_PATH_ADV, DATA_DIR_INSIDE_DOCKER) not defined. Ensure cell_2.py ran correctly.")
else:
    source_tflite_path_adv = TRAINED_MODELS_BASE_PATH_ADV / "wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
    # Save final model files directly into the /data root for easy access from host, with _adv suffix
    destination_tflite_path_adv = DATA_DIR_INSIDE_DOCKER / f"{target_word}_advanced_model.tflite"
    destination_json_path_adv = DATA_DIR_INSIDE_DOCKER / f"{target_word}_advanced_model.json"

    if source_tflite_path_adv.exists():
        shutil.copy(source_tflite_path_adv, destination_tflite_path_adv)
        print(f"Copied ADVANCED TFLite model to {destination_tflite_path_adv}")
    else:
        print(f"ERROR: Trained ADVANCED TFLite model not found at {source_tflite_path_adv}. Training might have failed or model path is incorrect.")

    json_data_adv = {
        "type": "micro",
        "wake_word": target_word,  # Using the target_word variable from cell_2
        "author": "kiwina", 
        "website": "https://github.com/kiwina/MicroWakeWord-Trainer-Docker",
        "model": f"{target_word}_advanced_model.tflite", # Relative path for use with ESPHome
        "trained_languages": ["en"],
        "version": 1, # Start version at 1 for a new advanced model, user can increment
        "micro": { # Example parameters for an "advanced" model, tune as needed
            "probability_cutoff": 0.90, # Potentially lower for better recall if FP is managed
            "sliding_window_size": 6,   # Might be larger/smaller depending on model architecture
            "feature_step_size": 10,
            "tensor_arena_size": 35000, # Potentially larger for a more complex model
            "minimum_esphome_version": "2024.7.0"
        }
    }

    with open(destination_json_path_adv, "w") as json_file_adv:
        json.dump(json_data_adv, json_file_adv, indent=2)
    print(f"Created ADVANCED JSON metadata at {destination_json_path_adv}")

    print("\n--- Advanced Script Finished ---")
    print(f"Output files for ADVANCED model are located in: {DATA_DIR_INSIDE_DOCKER.resolve().absolute()}")
    print("If running in Docker, this corresponds to the 'microwakeword-trainer-data' directory on your host machine.")

    if destination_tflite_path_adv.exists():
        print("\nADVANCED TFLite Model Link (for Jupyter environments):")
        display(FileLink(str(destination_tflite_path_adv)))
    if destination_json_path_adv.exists():
        print("\nADVANCED JSON Metadata Link (for Jupyter environments):")
        display(FileLink(str(destination_json_path_adv)))

# Cells 10, 11, 12 from the original advanced notebook are now effectively covered or made redundant
# by the new structure (cell_7 for YAML, cell_8 for training, this cell_9 for output).
# If they had other unique logic, it would need to be merged or placed in a new cell.
# For now, we assume this cell_9 is the last meaningful code cell for the advanced notebook.


# Cells 10 and 11 from the original advanced notebook have been removed as they are redundant
# Cell 10 duplicated the training functionality already in cell 8
# Cell 11 duplicated the model file preparation already in cell 9

