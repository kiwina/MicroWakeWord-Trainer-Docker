#!/usr/bin/env python
# coding: utf-8

# In[1]:


# CELL 1: Initial Setup, Dependency Checks, Data Preparation Call, and Path Definitions (Basic Notebook)
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

# --- Call External Data Preparation Script ---
PREPARE_DATA_SCRIPT_PATH = Path("/data/prepare_local_data.py")
DATA_DIR_INSIDE_DOCKER = Path("/data") # Consistent with prepare_local_data.py's default in Docker

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

# --- Define Global Paths and Variables for this Notebook ---
target_word = 'khum_puter'  # Phonetic spellings may produce better samples. User should change this.
print(f"Target wake word set to: {target_word}")

PIPER_SAMPLE_GENERATOR_DIR = DATA_DIR_INSIDE_DOCKER / "piper-sample-generator"
PIPER_SCRIPT_PATH = PIPER_SAMPLE_GENERATOR_DIR / "generate_samples.py"

# Updated generated samples directory structure as per feedback
GENERATED_SAMPLES_BASE_DIR = DATA_DIR_INSIDE_DOCKER / "generated_samples" / target_word
TEST_SAMPLE_OUTPUT_DIR = GENERATED_SAMPLES_BASE_DIR / "test"
WW_SAMPLES_OUTPUT_DIR = GENERATED_SAMPLES_BASE_DIR / "samples"

MIT_RIRS_PATH = DATA_DIR_INSIDE_DOCKER / "mit_rirs"
FMA_16K_PATH = DATA_DIR_INSIDE_DOCKER / "fma_16k"
AUDIOSONET_16K_PATH = DATA_DIR_INSIDE_DOCKER / "audioset_16k"
AUGMENTED_FEATURES_DIR = DATA_DIR_INSIDE_DOCKER / "generated_augmented_features"
NEGATIVE_DATASETS_PATH = DATA_DIR_INSIDE_DOCKER / "negative_datasets"
TRAINED_MODELS_BASE_PATH = DATA_DIR_INSIDE_DOCKER / "trained_models" # For basic notebook
TRAINING_CONFIG_PATH = DATA_DIR_INSIDE_DOCKER / "training_parameters.yaml" # For basic notebook

# Ensure key directories exist (prepare_local_data.py handles dataset dirs)
TEST_SAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WW_SAMPLES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUGMENTED_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
(TRAINED_MODELS_BASE_PATH / "wakeword").mkdir(parents=True, exist_ok=True)

# Function to check if assets exist in the data directory
def check_assets_exist(directory, pattern="*"):
    """Check if assets exist in the specified directory matching the pattern."""
    path = Path(directory)
    if not path.exists():
        return False, 0
    
    files = list(path.glob(pattern))
    return len(files) > 0, len(files)

# Add piper-sample-generator to sys.path if its modules are imported directly later
if str(PIPER_SAMPLE_GENERATOR_DIR) not in sys.path:
    sys.path.append(str(PIPER_SAMPLE_GENERATOR_DIR))
    print(f"Added {PIPER_SAMPLE_GENERATOR_DIR} to sys.path")

print("Initial setup cell (cell_1.py) complete: Data prep called and paths defined.")


# In[3]:


# CELL 2: Generate 1 test sample of the target word for manual verification.
import os
import sys
from pathlib import Path # Ensure Path is imported

# Conditional import for display
try:
    from IPython.display import Audio, display
except ImportError:
    def display(*args, **kwargs): pass
    def Audio(*args, **kwargs): pass

# Check for torch dependency which is required by piper-sample-generator
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    print("ERROR: PyTorch is not installed. Sample generation will fail.")
    print("Please install PyTorch with: pip install torch torchaudio torchvision")

# Variables from cell_1.py:
# target_word, PIPER_SCRIPT_PATH, TEST_SAMPLE_OUTPUT_DIR

print(f"\n--- Generating Test Sample for '{target_word}' ---")

if 'PIPER_SCRIPT_PATH' not in globals() or \
   'target_word' not in globals() or \
   'TEST_SAMPLE_OUTPUT_DIR' not in globals():
    print("ERROR: Essential variables (PIPER_SCRIPT_PATH, target_word, TEST_SAMPLE_OUTPUT_DIR) not defined. Ensure cell_1.py ran correctly.")
else:
    # Ensure the output directory exists (it should have been created in cell_1)
    TEST_SAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if test sample already exists (asset caching)
    assets_exist, file_count = check_assets_exist(TEST_SAMPLE_OUTPUT_DIR, "*.wav")
    if assets_exist:
        print(f"Found {file_count} existing test samples in {TEST_SAMPLE_OUTPUT_DIR}")
        audio_path_test = next(TEST_SAMPLE_OUTPUT_DIR.glob("*.wav"))
        print(f"Using existing test sample: {audio_path_test}")
        display(Audio(str(audio_path_test), autoplay=True))
    elif not torch_available:
        print("Skipping sample generation due to missing PyTorch dependency.")
    elif not PIPER_SCRIPT_PATH.exists():
        print(f"ERROR: Piper sample generator script not found at {PIPER_SCRIPT_PATH}. Check data preparation in cell_1.")
    else:
        # Construct the command string for the ! operator
        test_sample_cmd = f"\"{sys.executable}\" \"{str(PIPER_SCRIPT_PATH)}\" \"{target_word}\" \
--max-samples 1 \
--batch-size 1 \
--output-dir \"{str(TEST_SAMPLE_OUTPUT_DIR)}\""

        print(f"Executing: {test_sample_cmd}")
        try:
            if "get_ipython" in globals():
                get_ipython().system(test_sample_cmd)
            else:
                print("Warning: get_ipython() not available. Cannot execute command directly in this .py script.")
                # Fallback for pure .py script testing (though not ideal for notebook structure):
                # import subprocess
                # subprocess.run(test_sample_cmd, shell=True, check=True)
        except Exception as e:
            print(f"Error executing sample generation command: {e}")
            print("This might be due to missing dependencies or configuration issues.")

        audio_path_test = TEST_SAMPLE_OUTPUT_DIR / "0.wav"
        if audio_path_test.exists():
            print(f"Playing test sample: {audio_path_test}")
            display(Audio(str(audio_path_test), autoplay=True))
        else:
            print(f"Audio file not found at {audio_path_test}. Sample generation might have failed.")


# In[ ]:


# CELL 3: Generate a larger amount of wake word samples.
# Start here when trying to improve your model.
# See https://github.com/kiwina/piper-sample-generator for the full set of
# parameters. In particular, experiment with noise-scales and noise-scale-ws,
# generating negative samples similar to the wake word, and generating many more
# wake word samples, possibly with different phonetic pronunciations.

import os
import sys
import subprocess
from pathlib import Path # Ensure Path is imported if not already from cell_2

# Check for torch dependency which is required by piper-sample-generator
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    print("ERROR: PyTorch is not installed. Sample generation will fail.")
    print("Please install PyTorch with: pip install torch torchaudio torchvision")

# Variables like target_word, PIPER_SCRIPT_PATH, WW_SAMPLES_OUTPUT_DIR,
# DATA_DIR_INSIDE_DOCKER should be defined in cell_1.py and thus available here.

print("\n--- Generating Wake Word Samples ---")

# Ensure the output directory for wake word samples exists
if 'WW_SAMPLES_OUTPUT_DIR' not in globals() or 'PIPER_SCRIPT_PATH' not in globals() or 'target_word' not in globals():
    print("ERROR: Essential variables (WW_SAMPLES_OUTPUT_DIR, PIPER_SCRIPT_PATH, target_word) not defined. Ensure cell_1.py ran correctly.")
else:
    WW_SAMPLES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if wake word samples already exist (asset caching)
    assets_exist, file_count = check_assets_exist(WW_SAMPLES_OUTPUT_DIR, "*.wav")
    if assets_exist:
        print(f"Found {file_count} existing wake word samples in {WW_SAMPLES_OUTPUT_DIR}")
        print("Skipping generation of new samples. Delete the directory if you want to regenerate.")
    elif not torch_available:
        print("Skipping sample generation due to missing PyTorch dependency.")
    elif not PIPER_SCRIPT_PATH.exists():
        print(f"ERROR: Piper sample generator script not found at {PIPER_SCRIPT_PATH}. Check data preparation in cell_1.")
    else:
        print(f"Generating wake word samples for '{target_word}' into {WW_SAMPLES_OUTPUT_DIR}...")
        cmd_ww_samples = [
            sys.executable, str(PIPER_SCRIPT_PATH), target_word,
            "--max-samples", "1000", # As per original basic notebook
            "--batch-size", "100",  # As per original basic notebook
            "--output-dir", str(WW_SAMPLES_OUTPUT_DIR)
        ]
        try:
            subprocess.run(cmd_ww_samples, check=True, capture_output=True, text=True)
            print(f"Wake word samples generated in {WW_SAMPLES_OUTPUT_DIR}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR generating wake word samples: {e}")
            print("Stdout:", e.stdout)
            print("Stderr:", e.stderr)
            print("This might be due to missing dependencies or configuration issues.")
        except FileNotFoundError:
            print(f"ERROR: Python executable not found at {sys.executable} or script {PIPER_SCRIPT_PATH} not found.")
        except Exception as e:
            print(f"Unexpected error during sample generation: {e}")


# In[ ]:


# CELL 4: Sets up the augmentations.
# To improve your model, experiment with these settings and use more sources of
# background clips.

import os
from pathlib import Path # Ensure Path is imported
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.clips import Clips
# SpectrogramGeneration is imported in a later cell where it's used.

# These variables should be defined in cell_1.py and available globally in the notebook context:
# WW_SAMPLES_OUTPUT_DIR, MIT_RIRS_PATH, FMA_16K_PATH, AUDIOSONET_16K_PATH, DATA_DIR_INSIDE_DOCKER

print("\n--- Setting up Augmentations ---")

# Check if essential path variables from cell_1 exist
required_paths_for_cell_4 = {
    "WW_SAMPLES_OUTPUT_DIR": WW_SAMPLES_OUTPUT_DIR if 'WW_SAMPLES_OUTPUT_DIR' in globals() else None,
    "MIT_RIRS_PATH": MIT_RIRS_PATH if 'MIT_RIRS_PATH' in globals() else None,
    "FMA_16K_PATH": FMA_16K_PATH if 'FMA_16K_PATH' in globals() else None,
    "AUDIOSONET_16K_PATH": AUDIOSONET_16K_PATH if 'AUDIOSONET_16K_PATH' in globals() else None
}

missing_paths = [name for name, path in required_paths_for_cell_4.items() if path is None or not Path(path).exists()]

if missing_paths:
    print(f"ERROR: One or more required data paths are missing or not defined from cell_1: {', '.join(missing_paths)}")
    print("Please ensure cell_1.py (data preparation) ran successfully and all paths are correct.")
    # Depending on notebook execution flow, might want to raise an error or sys.exit()
    # For now, we'll let it proceed, but Clips/Augmentation might fail.
    clips = None
    augmenter = None
else:
    print(f"Using wake word samples from: {WW_SAMPLES_OUTPUT_DIR}")
    print(f"Using MIT RIRs from: {MIT_RIRS_PATH}")
    print(f"Using FMA 16k from: {FMA_16K_PATH}")
    print(f"Using Audioset 16k from: {AUDIOSONET_16K_PATH}")

    clips = Clips(input_directory=str(WW_SAMPLES_OUTPUT_DIR), # Use variable from cell_1
                  file_pattern='*.wav',
                  max_clip_duration_s=None,
                  remove_silence=False, # Basic notebook keeps silence for initial samples
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
                             impulse_paths = [str(MIT_RIRS_PATH)], # Use variable
                             background_paths = [str(FMA_16K_PATH), str(AUDIOSONET_16K_PATH)], # Use variables
                             background_min_snr_db = -5,
                             background_max_snr_db = 10,
                             min_jitter_s = 0.195,
                             max_jitter_s = 0.205,
                            )
    print("Augmentation setup complete.")


# In[ ]:


# CELL 5: Augment a random clip and play it back to verify it works well
from IPython.display import Audio, display
from microwakeword.audio.audio_utils import save_clip
import os

augmented_clip_path = "/data/augmented_clip_test.wav"

try:
    random_clip = clips.get_random_clip()
    augmented_clip = augmenter.augment_clip(random_clip)
    save_clip(augmented_clip, augmented_clip_path)
    print(f"Playing augmented test clip: {augmented_clip_path}")
    display(Audio(augmented_clip_path, autoplay=True))
except Exception as e:
    print(f"Error during test augmentation: {e}. Check if previous cells ran successfully and data paths are correct.")


# In[ ]:


# CELL 6: Augment samples and save the training, validation, and testing sets.
# Validating and testing samples generated the same way can make the model
# benchmark better than it performs in real-word use. Use real samples or TTS
# samples generated with a different TTS engine to potentially get more accurate
# benchmarks.
import os
from pathlib import Path # Ensure Path is imported
from mmap_ninja.ragged import RaggedMmap
from microwakeword.audio.spectrograms import SpectrogramGeneration # Ensure this is imported

# These variables should be defined in cell_1.py and cell_4.py:
# AUGMENTED_FEATURES_DIR, clips, augmenter

print("\n--- Generating Augmented Features for Training/Validation/Testing ---")

if 'AUGMENTED_FEATURES_DIR' not in globals() or 'clips' not in globals() or 'augmenter' not in globals():
    print("ERROR: Essential variables (AUGMENTED_FEATURES_DIR, clips, augmenter) not defined. Ensure cell_1.py and cell_4.py ran correctly.")
else:
    AUGMENTED_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    splits_config = ["training", "validation", "testing"]
    for split_item in splits_config:
      out_dir_split_item = AUGMENTED_FEATURES_DIR / split_item # Use variable from cell_1
      out_dir_split_item.mkdir(parents=True, exist_ok=True)

      current_split_name = "train"
      current_repetition = 2

      # Ensure clips and augmenter are not None (i.e., cell_4 executed successfully)
      if clips is None or augmenter is None:
          print(f"Skipping feature generation for {split_item} due to missing clips or augmenter setup from cell_4.")
          continue

      current_spectrograms = SpectrogramGeneration(clips=clips,
                                         augmenter=augmenter,
                                         slide_frames=10,    # Uses the same spectrogram repeatedly, just shifted over by one frame. This simulates the streaming inferences while training/validating in nonstreaming mode.
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
                                         slide_frames=1,    # The testing set uses the streaming version of the model, so no artificial repetition is necessary
                                         step_ms=10,
                                         )

      print(f"Generating augmented features for {current_split_name} set into {out_dir_split_item}...")
      try:
        RaggedMmap.from_generator(
            out_dir=str(out_dir_split_item / 'wakeword_mmap'), # Ensure path is string for older mmap_ninja if needed
            sample_generator=current_spectrograms.spectrogram_generator(split=current_split_name, repeat=current_repetition),
            batch_size=100,
            verbose=True,
        )
        print(f"Finished generating features for {current_split_name} set.")
      except Exception as e:
        print(f"Error generating features for {current_split_name} set: {e}")


# In[ ]:


# CELL 7: Save a yaml config that controls the training process
# These hyperparamters can make a huge different in model quality.
# Experiment with sampling and penalty weights and increasing the number of
# training steps.
import yaml
import os
from pathlib import Path # Ensure Path is imported

# These variables should be defined in cell_1.py and available globally:
# AUGMENTED_FEATURES_DIR, NEGATIVE_DATASETS_PATH, TRAINED_MODELS_BASE_PATH, TRAINING_CONFIG_PATH

print("\n--- Preparing Training Configuration ---")

if 'AUGMENTED_FEATURES_DIR' not in globals() or \
   'NEGATIVE_DATASETS_PATH' not in globals() or \
   'TRAINED_MODELS_BASE_PATH' not in globals() or \
   'TRAINING_CONFIG_PATH' not in globals():
    print("ERROR: Essential path variables for training config are not defined. Ensure cell_1.py ran correctly.")
    # Optionally, raise an error or sys.exit()
    config_train = {} # Create empty config to avoid further errors if notebook continues
else:
    # Ensure the base training directory exists
    (TRAINED_MODELS_BASE_PATH / "wakeword").mkdir(parents=True, exist_ok=True)

    config_train = {
        "window_step_ms": 10,
        "train_dir": str(TRAINED_MODELS_BASE_PATH / "wakeword"), # Use variable
        "features": [
            {
                "features_dir": str(AUGMENTED_FEATURES_DIR), # Use variable
                "sampling_weight": 2.0,
                "penalty_weight": 1.0,
                "truth": True,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            },
            {
                "features_dir": str(NEGATIVE_DATASETS_PATH / "speech"), # Use variable
                "sampling_weight": 10.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": str(NEGATIVE_DATASETS_PATH / "dinner_party"), # Use variable
                "sampling_weight": 10.0,
                "penalty_weight": 1.0,
                "truth": False,
                "truncation_strategy": "random",
                "type": "mmap",
            },
            {
                "features_dir": str(NEGATIVE_DATASETS_PATH / "no_speech"), # Use variable
                "sampling_weight": 5.0,
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
        "training_steps": [10000],
        "positive_class_weight": [1],
        "negative_class_weight": [20],
        "learning_rates": [0.001],
        "batch_size": 128,
        "time_mask_max_size": [0],
        "time_mask_count": [0],
        "freq_mask_max_size": [0],
        "freq_mask_count": [0],
        "eval_step_interval": 500,
        "clip_duration_ms": 1500,
        "target_minimization": 0.9,
        "minimization_metric": None,
        "maximization_metric": "average_viable_recall"
    }

    with open(TRAINING_CONFIG_PATH, "w") as file_yaml: # Use variable
        yaml.dump(config_train, file_yaml)
    print(f"Training parameters saved to {TRAINING_CONFIG_PATH}")


# In[ ]:


# CELL 8: Trains a model. When finished, it will quantize and convert the model to a
# streaming version suitable for on-device detection.
# It will resume if stopped, but it will start over at the configured training
# steps in the yaml file.
# Change --train 0 to only convert and test the best-weighted model.

import os
import sys
from pathlib import Path # Ensure Path is imported

# TRAINING_CONFIG_PATH should be defined in cell_1.py and created in cell_7.py
# Example: TRAINING_CONFIG_PATH = DATA_DIR_INSIDE_DOCKER / "training_parameters.yaml"

print("\n--- Starting Model Training ---")

if 'TRAINING_CONFIG_PATH' not in globals() or not TRAINING_CONFIG_PATH.exists():
    print(f"ERROR: Training config path {TRAINING_CONFIG_PATH if 'TRAINING_CONFIG_PATH' in globals() else 'TRAINING_CONFIG_PATH variable not found'} not found or variable not defined. Ensure cell_1.py and cell_7.py ran correctly.")
    # In a real notebook, execution might stop here or raise an error.
    # For a .py script representing a cell, we'll just print the error.
else:
    print(f"Starting model training using config: {TRAINING_CONFIG_PATH}")

    # Construct the command string for the ! operator
    # Ensure TRAINING_CONFIG_PATH is correctly inserted into the string.
    # Using f-string for clarity if sys.executable is needed, or direct string construction.

    training_command = f"\"{sys.executable}\" -m microwakeword.model_train_eval \\
    --training_config='{str(TRAINING_CONFIG_PATH)}' \\
    --train 1 \\
    --restore_checkpoint 1 \\
    --test_tf_nonstreaming 0 \\
    --test_tflite_nonstreaming 0 \\
    --test_tflite_nonstreaming_quantized 0 \\
    --test_tflite_streaming 0 \\
    --test_tflite_streaming_quantized 1 \\
    --use_weights \"best_weights\" \\
    mixednet \\
    --pointwise_filters \"64,64,64,64\" \\
    --repeat_in_block  \"1,1,1,1\" \\
    --mixconv_kernel_sizes '[5],[7,11],[9,15],[23]' \\
    --residual_connection \"0,0,0,0\" \\
    --first_conv_filters 32 \\
    --first_conv_kernel_size 5 \\
    --stride 3"

    print(f"Executing training command:\n{training_command}")

    # This is how you'd represent the ! command in a .py file if get_ipython() is available
    # For direct .py execution where get_ipython is not available, this would fail.
    # The user will convert this .py cell back to .ipynb where get_ipython().system() works.
    if "get_ipython" in globals():
        get_ipython().system(training_command)
    else:
        print("Warning: get_ipython() not available. Cannot execute training command directly in this .py script.")
        print("This cell is intended to be converted back to an .ipynb cell.")
        # As a fallback for pure .py script testing (though not ideal for notebook structure):
        # import subprocess
        # subprocess.run(training_command, shell=True, check=True)


print("Model training/evaluation finished (or command printed if not in IPython).")


# In[ ]:


# CELL 9: Prepare model files for download/use
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

# These variables should be defined in cell_1.py:
# target_word, TRAINED_MODELS_BASE_PATH, DATA_DIR_INSIDE_DOCKER

print("\n--- Preparing Output Model Files ---")

if 'target_word' not in globals() or \
   'TRAINED_MODELS_BASE_PATH' not in globals() or \
   'DATA_DIR_INSIDE_DOCKER' not in globals():
    print("ERROR: Essential variables (target_word, TRAINED_MODELS_BASE_PATH, DATA_DIR_INSIDE_DOCKER) not defined. Ensure cell_1.py ran correctly.")
else:
    source_tflite_path = TRAINED_MODELS_BASE_PATH / "wakeword/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite"
    # Save final model files directly into the /data root for easy access from host
    destination_tflite_path = DATA_DIR_INSIDE_DOCKER / f"{target_word}_basic_model.tflite"
    destination_json_path = DATA_DIR_INSIDE_DOCKER / f"{target_word}_basic_model.json"

    if source_tflite_path.exists():
        shutil.copy(source_tflite_path, destination_tflite_path)
        print(f"Copied TFLite model to {destination_tflite_path}")
    else:
        print(f"ERROR: Trained TFLite model not found at {source_tflite_path}. Training might have failed or model path is incorrect.")

    json_data = {
        "type": "micro",
        "wake_word": target_word,  # Using the target_word variable from cell_1
        "author": "kiwina", # Updated author
        "website": "https://github.com/kiwina/MicroWakeWord-Trainer-Docker",
        "model": f"{target_word}_basic_model.tflite", # Relative path for use with ESPHome
        "trained_languages": ["en"],
        "version": 1, # Start version at 1 for a new model
        "micro": {
            "probability_cutoff": 0.97, # User should adjust based on testing
            "sliding_window_size": 5,
            "feature_step_size": 10,
            "tensor_arena_size": 30000, # User should adjust based on model needs
            "minimum_esphome_version": "2024.7.0"
        }
    }

    with open(destination_json_path, "w") as json_file:
        json.dump(json_data, json_file, indent=2)
    print(f"Created JSON metadata at {destination_json_path}")

    print("\n--- Script Finished (Basic Training Notebook) ---")
    print(f"Output files are located in: {DATA_DIR_INSIDE_DOCKER.resolve().absolute()}")
    print("If running in Docker, this corresponds to the 'microwakeword-trainer-data' directory on your host machine.")

    if destination_tflite_path.exists():
        print("\nTFLite Model Link (for Jupyter environments):")
        display(FileLink(str(destination_tflite_path)))
    if destination_json_path.exists():
        print("\nJSON Metadata Link (for Jupyter environments):")
        display(FileLink(str(destination_json_path)))

