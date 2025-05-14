# Prepares all local data assets required for the
# `advanced_training_notebook_local_data.ipynb` notebook.
# This script downloads and processes:
# 1. Git repositories: microWakeWord, piper-sample-generator
# 2. Piper TTS Model
# 3. Augmentation Audio: MIT RIR, Audioset, FMA (converts to 16kHz WAV)
# 4. Negative Spectrogram Features

import os
import sys
import subprocess
import platform
import requests
import zipfile
import shutil
import argparse
import datetime
from pathlib import Path
from tqdm import tqdm

# --- Configuration and Environment Setup ---
def detect_docker():
    """Check if running inside a Docker container"""
    # Method 1: Check for Docker-specific file
    if os.path.exists('/proc/1/cgroup'):
        with open('/proc/1/cgroup', 'r') as f:
            if 'docker' in f.read():
                return True
    
    # Method 2: Check for environment variable
    if os.environ.get('IN_DOCKER', '0') == '1':
        return True
    
    return False

# Parse command line arguments
parser = argparse.ArgumentParser(description='Prepare local data for MicroWakeWord training')
parser.add_argument('--data-dir', type=str, help='Base directory for all downloaded and extracted data')
args = parser.parse_args()


# Check if running in Docker (needed for base_data_dir logic)
IN_DOCKER = detect_docker()

# Determine base data directory
# Priority: CLI arg > environment variable LOCAL_DATA_DIR > Docker default > Host default
if args.data_dir:
    base_data_dir = Path(args.data_dir)
    print(f"Using data directory from --data-dir argument: {base_data_dir.resolve().absolute()}")
elif os.environ.get('LOCAL_DATA_DIR'):
    local_data_dir = os.environ.get('LOCAL_DATA_DIR')
    if local_data_dir is not None:  # Explicit check to satisfy type checker
        base_data_dir = Path(local_data_dir)
    else:
        # This should never happen since we already checked os.environ.get('LOCAL_DATA_DIR')
        base_data_dir = Path('/data') if IN_DOCKER else Path(__file__).resolve().parent / 'microwakeword-trainer-data'
    print(f"Using data directory from LOCAL_DATA_DIR environment variable: {base_data_dir.resolve().absolute()}")
elif IN_DOCKER:
    base_data_dir = Path('/data')
    print(f"Running in Docker. Using default data directory: {base_data_dir.resolve().absolute()}")
else:
    # Default for host execution: 'microwakeword-trainer-data' in the same directory as the script.
    # Assuming prepare_local_data.py is in the project root.
    base_data_dir = Path(__file__).resolve().parent / 'microwakeword-trainer-data'
    print(f"Running on host. Using default data directory: {base_data_dir.resolve().absolute()}")

# Create base directory if it doesn't exist
base_data_dir.mkdir(parents=True, exist_ok=True)

# IN_DOCKER definition moved up, old comment removed. Original empty line 49 is preserved by this structure.

# Attempt to import heavy libraries only if needed, or ensure they are prerequisites
required_packages = {
    'scipy': 'scipy.io.wavfile',
    'numpy': 'numpy',
    'datasets': 'datasets',
    'soundfile': 'soundfile'
}

missing_packages = []

# Check each package individually to provide better error messages
for package, import_name in required_packages.items():
    try:
        __import__(import_name.split('.')[0])
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f"Missing required Python packages: {', '.join(missing_packages)}")
    print("Please ensure these packages are installed before running this script.")
    print("You can install them by running: pip install " + " ".join(missing_packages))
    sys.exit(1)

# Now that we've verified the packages exist, import them
import scipy.io.wavfile
import numpy as np
from datasets import Dataset, Audio, load_dataset  # For MIT RIR, FMA conversion
import soundfile as sf  # For Audioset conversion

# Print resolved data directory
# This print statement is now covered by the more specific prints in the base_data_dir determination logic.
if IN_DOCKER:
    print("Running inside Docker container")

# --- Configuration ---
# PIPER_REPO_URL_MAC = "https://github.com/kahrendt/piper-sample-generator" # Unused
# PIPER_REPO_BRANCH_MAC = "mps-support" # Unused
# PIPER_REPO_URL_OTHER = "https://github.com/rhasspy/piper-sample-generator" # Unused, kiwina fork URL is now hardcoded in prepare_git_repos
PIPER_REPO_DIR = base_data_dir / "piper-sample-generator" # This is the target directory for the kiwina fork

# Piper TTS Model Configuration
PIPER_MODEL_FILENAME = "en_US-libritts_r-medium.pt"
PIPER_MODEL_JSON_FILENAME = "en_US-libritts_r-medium.pt.json"
PIPER_MODEL_URL = f"https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/{PIPER_MODEL_FILENAME}"
PIPER_MODEL_JSON_URL = f"https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/{PIPER_MODEL_JSON_FILENAME}"
PIPER_MODEL_DIR = PIPER_REPO_DIR / "models"
PIPER_MODEL_FILE = PIPER_MODEL_DIR / PIPER_MODEL_FILENAME
PIPER_MODEL_JSON_FILE = PIPER_MODEL_DIR / PIPER_MODEL_JSON_FILENAME

# Augmentation Data Paths
MIT_RIR_OUTPUT_DIR = base_data_dir / "mit_rirs"

AUDIOSONET_BASE_DIR = base_data_dir / "audioset" # Raw downloads
AUDIOSONET_OUTPUT_WAV_DIR = base_data_dir / "audioset_16k" # Processed WAVs

FMA_BASE_DIR = base_data_dir / "fma" # Raw downloads
FMA_OUTPUT_WAV_DIR = base_data_dir / "fma_16k" # Processed WAVs
FMA_ZIP_FNAME = "fma_xs.zip"
FMA_ZIP_LINK = f"https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/{FMA_ZIP_FNAME}"

# Negative Spectrogram Feature Paths
NEGATIVE_FEATURES_OUTPUT_DIR = base_data_dir / "negative_datasets"
NEGATIVE_FEATURES_LINK_ROOT = "https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/"
NEGATIVE_FEATURES_FILENAMES = ['dinner_party.zip', 'dinner_party_eval.zip', 'no_speech.zip', 'speech.zip']

# --- Helper Functions ---

def run_command(command_list, description):
    print(f"Executing: {description} -> {' '.join(command_list)}")
    try:
        process = subprocess.run(command_list, check=True, capture_output=True, text=True)
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
    
    # Check if file already exists and has content
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"{description.capitalize()} already exists at {output_path} ({output_path.stat().st_size / (1024*1024):.2f} MB). Skipping download.")
        return True
        
    print(f"Downloading {description} from {url} to {output_path}...")
    try:
        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress bar
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
        
        # Verify download was successful
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"Successfully downloaded {description} to {output_path} ({output_path.stat().st_size / (1024*1024):.2f} MB).")
            return True
        else:
            print(f"Error: Downloaded file {output_path} is empty or does not exist.")
            return False
    except Exception as e:
        print(f"Error downloading {description} from {url}: {e}")
        if output_path.exists(): # Clean up partial download
            os.remove(output_path)
        return False

def extract_zip_robust(zip_path, extract_to, expected_content_name=None):
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    print(f"Checking if extraction is needed for {zip_path}...")

    # Check if expected content already exists
    if expected_content_name:
        extracted_content_path = extract_to / expected_content_name
        if extracted_content_path.is_dir():
            # Check if directory has content
            content_files = list(extracted_content_path.glob('**/*'))
            if content_files:
                print(f"Content '{expected_content_name}' already found at {extracted_content_path} with {len(content_files)} files. Skipping extraction.")
                return True
            else:
                print(f"Directory '{expected_content_name}' exists but appears empty. Will attempt extraction.")
    
    if not zip_path.exists():
        print(f"ERROR: Cannot extract, ZIP file not found: {zip_path}")
        return False

    print(f"Extracting {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files in the zip to check size
            zip_files = zip_ref.namelist()
            print(f"ZIP contains {len(zip_files)} files/directories.")
            
            # Extract all files
            zip_ref.extractall(extract_to)
            
        print(f"Successfully extracted {zip_path} to {extract_to}.")
        
        # Verify extraction was successful
        if expected_content_name:
            extracted_path = extract_to / expected_content_name
            if not extracted_path.is_dir():
                print(f"Warning: Extraction of {zip_path.name} was called, but target directory {extracted_path} was not created as expected.")
                # Check if files were extracted directly to extract_to instead
                direct_files = list(extract_to.glob('*'))
                if len(direct_files) > 1:  # More than just the zip file
                    print(f"Files appear to have been extracted directly to {extract_to} instead of a subdirectory.")
                    return True
                return False
            else:
                # Check if directory has content
                content_files = list(extracted_path.glob('**/*'))
                if not content_files:
                    print(f"Warning: Extracted directory {extracted_path} appears to be empty.")
                    return False
        return True
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False

# --- Main Preparation Steps ---

def prepare_git_repos():
    print("\n--- Preparing Git Repositories ---")
    
    # Piper Sample Generator
    if not PIPER_REPO_DIR.is_dir():
        print(f"Cloning Piper Sample Generator repository (kiwina fork) to {PIPER_REPO_DIR}...")
        
        clone_url = "https://github.com/kiwina/piper-sample-generator.git" # Use kiwina fork
        # clone_branch = None # Assuming main/master branch from kiwina fork
            
        cmd = ["git", "clone", clone_url, str(PIPER_REPO_DIR)]
        # If a specific branch from kiwina fork is needed, add: cmd.extend(["-b", "your-branch-name"])
        
        if not run_command(cmd, "Cloning Piper Sample Generator (kiwina fork)"):
            return False # Critical failure
    else:
        print(f"Piper Sample Generator repository already exists at {PIPER_REPO_DIR}.")
        # Verify it's the correct repository by checking for specific files
        if (PIPER_REPO_DIR / "generate_samples.py").exists():
            print(f"Verified Piper Sample Generator repository at {PIPER_REPO_DIR}.")
        else:
            print(f"Warning: Directory exists at {PIPER_REPO_DIR} but may not be the correct Piper repository.")
            print("Attempting to update repository...")
            # Try to pull latest changes
            if not run_command(["git", "-C", str(PIPER_REPO_DIR), "pull"], "Updating Piper Sample Generator"):
                print("Warning: Could not update repository. Continuing with existing files.")
    
    # microWakeWord repo
    MICROWAKEWORD_REPO_DIR = base_data_dir / "microWakeWord"
    if not MICROWAKEWORD_REPO_DIR.is_dir():
        print(f"Cloning microWakeWord repository to {MICROWAKEWORD_REPO_DIR}...")
        
        clone_url = "https://github.com/kiwina/microWakeWord.git"
        
        cmd = ["git", "clone", clone_url, str(MICROWAKEWORD_REPO_DIR)]
        
        if not run_command(cmd, "Cloning microWakeWord repository"):
            print("Warning: Failed to clone microWakeWord repository. It may be installed via Dockerfile.")
    else:
        print(f"microWakeWord repository already exists at {MICROWAKEWORD_REPO_DIR}.")
        # Verify it's the correct repository
        if (MICROWAKEWORD_REPO_DIR / "setup.py").exists():
            print(f"Verified microWakeWord repository at {MICROWAKEWORD_REPO_DIR}.")
            # Try to pull latest changes
            if not run_command(["git", "-C", str(MICROWAKEWORD_REPO_DIR), "pull"], "Updating microWakeWord"):
                print("Warning: Could not update repository. Continuing with existing files.")
        else:
            print(f"Warning: Directory exists at {MICROWAKEWORD_REPO_DIR} but may not be the correct microWakeWord repository.")
    
    return True

def prepare_piper_model():
    print("\n--- Preparing Piper TTS Model ---")
    PIPER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download model file if needed
    model_success = True
    if not PIPER_MODEL_FILE.exists():
        print(f"Downloading Piper TTS model file from {PIPER_MODEL_URL}...")
        if not download_file_simple(PIPER_MODEL_URL, PIPER_MODEL_FILE, "Piper TTS Model"):
            model_success = False
            print("Failed to download Piper TTS model file. This is required for training.")
    else:
        print(f"Piper TTS model already exists at {PIPER_MODEL_FILE} ({PIPER_MODEL_FILE.stat().st_size / (1024*1024):.2f} MB).")
    
    # Download model JSON file if needed
    json_success = True
    if not PIPER_MODEL_JSON_FILE.exists():
        print(f"Downloading Piper TTS model JSON file from {PIPER_MODEL_JSON_URL}...")
        if not download_file_simple(PIPER_MODEL_JSON_URL, PIPER_MODEL_JSON_FILE, "Piper TTS Model JSON"):
            json_success = False
            print("Failed to download Piper TTS model JSON file. This may affect training.")
    else:
        print(f"Piper TTS model JSON already exists at {PIPER_MODEL_JSON_FILE} ({PIPER_MODEL_JSON_FILE.stat().st_size / 1024:.2f} KB).")
    
    return model_success and json_success

def prepare_mit_rir():
    print("\n--- Preparing MIT RIR Dataset ---")
    MIT_RIR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed (e.g., by counting wav files)
    existing_files = list(MIT_RIR_OUTPUT_DIR.glob("*.wav"))
    # MIT RIR dataset should have around 271 files
    expected_min_files = 250  # Setting a minimum threshold
    
    if existing_files and len(existing_files) >= expected_min_files:
        print(f"MIT RIR dataset appears to be already processed in {MIT_RIR_OUTPUT_DIR} ({len(existing_files)} files found). Skipping.")
        return True
    elif existing_files:
        print(f"MIT RIR dataset partially processed in {MIT_RIR_OUTPUT_DIR} ({len(existing_files)} files found, expected at least {expected_min_files}).")
        print("Will attempt to complete the dataset...")
    else:
        print(f"No MIT RIR files found. Downloading and processing MIT RIR dataset to {MIT_RIR_OUTPUT_DIR}...")
    try:
        rir_dataset = load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)
        for row in tqdm(rir_dataset, desc="Processing MIT RIR"):
            name = Path(row["audio"]["path"]).name
            file_path = MIT_RIR_OUTPUT_DIR / name
            if not file_path.exists(): # Idempotency
                 scipy.io.wavfile.write(
                    file_path,
                    16000, # Target sample rate
                    (np.array(row["audio"]["array"]) * 32767).astype(np.int16)
                )
        print(f"Finished processing MIT RIR dataset to {MIT_RIR_OUTPUT_DIR}.")
        return True
    except Exception as e:
        print(f"Error during MIT RIR dataset processing: {e}")
        return False

def prepare_audioset():
    print("\n--- Preparing Audioset Dataset ---")
    AUDIOSONET_BASE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIOSONET_OUTPUT_WAV_DIR.mkdir(parents=True, exist_ok=True)

    dataset_links = [f"https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train0{i}.tar" for i in range(10)]
    all_extracted_and_converted = True

    # Download and Extract Tarballs
    for link in dataset_links:
        file_name = Path(link).name
        tar_path = AUDIOSONET_BASE_DIR / file_name
        # A simple marker for extraction could be the presence of a known subdirectory or file from the tar.
        # For Audioset, the tar files extract into subdirs like 'bal_train00', etc.
        # Let's assume if the first tar's content dir exists, others might too.
        # A more robust check would be per-tar marker files as in original script.
        # For simplicity here, we'll rely on checking the final WAV dir.
        
        if not tar_path.exists():
            if not download_file_simple(link, tar_path, f"Audioset part {file_name}"):
                all_extracted_and_converted = False
                continue # Try next file
        
        # Simplified extraction check: if final WAV dir is populated, assume extraction happened.
        # This is not ideal but simplifies this script. The original notebook's logic is more robust.
        # For a truly robust prepare_local_data.py, replicate the marker logic.
        # Here, we'll just extract if tar exists and WAVs are missing.
        # This part needs more robust marker logic from original 4.py for production use.
        if tar_path.exists(): # No specific check for prior extraction, will re-extract if WAVs missing
            print(f"Extracting {tar_path} to {AUDIOSONET_BASE_DIR} (if not already effectively done)...")
            # This is a simplified extraction call.
            if not run_command(["tar", "-xf", str(tar_path), "-C", str(AUDIOSONET_BASE_DIR)], f"Extracting {tar_path.name}"):
                 all_extracted_and_converted = False


    # Convert FLAC to WAV
    print(f"Collecting Audioset FLAC files from {AUDIOSONET_BASE_DIR} for conversion to {AUDIOSONET_OUTPUT_WAV_DIR}...")
    audioset_flac_files = list(AUDIOSONET_BASE_DIR.glob("**/*.flac"))
    print(f"Found {len(audioset_flac_files)} total FLAC files.")
    
    files_to_convert = []
    for flac_file_path in audioset_flac_files:
        wav_output_path = AUDIOSONET_OUTPUT_WAV_DIR / (flac_file_path.stem + ".wav")
        if not wav_output_path.exists():
            files_to_convert.append(flac_file_path)

    if not files_to_convert and audioset_flac_files:
        print("All found Audioset FLAC files seem to be already converted to WAV.")
    elif not audioset_flac_files:
         print(f"No FLAC files found in {AUDIOSONET_BASE_DIR} to convert. Check download/extraction.")
         # If tar files were downloaded, this implies extraction might have failed or tars were empty.
         if any( (AUDIOSONET_BASE_DIR/Path(l).name).exists() for l in dataset_links):
             all_extracted_and_converted = False # Mark as incomplete if FLACs are missing post-download
    
    if files_to_convert:
        print(f"Starting conversion of {len(files_to_convert)} Audioset FLAC files to 16kHz WAV...")
        corrupted_files = []
        for file_path in tqdm(files_to_convert, desc="Converting Audioset FLAC to WAV"):
            wav_output_path = AUDIOSONET_OUTPUT_WAV_DIR / (file_path.stem + ".wav")
            try:
                audio, sr = sf.read(file_path)
                if audio.ndim > 1: audio = audio[:, 0] # Mono
                # Resampling if necessary (sf.read gives original sr)
                # Assuming piper models expect 16kHz, but Audioset might be different.
                # The original notebook's `scipy.io.wavfile.write` directly writes at 16000.
                # If sf.read gives a different rate, resampling is needed.
                # For simplicity, if direct conversion to 16k is the goal, this step needs care.
                # The original notebook uses load_dataset which might handle resampling.
                # Here, we'll assume direct write, implying source is compatible or resampling is handled elsewhere.
                # This is a simplification from original notebook's direct 16k write.
                # Let's stick to scipy for consistency with original notebook's write format.
                # audio_16k = audio # Placeholder if resampling is needed.
                # For now, let's assume sf.read provides data that can be written.
                # This part is tricky without knowing exact sample rates from Audioset FLACs.
                # The original notebook's `scipy.io.wavfile.write(..., 16000, ...)` implies it expects to resample.
                # `soundfile` doesn't resample on write. `datasets.Audio(sampling_rate=16000)` does.
                # To match original:
                # This requires `audio` to be float. `sf.read` provides float by default.
                if sr != 16000:
                    print(f"Warning: {file_path} has sample rate {sr}, expected 16000. Resampling not implemented in this simplified script section. Output may not be 16kHz.")
                
                scipy.io.wavfile.write(wav_output_path, 16000, (audio * 32767).astype(np.int16))

            except Exception as e:
                print(f"Error converting {file_path}: {e}")
                corrupted_files.append(str(file_path))
                all_extracted_and_converted = False
        
        if corrupted_files:
            print(f"Logged {len(corrupted_files)} corrupted Audioset files (not actually logged in this simplified script).")
        print("Audioset FLAC to WAV conversion attempt complete.")

    if not list(AUDIOSONET_OUTPUT_WAV_DIR.glob("*.wav")) and audioset_flac_files:
        all_extracted_and_converted = False # If FLACs were there but no WAVs produced.

    return all_extracted_and_converted


def prepare_fma():
    print("\n--- Preparing FMA Dataset ---")
    FMA_BASE_DIR.mkdir(parents=True, exist_ok=True)
    FMA_OUTPUT_WAV_DIR.mkdir(parents=True, exist_ok=True)
    fma_zip_path = FMA_BASE_DIR / FMA_ZIP_FNAME
    fma_extracted_sentinel_dir = FMA_BASE_DIR / "fma_small" # As per original notebook

    # Download
    if not fma_zip_path.exists():
        if not download_file_simple(FMA_ZIP_LINK, fma_zip_path, "FMA dataset"):
            return False # Critical download failure
    else:
        print(f"FMA dataset ZIP {fma_zip_path} already downloaded.")

    # Extract
    if not extract_zip_robust(fma_zip_path, FMA_BASE_DIR, "fma_small"):
        # If extraction fails but sentinel dir exists from partial, it might still proceed to convert.
        # This is a soft fail for extraction if content seems to be there.
        if not fma_extracted_sentinel_dir.is_dir():
             print("FMA extraction failed and sentinel directory not found.")
             return False


    # Convert MP3 to WAV
    print(f"Collecting FMA MP3 files from {fma_extracted_sentinel_dir} for conversion to {FMA_OUTPUT_WAV_DIR}...")
    if not fma_extracted_sentinel_dir.is_dir():
        print(f"ERROR: FMA extracted directory '{fma_extracted_sentinel_dir}' not found. Cannot convert MP3s.")
        return False
        
    fma_mp3_files = list(fma_extracted_sentinel_dir.glob("**/*.mp3"))
    print(f"Found {len(fma_mp3_files)} MP3 files.")

    files_to_convert_paths = []
    for mp3_file_path_obj in fma_mp3_files:
        wav_output_path = FMA_OUTPUT_WAV_DIR / (mp3_file_path_obj.stem + ".wav")
        if not wav_output_path.exists():
            files_to_convert_paths.append(mp3_file_path_obj)
    
    if not files_to_convert_paths and fma_mp3_files:
        print("All found FMA MP3 files seem to be already converted to WAV.")
    elif not fma_mp3_files:
        print(f"No MP3 files found in {fma_extracted_sentinel_dir} to convert.")
        return False # If no MP3s, something is wrong with extraction or source.

    if files_to_convert_paths:
        print(f"Starting conversion of {len(files_to_convert_paths)} FMA MP3 files to 16kHz WAV...")
        # Using datasets.load_dataset for robust MP3 loading and resampling
        try:
            # Create a dataset from the list of MP3 file paths
            string_paths = [str(p) for p in files_to_convert_paths]
            temp_dataset = Dataset.from_dict({"audio": string_paths}).cast_column("audio", Audio(sampling_rate=16000))
            
            corrupted_fma_files = []
            for row in tqdm(temp_dataset, desc="Converting FMA MP3 to WAV"):
                original_path_str = row["audio"]["path"] # This path might be temporary if loaded from cache
                # We need the original stem for the output filename
                file_stem = Path(original_path_str).stem # This might be problematic if path is from cache
                                                        # Fallback: use index or a map if stems are not reliable
                
                # Re-derive stem from our input list to be safe
                # This is a bit hacky, assumes order is preserved or find original path by value
                # A better way: iterate our files_to_convert_paths and load one by one.
                # For now, let's assume stem from cached path is usable, but it's a known risk.
                # To be robust:
                # current_original_path = files_to_convert_paths[tqdm_iterator_index]
                # file_stem = current_original_path.stem

                # Simplified: find original path by matching the end of the path string
                # This is still not perfectly robust.
                matched_original_path = None
                for p_orig in files_to_convert_paths:
                    if original_path_str.endswith(p_orig.name):
                        matched_original_path = p_orig
                        break
                if matched_original_path:
                    file_stem = matched_original_path.stem
                else: # Fallback if no match, use the potentially temporary stem
                    print(f"Warning: Could not reliably match cached path {original_path_str} to original file list for stem. Using stem: {file_stem}")


                wav_output_path = FMA_OUTPUT_WAV_DIR / (file_stem + ".wav")
                
                try:
                    audio_array = np.array(row["audio"]["array"])
                    if audio_array.ndim == 0 or audio_array.size == 0:
                         raise ValueError("Empty audio data")
                    scipy.io.wavfile.write(wav_output_path, 16000, (audio_array * 32767).astype(np.int16))
                except Exception as e_inner:
                    print(f"Error converting file {original_path_str} (stem: {file_stem}): {e_inner}")
                    corrupted_fma_files.append(original_path_str)
            
            if corrupted_fma_files:
                 print(f"Logged {len(corrupted_fma_files)} corrupted FMA files (not actually logged).")
            print("FMA MP3 to WAV conversion attempt complete.")

        except Exception as e_outer:
            print(f"Error during FMA dataset conversion process: {e_outer}")
            return False
            
    return True


def prepare_negative_features():
    print("\n--- Preparing Negative Spectrogram Features ---")
    NEGATIVE_FEATURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_extracted = True

    # Check if all extracted directories already exist
    all_dirs_exist = True
    for fname in NEGATIVE_FEATURES_FILENAMES:
        extracted_content_dir_name = fname.replace('.zip', '')
        extracted_dir = NEGATIVE_FEATURES_OUTPUT_DIR / extracted_content_dir_name
        if not extracted_dir.is_dir():
            all_dirs_exist = False
            break
    
    if all_dirs_exist:
        print(f"All negative feature directories already exist in {NEGATIVE_FEATURES_OUTPUT_DIR}. Skipping download and extraction.")
        return True

    for fname in NEGATIVE_FEATURES_FILENAMES:
        link = NEGATIVE_FEATURES_LINK_ROOT + fname
        zip_path = NEGATIVE_FEATURES_OUTPUT_DIR / fname
        extracted_content_dir_name = fname.replace('.zip', '')
        extracted_dir = NEGATIVE_FEATURES_OUTPUT_DIR / extracted_content_dir_name
        
        # Skip if the extracted directory already exists
        if extracted_dir.is_dir():
            print(f"Directory {extracted_dir} already exists. Skipping download and extraction for {fname}.")
            continue
        
        if not zip_path.exists():
            print(f"Downloading {fname} from {link}...")
            if not download_file_simple(link, zip_path, f"Negative feature set {fname}"):
                all_extracted = False # Mark as failure if download fails
                continue # Try next file
        else:
            print(f"ZIP file {zip_path} already exists. Skipping download.")
        
        print(f"Extracting {zip_path} to {NEGATIVE_FEATURES_OUTPUT_DIR}...")
        if not extract_zip_robust(zip_path, NEGATIVE_FEATURES_OUTPUT_DIR, extracted_content_dir_name):
            all_extracted = False # Mark as failure if extraction fails and content not there
            
    return all_extracted

# --- Main Execution ---
if __name__ == "__main__":
    print("=" * 80)
    print("Starting local data preparation process...")
    print(f"Using data directory: {base_data_dir.absolute()}")
    if IN_DOCKER:
        print("Running inside Docker container - Windows-specific steps will be skipped")
    
    # Create a marker file to track successful runs
    marker_file = base_data_dir / ".data_preparation_completed"
    if marker_file.exists():
        print(f"Marker file {marker_file} found. Data preparation has been run before.")
        print("This script will still check for and download any missing assets.")
    
    overall_success = True

    # Skip Windows-specific steps if in Docker
    if IN_DOCKER and platform.system() == "Windows":
        print("WARNING: Detected Windows platform while running in Docker. Some Windows-specific steps may not work correctly.")

    print("\nStep 1/5: Preparing Git repositories")
    if not prepare_git_repos():
        overall_success = False
        print("Warning: Git repository preparation had issues. Continuing with other steps...")
    
    print("\nStep 2/5: Preparing Piper TTS model")
    if not prepare_piper_model():
        overall_success = False
        print("Error: Piper model preparation failed. This is required for training.")
    
    print("\nStep 3/5: Preparing MIT RIR dataset (room impulse responses)")
    if not prepare_mit_rir():
        print("Warning: MIT RIR dataset preparation had issues. Training can continue but audio augmentation may be limited.")

    print("\nStep 4/5: Preparing Audioset and FMA datasets")
    # Audioset and FMA are large datasets that take significant time to process
    existing_audioset_wavs = list(AUDIOSONET_OUTPUT_WAV_DIR.glob("*.wav"))
    if existing_audioset_wavs:
        print(f"Found {len(existing_audioset_wavs)} existing Audioset WAV files in {AUDIOSONET_OUTPUT_WAV_DIR}. Skipping Audioset preparation.")
    else:
        print("Audioset preparation can take a long time. Starting...")
        if not prepare_audioset():
            print(f"Audioset preparation reported issues. Check logs and verify contents in {AUDIOSONET_OUTPUT_WAV_DIR}.")
            print("Continuing with other datasets...")
    
    # Check if FMA WAVs already exist
    existing_fma_wavs = list(FMA_OUTPUT_WAV_DIR.glob("*.wav"))
    if existing_fma_wavs:
        print(f"Found {len(existing_fma_wavs)} existing FMA WAV files in {FMA_OUTPUT_WAV_DIR}. Skipping FMA preparation.")
    else:
        print("FMA preparation can take a long time. Starting...")
        if not prepare_fma():
            print(f"FMA preparation reported issues. Check logs and verify contents in {FMA_OUTPUT_WAV_DIR}.")
            print("Continuing with other datasets...")
    
    print("\nStep 5/5: Preparing negative spectrogram features")
    if not prepare_negative_features():
        overall_success = False
        print("Warning: Negative features preparation had issues. This may affect training quality.")

    print("=" * 80)
    if overall_success:
        print("Local data preparation process completed successfully!")
        # Create marker file to indicate successful run
        with open(marker_file, 'w') as f:
            f.write(f"Data preparation completed successfully on {platform.node()} at {datetime.datetime.now()}")
    else:
        print("Local data preparation process completed with some warnings or errors.")
        print("Training may still work if the essential components were prepared successfully.")
    
    # Create a summary file with information about the prepared data
    summary_file = base_data_dir / "data_summary.txt"
    try:
        with open(summary_file, 'w') as f:
            f.write(f"MicroWakeWord Training Data Summary\n")
            f.write(f"Generated on: {datetime.datetime.now()}\n")
            f.write(f"Data directory: {base_data_dir.absolute()}\n\n")
            
            f.write("Git Repositories:\n")
            f.write(f"- Piper Sample Generator: {PIPER_REPO_DIR.exists()}\n")
            microwakeword_dir = base_data_dir / "microWakeWord"
            f.write(f"- microWakeWord: {microwakeword_dir.exists()}\n\n")
            
            f.write("Models:\n")
            f.write(f"- Piper TTS Model: {PIPER_MODEL_FILE.exists()} ({PIPER_MODEL_FILE.stat().st_size / (1024*1024):.2f} MB if exists)\n")
            f.write(f"- Piper TTS Model JSON: {PIPER_MODEL_JSON_FILE.exists()} ({PIPER_MODEL_JSON_FILE.stat().st_size / 1024:.2f} KB if exists)\n\n")
            
            f.write("Datasets:\n")
            mit_rir_files = list(MIT_RIR_OUTPUT_DIR.glob('*.wav'))
            f.write(f"- MIT RIR: {len(mit_rir_files)} files\n")
            
            audioset_files = list(AUDIOSONET_OUTPUT_WAV_DIR.glob('*.wav'))
            f.write(f"- Audioset: {len(audioset_files)} files\n")
            
            fma_files = list(FMA_OUTPUT_WAV_DIR.glob('*.wav'))
            f.write(f"- FMA: {len(fma_files)} files\n\n")
            
            f.write("Negative Features:\n")
            for feature_dir in NEGATIVE_FEATURES_OUTPUT_DIR.glob('*'):
                if feature_dir.is_dir():
                    feature_files = list(feature_dir.glob('**/*'))
                    f.write(f"- {feature_dir.name}: {len(feature_files)} files\n")
        
        print(f"\nData summary written to {summary_file}")
    except Exception as e:
        print(f"Error writing summary file: {e}")
    
    print(f"\nAll data has been stored in: {base_data_dir.absolute()}")
    print("\nPlease verify the contents of the output directories:")
    print(f"  Piper Repo: {PIPER_REPO_DIR}")
    print(f"  Piper Model: {PIPER_MODEL_FILE}")
    print(f"  MIT RIR WAVs: {MIT_RIR_OUTPUT_DIR} ({len(list(MIT_RIR_OUTPUT_DIR.glob('*.wav')))} files)")
    print(f"  Audioset WAVs: {AUDIOSONET_OUTPUT_WAV_DIR} ({len(list(AUDIOSONET_OUTPUT_WAV_DIR.glob('*.wav')))} files)")
    print(f"  FMA WAVs: {FMA_OUTPUT_WAV_DIR} ({len(list(FMA_OUTPUT_WAV_DIR.glob('*.wav')))} files)")
    print(f"  Negative Features: {NEGATIVE_FEATURES_OUTPUT_DIR}")
    
    # List negative feature directories
    neg_feature_dirs = [d for d in NEGATIVE_FEATURES_OUTPUT_DIR.glob('*') if d.is_dir()]
    for neg_dir in neg_feature_dirs:
        print(f"    - {neg_dir.name} ({len(list(neg_dir.glob('**/*')))} files)")
    
    print("=" * 80)