# MicroWakeWord Trainer Docker

Easily train MicroWakeWord detection models with this Docker image. This setup uses Python 3.11 and is based on the `tensorflow/tensorflow:2.19.0-gpu-jupyter` image.

## Prerequisites

- Docker installed on your system.
- An NVIDIA GPU with CUDA support (optional but highly recommended for faster training).
- The `prepare_local_data.py` script should be present at the root of your workspace (e.g., alongside the `MicroWakeWord-Trainer-Docker` directory) before building the Docker image.

## Project Setup within Docker

- **`microWakeWord` library**: Cloned from `https://github.com/kiwina/microWakeWord.git` and installed in editable mode (`pip install -e .`) during the Docker image build.
- **`piper-sample-generator`**: Cloned from `https://github.com/kiwina/piper-sample-generator.git` into the `/data` volume by the `prepare_local_data.py` script when the notebooks are first run. The notebooks then use scripts from `/data/piper-sample-generator`.
- **`prepare_local_data.py`**: This script is copied into the Docker image and then into the `/data` volume by `startup.sh`. The training notebooks execute this script to download all necessary datasets and the `piper-sample-generator`.

## Quick Start

Follow these steps to get started with the microWakeWord Trainer:

### 1. Build the Docker Image Locally

Navigate to the `MicroWakeWord-Trainer-Docker` directory (this directory) in your terminal and run:

```bash
docker build -t kiwina/microwakeword-trainer .
```

_(You can replace `kiwina/microwakeword-trainer` with your preferred image name and tag.)_

### 2. Run the Docker Container

Start the container with a mapped volume for saving your data and expose the Jupyter Notebook port. Create a `microwakeword-trainer-data` directory if it doesn't exist, typically inside your `MicroWakeWord-Trainer-Docker` project directory or at your workspace root.

**Example (if `microwakeword-trainer-data` is inside `MicroWakeWord-Trainer-Docker` directory):**

```bash
# From within MicroWakeWord-Trainer-Docker directory:
mkdir -p microwakeword-trainer-data
docker run --rm -it \
    --gpus all \
    -p 8888:8888 \
    -v "$(pwd)/microwakeword-trainer-data:/data" \
    kiwina/microwakeword-trainer
```

**Example (if `microwakeword-trainer-data` is at the workspace root, sibling to `MicroWakeWord-Trainer-Docker`):**

```bash
# From within MicroWakeWord-Trainer-Docker directory:
# mkdir -p ../microwakeword-trainer-data # Create it if it doesn't exist
# docker run --rm -it \
#     --gpus all \
#     -p 8888:8888 \
#     -v "$(pwd)/../microwakeword-trainer-data:/data" \
#     kiwina/microwakeword-trainer
```

_Choose the volume mapping (`-v`) that matches your `microwakeword-trainer-data` directory location._

- `--gpus all`: Enables GPU acceleration (optional, remove if not using a GPU).
- `-p 8888:8888`: Exposes the Jupyter Notebook on port 8888.
- `-v ...:/data`: Maps your local `microwakeword-trainer-data` directory to the container's `/data` directory. This is where all generated samples, datasets, trained models, and notebooks will be stored, ensuring persistence.

### 3. Access Jupyter Notebook

Open your web browser and navigate to:

```bash
http://localhost:8888
```

(The current Dockerfile CMD disables the token, so one might not be required.)

The notebook interface should appear, showing the contents of the `/data` directory.

### 4. Prepare Notebooks & Data (if first time)

The `startup.sh` script within the container will automatically copy:

- `basic_training_notebook.ipynb`
- `advanced_training_notebook.ipynb`
- `prepare_local_data.py`
  from the image's `/root/` directory to `/data/` if they are not already present.

You should generate the `.ipynb` files from their corresponding `.py` versions (which were modified in this project) and ensure they, along with `prepare_local_data.py` (from your workspace root), are correctly placed for the Docker `COPY` commands during the build.

When you first run a notebook (`.py` or `.ipynb` version), its initial cells will execute `/data/prepare_local_data.py`. This script downloads and prepares all necessary datasets (Piper TTS model, augmentation audio, negative features) and clones `piper-sample-generator` into subdirectories within `/data`. This process is idempotent.

### 5. Edit the Wake Word

Open either `basic_training_notebook.py` or `advanced_training_notebook.py` (or their `.ipynb` versions) from the Jupyter interface (or run the `.py` scripts via a terminal in the container). Locate the line defining `target_word` and change it:

```python
target_word = 'your_wake_word_here'  # Phonetic spellings may produce better samples
```

### 6. Run the Notebook/Script

Execute the cells or run the Python script. The process will:

- Utilize the data prepared by `prepare_local_data.py`.
- Generate wake word samples into `/data/generated_samples/{target_word}/...`.
- Train a detection model.
- Output a quantized `.tflite` model and a corresponding JSON manifest file into the `/data` directory.

### 7. Retrieve the Trained Model and JSON

The trained `.tflite` model and `.json` manifest will be available in your mapped local `microwakeword-trainer-data` directory.

## Resetting to a Clean State

If you need to start fresh with datasets or generated content:

- **Delete specific subfolders** within your local `microwakeword-trainer-data` directory (e.g., `generated_samples`, `mit_rirs`, `piper-sample-generator`, `trained_models_adv`). The `prepare_local_data.py` script will then re-download/re-process only the missing parts when a notebook is run.
- To reset the notebooks or `prepare_local_data.py` script themselves, delete them from your local `microwakeword-trainer-data` directory. When you restart the Docker container, `startup.sh` will copy fresh versions from the image.

## Credits

This project builds upon the excellent work of [kahrendt/microWakeWord](https://github.com/kahrendt/microWakeWord).
This version is maintained in the [kiwina forks](https://github.com/kiwina) (e.g., `kiwina/MicroWakeWord-Trainer-Docker`, `kiwina/microWakeWord`).
