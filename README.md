# MicroWakeWord Trainer Docker

Easily train MicroWakeWord detection models with this Docker image. This setup uses Python 3.11 and is based on the `tensorflow/tensorflow:2.19.0-gpu-jupyter` image.

## Prerequisites

- Docker installed on your system.
- An NVIDIA GPU with CUDA support (optional but highly recommended for faster training).
- Local clones of the `kiwina/microWakeWord` and `kiwina/piper-sample-generator` repositories in the parent directory relative to this `MicroWakeWord-Trainer-Docker` project folder. The Docker build process will copy these local projects into the image.

## Quick Start

Follow these steps to get started with the microWakeWord Trainer:

### 1. Build the Docker Image Locally

Navigate to the `MicroWakeWord-Trainer-Docker` directory (this directory) in your terminal and run:

```bash
docker build -t kiwina/microwakeword-trainer .
```

_(You can replace `kiwina/microwakeword-trainer` with your preferred image name and tag.)_

### 2. Run the Docker Container

Start the container with a mapped volume for saving your data and expose the Jupyter Notebook port:

```bash
docker run --rm -it \
    --gpus all \
    -p 8888:8888 \
    -v $(pwd)/../microwakeword-trainer-data:/data \
    kiwina/microwakeword-trainer
```

_(Note: `$(pwd)/../microwakeword-trainer-data` assumes `microwakeword-trainer-data` is in the parent directory of `MicroWakeWord-Trainer-Docker`. Adjust the path if your `microwakeword-trainer-data` directory is located elsewhere.)_

- `--gpus all`: Enables GPU acceleration (optional, remove if not using a GPU).
- `-p 8888:8888`: Exposes the Jupyter Notebook on port 8888.
- `-v ...:/data`: Maps a local directory (e.g., `microwakeword-trainer-data`) to the container's `/data` directory. This is where all generated samples, datasets, trained models, and notebooks will be stored, ensuring persistence.

### 3. Access Jupyter Notebook

Open your web browser and navigate to:

```bash
http://localhost:8888
```

(A token might be required if you don't disable it in the CMD of the Dockerfile; the current Dockerfile disables the token.)

The notebook interface should appear, showing the contents of the `/data` directory.

### 4. Prepare Notebooks (if first time)

The `startup.sh` script within the container will automatically copy the `basic_training_notebook.ipynb` and `advanced_training_notebook.ipynb` from the image's `/root/` directory to `/data/` if they are not already present. You should generate these `.ipynb` files from their corresponding `.py` versions (which were modified in this project) and ensure they are present in the `MicroWakeWord-Trainer-Docker` directory before building the Docker image so they are correctly copied.

### 5. Edit the Wake Word

Open either `basic_training_notebook.ipynb` or `advanced_training_notebook.ipynb` from the Jupyter interface. Locate the cell defining `target_word` and change it to your desired wake word:

```python
target_word = 'your_wake_word_here'  # Phonetic spellings may produce better samples
```

### 6. Run the Notebook

Run all cells in the chosen notebook. The process will:

- **Prepare Data**: The initial cells in the notebook now handle the download and preparation of all necessary datasets (Piper TTS model, augmentation audio, negative features) directly into subdirectories within `/data`. This process is idempotent; if data already exists, it will be skipped.
- Generate wake word samples.
- Train a detection model.
- Output a quantized `.tflite` model and a corresponding JSON manifest file into the `/data` directory (specifically, within subfolders like `/data/trained_models/wakeword/` or directly in `/data` for the final output).

### 7. Retrieve the Trained Model and JSON

Once the training is complete, the quantized `.tflite` model and `.json` manifest will be available in your mapped local directory (e.g., `microwakeword-trainer-data`).

## Resetting to a Clean State

If you need to start fresh with datasets or generated content:

- **Delete specific subfolders** within your local `microwakeword-trainer-data` directory (e.g., `generated_samples_ww`, `mit_rirs`, `trained_models`). The data preparation logic in the notebooks will then re-download/re-process only the missing parts.
- To reset the notebooks themselves, delete them from your local `microwakeword-trainer-data` directory. When you restart the Docker container, the `startup.sh` script will copy fresh versions from the image.

## Credits

This project builds upon the excellent work of [kahrendt/microWakeWord](https://github.com/kahrendt/microWakeWord). A huge thank you to the original authors for their contributions to the open-source community!
This version is maintained in the [kiwina forks](https://github.com/kiwina).
