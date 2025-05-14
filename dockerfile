# Use TensorFlow 2.19.0 as the base image (includes Python 3.11, CUDA, CuDNN)
FROM tensorflow/tensorflow:2.19.0-gpu-jupyter

# Set environment variables for non-interactive installations and Python buffering
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install essential system dependencies that might not be in the base image.
# The base TF image is Ubuntu based.
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git unzip build-essential \
    libsndfile1 libffi-dev g++ cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies from local requirements.txt
# These are for the trainer environment itself.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy local microWakeWord project and install it
COPY ../microWakeWord /app/microWakeWord
RUN pip install --no-cache-dir -e /app/microWakeWord

# Copy local piper-sample-generator project
# It's used as scripts, but if it had its own installable package, you'd pip install -e it too.
# For now, we ensure its scripts are accessible. The notebooks will add it to sys.path if needed,
# or call scripts directly from /data/piper-sample-generator after prepare_local_data.py clones it there.
# However, to ensure its direct dependencies (like piper-phonemize) are met if not covered by main reqs:
COPY ../piper-sample-generator /app/piper-sample-generator
# If piper-sample-generator had its own setup.py for broader use:
# RUN pip install --no-cache-dir -e /app/piper-sample-generator
# For now, ensure its requirements (if any beyond torch/numpy which should be handled by trainer's env) are met.
# The piper-sample-generator/requirements.txt we edited mainly lists torch, numpy, audiomentations, etc.
# which should be covered by the main requirements.txt or base image.
# If it has unique small dependencies, list them in MicroWakeWord-Trainer-Docker/requirements.txt or install its reqs:
# RUN pip install --no-cache-dir -r /app/piper-sample-generator/requirements.txt

# Create a data directory for external mapping and notebook execution
RUN mkdir -p /data
WORKDIR /data

# Copy the notebooks to a fallback location in the container (used by startup.sh)
# These .ipynb files should be generated from the .py versions and placed in the
# MicroWakeWord-Trainer-Docker directory before building the image.
COPY basic_training_notebook.ipynb /root/basic_training_notebook.ipynb
COPY advanced_training_notebook.ipynb /root/advanced_training_notebook.ipynb
# COPY prepare_local_data.py /root/prepare_local_data.py # If startup.sh also copies this

# Add the startup script from local file
COPY startup.sh /usr/local/bin/startup.sh
RUN chmod +x /usr/local/bin/startup.sh

# Expose the Jupyter Notebook port
EXPOSE 8888

# Run the startup script and start Jupyter Notebook
# Startup script will copy notebooks and prepare_local_data.py to /data if not present.
CMD ["/bin/bash", "-c", "/usr/local/bin/startup.sh && jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --notebook-dir=/data"]
