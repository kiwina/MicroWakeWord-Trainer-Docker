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

# Clone and install the kiwina/microWakeWord project from GitHub
RUN git clone https://github.com/kiwina/microWakeWord.git /app/microWakeWord
RUN pip install --no-cache-dir -e /app/microWakeWord

# piper-sample-generator will be cloned into /data by prepare_local_data.py, so no global install here.

# Create a data directory for external mapping and notebook execution
RUN mkdir -p /data
WORKDIR /data

# Copy the notebooks and prepare_local_data.py to a fallback location in the container (used by startup.sh)
# These .ipynb files should be generated from the .py versions and placed in the
# MicroWakeWord-Trainer-Docker directory before building the image.
# prepare_local_data.py is assumed to be at the root of the build context (c:/_ai/trash/train)
COPY basic_training_notebook.ipynb /root/basic_training_notebook.ipynb
COPY advanced_training_notebook.ipynb /root/advanced_training_notebook.ipynb
# Assuming prepare_local_data.py is at workspace root
COPY prepare_local_data.py /root/prepare_local_data.py
COPY advanced_training_notebook.py /root/advanced_training_notebook.py
COPY basic_training_notebook.py /root/basic_training_notebook.py
# Add the startup script from local file
COPY startup.sh /usr/local/bin/startup.sh
RUN chmod +x /usr/local/bin/startup.sh

# Expose the Jupyter Notebook port
EXPOSE 8888

# Run the startup script and start Jupyter Notebook
# Startup script will copy notebooks and prepare_local_data.py to /data if not present.
CMD ["/bin/bash", "-c", "/usr/local/bin/startup.sh && jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --notebook-dir=/data"]
