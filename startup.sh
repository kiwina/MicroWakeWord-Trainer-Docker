#!/bin/bash

# Check if basic training notebook exists in /data
if [ ! -f /data/basic_training_notebook_docker.ipynb ]; then
    echo "Basic training notebook not found in /data. Copying the default notebook..."
    cp /root/basic_training_notebook_docker.ipynb /data/basic_training_notebook_docker.ipynb
else
    echo "Basic training notebook already exists in /data. Skipping copy."
fi

# Check if advanced training notebook exists in /data
if [ ! -f /data/advanced_training_notebook_docker.ipynb ]; then
    echo "Advanced training notebook not found in /data. Copying the default notebook..."
    cp /root/advanced_training_notebook_docker.ipynb /data/advanced_training_notebook_docker.ipynb
else
    echo "Advanced training notebook already exists in /data. Skipping copy."
fi

# Check if prepare_local_data.py exists in /data
if [ ! -f /data/prepare_local_data.py ]; then
    echo "prepare_local_data.py not found in /data. Copying the default script..."
    if [ -f /root/prepare_local_data.py ]; then
        cp /root/prepare_local_data.py /data/prepare_local_data.py
        chmod +x /data/prepare_local_data.py # Ensure it's executable
    else
        echo "ERROR: /root/prepare_local_data.py not found in the image. Cannot copy."
    fi
else
    echo "prepare_local_data.py already exists in /data. Skipping copy."
fi

exec "$@"
