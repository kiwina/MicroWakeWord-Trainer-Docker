#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Function to log messages with timestamps
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to copy a file with error handling
copy_file() {
    local source="$1"
    local destination="$2"
    local description="$3"
    
    if [ ! -f "$destination" ]; then
        log_message "$description not found in /data. Copying the default file..."
        if [ -f "$source" ]; then
            cp "$source" "$destination"
            if [ $? -eq 0 ]; then
                log_message "Successfully copied $description to $destination"
                # Make Python scripts executable
                if [[ "$destination" == *.py ]]; then
                    chmod +x "$destination"
                    log_message "Made $destination executable"
                fi
                return 0
            else
                log_message "ERROR: Failed to copy $description to $destination"
                return 1
            fi
        else
            log_message "ERROR: Source file $source not found in the image. Cannot copy $description."
            return 1
        fi
    else
        log_message "$description already exists in $destination. Skipping copy."
        return 0
    fi
}

# Verify Python 3.11 is available
PYTHON_VERSION=$(python3 --version 2>&1)
log_message "Using $PYTHON_VERSION"
if [[ "$PYTHON_VERSION" != *"Python 3.11"* ]]; then
    log_message "WARNING: Expected Python 3.11, but found $PYTHON_VERSION. Some functionality may not work correctly."
fi

# Create data directory if it doesn't exist
if [ ! -d /data ]; then
    log_message "Creating /data directory..."
    mkdir -p /data
fi

# Copy notebook files
copy_file "/root/basic_training_notebook_docker.ipynb" "/data/basic_training_notebook_docker.ipynb" "Basic training notebook"
copy_file "/root/advanced_training_notebook_docker.ipynb" "/data/advanced_training_notebook_docker.ipynb" "Advanced training notebook"

# Copy prepare_local_data.py script
copy_file "/root/prepare_local_data.py" "/data/prepare_local_data.py" "prepare_local_data.py script"

# Verify all required files are present
MISSING_FILES=0
for file in "/data/basic_training_notebook_docker.ipynb" "/data/advanced_training_notebook_docker.ipynb" "/data/prepare_local_data.py"; do
    if [ ! -f "$file" ]; then
        log_message "ERROR: Required file $file is missing. Training may not work correctly."
        MISSING_FILES=1
    fi
done

if [ $MISSING_FILES -eq 1 ]; then
    log_message "WARNING: Some required files are missing. Please check the logs above."
else
    log_message "All required files are present in /data directory."
fi

# Continue with the command passed to the container
log_message "Startup script completed. Executing: $@"
exec "$@"
