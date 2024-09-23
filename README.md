# TensorFlow Installation Guide

This guide will help you install TensorFlow, an open-source machine learning framework, using Python.

## Prerequisites

-   Python 3.6-3.9
-   pip (Python package installer)

## Step 1: Create a Virtual Environment

It is recommended to install TensorFlow in a virtual environment to avoid conflicts with other packages.

# Install virtualenv if you don't have it
``` markdown
``` bash

pip install virtualenv

# Create a virtual environment named 'tensorflow_env'
virtualenv tensorflow_env

# Activate the virtual environment
# On Linux or macOS
source tensorflow_env/bin/activate

# On Windows
tensorflow_env\Scripts\activate
```
## Step 2: InstallTensorFlow
Once the virtual environment is activated, you can install TensorFlow.
``` bash
# Install the latest version of TensorFlow
pip install tensorflow
```
## Step 3: Verify the Installation
To verify that TensorFlow is installed correctly, you can run a simple Python script.

``` bash
import tensorflow as tf

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Create a simple constant tensor
hello = tf.constant('Hello, TensorFlow!')
print(hello.numpy())
```
Save the aboe script as
```bash
verify_tensorflow.py and run it:
python verify_tensorflow.py
```

Additional Resources
* [TensorFlow Official Documentation](https://www.tensorflow.org/learn)
* [TensorFlow GitHub Repository](https://www.github.com/tensorflow/tensorflow)



## Setting Up TensorFlow with GPU Support on Ubuntu
To set up TensorFlow with GPU support on Ubuntu, follow these steps:

``` bash
# Upgrade pip
pip install --upgrade pip

# Check for GPU devices
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install TensorFlow with GPU support
python3 -m pip install tensorflow[and-cuda]

# Verify TensorFlow installation with GPU support
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Install CUDA toolkit
sudo apt-get install nvcc

# Download and install CUDA repository pin
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Download and install CUDA repository
wget https://developer.download.nvidia.com/compute/cuda/12.6.1/local_installers/cuda-repo-wsl-ubuntu-12-6-local_12.6.1-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-6-local_12.6.1-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/

# Update package lists and install CUDA toolkit
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

# Verify TensorFlow installation with GPU support again
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Check NVIDIA driver installation
nvidia-smi
```
This will guide you through setting up TensorFlow with GPU support on Ubuntu.

## Run Script
``` bash
python3 tensorFlowChurnTelco.py
```