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
### Step 3: Verify the Installation
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

