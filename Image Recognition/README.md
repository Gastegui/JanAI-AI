# This is a guide for setting up the environment to work with PyTorch in Windows using WSL

### If you don't have an Nvidia GPU, don't bother doing all this

#### There are some blocks of commands that you can run at the same time, but it's better if you run them all one by one as some of them require being modified and some others prompt questions

1. Download Ubuntu 20.04LTS version from Microsoft Store

1. Download the latest drivers for your GPU from [here](https://www.nvidia.com/en-us/drivers/) or from the GeForce application

1. Search Ubuntu in the Windows search bar and finish the installation

1. Download the WSL extension in VS Code

1. Open the Ubuntu machine from VS Code using the extension

1. Download all the recomended extensions (at least should ask for Python and Jupyter)

1. Run the next commands in the terminal to create the Python enviroment and instaling PyTorch (make sure you are running the commands in WSL)
    ```
    sudo apt update && sudo apt upgrade -y

    mkdir ~/Downloads

    cd ~/Downloads

    sudo apt install curl

    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh

    bash Miniconda3-latest-Linux-x86_64.sh
    ```

    When prompted, press enter then scroll all the way down and input 'yes', then press enter without writing anything, and the input 'yes' again
    ```
    sudo reboot
    ```
    Wait until it reboots (maybe you need to restart VS Code)

    The terminal prompt should start with '(base)' now. If so, Conda is properly installed
    ```
    conda create --name pt python=3.9

    conda activate pd

    pip install --upgrade pip

    pip install matplotlib

    pip install tqdm
    ```
    Now the conda environment is created and almost finished, PyTorch is missing. Run the next command in windows PowerShell (if it doesn't work, try Admin PowerShell)
    ```
    nvidia-smi
    ```
    Check the top right corner for the CUDA version and run the next commands in WSL again (make sure you are in the Conda 'pd' environment)
    
    1. If it says 11.8:
        ```
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
        ```
    1. If it says 12.1:
        ```
        conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
        ```
    1. If it says 12.4 or later (12.7 for example):
        ```
        conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
        ```
1. PyTorch should be installed. Run the next commands to check (don't worry about the warnings):
    ```
    python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
    ```
    Should have said something like: tf.Tensor(1025.14, shape=(), dtype=float32)
    ```
    python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ```
    Should have said something like: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

1. At VS Code, when selecting the kernel for the Python Notebook, select "tf (Python 3.9.x)

1. Enjoy!