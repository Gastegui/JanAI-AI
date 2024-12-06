# This is a guide for setting up the environment to work with TensorFlow in Windows using WSL

### If you don't have an Nvidia GPU, don't bother doing all this

#### There are some blocks of commands that you can run at the same time, but it's better if you run them all one by one as some of them require being modified and some others prompt questions

1. Download Ubuntu 20.04LTS version from Microsoft Store

1. Download the latest drivers for your GPU from [here](https://www.nvidia.com/en-us/drivers/) or from the GeForce application

1. Search Ubuntu in the Windows search bar and finish the installation

1. Download the WSL extension in VS Code

1. Open the Ubuntu machine from VS Code using the extension

1. Run the next commands on the terminal (make sure the terminal is the Ubuntu WSL):

    ```
    sudo apt update && sudo apt upgrade -y

    mkdir ~/Downloads

    cd ~/Downloads

    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin 

    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

    wget https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.1-470.57.02-1_amd64.deb

    sudo dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.1-470.57.02-1_amd64.deb

    sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub

    sudo apt-get update

    sudo apt-get -y install cuda
    ```
1. Run "nvidia-smi" and check if it displays info about the graphics card

1. Check the Cuda version displayed on the top right corner and remember that

    1. If it says 11, run:
        ```
        wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.6.0.74_cuda11-archive.tar.xz
        ```
    1. If it says 12, run:
        ```
        wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.6.0.74_cuda12-archive.tar.xz
        ```
        If it says something else, you'r coocked

1. If you downloaded something, run the next commands:
    ```
    tar -xvf cudnn-(PRESS TAB)

    ls -l (CHECK FOR THE NAME OF THE UNCOMPRESSED FOLDER, CUDNN-something)

    sudo cp (NAME OF THE CUDNN- FOLDER)/include/cudnn*.h /usr/local/cuda/include

    sudo cp (NAME OF THE CUDNN- FOLDER)/lib/libcudnn* /usr/local/cuda/lib64

    sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc

    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

    source ~/.bashrc
    ```
1. Cuda drivers should be installed. To check, run this command (should display the version installed, if not, you'r also coocked)
    ```
    nvcc --version
    ```
1. Run the next commands to install Python and create the environmnent
    ```
    sudo apt install python

    cd ~/Downloads

    sudo apt install curl

    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh

    bash Miniconda3-latest-Linux-x86_64.sh
    ```

    When prompted, press ENTER and then write YES
    ```
    sudo reboot
    ```
    Wait until it reboots (maybe you need to restart VS Code)
    ```
    conda create --name tf python=3.9

    conda activate tf

    conda install -c conda-forge cudatoolkit=11.8.0

    pip install nvidia-cudnn-cu11==8.6.0.163

    mkdir -p $CONDA_PREFIX/etc/conda/activate.d

    echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

    echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

    pip install --upgrade pip

    python3 -m install tensorflow[and-cuda]

    pip install TensorRT
    ```
1. Hopefully, tensorflow got installed. Run the next commands to check (don't worry about the warnings):
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