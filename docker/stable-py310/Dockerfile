#!/bin/bash

# Cloned from official stable version @2023-06-15 (https://github.com/gradient-ai/base-container/blob/main/pt112-tf29-jax0314-py39/Dockerfile)

# Paperspace Dockerfile for Gradient base image


# [Note]
# Colab versions @2023-05-06
# Python:                 3.10.11
# NumPy:      numpy       1.22.4
# PyTorch:    torch       2.0.0+cu118
# Torchaudio: torchaudio  2.0.1+cu118


# ==================================================================
# Initial setup
# ------------------------------------------------------------------

    # Ubuntu 20.04 as base image
    FROM ubuntu:20.04
    RUN yes| unminimize

    # Set ENV variables
    ENV LANG C.UTF-8
    ENV SHELL=/bin/bash
    ENV DEBIAN_FRONTEND=noninteractive

    ENV APT_INSTALL="apt-get install -y --no-install-recommends"
    ENV PIP_INSTALL="python3 -m pip --no-cache-dir install --upgrade"
    ENV GIT_CLONE="git clone --depth 10"


# ==================================================================
# Tools
# ------------------------------------------------------------------

    RUN apt-get update && \
        $APT_INSTALL \
        apt-utils \
        gcc \
        make \
        pkg-config \
        apt-transport-https \
        build-essential \
        ca-certificates \
        wget \
        rsync \
        git \
        vim \
        mlocate \
        libssl-dev \
        curl \
        openssh-client \
        unzip \
        unrar \
        zip \
        csvkit \
        emacs \
        joe \
        jq \
        dialog \
        man-db \
        manpages \
        manpages-dev \
        manpages-posix \
        manpages-posix-dev \
        nano \
        iputils-ping \
        sudo \
        ffmpeg \
        libsm6 \
        libxext6 \
        libboost-all-dev \
        cifs-utils \
        software-properties-common


# ==================================================================
# Python: 3.10
# ------------------------------------------------------------------

    #Based on https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa

    # Adding repository for python3.10
    RUN add-apt-repository ppa:deadsnakes/ppa -y && \

    # Installing python3.10
        $APT_INSTALL \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3-distutils-extra

    # Add symlink so python and python3 commands use same python3.10 executable
    RUN ln -s /usr/bin/python3.10 /usr/local/bin/python3 && \
        ln -s /usr/bin/python3.10 /usr/local/bin/python

    # Installing pip
    RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
    ENV PATH=$PATH:/root/.local/bin


# ==================================================================
# Installing CUDA packages (CUDA Toolkit 11.6.2 & CUDNN 8.4.1)
#
# Note:
#     In Gradient Notebook, paperspace provide/configure the underlying machines and users specify containers on the machines.
#     This means that 
#         - CUDA driver is given by paperspace
#         - Corresponding CUDA toolkit and cuDNN should be properly configured by users for custom runtime's container.
#     Bacause there is no way to know exact host environment on paperspace machines, we should follow official `gradient-base` image for this part.
# ------------------------------------------------------------------
    # Based on https://developer.nvidia.com/cuda-toolkit-archive
    # Based on https://developer.nvidia.com/rdp/cudnn-archive
    # Based on https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#package-manager-ubuntu-install

    # Installing CUDA Toolkit
    RUN wget https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run && \
        bash cuda_11.6.2_510.47.03_linux.run --silent --toolkit && \
        rm cuda_11.6.2_510.47.03_linux.run
    ENV PATH=$PATH:/usr/local/cuda-11.6/bin
    ENV LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64

    # Installing CUDNN
    RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
        mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
        apt-get install dirmngr -y && \
        apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
        add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" && \
        apt-get update && \
        apt-get install libcudnn8=8.4.1.*-1+cuda11.6 -y && \
        apt-get install libcudnn8-dev=8.4.1.*-1+cuda11.6 -y && \
        rm /etc/apt/preferences.d/cuda-repository-pin-600


# ==================================================================
# Python packages
# ------------------------------------------------------------------

    # Based on https://pytorch.org/get-started/locally/
    # Based on https://www.tensorflow.org/install/pip
    # Based on https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html#pip

    RUN $PIP_INSTALL torch==2.0.1 torchaudio==2.0.2 --extra-index-url https://download.pytorch.org/whl/cu116 && \
        $PIP_INSTALL \
            tensorflow==2.9.2 \
            jupyterlab==3.4.6 \
            cython==0.29.32 \
            ipython==8.5.0 \
            ipykernel==6.16.0 \
            ipywidgets==8.0.2 \
            jupyterlab-snippets==0.4.1 \
            gradient==2.0.6 \
            numpy==1.23.4 \
            scipy==1.9.2 \
            gdown==4.5.1 \
            boto3==1.24.90 \
            awscli==1.25.91


# ==================================================================
# Node.js and Jupyter Notebook Extensions
# ------------------------------------------------------------------

    RUN curl -sL https://deb.nodesource.com/setup_16.x | bash  && \
        $APT_INSTALL nodejs  && \
        $PIP_INSTALL jupyter_contrib_nbextensions jupyterlab-git && \
        jupyter contrib nbextension install --user
                

# ==================================================================
# Startup
# ------------------------------------------------------------------

    EXPOSE 8888 6006

    CMD jupyter lab --allow-root --ip=0.0.0.0 --no-browser --ServerApp.trust_xheaders=True --ServerApp.disable_check_xsrf=False --ServerApp.allow_remote_access=True --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True