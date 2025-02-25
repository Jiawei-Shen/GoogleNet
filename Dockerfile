# Base image with Miniconda and CUDA support
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
SHELL ["/bin/bash", "-c"]

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# Create and activate the Conda environment
RUN conda create -n googlenet_env python=3.8 -y && \
    conda init && \
    echo "conda activate googlenet_env" >> ~/.bashrc

# Set default shell
SHELL ["conda", "run", "-n", "googlenet_env", "/bin/bash", "-c"]

# Install Python dependencies
COPY environment.yml /workspace/environment.yml
RUN conda env update --name googlenet_env --file /workspace/environment.yml && conda clean --all -y

# Set workspace as working directory
WORKDIR /workspace

# Copy project files into the container
COPY . .

# Ensure scripts are executable
RUN chmod +x scripts/*.py

# Install the project as an editable package
RUN pip install -e .

# Set entry point for the container
ENTRYPOINT ["/bin/bash"]
