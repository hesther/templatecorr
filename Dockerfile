FROM mambaorg/micromamba:1.4.7

USER root
# Keep the base environment activated
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN apt update && apt -y install git gcc g++ make

# Use micromamba to resolve conda-forge, much faster than conda
RUN micromamba install -y python=3.8.17 pip=23.2.1 rdkit=2020.09.5 -c conda-forge
RUN micromamba install -y numpy pandas joblib tqdm -c conda-forge
RUN micromamba install -y rdchiral_cpp=1.1.2 -c conda-forge

WORKDIR /app

COPY . /app