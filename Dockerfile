FROM continuumio/miniconda3:latest
WORKDIR /workspace
COPY environment.yml .
RUN conda env create -f environment.yml
ENV PATH /opt/conda/envs/batter-env/bin:$PATH

