FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

COPY requirements.txt /root/requirements.txt
COPY hf_token.txt /root/hf_token.txt

RUN apt-key del 7fa2af80 \
    && apt-key del 3bf863cc \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN apt-get update
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install git wget vim pkg-config cmake -y
RUN cd /root
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN /opt/miniconda3/bin/conda create -n py3.8 python=3.8 -y
ENV PATH /opt/miniconda3/envs/py3.8/bin:$PATH
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r /root/requirements.txt
RUN huggingface-cli login --token $(cat /root/hf_token.txt)
RUN wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.0/tars/TensorRT-10.0.0.6.Linux.x86_64-gnu.cuda-11.8.tar.gz \
    && tar -xvzf TensorRT-10.0.0.6.Linux.x86_64-gnu.cuda-11.8.tar.gz -C /root
ENV CUDA_PATH /usr/local/cuda
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64:/root/TensorRT-10.0.0.6/lib
RUN cd /root/TensorRT-10.0.0.6/python \
    && python -m pip install tensorrt-*-cp38-none-linux_x86_64.whl \
    && cd ../onnx_graphsurgeon/ \
    && python -m pip install onnx_graphsurgeon-0.5.0-py2.py3-none-any.whl