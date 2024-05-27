FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

COPY requirements.txt /root/requirements.txt
COPY hf_token.txt /root/hf_token.txt

RUN apt-key del 7fa2af80
RUN apt-key del 3bf863cc
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN apt-get update
RUN apt-get install git wget vim -y
RUN cd root
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN source miniconda3/bin/activate
RUN conda create -n py3.8 \\
    && conda create -n py3.8 python=3.8 -y \\
    && conda activate py3.8
RUN conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
RUN pip install -r requirements.txt
RUN huggingface-cli login --token $(cat hf_token.txt)
RUN wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.0/tars/TensorRT-10.0.0.6.Linux.x86_64-gnu.cuda-11.8.tar.gz \\
    && tar -xvzf TensorRT-10.0.0.6.Linux.x86_64-gnu.cuda-11.8.tar.gz \\
    && cd /root/TensorRT-10.0.0.6/python \\
    && python3 -m pip install tensorrt-*-cp38-none-linux_x86_64.whl \\
    && cd ../onnx_graphsurgeon/ \\
    && python3 -m pip install onnx_graphsurgeon-0.5.0-py2.py3-none-any.whl
RUN export CUDA_PATH=/usr/local/cuda
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64:/root/TensorRT-10.0.0.6/lib