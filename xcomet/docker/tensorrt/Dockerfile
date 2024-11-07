FROM nvcr.io/nvidia/pytorch:23.07-py3
COPY requirements.txt /workspace/requirements.txt
COPY hf_token.txt /workspace/hf_token.txt

RUN huggingface-cli login --token $(cat /workspace/hf_token.txt)