# efficient-llm-metrics
Efficient &amp; Open LLM-based metrics for NLG tasks


## xcomet

setup
```
conda create -n metric python=3.11.5 pandas=2.1.1 pytorch=2.1.1 torchvision=0.16.1 pytorch-cuda=11.8 scipy=1.11.3 -c pytorch -c nvidia

conda activate metric

# to use GPTQ quantization
pip install auto-gptq==0.5.1
pip install optimum==1.14.1 accelerate==0.24.1
pip install --upgrade git+https://github.com/huggingface/transformers.git

# to access xCOMET models
pip install --upgrade pip
pip install "unbabel-comet>=2.2.0"
huggingface-cli login               # will have to enter your huggingface access token

# To train distilled models
pip install lightning==2.1.2

# Visualization
pip install jupyterlab==4.0.9
```

note: it gives 
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
datasets 2.15.0 requires huggingface-hub>=0.18.0, but you have huggingface-hub 0.16.4 which is incompatible.
```
However, it didn't affect anything so far.