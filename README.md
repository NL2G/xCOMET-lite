# efficient-llm-metrics
Efficient &amp; Open LLM-based metrics for NLG tasks


## xcomet

setup
```
conda create -n metric python numpy pandas pytorch torchvision pytorch-cuda=11.8 
scipy -c pytorch -c nvidia
conda activate metric

# to use GPTQ quantization
pip install auto-gptq
pip install --upgrade optimum
pip install --upgrade git+https://github.com/huggingface/transformers.git
pip install --upgrade accelerate

# to access xCOMET models
pip install --upgrade pip
pip install "unbabel-comet>=2.2.0"
huggingface-cli login               # will have to enter your huggingface access token
```