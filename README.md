# Tutorial 

## Setup

```bash
git clone https://github.com/simonucl/tutorial.git
cd tutorial

bash setup-vast.sh

# Run inference
CUDA_VISIBLE_DEVICES=0 python3 inference/example_hf.py
CUDA_VISIBLE_DEVICES=1 python3 inference/example_vllm.py
```
