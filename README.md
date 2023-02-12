Using Large Language Models (starting with Bloomz) slowly, on commodity GPU-free desktops.

## Use cases

- No way to do text generation, this would be far too slow with autoregressive models;
may be in the future  with text diffusion models, but definitely not now.
- So there are 2 main use cases:
    - generate text embeddings for further processing
    - answering binary yes/no question-answering tasks

This may seem very limited (and it is, yes), but there are many tasks that can be framed as a
yes/no question and could thus benefit from the power of LLM.

## Speed and requirements

- RAM: 25GB (but it should be possible to reduce it to 16GB)
- passing data through 1 layer: 0.5s (I tested with 16x Intel(R) Xeon(R) CPU E5-2609 v4 @ 1.70GHz)
- loading weights of 1 layer: 55s (I tested with a very slow NAS drive, you should get much better speed)
- There are 70 layers, so in my case, processing 1 input requires 70 minutes;
You can gain a lot of speed by putting the model's parameters onto an NVMe SSD disk.
Also, remember to put as much questions as possible in the input file so that they all pass into the first
layer before the second layer weights are loaded.

