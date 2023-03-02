## What is this soft about?

Using Large Language Models (starting with Bloom-176b and Bloomz-176b) slowly, but on commodity GPU-free personal computer.

There are 3 different pieces of code in this repo:
- The main code (in code/) loads Bloom's layers one by one to perform inference on CPU-only 16GB-RAM personal computer
- Another version of this code is an adaptation of [Arteaga's blog](https://nbviewer.org/urls/arteagac.github.io/blog/bloom_local.ipynb) that fixes some bugs with the latest version of transformers.
- Another code exploits collaborative inference, where each node hosts a few layers and communication is achieved through web sockets.

I compared this main code (for local inference with Bloom on CPU and 16GB RAM) with two alternatives (see at the end): the accelerate library and
Arteaga's code, and this main code is the only one that achieves inference in less than 15' on my personal computer.

In this README, I only talk about the main code.

The principle is very simple: load the layers in RAM one by one, and process all the data through one layer,
the pass to the next layer, and so on until the top layer and next word prediction.
A similar behaviour may be obtained with [offloading and the accelerate library](https://huggingface.co/docs/accelerate/usage_guides/big_modeling),
but this code is more specialized as it has been designed from the ground up for this specific use case
and a given model.

## Requirements

- desktop or laptop with at least 16GB of RAM
- harddrive with 400GB free to store Bloom's parameters (the fastest drive the better; NVMe SSD welcome)
- that's it! no GPU is needed

## How to use

- download the Bloom-176b weights [from Huggingface Hub](https://huggingface.co/bigscience/bloom). You may also reuse or adapt the script downloadBloomz.sh available in this repo in code/ for linux.
- download or clone this github repo
- install in a conda/pip environment: pytorch + transformers + accelerate + datasets + promptsource
- write in the text file code/sentences.txt one sentence per line
- cd code; python slowLLM.py
    - by default, this code makes a forward pass on every sentence in "sentences.txt", but there are a few other scripts, e.g., to evalute Bloomz on the BoolQ dataset.

## Limitations

- **This version does not really support real text generation**, which is too slow with autoregressive models;
may be in the future with text diffusion models, but not now.
- So it is limited for now to compute likelihood and generate 1, 2 or a few tokens maximum, such as for answering yes/no questions: many tasks can be framed as yes/no questions,
but this may require some creative thinking.
- Pipeline parallelism is limited to about 50 input sentences maximum, if you want to stay within 16GB of RAM

## Detailed speed and requirements

- RAM: 16GB 

The speed you may get greatly depends on your hardware.
For instance, on a very very slow network drive, I got:
- passing data through 1 layer: 0.5s (I tested with 16x Intel(R) Xeon(R) CPU E5-2609 v4 @ 1.70GHz)
- loading weights of 1 layer: 55s (with a very slow NAS drive, you should get much better speed)
- There are 70 layers, so in my case, processing 1 input requires 70 minutes;

You can gain a lot of speed by putting the model's parameters onto an NVMe SSD disk.
For instance, the forward pass on a single sentence (13 tokens) with slowLLM, using less than 16GB of RAM, no gpu, with NVMe SSD (Micron/Crucial Technology P2 NVMe PCIe SSD (rev 01)) and cpu= AMD Ryzen 5 3600 6-Core Processor:
- total time = 791s; 

## FAQ, TODO and bugs to fix

- The current approach to save every layer output to disk does not scale beyond 50 examples; a better usage of
the RAM should be realized to process more examples.
- Adapt the code to perform prompt tuning; this is rather straightforward.
- Although it'd be extremely slow, nothing prevent this approach to perform generation; this option shall be at least enabled.

## Benchmarks

- forward pass on a single sentence (13 tokens) with slowLLM, using less than 16GB of RAM, no gpu, with NVMe SSD (Micron/Crucial Technology P2 NVMe PCIe SSD (rev 01)) and cpu= AMD Ryzen 5 3600 6-Core Processor
	- total time = 791s; 

- baseline: [vanilla accelerate recipe](https://huggingface.co/blog/bloom-inference-pytorch-scripts) on the same computer:
note: I'm not using the safetensors version of bloomz here, so accelerate initially copies the weights into the offload dir, which roughly 
accounts for an initial delay of. Another difference in favor of accelerate is that the computer I'm using actually has a Titan X 12GB GPU
as well as 24GB of RAM: accelerate is exploiting the additional RAM, while slowLLM figures are computed with <16GB of RAM and no GPU. However, we tried to let accelerate also exploit the GPU, but it crashed with cuda out-of-memory error.
	- total time = *crashed (cuda oom with gpu; kernel killed without gpu)*

- code derived from Arteaga's blog: this code achieves the same process than mine, but reimplements the forward pass while I'm hacking into the Bloom module. The advantage of hacking into the Bloom module is that I can still benefit from the advanced functionalities empowering the transformers library, such as beam-search generation.
	- total time = 4200s (because by reimplementing the forward pass, a single CPU core is working at a time)


