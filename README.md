## What is this soft about?

Using Large Language Models (starting with Bloom-176b and Bloomz-176b) slowly, but on commodity GPU-free personal computer.

There are 3 different pieces of code to achieve that in this repo:
- The main code (in code/) loads Bloom's layers one by one to perform inference on CPU-only 16GB-RAM personal computer.
There is also a development branch for training (requires for now 25GB of RAM and roughly doubles the time of inference).
- Another version of this code is an adaptation of [Arteaga's blog](https://nbviewer.org/urls/arteagac.github.io/blog/bloom_local.ipynb) that fixes some bugs with the latest version of transformers.
- Another code exploits collaborative inference, where each node hosts a few layers and communication is achieved through web sockets.

I compared my main code (for local inference with Bloom on CPU and 16GB RAM) with both alternatives (see at the end): the accelerate library and
Arteaga's code, and this main code is the only one that achieves inference in less than 15' on my personal computer.
It is also the only code to perform training in less than 30' on the same computer.

The principle is very simple: load the layers in RAM one by one, and process all the data through one layer,
then pass the activation to the next layer, and so on until the top layer and next word prediction.
Training may then proceed by going backward through the same process.
A similar behaviour may be obtained with [offloading and the accelerate library](https://huggingface.co/docs/accelerate/usage_guides/big_modeling),
but my code is more specialized as it has been designed from the ground up for this specific use case
and a given model.

## Requirements

- desktop or laptop with at least 16GB (for inference) or 25GB (for training) of RAM
- harddrive with 400GB free to store Bloom's parameters (the fastest drive the better; NVMe SSD welcome)
- that's it! no GPU is needed

## How to use

- download the Bloom-176b weights [from Huggingface Hub](https://huggingface.co/bigscience/bloom). You may also reuse or adapt the script downloadBloomz.sh available in this repo in code/ for linux.
- download or clone this github repo
- install in a conda/pip environment: pytorch + transformers + accelerate + datasets + promptsource
- write in the text file code/sentences.txt one yes/no question per line
- cd code; python slowLLM.py
    - by default, this code now evaluates Bloomz on the BoolQ dataset; but there's another simpler function 'run_test_0()' that you can adapt to run bloomz on your own examples

Example of yes/no questions it may answer:
```
John: "We need so much fossil energy to make a plane fly". Mary: "Do the birds can fly?" Is this question a rhetorical question, yes or no?
Mary: "you know when your machine learning experiments give very good results, and you fix a bug, and then nothing works any more?" Is Mary's question a rhetorical question, yes or no?  
```

## Limitations

- **This version does not support real text generation**, this would be far too slow with autoregressive models;
may be in the future with text diffusion models, but not now.
- So it is limited for now to only answer yes/no questions: many tasks can be framed as yes/no questions,
but this may require some creative thinking.
- Answering 10 questions with Bloom takes about the same time as answering one question:
this time ranges between 7' to 70', depending on the speed of your hard drive.
So it's best to write all questions at once before calling the program.

## Detailed speed and requirements

- RAM: 25GB (but it should be possible to reduce it to 16GB)
- passing data through 1 layer: 0.5s (I tested with 16x Intel(R) Xeon(R) CPU E5-2609 v4 @ 1.70GHz)
- loading weights of 1 layer: 55s (I tested with a very slow NAS drive, you should get much better speed)
- There are 70 layers, so in my case, processing 1 input requires 70 minutes;
You can gain a lot of speed by putting the model's parameters onto an NVMe SSD disk.
Also, remember to put as much questions as possible in the input file so that they all pass into the first
layer before the second layer weights are loaded.

## FAQ, TODO and bugs to fix

- The current approach to save every layer output to disk does not scale beyond 50 examples; a better usage of
the RAM should be realized to process more examples.
- Although it'd be extremely slow, nothing prevent this approach to perform generation; this option shall be at least enabled.

## Benchmarks

- forward pass on a single sentence (13 tokens) with slowLLM, using less than 16GB of RAM, no gpu, with NVMe sdd (Micron/Crucial Technology P2 NVMe PCIe SSD (rev 01)) and cpu= AMD Ryzen 5 3600 6-Core Processor
	- total time = 791s; 

- baseline: [vanilla accelerate recipe](https://huggingface.co/blog/bloom-inference-pytorch-scripts) on the same computer:
note: I'm not using the safetensors version of bloomz here, so accelerate initially copies the weights into the offload dir, which roughly 
accounts for an initial delay of. Another difference in favor of accelerate is that the computer I'm using actually has a Titan X 12GB GPU
as well as 24GB of RAM: accelerate is exploiting the additional RAM, while slowLLM figures are computed with <16GB of RAM and no GPU. However, we tried to let accelerate also exploit the GPU, but it crashed with cuda out-of-memory error.
	- total time = *crashed (cuda oom with gpu; kernel killed without gpu)*

- code derived from Arteaga's blog: this code achieves the same process than mine, but reimplements the forward pass while I'm hacking into the Bloom module. The advantage of hacking into the Bloom module is that I can still benefit from the advanced functionalities empowering the transformers library, such as beam-search generation.
	- total time = 4200s (because by reimplementing the forward pass, a single CPU core is working at a time)


