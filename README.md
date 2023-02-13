## What is this soft about?

Using Large Language Models (starting with Bloom) slowly, on commodity GPU-free desktops.

## Requirements

- desktop or laptop with at least 25GB of RAM (could be reduced <16GB, PR welcome)
- harddrive with 400GB free to store Bloom's parameters
- that's it! no GPU is needed
- optional: a fast hard drive (the best would be a NVMe ssd)

## How to use

- download or clone this repo
- install in conda/pip pytorch + transformers + accelerate
- write in the text file code/questions.txt one yes/no question per line
- cd code; python slowLLM.py

Example of usage:
```

```

## Limitations

- It does not support real text generation, this would be far too slow with autoregressive models;
may be in the future  with text diffusion models, but not now.
- So it is limited for now to only answer yes/no questions: many tasks can be framed as yes/no questions,
but this may require some thinking.
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

