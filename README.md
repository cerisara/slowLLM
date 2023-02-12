Using Large Language Models (starting with Bloomz) slowly, on commodity GPU-free desktops.

## Use cases

- No way to do text generation, this would be far too slow with autoregressive models;
may be in the future  with text diffusion models, but definitely not now.
- So there are 2 main use cases:
    - generate text embeddings for further processing
    - answering binary yes/no question-answering tasks

This may seem very limited (and it is, yes), but there are many tasks that can be framed as a
yes/no question and could thus benefit from the power of LLM.

