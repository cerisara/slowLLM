# What is it and why?

Large Pretrained Languague Models (PLM) are essential components to any
Natural Language Processing application.
However, the best results can only be obtained with the largest PLMs,
which require non-standard computing power to use.

Common solutions to this issue consist in using cloud-based GPUs to run
the PLMs, but it may become costly in the long term.

The solution we propose here is to run such PLMs **collaboratively** on
standard personal desktops and laptops.
This open-source code implements a standard model-parallelism, i.e.,
each node in the network hosts one (or a few) layers of a PLM.
An adaptation of pipeline-parallelism shall also be implemented, i.e.,
successive requests are pushed into the first layer as soon as possible, before the computation
of the previous request has made its way up to the top layer.

One of the main drawback is communication delays, because every post-layer activation
is transmitted through unreliable low-latency web-sockets.
Because of this delay, the code only currently implements the forward pass, i.e., computing
predictions, while training is not supported for now.
A request hence requires a few seconds to pass through the 44 layers of the GPT-NeoX-20GB model,
which we believe is acceptable for a number of applications.

The software is based on a client-server architecture, where the server manages passing the
information through all layers. But both the server and client codes are open-source, and so
it is relatively easy to adapt the server code to host your own network of private and trusted nodes
(the documentation to do that is not yet written though).

Another limitation of the code is that there is for now not any load-balancing, priority list nor
anti-malevolent node codes implemented: these are all future works; pull requests are welcome.

The code is still very alpha; use it at your own risks.

# Quickstart

- Download all files for GPT-NeoX-20G from Huggingface and put them directly in this directory (no sub-directory)
- You may check how many clients already serve each layer with:
```
curl http://fb.cerisara.fr/status
```
- Run this command 1 or several times (check your RAM!) if you want to serve 1 or several layers:
```
python3 client.py <num_layer_within_[0:43]>
```

