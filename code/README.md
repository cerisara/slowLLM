in this branch a100, I'll use the same implementation
of pipeline parallelism + gradient checkpointing on 8xA100
for soft prompt tuning, using another optimization:
when the gradient at the input of a layer L has been computed, then
all gradients inside this layer L can be freed.

I'll need to place 9 layers per GPU, and will use only SGD
- so optimizer state = 0B
- each layer is 5BG in bf16, so 9 layers = 45GB for the parameters in 1 GPU
- gradient checkpointing: during the forward pass, I only keep the activations at the output of each layer
  = 120MB/layer (for 2048 tokens and 14337 vec dim) = 1.2GB per GPU

In total, for one GPU, forward requires 57GB (9x theta + 9x activation checkpoints + 1x in-layer-graph with activations and gradients)

--------------------------------

- seqlen = 2048
- hidden size = 14336

- storing activations per layer =
  - fp32: 2048 x 14336 x 4 = 117MB / layer = 8.2GB total
  - bf16: 2048 x 14336 x 2 = 59MB/layer    = 4.1GB total

params 1 layer = 
14336	h.3.input_layernorm.weight torch.Size([14336]) torch.bfloat16
14336	h.3.input_layernorm.bias torch.Size([14336]) torch.bfloat16
616562688	h.3.self_attention.query_key_value.weight torch.Size([43008, 14336]) torch.bfloat16
43008	h.3.self_attention.query_key_value.bias torch.Size([43008]) torch.bfloat16
205520896	h.3.self_attention.dense.weight torch.Size([14336, 14336]) torch.bfloat16
14336	h.3.self_attention.dense.bias torch.Size([14336]) torch.bfloat16
14336	h.3.post_attention_layernorm.weight torch.Size([14336]) torch.bfloat16
14336	h.3.post_attention_layernorm.bias torch.Size([14336]) torch.bfloat16
822083584	h.3.mlp.dense_h_to_4h.weight torch.Size([57344, 14336]) torch.bfloat16
57344	h.3.mlp.dense_h_to_4h.bias torch.Size([57344]) torch.bfloat16
822083584	h.3.mlp.dense_4h_to_h.weight torch.Size([14336, 57344]) torch.bfloat16
14336	h.3.mlp.dense_4h_to_h.bias torch.Size([14336]) torch.bfloat16


En conservant en bf16: selfatt.query_k_v.w mlp.dense_h_to_4h.w mlp.dense_4h_to_h.w:
	- on devrait gagner 4.5GB ??
	- RAM = 8.82GB (enough to store gradients for backward?)
	- temps par layer = 3.63s
	- total time = 986s = 16.5' (lost 100" total compared to all parms in fp32)
	- time in all layers = 280.546"
	- time to load parms = 613.361" (both combined = 894") (note: we should be able to gain 1' by avoiding the last LMhead in bf16)
En conservant en bf16: selfatt.query_k_v.w mlp.dense_h_to_4h.w:
	- RAM = 10.4GB
	- temps par layer = 3.2s
En conservant en bf16: rien
	- RAM = 11.6GB
	- temps par layer = 2.08s

lecture d'une layer (2.466b params = 4.93GB)
	- hors cache: 8.927s
	- en   cache: 3.3s

Pass backward fonctionne pour soft prompt tuning avec 1 seule sentence !
- RAM max = 22GB
- total time = 2108s = 35'
- time forward = 966 = 16'
- time backward = 1142

Training is working with a single sentence:
time forward 1096.3856613636017 [tensor([5.2762], grad_fn=<ViewBackward0>)]
time backward 1259.036093711853
time forward 1036.817584514618 [tensor([5.2754], grad_fn=<ViewBackward0>)]
total time required 3393.0070238113403
