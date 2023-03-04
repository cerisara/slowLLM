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
	h.3.mlp.dense_h_to_4h.weight torch.Size([57344, 14336]) torch.bfloat16
	h.3.mlp.dense_h_to_4h.bias torch.Size([57344]) torch.bfloat16
	h.3.mlp.dense_4h_to_h.weight torch.Size([14336, 57344]) torch.bfloat16
14336	h.3.mlp.dense_4h_to_h.bias torch.Size([14336]) torch.bfloat16
