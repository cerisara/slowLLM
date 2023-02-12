import time
import resource
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BloomConfig
from accelerate import init_empty_weights
from typing import Optional, Tuple, Union
import torch

wd = "/home/xtof/sda5/bloom/"
wd = "/mnt/dos/xtof/"
singletonBlock = None
curlayer = 0
allblocks = []

class MyBloomBlock(transformers.models.bloom.modeling_bloom.BloomBlock):
    def __init__(self, config):
        super().__init__(config)
        self.memAllocated = False
        # there's one such block created per layer
        # when the first one is created, it'll be the only one with actual parameters
        global singletonBlock, curlayer, allblocks
        self.numLayer = curlayer
        curlayer += 1
        if singletonBlock==None:
            singletonBlock = self
            self.memAllocated = False
        else:
            self.memAllocated = True
        allblocks.append(self)

    def loadLayer(self,l):
        if not self.memAllocated:
            print("allocate RAM for one layer")
            nparms = sum(p.numel() for _,p in self.named_parameters())
            print("parmsize for one layer",4.*nparms/1000000000.,"GB")

            p=self.input_layernorm.weight
            self.input_layernorm.weight = torch.nn.Parameter(torch.zeros_like(p,device="cpu",dtype=torch.float32,requires_grad=False))
            p=self.input_layernorm.bias
            self.input_layernorm.bias = torch.nn.Parameter(torch.zeros_like(p,device="cpu",dtype=torch.float32,requires_grad=False))
            p=self.self_attention.query_key_value.weight
            self.self_attention.query_key_value.weight = torch.nn.Parameter(torch.zeros_like(p,device="cpu",dtype=torch.float32,requires_grad=False))
            p=self.self_attention.query_key_value.bias
            self.self_attention.query_key_value.bias = torch.nn.Parameter(torch.zeros_like(p,device="cpu",dtype=torch.float32,requires_grad=False))
            p=self.self_attention.dense.weight
            self.self_attention.dense.weight = torch.nn.Parameter(torch.zeros_like(p,device="cpu",dtype=torch.float32,requires_grad=False))
            p=self.self_attention.dense.bias
            self.self_attention.dense.bias = torch.nn.Parameter(torch.zeros_like(p,device="cpu",dtype=torch.float32,requires_grad=False))
            p=self.post_attention_layernorm.weight
            self.post_attention_layernorm.weight = torch.nn.Parameter(torch.zeros_like(p,device="cpu",dtype=torch.float32,requires_grad=False))
            p=self.post_attention_layernorm.bias
            self.post_attention_layernorm.bias = torch.nn.Parameter(torch.zeros_like(p,device="cpu",dtype=torch.float32,requires_grad=False))
            p=self.mlp.dense_h_to_4h.weight
            self.mlp.dense_h_to_4h.weight = torch.nn.Parameter(torch.zeros_like(p,device="cpu",dtype=torch.float32,requires_grad=False))
            p=self.mlp.dense_h_to_4h.bias
            self.mlp.dense_h_to_4h.bias = torch.nn.Parameter(torch.zeros_like(p,device="cpu",dtype=torch.float32,requires_grad=False))
            p=self.mlp.dense_4h_to_h.weight
            self.mlp.dense_4h_to_h.weight = torch.nn.Parameter(torch.zeros_like(p,device="cpu",dtype=torch.float32,requires_grad=False))
            p=self.mlp.dense_4h_to_h.bias
            self.mlp.dense_4h_to_h.bias = torch.nn.Parameter(torch.zeros_like(p,device="cpu",dtype=torch.float32,requires_grad=False))
            for p in self.parameters(): p.requires_grad=False

            for layer in allblocks[1:]:
                # use the same parms tensor than the first block ! Just overwrite the values inside...
                layer.input_layernorm.weight = singletonBlock.input_layernorm.weight
                layer.input_layernorm.bias = singletonBlock.input_layernorm.bias
                layer.self_attention.query_key_value.weight = singletonBlock.self_attention.query_key_value.weight
                layer.self_attention.query_key_value.bias = singletonBlock.self_attention.query_key_value.bias
                layer.self_attention.dense.weight = singletonBlock.self_attention.dense.weight
                layer.self_attention.dense.bias = singletonBlock.self_attention.dense.bias
                layer.post_attention_layernorm.weight = singletonBlock.post_attention_layernorm.weight
                layer.post_attention_layernorm.bias = singletonBlock.post_attention_layernorm.bias
                layer.mlp.dense_h_to_4h.weight = singletonBlock.mlp.dense_h_to_4h.weight
                layer.mlp.dense_h_to_4h.bias = singletonBlock.mlp.dense_h_to_4h.bias
                layer.mlp.dense_4h_to_h.weight = singletonBlock.mlp.dense_4h_to_h.weight
                layer.mlp.dense_4h_to_h.bias = singletonBlock.mlp.dense_4h_to_h.bias

        print("load weights from disk")
        t0 = time.time()
        f = "0000"+str(l+2) if l<8 else "000"+str(l+2)
        prebloc = [None]*70
        parms = torch.load(wd+"pytorch_model_"+f+"-of-00072.bin")
        prebloc[0] = parms['h.'+str(l)+'.input_layernorm.weight'].to(dtype=torch.float32)
        prebloc[1] = parms['h.'+str(l)+'.input_layernorm.bias'].to(dtype=torch.float32)
        prebloc[2] = parms['h.'+str(l)+'.self_attention.query_key_value.weight'].to(dtype=torch.float32)
        prebloc[3] = parms['h.'+str(l)+'.self_attention.query_key_value.bias'].to(dtype=torch.float32)
        prebloc[4] = parms['h.'+str(l)+'.self_attention.dense.weight'].to(dtype=torch.float32)
        prebloc[5] = parms['h.'+str(l)+'.self_attention.dense.bias'].to(dtype=torch.float32)
        prebloc[6] = parms['h.'+str(l)+'.post_attention_layernorm.weight'].to(dtype=torch.float32)
        prebloc[7] = parms['h.'+str(l)+'.post_attention_layernorm.bias'].to(dtype=torch.float32)
        prebloc[8] = parms['h.'+str(l)+'.mlp.dense_h_to_4h.weight'].to(dtype=torch.float32)
        prebloc[9] = parms['h.'+str(l)+'.mlp.dense_h_to_4h.bias'].to(dtype=torch.float32)
        prebloc[10] = parms['h.'+str(l)+'.mlp.dense_4h_to_h.weight'].to(dtype=torch.float32)
        prebloc[11] = parms['h.'+str(l)+'.mlp.dense_4h_to_h.bias'].to(dtype=torch.float32)
        del parms
        for i in range(12): prebloc[i] = torch.nn.Parameter(prebloc[i],requires_grad=False)
        t1 = time.time()
        print("preloaded OK",l,t1-t0)

        print("copy parms in block",l)
        singletonBlock.input_layernorm.weight.copy_(prebloc[0])
        singletonBlock.input_layernorm.bias.copy_(prebloc[1])
        singletonBlock.self_attention.query_key_value.weight.copy_(prebloc[2])
        singletonBlock.self_attention.query_key_value.bias.copy_(prebloc[3])
        singletonBlock.self_attention.dense.weight.copy_(prebloc[4])
        singletonBlock.self_attention.dense.bias.copy_(prebloc[5])
        singletonBlock.post_attention_layernorm.weight.copy_(prebloc[6])
        singletonBlock.post_attention_layernorm.bias.copy_(prebloc[7])
        singletonBlock.mlp.dense_h_to_4h.weight.copy_(prebloc[8])
        singletonBlock.mlp.dense_h_to_4h.bias.copy_(prebloc[9])
        singletonBlock.mlp.dense_4h_to_h.weight.copy_(prebloc[10])
        singletonBlock.mlp.dense_4h_to_h.bias.copy_(prebloc[11])
        del prebloc 

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # to free the computation graph ???????
        # hidden_states = hidden_states.detach()
        # if not alibi==None: alibi = alibi.detach()
        # if not attention_mask==None: attention_mask = attention_mask.detach()

        t0 = time.time()
        print("debug0",torch.norm(self.input_layernorm.weight).item())
        y = super().forward(hidden_states, alibi, attention_mask, layer_past, head_mask, use_cache, output_attentions)
        t1 = time.time()
        print("called forward",self.numlayer,t1-t0,torch.norm(hidden_states).item(),torch.norm(y[0]).item())
        return y


print("loading empty model")
t0 = time.time()
model=None
with init_empty_weights():
    config = BloomConfig.from_pretrained(wd)
    transformers.models.bloom.modeling_bloom.BloomBlock = MyBloomBlock
    model = transformers.models.bloom.modeling_bloom.BloomForCausalLM(config)
t1 = time.time()
t1-=t0
print("empty model loaded",t1,"RAM",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

singletonBlock.loadLayer(49)

