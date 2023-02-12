import time
import resource
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BloomConfig
from accelerate import init_empty_weights
from typing import Optional, Tuple, Union
import torch
import gc

wd = "/home/xtof/sda5/bloom/"
wd = "/mnt/dos/xtof/"
curlayer = 0
allblocks = []

pnames = (
        'input_layernorm.weight',
        'input_layernorm.bias',
        'self_attention.query_key_value.weight',
        'self_attention.query_key_value.bias',
        'self_attention.dense.weight',
        'self_attention.dense.bias',
        'post_attention_layernorm.weight',
        'post_attention_layernorm.bias',
        'mlp.dense_h_to_4h.weight',
        'mlp.dense_h_to_4h.bias',
        'mlp.dense_4h_to_h.weight',
        'mlp.dense_4h_to_h.bias',
        )

class MyBloomBlock(transformers.models.bloom.modeling_bloom.BloomBlock):
    def __init__(self, config):
        super().__init__(config)
        self.memAllocated = False
        # there's one such block created per layer
        # when the first one is created, it'll be the only one with actual parameters
        global curlayer, allblocks
        self.numLayer = curlayer
        curlayer += 1
        allblocks.append(self)
        self.emptyParms = [p for p in self.parameters()]
        self.hasParms = False

    def assignParms(self,pname,v):
        if pname==pnames[0]: self.input_layernorm.weight = v
        if pname==pnames[1]: self.input_layernorm.bias = v
        if pname==pnames[2]: self.self_attention.query_key_value.weight = v
        if pname==pnames[3]: self.self_attention.query_key_value.bias = v
        if pname==pnames[4]: self.self_attention.dense.weight = v
        if pname==pnames[5]: self.self_attention.dense.bias = v
        if pname==pnames[6]: self.post_attention_layernorm.weight = v
        if pname==pnames[7]: self.post_attention_layernorm.bias = v
        if pname==pnames[8]: self.mlp.dense_h_to_4h.weight = v
        if pname==pnames[9]: self.mlp.dense_h_to_4h.bias = v
        if pname==pnames[10]: self.mlp.dense_4h_to_h.weight = v
        if pname==pnames[11]: self.mlp.dense_4h_to_h.bias = v

    def emptyLayer(self):
        for i in range(len(pnames)):self.assignParms(pnames[i],self.emptyParms[i])
        gc.collect()
        self.hasParms = False

    def loadLayer(self,l):
        print("load weights from disk")
        t0 = time.time()
        f = "0000"+str(l+2) if l<8 else "000"+str(l+2)
        parms = torch.load(wd+"pytorch_model_"+f+"-of-00072.bin")
        for i in range(len(pnames)):
            prebloc = parms['h.'+str(l)+'.'+pnames[i]].to(dtype=torch.float32)
            del parms['h.'+str(l)+'.'+pnames[i]]
            prebloc = torch.nn.Parameter(prebloc,requires_grad=False)
            self.assignParms(pnames[i],prebloc)
        t1 = time.time()
        print("preloaded OK",l,t1-t0)
        self.hasParms = True

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

allblocks[0].loadLayer(49)
allblocks[0].emptyLayer()

toker = transformers.models.bloom.tokenization_bloom_fast.BloomTokenizerFast.from_pretrained(wd)
prompt = toker('"Do the rich need more money?" Is this question a rhetorical question, yes or no?')
plen = len(prompt['input_ids'][0])
print("prompt len",plen)
input()
out = model(**prompt)
print("model generate finished",out)

