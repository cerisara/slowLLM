import time
import resource
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BloomConfig
from accelerate import init_empty_weights

wd = "/home/xtof/sda5/bloom/"
wd = "/mnt/dos/xtof/"

print("loading empty model")
t0 = time.time()
model=None
with init_empty_weights():
    config = BloomConfig.from_pretrained(wd)
    model = transformers.models.bloom.modeling_bloom.BloomForCausalLM(config)
t1 = time.time()
t1-=t0
print("empty model loaded",t1,"RAM",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

class MyBloomBlock(transformers.models.bloom.modeling_bloom.BloomBlock):
    def __init__(self, config):
        super().__init__(config)
        self.numlayer = 51 # must be set by calling process

    def loadLayer(self,numlayer):
        self.numlayer = numlayer
            bloomblock0 = self
            nparms = sum(p.numel() for _,p in self.named_parameters())
            print("parmsize",2.*nparms/1000000000.,"GB")

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
        else:
            # use the same parms tensor than the other blocks ! Just overwrite the values inside...
            self.input_layernorm.weight = bloomblock0.input_layernorm.weight
            self.input_layernorm.bias = bloomblock0.input_layernorm.bias
            self.self_attention.query_key_value.weight = bloomblock0.self_attention.query_key_value.weight
            self.self_attention.query_key_value.bias = bloomblock0.self_attention.query_key_value.bias
            self.self_attention.dense.weight = bloomblock0.self_attention.dense.weight
            self.self_attention.dense.bias = bloomblock0.self_attention.dense.bias
            self.post_attention_layernorm.weight = bloomblock0.post_attention_layernorm.weight
            self.post_attention_layernorm.bias = bloomblock0.post_attention_layernorm.bias
            self.mlp.dense_h_to_4h.weight = bloomblock0.mlp.dense_h_to_4h.weight
            self.mlp.dense_h_to_4h.bias = bloomblock0.mlp.dense_h_to_4h.bias
            self.mlp.dense_4h_to_h.weight = bloomblock0.mlp.dense_4h_to_h.weight
            self.mlp.dense_4h_to_h.bias = bloomblock0.mlp.dense_4h_to_h.bias
        wantLayer(self.numlayer)
 
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
        self.getParms()
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


