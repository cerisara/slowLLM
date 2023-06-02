import torch
import transformers
from transformers import AutoTokenizer, BloomConfig
from accelerate import init_empty_weights

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
        global allblocks
        self.numLayer = len(allblocks)
        allblocks.append(self)
        if self.numLayer<10: self.device = "cuda:0"
        elif self.numLayer<40: self.device = "cuda:1"
        else: self.device = "cuda:2"

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

    def loadLayer(self):
        f = "0000"+str(self.numLayer+2) if self.numLayer<8 else "000"+str(self.numLayer+2)
        parms = torch.load(wd+"pytorch_model_"+f+"-of-00072.bin")
        for i in range(len(pnames)):
            prebloc = parms['h.'+str(self.numLayer)+'.'+pnames[i]].to(self.device)
            # TODO: make it 4 bits
            prebloc = torch.nn.Parameter(prebloc,requires_grad=False)
            self.assignParms(pnames[i],prebloc)

transformers.models.bloom.modeling_bloom.BloomBlock = MyBloomBlock
with init_empty_weights():
    config = BloomConfig.from_pretrained(wd)
    transformers.models.bloom.modeling_bloom.BloomBlock = MyBloomBlock
    model = transformers.models.bloom.modeling_bloom.BloomForCausalLM(config)
for l in range(len(allblocks)): allblocks[l].loadLayer()

