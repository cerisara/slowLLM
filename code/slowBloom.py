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
wd = "/home/xtof/nas1/TALC/Synalp/Models/bloom/bloom/"

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

class MyLinear(torch.nn.Linear):
    def __init__(self, *args):
        super().__init__(1,1)
        self.isLoaded = False

    def forward(self, x):
        if self.isLoaded:
            return super().forward(x)
        # I dont know exactly the lexicon size, but I only query yes/no anyway so...
        return torch.zeros((1,250000))

class MyEmbeddings(torch.nn.Embedding):
    def __init__(self, *args):
        super().__init__(1,1)
        self.dummy = torch.zeros((1,14336))
        self.isLoaded = False

    def forward(self, x):
        if self.isLoaded:
            return super().forward(x)
        else: return self.dummy.expand((x.size(0),14336))

class MyBloomBlock(transformers.models.bloom.modeling_bloom.BloomBlock):
    def __init__(self, config):
        super().__init__(config)
        self.memAllocated = False
        # there's one such block created per layer
        # when the first one is created, it'll be the only one with actual parameters
        global allblocks
        self.numLayer = len(allblocks)
        allblocks.append(self)
        self.emptyParms = [p for p in self.parameters()]
        self.hasParms = False
        self.loadInputs = False

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
        print("preloaded OK",l,t1-t0,"RAM",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
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
        
        print("calling forward layer",self.numLayer,self.hasParms, hidden_states.device)
        t0 = time.time()

        if self.hasParms:
            if self.loadInputs:
                hidden_states = torch.load("layerout."+str(self.numLayer-1)+".0")
                # attention_mask = torch.load("layerout."+str(self.numLayer-1)+".1")
            y0 = super().forward(hidden_states, alibi, attention_mask, layer_past, head_mask, use_cache=False, output_attentions=False)
            self.saveOutputs(y0)
            y = (y0[0], attention_mask)
        else:
            y=(hidden_states, attention_mask)
        t1 = time.time()
        print("called forward",self.numLayer,len(y),t1-t0,self.hasParms,self.loadInputs)
        return y

    def saveOutputs(self,y):
        torch.save(y[0],"layerout."+str(self.numLayer)+".0")
        # I have no cue what's in this tuple ??
        # torch.save(y[1],"layerout."+str(self.numLayer)+".1")

def initModel():
    print("loading empty model")
    t0 = time.time()
    model=None
    with init_empty_weights():
        config = BloomConfig.from_pretrained(wd)
        transformers.models.bloom.modeling_bloom.BloomBlock = MyBloomBlock
        model = transformers.models.bloom.modeling_bloom.BloomForCausalLM(config)

    model.transformer.word_embeddings = MyEmbeddings()
    model.transformer.word_embeddings_layernorm.weight = torch.nn.Parameter(torch.zeros(model.transformer.word_embeddings_layernorm.weight.size()),requires_grad=False)
    model.transformer.word_embeddings_layernorm.bias = torch.nn.Parameter(torch.zeros(model.transformer.word_embeddings_layernorm.bias.size()),requires_grad=False)
    model.transformer.ln_f.weight = torch.nn.Parameter(torch.zeros(model.transformer.ln_f.weight.size()),requires_grad=False)
    model.transformer.ln_f.bias = torch.nn.Parameter(torch.zeros(model.transformer.ln_f.bias.size()),requires_grad=False)
    model.lm_head = MyLinear()

    t1 = time.time()
    t1-=t0
    print("empty model loaded",t1,"RAM",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return model

def loadEmbeddings(model):
    # takes approx. 21MB
    parms = torch.load(wd+"pytorch_model_00001-of-00072.bin")
    vparms = parms['word_embeddings.weight'].to(dtype=torch.float32)
    model.transformer.word_embeddings = torch.nn.Embedding.from_pretrained(vparms,freeze=True)
    model.transformer.word_embeddings.isLoaded = True
    vparms = parms['word_embeddings_layernorm.weight'].to(dtype=torch.float32)
    model.transformer.word_embeddings_layernorm.weight = torch.nn.Parameter(vparms,requires_grad=False)
    vparms = parms['word_embeddings_layernorm.bias'].to(dtype=torch.float32)
    model.transformer.word_embeddings_layernorm.bias = torch.nn.Parameter(vparms,requires_grad=False)
    model.lm_head.weight = model.transformer.word_embeddings.weight
    model.lm_head.isLoaded = True
    del parms
    gc.collect()
    print("embeddings created","RAM",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def loadLMHead(model):
    parms = torch.load(wd+"pytorch_model_00072-of-00072.bin")
    vparms = parms['ln_f.weight'].to(dtype=torch.float32)
    model.transformer.ln_f.weight = torch.nn.Parameter(vparms,requires_grad=False)
    vparms = parms['ln_f.bias'].to(dtype=torch.float32)
    model.transformer.ln_f.bias = torch.nn.Parameter(vparms,requires_grad=False)
    del parms
    gc.collect()
    print("LMhead loaded","RAM",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

toker = transformers.models.bloom.tokenization_bloom_fast.BloomTokenizerFast.from_pretrained(wd)
prompt = toker('"Do the rich need more money?" Is this question a rhetorical question, yes or no?')
x = torch.LongTensor([prompt['input_ids']])
print("prompt",x)

if True:
    model = initModel()
    loadEmbeddings(model)
    allblocks[0].loadLayer(0)
    out = model(x) # save layer 0 output to disk
    allblocks[0].emptyLayer()

    for i in range(1,69):
        allblocks[i].loadInputs = True
        allblocks[i].loadLayer(i)
        out = model(x) # save layer i output to disk
        allblocks[i].emptyLayer()
        allblocks[i].loadInputs = False

    loadLMHead(model)
    allblocks[69].loadInputs = True
    allblocks[69].loadLayer(69)

out = model(x)
print("model forward finished")
print("out",out.logits.shape, out.logits.device)
logits = out.logits.view(-1)
print("yes",logits[18260].item())
print("no",logits[654].item())

# token of 'yes': 18260
# token of 'no' : 654

