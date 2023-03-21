import time
import resource
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BloomConfig
from accelerate import init_empty_weights
from typing import Optional, Tuple, Union
import torch
import gc
from pathlib import Path
from pynvml import *

# Put here the directory where you downloaded the Bloom's parameters
wd = "/home/xtof/nas1/TALC/Synalp/Models/bloom/bloom/"
wd = "/media/xtof/556E99561C655FA8/bloomz/"
wd = "/mnt/dos/xtof/"
wd = "/home/xtof/nas1/TALC/Synalp/Models/bloomz/"
wd = "/home/xtof/models/bloomz/"
# this version of Bloom on JZ does not assign one layer to one file ! So I use my own version next:
wd = "/gpfsdswork/dataset/HuggingFace_Models/bigscience/bloom/"
wd = "/gpfswork/rech/knb/uyr14tk/home/bloom/model176b/"

prefix = 0
# TODO: load prefix

# this class wraps the Linear class of some weights in bf16 by converting them to fp32 first
class MyLinearAtt(torch.nn.Module):
    def __init__(self, nom, lin):
        super().__init__()
        self.nom = nom
        self.lin = lin
        self.weight = self.lin.weight.data
        self.bias   = self.lin.bias.data
        self.passthru = False

    def forward(self,x):
        if self.passthru: return x
        x32 = x.to(dtype=torch.float32)
        w32 = self.weight.data.to(dtype=torch.float32).t()
        b32 = self.bias.data.to(dtype=torch.float32)
        y = torch.matmul(x32,w32)
        y = y+b32
        # w32.requires_grad = True
        # got 2 sets of parms: lin.* are meta + requires_grad
        #                      weight,bias are allocated + does not require grad
        return y

# Do not modify below

allblocks = []
filesuffix = ""
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

class MyLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.isLoaded = False
        self.passthru = False

    def forward(self, x):
        if self.isLoaded and not self.passthru:
            print("enter linear",x.storage().data_ptr(),x.requires_grad,x.device)
            xx = x.squeeze()#.to(dtype=torch.bfloat16)
            # matmul in bf16 takes 12s for 3 tokens because it only uses 1 core, while it uses all cores in fp32
            # so I rather convert the matrix to fp32
            y = torch.matmul(xx,self.weight.T.to(dtype=torch.float32))
            y = y.unsqueeze(0)
            print("in linear",y.shape)
            return y#.to(torch.float32)
        # I dont know exactly the lexicon size, but I only query yes/no anyway so... TODO: fix that!
        print("in linear pass thru")
        return torch.zeros((1,250000))
 
    def emptyLayer(self):
        print("ERR called emptylayer")

def print_usage_gpu():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print("GPU mem(MB)", info.used//1024**2)

class LatentOutputs():
    # rather than saving all latent outputs (activations) to disk, which is slow, it's best to keep it in RAM
    # we need to keep only Nutts*Ntoks*14336 float32
    # we know that embeddings in bf16 take about 8GB, and one layer in f32 takes about 10GB (less with MyLinearAtt)
    # assuming Ntoks = 1000, then each latent output per utterance takes about 50MB
    # so in 6GB, we can store 120 latent outputs, so let's max at 100 examples

    def __init__(self):
        self.latent = []
        self.getidx = -1

    def store(self,z,keepgraph=False):
        # this is called once per utterance
        if keepgraph:
            if z.is_leaf: z.requires_grad=True
            self.latent.append(z)
            print("storing",z.requires_grad,z.storage().data_ptr())
        else: self.latent.append(z.detach())

    def get(self):
        # in the second phase, we pop out all latent FIFO-style
        if len(self.latent)<=0: return None
        # instead of popping, I keep all latents in RAM (8GB max) to facilitate backward()
        self.getidx += 1
        if self.getidx>=len(self.latent): self.getidx=0
        return self.latent[self.getidx]
        # return self.latent.pop(0)

    def tostr(self):
        return ' '.join([str(tuple(z.shape)) for z in self.latent])

class MyEmbeddings(torch.nn.Embedding):
    # store embeddings in bf16, and convert them to fp32 on the fly
    def __init__(self):
        super().__init__(1,1)
        self.dummy = torch.zeros((1,14336))
        self.isLoaded = False
        self.latentOutputs = None
        if prefix>0: self.prefv = torch.nn.Parameter(torch.randn(prefix,14336).to(dtype=torch.bfloat16))
        self.keepgraph=False
        self.passthru = False

    def setPoids(self,poids):
        self.isLoaded = True
        self.weight = torch.nn.Parameter(poids.to(dtype=torch.bfloat16),requires_grad=False)

    def forward(self, x):
        if self.isLoaded and not self.passthru:
            e=super().forward(x)
            if prefix>0:
                # tried first to append the prefix, but this creates issue when computing alibi
                # e=torch.cat((self.prefv.expand(e.size(0),-1,-1),e),dim=1)
                emask = torch.tensor([True]*prefix+[False]*(x.size(1)-prefix))
                pref0 = torch.zeros(e.size(0),e.size(1)-prefix,e.size(2)).to(dtype=torch.bfloat16)
                prefm = torch.cat((self.prefv.expand(e.size(0),-1,-1),pref0),dim=1)
                e[:,emask,:] = prefm[:,emask,:]
                print("pass in embeds prefv",self.prefv.requires_grad)
            if self.latentOutputs!=None:
                self.latentOutputs.store(e,keepgraph=self.keepgraph)
            e=e.to(dtype=torch.float32)
            return e
        elif self.latentOutputs != None:
            # in the second phase, we poll previously computed outputs to pass them to the first layer
            e=self.latentOutputs.get()
            if e==None: return self.dummy.expand((x.size(0),14336))
            e=e.to(dtype=torch.float32)
            return e
        else: return self.dummy.expand((x.size(0),14336))

    def emptyLayer(self):
        print("ERR called emptylayer")

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
        self.latentOutputs = None
        self.self_attention.query_key_value = MyLinearAtt('satt',self.self_attention.query_key_value)
        self.mlp.dense_h_to_4h = MyLinearAtt('h_4h',self.mlp.dense_h_to_4h)
        self.mlp.dense_4h_to_h = MyLinearAtt('4h_h',self.mlp.dense_4h_to_h)
        self.passthru = False
        self.keepgraph = False
        self.nextBlockDev = None

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
        print("ERR called emptylayer")

    def loadLayer(self,dev="cpu"):
        print("load weights from disk")
        self.dev = dev
        t0 = time.time()
        # attention: les fichiers sur JZ ne sont pas comme chez moi: il faudrait utiliser le json!
        f = "0000"+str(self.numLayer+2) if self.numLayer<8 else "000"+str(self.numLayer+2)
        print("loading",f)
        parms = torch.load(wd+"pytorch_model_"+f+"-of-00072.bin")
        for i in range(len(pnames)):
            if 'value.weight' in pnames[i] or 'h.weight' in pnames[i]: # or "dense.weight" in pnames[i]:
                prebloc = parms['h.'+str(self.numLayer)+'.'+pnames[i]] # keep them in bf16 !
            else:
                prebloc = parms['h.'+str(self.numLayer)+'.'+pnames[i]].to(dtype=torch.float32)
                del parms['h.'+str(self.numLayer)+'.'+pnames[i]]
            prebloc = prebloc.to(dev)
            prebloc = torch.nn.Parameter(prebloc,requires_grad=False)
            self.assignParms(pnames[i],prebloc)

        t1 = time.time()
        print("preloaded OK",self.numLayer,t1-t0,"RAM",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
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
        print("calling forward layer",self.numLayer,self.hasParms, hidden_states.device)
        print("layer input",torch.norm(hidden_states).item())

        t0 = time.time()
        if self.hasParms and not self.passthru:
            hidden_states = hidden_states.to(self.dev)
            if alibi!=None: alibi = alibi.to(self.dev)
            if attention_mask!=None: attention_mask = attention_mask.to(self.dev)
            y0 = super().forward(hidden_states, alibi, attention_mask, layer_past, head_mask, use_cache=False, output_attentions=False)
            if self.latentOutputs!=None: self.latentOutputs.store(y0[0],keepgraph=self.keepgraph)
            y = (y0[0], attention_mask)
            print("just computed output",y0[0].storage().data_ptr(),y0[0].device,self.keepgraph,self.numLayer)
            if not self.keepgraph:
                # detach so that VRAM is freed from internal layer activations
                y = (y[0].detach(),y[1].detach())
            else: print("KEEPGRAPH",self.numLayer,y[0].storage().data_ptr())
        elif self.latentOutputs!=None:
            h = self.latentOutputs.get()
            if h==None: y=(hidden_states, attention_mask)
            else: y = (h, attention_mask)
        else:
            # when the layer is empty, just pass the input unchanged
            y=(hidden_states, attention_mask)
        t1 = time.time()
        if self.hasParms: print("TIME in 1 layer",t1-t0)

        # the LM head is on the CPU, as well as the embeddings...
        if self.numLayer>=69:
            print("passing to CPU",y[0].storage().data_ptr(),y[0].requires_grad,self.keepgraph)
            y = (y[0].to("cpu"),y[1].to("cpu"))
        if self.nextBlockDev!=None:
            print("passing to GPU",self.nextBlockDev,y[0].storage().data_ptr(),y[0].requires_grad,self.keepgraph)
            y = (y[0].to(self.nextBlockDev),y[1].to(self.nextBlockDev))
        return y

def initModel():
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
    parms = torch.load(wd+"pytorch_model_00001-of-00072.bin")
    vparms = parms['word_embeddings.weight']
    model.transformer.word_embeddings.setPoids(vparms)
    model.transformer.word_embeddings.latentOutputs = LatentOutputs()
    vparms = parms['word_embeddings_layernorm.weight'].to(dtype=torch.float32)
    model.transformer.word_embeddings_layernorm.weight = torch.nn.Parameter(vparms,requires_grad=False)
    vparms = parms['word_embeddings_layernorm.bias'].to(dtype=torch.float32)
    model.transformer.word_embeddings_layernorm.bias = torch.nn.Parameter(vparms,requires_grad=False)
    del parms
    gc.collect()
    print("embeddings loaded","RAM",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def loadLMHead(model):
    loadEmbeddings(model)
    model.lm_head.weight = model.transformer.word_embeddings.weight
    model.lm_head.isLoaded = True
    # ln_f est tout petit
    parms = torch.load(wd+"pytorch_model_00072-of-00072.bin")
    vparms = parms['ln_f.weight'].to(dtype=torch.float32)
    model.transformer.ln_f.weight = torch.nn.Parameter(vparms,requires_grad=False)
    vparms = parms['ln_f.bias'].to(dtype=torch.float32)
    model.transformer.ln_f.bias = torch.nn.Parameter(vparms,requires_grad=False)
    del parms
    gc.collect()
    print("LMhead loaded","RAM",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def inference_soft_prompt():
    s = "Le COVID est"
    run_forward_oneutt(s)

def run_forward_oneutt(utt):
    model.transformer.word_embeddings.passthru = False
    model.transformer.word_embeddings.latentOutputs = None
    model.lm_head.passthru = False
    for l in range(len(allblocks)):
        allblocks[l].latentOutputs = None
        allblocks[l].passthru = False
        allblocks[l].keepgraph=False
    prompt = toker(utt)
    tokids = prompt['input_ids']
    if prefix>0: tokids = [0]*prefix+tokids
    generate_kwargs = dict(max_new_tokens=50, do_sample=False)
    x = torch.LongTensor([tokids])
    outputs = model.generate(x, **generate_kwargs)
    sout = toker.batch_decode(outputs, skip_special_tokens=True)
    print("GENERATE",sout)
    print_usage_gpu()

    outl1 = allblocks[-1].latentOutputs.latent[0]
    print("last activ checkpoint",outl1.device,torch.norm(outl1).item())

def save_prefix():
    torch.save(model.transformer.word_embeddings.prefv,"prefv.pt")

# ###################################

model = initModel()
loadLMHead(model)

i=0
for j in range(9):
    allblocks[i].loadLayer("cuda:0")
    i+=1
allblocks[i-1].nextBlockDev="cuda:1"
for j in range(9):
    allblocks[i].loadLayer("cuda:1")
    i+=1
allblocks[i-1].nextBlockDev="cuda:2"
for j in range(9):
    allblocks[i].loadLayer("cuda:2")
    i+=1
allblocks[i-1].nextBlockDev="cuda:3"
for j in range(9):
    allblocks[i].loadLayer("cuda:3")
    i+=1
allblocks[i-1].nextBlockDev="cuda:4"
for j in range(9):
    allblocks[i].loadLayer("cuda:4")
    i+=1
allblocks[i-1].nextBlockDev="cuda:5"
for j in range(9):
    allblocks[i].loadLayer("cuda:5")
    i+=1
allblocks[i-1].nextBlockDev="cuda:6"
for j in range(9):
    allblocks[i].loadLayer("cuda:6")
    i+=1
allblocks[i-1].nextBlockDev="cuda:7"
while i<70:
    allblocks[i].loadLayer("cuda:7")
    i+=1

print_usage_gpu()

toker = transformers.models.bloom.tokenization_bloom_fast.BloomTokenizerFast.from_pretrained(wd)

t0 = time.time()
inference_soft_prompt()
t1 = time.time()
print("total time required",t1-t0)
