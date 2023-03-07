import time
import resource
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BloomConfig
from accelerate import init_empty_weights
from typing import Optional, Tuple, Union
import torch
import gc
from pathlib import Path

# Put here the directory where you downloaded the Bloom's parameters
wd = "/home/xtof/nas1/TALC/Synalp/Models/bloom/bloom/"
wd = "/media/xtof/556E99561C655FA8/bloomz/"
wd = "/mnt/dos/xtof/"
wd = "/home/xtof/nas1/TALC/Synalp/Models/bloomz/"
wd = "/home/xtof/models/bloomz/"

prefix = 1

# note: matmul btw 57344x14336 takes 0.62s in fp32 but 3.52s in bf16 !
# pour pouvoir stocker les + gros poids en bf16, l'idee est de convertir en fp32 juste avant
# convert the largest matrix takes 0.6534 seconds and 822MB of RAM

# les poids sont les suivants:
# h.3.input_layernorm.weight torch.Size([14336])
# h.3.input_layernorm.bias torch.Size([14336])
# h.3.self_attention.query_key_value.weight torch.Size([43008, 14336]) ==> BIG
# h.3.self_attention.query_key_value.bias torch.Size([43008])
# h.3.self_attention.dense.weight torch.Size([14336, 14336]) ==> BIG
# h.3.self_attention.dense.bias torch.Size([14336])
# h.3.post_attention_layernorm.weight torch.Size([14336])
# h.3.post_attention_layernorm.bias torch.Size([14336])
# h.3.mlp.dense_h_to_4h.weight torch.Size([57344, 14336]) ==> BIG
# h.3.mlp.dense_h_to_4h.bias torch.Size([57344])
# h.3.mlp.dense_4h_to_h.weight torch.Size([14336, 57344]) ==> BIG
# h.3.mlp.dense_4h_to_h.bias torch.Size([14336])

# this class wraps the Linear class of some weights in bf16 by converting them to fp32 first
class MyLinearAtt(torch.nn.Module):
    def __init__(self, nom, lin):
        super().__init__()
        self.nom = nom
        self.lin = lin
        self.weight = self.lin.weight.data
        self.bias   = self.lin.bias.data

    def forward(self,x):
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

    def forward(self, x):
        if self.isLoaded:
            print("enter linear")
            xx = x.squeeze()#.to(dtype=torch.bfloat16)
            # matmul in bf16 takes 12s for 3 tokens :-(
            # so I rather convert the matrix to fp32
            y = torch.matmul(xx,self.weight.T.to(dtype=torch.float32))
            y = y.unsqueeze(0)
            print("in linear",y.shape)
            return y#.to(torch.float32)
        # I dont know exactly the lexicon size, but I only query yes/no anyway so... TODO: fix that!
        return torch.zeros((1,250000))
 
    def emptyLayer(self):
        if hasattr(self,'weight'): del self.weight
        gc.collect()
        self.isLoaded=False

class LatentOutputs():
    # rather than saving all latent outputs (activations) to disk, which is slow, it's best to keep it in RAM
    # we need to keep only Nutts*Ntoks*14336 float32
    # we know that embeddings in bf16 take about 8GB, and one layer in f32 takes about 10GB (less with MyLinearAtt)
    # assuming Ntoks = 1000, then each latent output per utterance takes about 50MB
    # so in 6GB, we can store 120 latent outputs, so let's max at 100 examples

    def __init__(self):
        self.latent = []
        self.dostore = True

    def store(self,z,keepgrad=False):
        # this is called once per utterance
        if self.dostore:
            if keepgrad: self.latent.append(z)
            else: self.latent.append(z.detach())

    def get(self):
        # in the second phase, we pop out all latent FIFO-style
        if len(self.latent)<=0: return None
        return self.latent.pop(0)

    def tostr(self):
        return ' '.join([str(tuple(z.shape)) for z in self.latent])

class MyEmbeddings(torch.nn.Embedding):
    # store embeddings in bf16, and convert them to fp32 on the fly
    def __init__(self, poids):
        super().__init__(1,1)
        self.dummy = torch.zeros((1,14336))
        if poids==None:
            self.isLoaded = False
        else:
            self.isLoaded = True
            self.weight = torch.nn.Parameter(poids.to(dtype=torch.bfloat16),requires_grad=False)
        self.latentOutputs = None
        if prefix>0: self.prefv = torch.nn.Parameter(torch.randn(prefix,14336).to(dtype=torch.bfloat16))

    def forward(self, x):
        if self.isLoaded:
            e=super().forward(x)
            if prefix>0:
                # tried first to append the prefix, but this creates issue when computing alibi
                # e=torch.cat((self.prefv.expand(e.size(0),-1,-1),e),dim=1)
                emask = torch.tensor([True]*prefix+[False]*(x.size(1)-prefix))
                pref0 = torch.zeros(e.size(0),e.size(1)-prefix,e.size(2)).to(dtype=torch.bfloat16)
                prefm = torch.cat((self.prefv.expand(e.size(0),-1,-1),pref0),dim=1)
                e[:,emask,:] = prefm[:,emask,:]
            self.latentOutputs.store(e)
            e=e.to(dtype=torch.float32)
            return e
        elif self.latentOutputs != None:
            # in the second phase, we poll previously computed outputs to pass them to the first layer
            e=self.latentOutputs.get()
            if e==None: return self.dummy.expand((x.size(0),14336))
            e=e.to(dtype=torch.float32)
            e.requires_grad=True
            return e
        else: return self.dummy.expand((x.size(0),14336))

    def emptyLayer(self):
        del self.weight
        gc.collect()
        self.isLoaded=False

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

    def saveOutputs(self,b):
        if b: self.latentOutputs = LatentOutputs()
        else: self.latentOutputs.dostore=False

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

    def loadLayer(self):
        print("load weights from disk")
        t0 = time.time()
        f = "0000"+str(self.numLayer+2) if self.numLayer<8 else "000"+str(self.numLayer+2)
        parms = torch.load(wd+"pytorch_model_"+f+"-of-00072.bin")
        for i in range(len(pnames)):
            if 'value.weight' in pnames[i] or 'h.weight' in pnames[i]: # or "dense.weight" in pnames[i]:
                prebloc = parms['h.'+str(self.numLayer)+'.'+pnames[i]] # keep them in bf16 !
            else:
                prebloc = parms['h.'+str(self.numLayer)+'.'+pnames[i]].to(dtype=torch.float32)
                del parms['h.'+str(self.numLayer)+'.'+pnames[i]]
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
        # print("calling forward layer",self.numLayer,self.hasParms, hidden_states.device)
        # print("layer input",torch.norm(hidden_states).item())

        t0 = time.time()
        if self.hasParms:
            y0 = super().forward(hidden_states, alibi, attention_mask, layer_past, head_mask, use_cache=False, output_attentions=False)
            if self.latentOutputs!=None: self.latentOutputs.store(y0[0],keepgrad=self.keepgraph)
            y = (y0[0], attention_mask)
        elif self.latentOutputs!=None:
            h = self.latentOutputs.get()
            if h==None: y=(hidden_states, attention_mask)
            else: y = (h, attention_mask)
        else:
            # when the layer is empty, just pass the input unchanged
            y=(hidden_states, attention_mask)
        t1 = time.time()
        if self.hasParms: print("TIME in 1 layer",t1-t0)
        return y

def initModel():
    t0 = time.time()
    model=None
    with init_empty_weights():
        config = BloomConfig.from_pretrained(wd)
        transformers.models.bloom.modeling_bloom.BloomBlock = MyBloomBlock
        model = transformers.models.bloom.modeling_bloom.BloomForCausalLM(config)

    torch.nn.Embedding = MyEmbeddings(None)
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
    model.transformer.word_embeddings = MyEmbeddings(vparms)
    model.transformer.word_embeddings.isLoaded = True
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

# ###################################

model = initModel()
toker = transformers.models.bloom.tokenization_bloom_fast.BloomTokenizerFast.from_pretrained(wd)

def showLatents():
    if model.transformer.word_embeddings.latentOutputs != None: print("LAT","EE", model.transformer.word_embeddings.latentOutputs.tostr())
    for l in range(len(allblocks)):
        if allblocks[l].latentOutputs != None:
            print("LAT",l,allblocks[l].latentOutputs.tostr())

def run_forward():
    loadEmbeddings(model)
    with open("sentences.txt","r") as f:
        for l in f.readlines():
            prompt = toker(l)
            tokids = prompt['input_ids']
            if prefix>0: tokids = [0]*prefix+tokids
            x = torch.LongTensor([tokids])
            out = model(x)
    showLatents()
    model.transformer.word_embeddings.emptyLayer()
    model.lm_head.emptyLayer()

    # for l in range(len(allblocks)):
    for l in range(1):
        allblocks[l].loadLayer()
        allblocks[l].saveOutputs(True)
        with open("sentences.txt","r") as f:
            for s in f.readlines():
                prompt = toker(s)
                tokids = prompt['input_ids']
                if prefix>0: tokids = [0]*prefix+tokids
                x = torch.LongTensor([tokids])
                allblocks[l].keepgraph=False
                out = model(x)
        showLatents()
        allblocks[l].emptyLayer()
        allblocks[l].saveOutputs(False)

    loadLMHead(model)
    with open("sentences.txt","r") as f:
        for ui,s in enumerate(f.readlines()):
            prompt = toker(s)
            tokids = prompt['input_ids']
            if prefix>0: tokids = [0]*prefix+tokids
            x = torch.LongTensor([tokids])
            labels = x.clone()
            out = model(x,labels=labels)
            print(ui,s)
            print("LOSS",ui,out.loss.view(-1))

def run_backward():
    loadEmbeddings(model)
    with open("sentences.txt","r") as f:
        for l in f.readlines():
            prompt = toker(l)
            tokids = prompt['input_ids']
            if prefix>0: tokids = [0]*prefix+tokids
            x = torch.LongTensor([tokids])
            out = model(x)
    showLatents()
    model.transformer.word_embeddings.emptyLayer()
    model.lm_head.emptyLayer()

    for l in range(len(allblocks)):
        allblocks[l].loadLayer()
        allblocks[l].saveOutputs(True)
        with open("sentences.txt","r") as f:
            for s in f.readlines():
                prompt = toker(s)
                tokids = prompt['input_ids']
                if prefix>0: tokids = [0]*prefix+tokids
                x = torch.LongTensor([tokids])
                allblocks[l].keepgraph=False
                out = model(x)

                if l==0:
                    outl1 = allblocks[l].latentOutputs.latent[0]
                    print("outl1",outl1.shape,outl1.requires_grad)
                    outl1.retain_grad()
                    loss = torch.norm(outl1)
                    print("loss",loss.item())
                    loss.backward()
                    prefixparms = model.transformer.word_embeddings.prefv
                    print("prefixgrad",prefixparms.requires_grad,prefixparms.grad)
                    print("latentgrad",outl1.requires_grad,outl1.grad)
                    exit()
        showLatents()
        allblocks[l].emptyLayer()
        allblocks[l].saveOutputs(False)

    loadLMHead(model)
    with open("sentences.txt","r") as f:
        for ui,s in enumerate(f.readlines()):
            prompt = toker(s)
            tokids = prompt['input_ids']
            if prefix>0: tokids = [0]*prefix+tokids
            x = torch.LongTensor([tokids])
            labels = x.clone()
            out = model(x,labels=labels)
            print(ui,s)
            print("LOSS",ui,out.loss.view(-1))


def run_free_utts():
    run_forward()
    # run_backward()

"""
def run_BoolQ():
    from datasets import load_dataset
    from promptsource.templates import DatasetTemplates

    global filesuffix
    pro = DatasetTemplates('super_glue/boolq')
    # just pick the first prompt for now TODO: sample randomly prompts
    protemp = list(pro.templates.values())[0]

    dataset = load_dataset('boolq',split="validation[0:100]")
    # start by loading the first layer in RAM and process all sentences through the first layer
    allblocks[0].loadLayer()
    for ui,ex in enumerate(dataset):
        filesuffix = "."+str(ui)
        prompt = toker(protemp.apply(ex)[0])
        x = torch.LongTensor([prompt['input_ids']])
        out = model(x) # save layer 0 output to disk
    allblocks[0].emptyLayer()

    # then do the same for second layer, and then 3rd...
    for i in range(1,len(allblocks)-1):
        # reload the input to the i^th layer from disk
        allblocks[i].loadInputs = True
        allblocks[i].loadLayer()
        for ui,ex in enumerate(dataset):
            filesuffix = "."+str(ui)
            prompt = toker(protemp.apply(ex)[0])
            x = torch.LongTensor([prompt['input_ids']])
            out = model(x) # save layer i output to disk
        allblocks[i].emptyLayer()
        allblocks[i].loadInputs = False

    # finally pass penultimate input into the last layer and get answers
    nok,ntot=0,0
    allblocks[-1].loadInputs = True
    allblocks[-1].loadLayer()
    for ui,ex in enumerate(dataset):
        filesuffix = "."+str(ui)
        l = protemp.apply(ex)[0]
        prompt = toker(l)
        x = torch.LongTensor([prompt['input_ids']])
        out = model(x)
        logits = out.logits.view(-1)

        # reporting:
        print("prompt",ui,l)
        print("GOLD",ex['answer'],ui)
        print("yes",logits[18260].item(),ui)
        print("no",logits[654].item(),ui)
        truesc, falsesc = logits[17867].item(), logits[32349].item()
        print("True",truesc,ui)
        print("False",falsesc,ui)
        # let's go slightly beyond yes/no questions...
        besttok = torch.argmax(logits).item()
        print("maxtoken",logits[besttok].item(),besttok,ui)

        denom = (truesc+falsesc)
        truesc /= denom
        falsesc /= denom
        ntot+=1
        if ex['answer'] and truesc>=0.5: nok+=1
        elif (not ex['answer']) and falsesc>0.5: nok+=1
        acc = float(nok)/float(ntot)
        print("ACC",acc,nok,ntot)
    allblocks[-1].emptyLayer()
    allblocks[-1].loadInputs = False

    # token of 'yes': 18260
    # token of 'no' : 654
    # token of 'True':  17867
    # token of 'False': 32349
"""

t0 = time.time()
run_free_utts()
# run_BoolQ()
t1 = time.time()
print("total time required",t1-t0)
