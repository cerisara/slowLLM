# je reprends le code slowBloom et je le nettoie pour tourner sur CPU

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
wd = "/home/xtof/models/bloomz/"
wd = "/home/xtof/nas1/TALC/Synalp/Models/bloomz/"
wd = "/home/xtof/nvme/bloomz/"

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

def _draw_graph(var, watch=[], seen=[], indent="", pobj=None):
    from rich import print
    if hasattr(var, "next_functions"):
        for fun in var.next_functions:
            joy = fun[0]
            if joy is not None:
                if joy not in seen:
                    label = str(type(joy)).replace("class", "").replace("'", "").replace(" ", "")
                    label_graph = label
                    colour_graph = ""
                    seen.append(joy)
                    if hasattr(joy, 'variable'):
                        happy = joy.variable
                        if happy.is_leaf:
                            label += " \U0001F343"
                            colour_graph = "green"                            
                            for (name, obj) in watch:
                                if obj is happy:
                                    label += " \U000023E9 " + "[b][u][color=#FF00FF]" + name + "[/color][/u][/b]"
                                    label_graph += name                                    
                                    colour_graph = "blue"
                                    break                            
                                vv = [str(obj.shape[x]) for x in range(len(obj.shape))]
                            label += " [["
                            label += ', '.join(vv)
                            label += "]]"
                            label += " " + str(happy.var())                    
                    print(indent + label)
                    _draw_graph(joy, watch, seen, indent + ".", joy)

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

# just for the last lm_head ??
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
        self.keepgraph=False
        self.passthru = False

    def setPoids(self,poids):
        self.isLoaded = True
        self.weight = torch.nn.Parameter(poids.to(dtype=torch.bfloat16),requires_grad=False)

    def forward(self, x):
        if self.isLoaded and not self.passthru:
            e=super().forward(x)
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
        # attention: les fichiers sur JZ ne sont pas comme chez moi: TODO il faudrait utiliser le json!
        f = "0000"+str(self.numLayer+2) if self.numLayer<8 else "000"+str(self.numLayer+2)
        print("loading",f)
        parms = torch.load(wd+"pytorch_model_"+f+"-of-00072.bin")
        for i in range(len(pnames)):
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


def showLatents():
    if model.transformer.word_embeddings.latentOutputs != None: print("LAT","EE", model.transformer.word_embeddings.latentOutputs.tostr())
    for l in range(len(allblocks)):
        if allblocks[l].latentOutputs != None:
            print("LAT",l,allblocks[l].latentOutputs.tostr())

def run_forward_oneutt(utt):
    model.transformer.word_embeddings.passthru = False
    model.lm_head.passthru = False
    losses = []
    for l in range(len(allblocks)):
        allblocks[l].latentOutputs = LatentOutputs()
        allblocks[l].passthru = False
    allblocks[-1].keepgraph=True
    prompt = toker(utt)
    tokids = prompt['input_ids']
    x = torch.LongTensor([tokids])
    labels = x.clone()
    out = model(x,labels=labels)
    losses.append(out.loss.view(-1))
    print("LOSS",out.loss.view(-1))
    # showLatents()

    # prepare backward: if we only want forward, we shouldn't set requires_grad here
    outl1 = allblocks[-1].latentOutputs.latent[0]
    outl1.requires_grad=True
    print("last activ checkpoint",outl1.device,torch.norm(outl1).item())
    return losses,x

def run_backward(losses,x):
    # faut-il detach x du graphe ?
    # TODO: fix for multiple sentences
    outl1 = allblocks[-1].latentOutputs.latent.pop(0)
    print("latentnode",outl1.storage().data_ptr(),outl1.requires_grad)
    outl1.retain_grad()
    loss = losses[0]
    loss.backward()
    latentgrad=outl1.grad
    print("latentgrad",latentgrad)
    del outl1
    print("just computed grad at the output of layer",len(allblocks)-1,torch.norm(latentgrad),latentgrad.shape)

    model.transformer.word_embeddings.passthru = True
    model.lm_head.passthru = True
    for l in range(len(allblocks)-1,0,-1):
        allblocks[l].latentOutputs = LatentOutputs()
        for ll in range(len(allblocks)): allblocks[ll].passthru = True
        allblocks[l].passthru = False
        allblocks[l].keepgraph=True
        inl1 = allblocks[l-1].latentOutputs.latent[0]
        inl1.requires_grad=True
        model(x)
        inl1.retain_grad()
        outl = allblocks[l].latentOutputs.latent.pop(0)
        outl.backward(latentgrad,inputs=(inl1,))
        latentgrad = inl1.grad
        del inl1
        print("just computed grad",l-1,torch.norm(latentgrad),latentgrad.shape)
        allblocks[l].keepgraph=False

    l=0
    allblocks[l].latentOutputs = LatentOutputs()
    for ll in range(len(allblocks)): allblocks[ll].passthru = True
    allblocks[l].passthru = False
    model.transformer.word_embeddings.passthru=False
    model.transformer.word_embeddings.keepgraph=True
    allblocks[l].keepgraph=True
    inl1 = model.transformer.word_embeddings.prefv
    inl1.requires_grad=True
    model(x)
    inl1.retain_grad()
    outl = allblocks[l].latentOutputs.latent.pop(0)
    outl.backward(latentgrad,inputs=(inl1,))
    latentgrad = inl1.grad
    print("just computed grad at the output of embeddings",torch.norm(latentgrad),latentgrad.shape)
    latentgrad = model.transformer.word_embeddings.prefv.grad
    model.transformer.word_embeddings.keepgraph=False
    allblocks[l].keepgraph=False
    del inl1

def save_prefix():
    torch.save(model.transformer.word_embeddings.prefv,"prefv.pt")

class QtLinear(torch.nn.Linear):
    def quantize(self):
        # TODO
        # replace WX by softmax(Theta)VX during training
        # and in the end by V[i]X; so replace 2S|W| by:
        # if V in 4bits: 4*128+S|W|/2
        # if V in 8bits: 4*256+S|W|
        # where S = nb of Linear that shares the same V
        # V = set of quantized values (|V|=2,4,8,16,...,128), shared across Linears? Layers?
        # Theta = 1 theta per value; will be removed after training (0B)
        # when S=1, we already gain a lot: 2|W| >> 4*128+|W|/2 so we can store a specific V per Linear
        pass

    def forward(self,x):
        # OK, j'ai verifie que ces 2 lignes sont bien la meme chose:
        # y= torch.nn.functional.linear(x, self.weight, self.bias)
        y = self.bias + x @ self.weight.t()
        print("detlinear",x.shape,y.norm(),y.shape)
        return y

# ###################################

model = initModel()
loadLMHead(model)

allblocks[0].loadLayer()
# indique la RAM en kB
print("one layer loaded",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
for n,p in allblocks[0].named_parameters():
    print(n,p.shape,p.device,p.dtype)

# just compute the output of layer 1
model.transformer.word_embeddings.passthru = False
model.lm_head.passthru = True
allblocks[0].latentOutputs = LatentOutputs()
allblocks[0].passthru = False
allblocks[-1].keepgraph=False

linear0 = allblocks[0].self_attention.query_key_value
linear0.__class__ = QtLinear

toker = transformers.models.bloom.tokenization_bloom_fast.BloomTokenizerFast.from_pretrained(wd)
utt = "the time has come to apologize to Nature"
prompt = toker(utt)
tokids = prompt['input_ids']
x = torch.LongTensor([tokids])
out = model(x)
print("LAT",allblocks[0].latentOutputs.tostr())

print("press enter to finish")
input()


# debug
# allblocks = allblocks[0:18]

