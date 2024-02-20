import time
import resource
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BloomConfig
from accelerate import init_empty_weights
from typing import Optional, Tuple, Union
import torch
import gc
from pathlib import Path
import random

# Put here the directory where you downloaded the Bloom's parameters
wd = "/home/xtof/nas1/TALC/Synalp/Models/bloom/bloom/"
wd = "/media/xtof/556E99561C655FA8/bloomz/"
wd = "/mnt/dos/xtof/"
wd = "/home/xtof/nas1/TALC/Synalp/Models/bloomz/"
wd = "/home/xtof/nvme/bloomz/"

prefix = 5
niters = 100
LR = 0.1
pruning_sparsity = 0.4

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
# TODO: it keeps both matrices in RAM, could avoid that?
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
            # matmul in bf16 takes 12s for 3 tokens because it only uses 1 core, while it uses all cores in fp32
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
        self.getidx = -1

    def store(self,z,keepgraph=False):
        # this is called once per utterance
        if self.dostore:
            if keepgraph: self.latent.append(z)
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

    def setPoids(self,poids):
        self.isLoaded = True
        self.weight = torch.nn.Parameter(poids.to(dtype=torch.bfloat16),requires_grad=False)

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
                print("pass in embeds prefv",self.prefv.requires_grad)
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
        self.dopruning = False

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
            if self.latentOutputs!=None: self.latentOutputs.store(y0[0],keepgraph=self.keepgraph)
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


def showLatents(model):
    if model.transformer.word_embeddings.latentOutputs != None: print("LAT","EE", model.transformer.word_embeddings.latentOutputs.tostr())
    for l in range(len(allblocks)):
        if allblocks[l].latentOutputs != None:
            print("LAT",l,allblocks[l].latentOutputs.tostr())

allutts = None
numinput = 0
def getInputs():
    global allutts
    global numinput
    if allutts==None:
        with open("sentences.txt","r") as f: lines = f.readlines()
        utts,s = [],""
        for l in lines:
            if l[0]=='\n' and len(s)>0:
                utts.append(s)
                s=""
            else: s+=l
        if len(s)>0: utts.append(s)
        random.shuffle(utts)
        allutts = utts
    oneutt = allutts[numinput:numinput+1]
    print("training utts:",oneutt)
    return oneutt

def nextInput():
    global allutts
    global numinput
    numinput+=1
    if numinput>len(allutts)-1: numinput=0

def run_forward(model, toker, utts, prepare_backward=False):
    loadEmbeddings(model)
    model.transformer.word_embeddings.keepgraph=False
    for s in utts:
        prompt = toker(s)
        tokids = prompt['input_ids']
        if prefix>0: tokids = [0]*prefix+tokids
        x = torch.LongTensor([tokids])
        out = model(x)
    # showLatents()
    model.transformer.word_embeddings.emptyLayer()
    model.lm_head.emptyLayer()

    for l in range(len(allblocks)):
        allblocks[l].loadLayer()
        if allblocks[l].dopruning: magnitude_pruning(allblocks[l])
        allblocks[l].saveOutputs(True)
        for s in utts:
            prompt = toker(s)
            tokids = prompt['input_ids']
            if prefix>0: tokids = [0]*prefix+tokids
            x = torch.LongTensor([tokids])
            allblocks[l].keepgraph=False
            out = model(x)
        # showLatents()
        allblocks[l].emptyLayer()
        allblocks[l].saveOutputs(False)

    if prepare_backward:
        for vout in allblocks[-1].latentOutputs.latent: vout.requires_grad=True

    loadLMHead(model)
    losses = []
    for ui,s in enumerate(utts):
        prompt = toker(s)
        tokids = prompt['input_ids']
        if prefix>0: tokids = [0]*prefix+tokids
        x = torch.LongTensor([tokids])
        # we can just clone the labels, they're shifted in the forward pass: https://github.com/seungeunrho/transformers/blob/988fab92806dc8db0b0218018ee5a582f4545193/src/transformers/models/bloom/modeling_bloom.py#L907
        labels = x.clone()
        out = model(x,labels=labels)
        losses.append(out.loss.view(-1))
        print("LOSS",ui,out.loss.view(-1),s)
    return losses

def run_backward(model, losses,nit):
    latentgrad = []
    for i in range(len(allblocks[-1].latentOutputs.latent)):
        outl1 = allblocks[-1].latentOutputs.latent.pop(0)
        outl1.retain_grad()
        loss = losses[i]
        loss.backward()
        latentgrad.append(outl1.grad)
        del outl1
        print("just computed grad at the output of layer",len(allblocks)-1,torch.norm(latentgrad[i]),latentgrad[i].shape)

    model.transformer.word_embeddings.emptyLayer()
    model.transformer.word_embeddings.keepgraph=False
    model.lm_head.emptyLayer()
    for l in range(len(allblocks)-1,0,-1):
        allblocks[l].loadLayer()
        allblocks[l].saveOutputs(True) # reset the latents of this layer
        for si,s in enumerate(getInputs()):
            prompt = toker(s)
            tokids = prompt['input_ids']
            if prefix>0: tokids = [0]*prefix+tokids
            x = torch.LongTensor([tokids])
            allblocks[l].keepgraph=True
            inl1 = allblocks[l-1].latentOutputs.latent[si]
            inl1.requires_grad=True
            model(x)
            inl1.retain_grad()
            outl = allblocks[l].latentOutputs.latent.pop(0)
            outl.backward(latentgrad[si],inputs=(inl1,))
            latentgrad[si] = inl1.grad
            del inl1
            print("just computed grad",l-1,si,torch.norm(latentgrad[si]),latentgrad[si].shape)
            allblocks[l].keepgraph=False
        allblocks[l].emptyLayer()

    l=0
    allblocks[l].loadLayer()
    allblocks[l].saveOutputs(True) # reset the latents of this layer
    for si,s in enumerate(getInputs()):
        prompt = toker(s)
        tokids = prompt['input_ids']
        if prefix>0: tokids = [0]*prefix+tokids
        x = torch.LongTensor([tokids])
        allblocks[l].keepgraph=True
        inl1 = model.transformer.word_embeddings.latentOutputs.latent[si]
        inl1.requires_grad=True
        model(x)
        inl1.retain_grad()
        outl = allblocks[l].latentOutputs.latent.pop(0)
        outl.backward(latentgrad[si],inputs=(inl1,))
        latentgrad[si] = inl1.grad
        del inl1
        print("just computed grad at the output of embeddings",torch.norm(latentgrad[si]),latentgrad[si].shape)
        allblocks[l].keepgraph=False
    allblocks[l].emptyLayer()

    loadEmbeddings(model) # also resets latentOutputs
    for si,s in enumerate(getInputs()):
        prompt = toker(s)
        tokids = prompt['input_ids']
        if prefix>0: tokids = [0]*prefix+tokids
        x = torch.LongTensor([tokids])
        model.transformer.word_embeddings.prefv.requires_grad=True
        model.transformer.word_embeddings.keepgraph=True
        model(x)
        model.transformer.word_embeddings.prefv.retain_grad()
        outl = model.transformer.word_embeddings.latentOutputs.latent.pop(0)
        outl.backward(latentgrad[si],inputs=(model.transformer.word_embeddings.prefv,))
        latentgrad[si] = model.transformer.word_embeddings.prefv.grad
        print("just computed grad in the prefix",torch.norm(latentgrad[si]),latentgrad[si].shape)
        torch.save(model.transformer.word_embeddings.prefv,"prefv_"+str(nit)+".pt")

def wikitextPerplexity(model):
    # perplexity of BloomZ here on Wikitext-2-raw test is 198, which is too high...
    # on C4 perplexity is about 38, still quite high...

    # comparison with bloomz-7b on Jean Zay
    # bloomz-7b = PPL(C4) = 27
    # bloom-7b = PPL(C4) = 22
    # mistral-7b = PPL(C4) = 12
    # TODO: rather use bits-per-byte to compare across tokenizers!!

    # mistral-7b PPL(Wikitext-all) = 26.6
    # bloom-7b PPL(Wikitext-all) = 55 (plus 7 nan)
    # bloomz-7b PPL(Wikitext-all) = 58.9 (plus 7 nan)

    # perplexity des 10 premieres utts de wikitext avec bloomz-7b: 
    # 4456.86
    # 286.288
    # 109.33
    # 111.718
    # 96.9563
    # 77.2953
    # 64.8941
    # 65.9469
    # 56.8611
    # 51.063

    # et avec slow-Bloomz-176b: (certaines phrases sont tres mal predites, la plupart des autres bien mieux !)
    # !!! en fait, ce sont les phrases courtes, les "titres markdown", qui sont tres mal predites !
    # 
    # base      MP 10%      MP 20%      MP 30%      MP 40%      MP 50%
    # 785048
    # 25.1058   24.1673     23.3851     23.7884     22.0806     26.0678
    # 16.1045   15.6928     14.7302     15.2427     15.3023     17.3831
    # 237139    
    # 36640.2
    # 30.487    29.2389     30.4717     31.3935     34.8168     38.5286
    # 25.8084   25.3759     25.2443     25.3581     24.2181     28.1121
    # 27617.4
    # 18.8705   18.6044     17.8785     17.4772     18.1233     20.9513
    # 19.5916   19.2132     18.4784     19.0907     19.244      22.9496
    #
    # conclusion: it's clear that the reasonable limit is around 40% for unstructured magnitude pruning, as stated in the litarature

    utts = []
    if True:
        utts = ["Robert Boulter is an English film , television and theatre actor . He had a guest @-@ starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . He had a guest role in the television series Judge John Deed in 2002 . In 2004 Boulter landed a role as \" Craig \" in the episode \" Teddy 's Story \" of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi . He was cast in the 2005 theatre productions of the Philip Ridley play Mercury Fur , which was performed at the Drum Theatre in Plymouth and the Menier Chocolate Factory in London . He was directed by John Tiffany and starred alongside Ben Whishaw , Shane Zaza , Harry Kent , Fraser Ayres , Sophie Stanton and Dominic Hall .", "In 2006 , Boulter starred alongside Whishaw in the play Citizenship written by Mark Ravenhill . He appeared on a 2006 episode of the television series , Doctors , followed by a role in the 2007 theatre production of How to Curse directed by Josie Rourke . How to Curse was performed at Bush Theatre in the London Borough of Hammersmith and Fulham . Boulter starred in two films in 2008 , Daylight Robbery by filmmaker Paris Leonti , and Donkey Punch directed by Olly Blackburn . In May 2008 , Boulter made a guest appearance on a two @-@ part episode arc of the television series Waking the Dead , followed by an appearance on the television series Survivors in November 2008 . He had a recurring role in ten episodes of the television series Casualty in 2010 , as \" Kieron Fletcher \" . Boulter starred in the 2011 film Mercenaries directed by Paris Leonti .", "In 2000 Boulter had a guest @-@ starring role on the television series The Bill ; he portrayed \" Scott Parry \" in the episode , \" In Safe Hands \" . Boulter starred as \" Scott \" in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . A review of Boulter 's performance in The Independent on Sunday described him as \" horribly menacing \" in the role , and he received critical reviews in The Herald , and Evening Standard . He appeared in the television series Judge John Deed in 2002 as \" Addem Armitage \" in the episode \" Political Expediency \" , and had a role as a different character \" Toby Steele \" on The Bill .", "He had a recurring role in 2003 on two episodes of The Bill , as character \" Connor Price \" . In 2004 Boulter landed a role as \" Craig \" in the episode \" Teddy 's Story \" of the television series The Long Firm ; he starred alongside actors Mark Strong and Derek Jacobi . Boulter starred as \" Darren \" , in the 2005 theatre productions of the Philip Ridley play Mercury Fur . It was performed at the Drum Theatre in Plymouth , and the Menier Chocolate Factory in London . He was directed by John Tiffany and starred alongside Ben Whishaw , Shane Zaza , Harry Kent , Fraser Ayres , Sophie Stanton and Dominic Hall . Boulter received a favorable review in The Daily Telegraph : \" The acting is shatteringly intense , with wired performances from Ben Whishaw ( now unrecognisable from his performance as Trevor Nunn 's Hamlet ) , Robert Boulter , Shane Zaza and Fraser Ayres . \" The Guardian noted , \" Ben Whishaw and Robert Boulter offer tenderness amid the savagery . \"", "In 2006 Boulter starred in the play Citizenship written by Mark Ravenhill . The play was part of a series which featured different playwrights , titled Burn / Chatroom / Citizenship . In a 2006 interview , fellow actor Ben Whishaw identified Boulter as one of his favorite co @-@ stars : \" I loved working with a guy called Robert Boulter , who was in the triple bill of Burn , Chatroom and Citizenship at the National . He played my brother in Mercury Fur . \" He portrayed \" Jason Tyler \" on the 2006 episode of the television series , Doctors , titled \" Something I Ate \" . Boulter starred as \" William \" in the 2007 production of How to Curse directed by Josie Rourke . How to Curse was performed at Bush Theatre in the London Borough of Hammersmith and Fulham . In a review of the production for The Daily Telegraph , theatre critic Charles Spencer noted , \" Robert Boulter brings a touching vulnerability to the stage as William . \"", "Boulter starred in two films in 2008 , Daylight Robbery by filmmaker Paris Leonti , and Donkey Punch directed by Olly Blackburn . Boulter portrayed a character named \" Sean \" in Donkey Punch , who tags along with character \" Josh \" as the \" quiet brother ... who hits it off with Tammi \" . Boulter guest starred on a two @-@ part episode arc \" Wounds \" in May 2008 of the television series Waking the Dead as character \" Jimmy Dearden \" . He appeared on the television series Survivors as \" Neil \" in November 2008 . He had a recurring role in ten episodes of the television series Casualty in 2010 , as \" Kieron Fletcher \" . He portrayed an emergency physician applying for a medical fellowship . He commented on the inherent difficulties in portraying a physician on television : \" Playing a doctor is a strange experience . Pretending you know what you 're talking about when you don 't is very bizarre but there are advisers on set who are fantastic at taking you through procedures and giving you the confidence to stand there and look like you know what you 're doing . \" Boulter starred in the 2011 film Mercenaries directed by Paris Leonti ."]
        # from datasets import load_dataset
        # dataset = load_dataset("wikitext",'wikitext-2-raw-v1')
        # dataset = dataset['test']
        # for i in range(len(dataset)):
        #     s=dataset[i]['text'].strip()
        #     if len(s)>0: utts.append(s)
        # random.shuffle(utts)
    else:
        # rather test on C4
        utts.append("Beginners BBQ Class Taking Place in Missoula! Do you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers. He will be teaching a beginner level class for everyone who wants to get better with their culinary skills. He will teach you everything you need to know to compete in a KCBS BBQ competition, including techniques, recipes, timelines, meat selection and trimming, plus smoker and fire information. The cost to be in the class is $35 per person, and for spectators it is free. Included in the cost will be either a t-shirt or apron and you will be tasting samples of each meat that is prepared.")
        utts.append("Discussion in 'Mac OS X Lion (10.7)' started by axboi87, Jan 20, 2012. I've got a 500gb internal drive and a 240gb SSD. When trying to restore using disk utility i'm given the error \"Not enough space on disk ____ to restore\" But I shouldn't have to do that!!! Any ideas or workarounds before resorting to the above? Use Carbon Copy Cloner to copy one drive to the other. I've done this several times going from larger HDD to smaller SSD and I wound up with a bootable SSD drive. One step you have to remember not to skip is to use Disk Utility to partition the SSD as GUID partition scheme HFS+ before doing the clone. If it came Apple Partition Scheme, even if you let CCC do the clone, the resulting drive won't be bootable. CCC usually works in \"file mode\" and it can easily copy a larger drive (that's mostly empty) onto a smaller drive. If you tell CCC to clone a drive you did NOT boot from, it can work in block copy mode where the destination drive must be the same size or larger than the drive you are cloning from (if I recall). I've actually done this somehow on Disk Utility several times (booting from a different drive (or even the dvd) so not running disk utility from the drive your cloning) and had it work just fine from larger to smaller bootable clone. Definitely format the drive cloning to first, as bootable Apple etc.. Thanks for pointing this out. My only experience using DU to go larger to smaller was when I was trying to make a Lion install stick and I was unable to restore InstallESD.dmg to a 4 GB USB stick but of course the reason that wouldn't fit is there was slightly more than 4 GB of data.")
        utts.append("Foil plaid lycra and spandex shortall with metallic slinky insets. Attached metallic elastic belt with O-ring. Headband included. Great hip hop or jazz dance costume. Made in the USA.")
        utts.append("How many backlinks per day for new site? Discussion in 'Black Hat SEO' started by Omoplata, Dec 3, 2010. 1) for a newly created site, what's the max # backlinks per day I should do to be safe? 2) how long do I have to let my site age before I can start making more blinks? I did about 6000 forum profiles every 24 hours for 10 days for one of my sites which had a brand new domain. There is three backlinks for every of these forum profile so thats 18 000 backlinks every 24 hours and nothing happened in terms of being penalized or sandboxed. This is now maybe 3 months ago and the site is ranking on first page for a lot of my targeted keywords. build more you can in starting but do manual submission and not spammy type means manual + relevant to the post.. then after 1 month you can make a big blast.. Wow, dude, you built 18k backlinks a day on a brand new site? How quickly did you rank up? What kind of competition/searches did those keywords have?")
        utts.append("The Denver Board of Education opened the 2017-18 school year with an update on projects that include new construction, upgrades, heat mitigation and quality learning environments. We are excited that Denver students will be the beneficiaries of a four year, $572 million General Obligation Bond. Since the passage of the bond, our construction team has worked to schedule the projects over the four-year term of the bond. Denver voters on Tuesday approved bond and mill funding measures for students in Denver Public Schools, agreeing to invest $572 million in bond funding to build and improve schools and $56.6 million in operating dollars to support proven initiatives, such as early literacy. Denver voters say yes to bond and mill levy funding support for DPS students and schools. Click to learn more about the details of the voter-approved bond measure. Denver voters on Nov. 8 approved bond and mill funding measures for DPS students and schools. Learn more about what’s included in the mill levy measure.")
    toker = transformers.models.bloom.tokenization_bloom_fast.BloomTokenizerFast.from_pretrained(wd)
    tk = time.time()
    losses = run_forward(model, toker, utts[0:20])
    tl = time.time()
    ppl = torch.exp(torch.stack(losses).mean())
    print("time forward PPL",tl-tk, ppl, losses)

def train_soft_prompt(nit=0):
    model = initModel()
    toker = transformers.models.bloom.tokenization_bloom_fast.BloomTokenizerFast.from_pretrained(wd)
    print("train a soft prompt on sentences.txt")
    tk = time.time()
    losses = run_forward(model, toker, allutts)
    tl = time.time()
    print("time forward",tl-tk,losses)
    run_backward(model, losses,nit)
    tk = time.time()
    print("time backward",tk-tl)
    prefv0 = model.transformer.word_embeddings.prefv.clone()
    opt = torch.optim.SGD([model.transformer.word_embeddings.prefv], lr=LR)
    opt.step()
    print("delta_prefix",torch.norm(model.transformer.word_embeddings.prefv-prefv0).item())

def magnitude_pruning(b):
    print("pruning layer",b.numLayer,pruning_sparsity)
    with torch.no_grad():
        for n,p in b.named_parameters():
            if len(p.shape)==2:
                print(n,p.shape)
                # prune row-wise as suggested in Wanda paper
                metric = p.abs()
                _, sorted_idx = torch.sort(metric, dim=1)
                pruned_idx = sorted_idx[:,:int(p.shape[1] * pruning_sparsity)]
                p.scatter_(dim=1, index=pruned_idx, value=0)
                ra = torch.linalg.matrix_rank(p).item()
                print("pruning",p.requires_grad,ra)

# ###################################

model = initModel()
for b in allblocks: b.dopruning = True
wikitextPerplexity(model)

exit()


# debug
# allblocks = allblocks[0:2]

t0 = time.time()
for nit in range(niters):
    train_soft_prompt(model, nit)
    nextInput()
# run_inference()
t1 = time.time()
print("total time required",t1-t0)

