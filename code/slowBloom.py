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
wd = "/home/xtof/nas1/TALC/Synalp/Models/bloomz/"

# subdirectory that will contain the outputs at every layer (does not need to be emptied)
tmpdir = "tmpdir/"
Path(tmpdir).mkdir(parents=True, exist_ok=True)

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

class MyLinear(torch.nn.Linear):
    def __init__(self, *args):
        super().__init__(1,1)
        self.isLoaded = False

    def forward(self, x):
        if self.isLoaded:
            return super().forward(x)
        # I dont know exactly the lexicon size, but I only query yes/no anyway so... TODO: fix that!
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

    def loadLayer(self):
        print("load weights from disk")
        t0 = time.time()
        f = "0000"+str(self.numLayer+2) if self.numLayer<8 else "000"+str(self.numLayer+2)
        parms = torch.load(wd+"pytorch_model_"+f+"-of-00072.bin")
        for i in range(len(pnames)):
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
        print("layer input",torch.norm(hidden_states).item())

        t0 = time.time()
        if self.hasParms:
            if self.loadInputs:
                hidden_states = torch.load(tmpdir+"layerout."+str(self.numLayer-1)+filesuffix)
                print("loading input",self.numLayer-1,torch.norm(hidden_states).item())
            y0 = super().forward(hidden_states, alibi, attention_mask, layer_past, head_mask, use_cache=False, output_attentions=False)
            self.saveOutputs(y0)
            y = (y0[0], attention_mask)
        else:
            # when the layer is empty, just pass the input unchanged
            y=(hidden_states, attention_mask)
        t1 = time.time()
        # print("called forward",self.numLayer,len(y),t1-t0,self.hasParms,self.loadInputs)
        return y

    def saveOutputs(self,y):
        torch.save(y[0],tmpdir+"layerout."+str(self.numLayer)+filesuffix)
        # what's next in this tuple ?

def initModel():
    t0 = time.time()
    model=None
    with init_empty_weights():
        config = BloomConfig.from_pretrained(wd)
        transformers.models.bloom.modeling_bloom.BloomBlock = MyBloomBlock
        model = transformers.models.bloom.modeling_bloom.BloomForCausalLM(config)

    # we keep the embeddings and final head always in memory: TODO: free them to reduce RAM requirements
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
    vparms = parms['word_embeddings.weight'].to(dtype=torch.float32)
    model.transformer.word_embeddings = torch.nn.Embedding.from_pretrained(vparms,freeze=True)
    model.transformer.word_embeddings.isLoaded = True
    vparms = parms['word_embeddings_layernorm.weight'].to(dtype=torch.float32)
    model.transformer.word_embeddings_layernorm.weight = torch.nn.Parameter(vparms,requires_grad=False)
    vparms = parms['word_embeddings_layernorm.bias'].to(dtype=torch.float32)
    model.transformer.word_embeddings_layernorm.bias = torch.nn.Parameter(vparms,requires_grad=False)
    model.lm_head.weight = model.transformer.word_embeddings.weight
    model.lm_head.isLoaded = True
    print("embeddings loaded","RAM",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    del parms
    gc.collect()
    print("embeddings loaded","RAM",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

def loadLMHead(model):
    parms = torch.load(wd+"pytorch_model_00072-of-00072.bin")
    vparms = parms['ln_f.weight'].to(dtype=torch.float32)
    model.transformer.ln_f.weight = torch.nn.Parameter(vparms,requires_grad=False)
    vparms = parms['ln_f.bias'].to(dtype=torch.float32)
    model.transformer.ln_f.bias = torch.nn.Parameter(vparms,requires_grad=False)
    del parms
    gc.collect()
    print("LMhead loaded","RAM",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

# ###################################

toker = transformers.models.bloom.tokenization_bloom_fast.BloomTokenizerFast.from_pretrained(wd)
model = initModel()
loadEmbeddings(model)
loadLMHead(model)

def test_end():
    with open("sentences.txt","r") as f: lines = f.readlines()
    utt = lines[0]
    prompt = toker(utt)
    x = torch.LongTensor([prompt['input_ids']])

    allblocks[0].loadLayer()
    out = model(x)
    logits = out.logits.view(-1)
    print("yes",logits[18260].item())
    allblocks[0].emptyLayer()

    for i in range(1,len(allblocks)-1):
        allblocks[i].loadInputs = True
        allblocks[i].loadLayer()
        out = model(x)
        logits = out.logits.view(-1)
        print("yes",logits[18260].item())
        allblocks[i].emptyLayer()
        allblocks[i].loadInputs = False

    allblocks[-1].loadInputs = True
    allblocks[-1].loadLayer()
    out = model(x)
    logits = out.logits.view(-1)
    print("yes",logits[18260].item())
    allblocks[-1].emptyLayer()
    allblocks[-1].loadInputs = False


def run_test_0():
    global filesuffix
    # start by loading the first layer in RAM and process all sentences through the first layer
    allblocks[0].loadLayer()
    with open("sentences.txt","r") as f:
        for ui,l in enumerate(f.readlines()):
            filesuffix = "."+str(ui)
            prompt = toker(l)
            x = torch.LongTensor([prompt['input_ids']])
            out = model(x) # save layer 0 output to disk
    allblocks[0].emptyLayer()

    # then do the same for second layer, and then 3rd...
    for i in range(1,len(allblocks)-1):
        # reload the input to the i^th layer from disk
        allblocks[i].loadInputs = True
        allblocks[i].loadLayer()
        with open("sentences.txt","r") as f:
            for ui,l in enumerate(f.readlines()):
                filesuffix = "."+str(ui)
                prompt = toker(l)
                x = torch.LongTensor([prompt['input_ids']])
                out = model(x) # save layer i output to disk
        allblocks[i].emptyLayer()
        allblocks[i].loadInputs = False

    # finally pass penultimate input into the last layer and get answers
    allblocks[-1].loadInputs = True
    allblocks[-1].loadLayer()
    with open("sentences.txt","r") as f:
        for ui,l in enumerate(f.readlines()):
            filesuffix = "."+str(ui)
            prompt = toker(l)
            x = torch.LongTensor([prompt['input_ids']])
            print("prompt",l)
            out = model(x)
            logits = out.logits.view(-1)
            print("yes",logits[18260].item(),ui)
            print("no",logits[654].item(),ui)
            # let's go slightly beyond yes/no questions...
            besttok = torch.argmax(logits).item()
            print("maxtoken",logits[besttok].item(),besttok,ui)
    allblocks[-1].emptyLayer()
    allblocks[-1].loadInputs = False

    # token of 'yes': 18260
    # token of 'no' : 654

    # output of a block is a tuple with:
    # - hidden_states
    # - optional: presents
    # - optional: self-att

def run_BoolQ():
    from datasets import load_dataset
    from promptsource.templates import DatasetTemplates

    global filesuffix
    pro = DatasetTemplates('super_glue/boolq')
    # just pick the first prompt for now TODO: sample randomly prompts
    protemp = list(pro.templates.values())[0]

    dataset = load_dataset('boolq',split="validation[150:250]")
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

test_end()
# run_BoolQ()

