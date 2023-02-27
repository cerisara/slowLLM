import torch
import torch.nn as nn
import transformers
import pickle
from utils import *
import numpy as np
import threading
import time
import sys
import json
import psutil
from queue import Queue
import xtof
import os
import requests
from flask import Flask, render_template, request
import simple_websocket
import logging

"""
roadmap:
    X create API endpoint to register client
    X test GPT2 model dispatched on several physical nodes inside lab
    X open API to outside : OK ! il faut lancer sur lully: ssh -N olkihost -R 9487:152.81.7.89:5000
    X test GPT2 model dispatched on several physical nodes outside lab
    X may load several layers on one node
    X web page to get status
    X test (and switch to) GPT-Neox-20GB
    X enhancement: same layers on multiple nodes
    X BUG: generation crashes: second autoregres. step X not good shape
    X test simple generation with ws-collab on G5K !
    X client automatically load layer files
    X enable client to submit request to process
    X tune maximum length as prompt length
    - quit all clients when server die
    - test: 1 process per request to enable jobs to follow each other through layers
    - automate evaluation (like CI)
    - when client disconnect, stop and delete its running requests
    - implement as a pip library
    - test, test, test
    - release v0.1

    - enable fine-tuning
    - test finetuning with gpt-neox-20b

    - batch_size>1 ??? not needed when parallelizing process per request !!
        - bug: batch_size>1 fails
        - implement "pipeline parallelism": split X-batch to have all nodes working at the same time
    - implement karma-based submission queue
    - implement load-balancing
    - implement hybrid "local + federation" that adjusts dynamically to be as fast as possible
    - split by column instead of layers

----------------

SERVER:
- 1 thread lambda for the wss socket that serves the @app.routes
- 1 thread for "checkClients": rm clients that have no pong since more than 9 seconds
- launchGenerateTask() is launched independently: loops:
    - takes an UTT from jobq and feeds it to the model
- "newClient()" called when a client register via the @app.route("/gpt")
    - stores the Comm object that links the server to the client
    - wait for the end (don't send anything else to the client!)
- MyGPTNeoLayer sends a XNeoXMsg to the client when it gets some input
    - then it waits for an answer, and pass Y to the main model

CLIENT:
- 1 thread "sendthread" to send msgs 1 after the other through the link
- 1 thread "pingpong" to send "ping" every 2 seconds
- loops and gets msgs from the link: "X" or "R"
    - if "X": compute Y locally and send "Y" msg
    - if "R": TODO
- offers method "sendRequest" to push external sentences into the send queue

----------------
papier sur GPT-NeoX: https://arxiv.org/pdf/2204.06745.pdf

1er run complet des 44 layers de GPT-NeoX-20GB: OK !

----------------

La config suivante sur olkihost permet d'avoir both la page web http://fb.cerisara.fr/collabgpt
et le websocket (cf. client.py)
<VirtualHost *:80>
  ServerName fb.cerisara.fr
  ServerAdmin christophe.cerisara@loria.fr
  ProxyPass /status http://localhost:9487/status
  ProxyPassReverse /status http://localhost:9487/status
  ProxyPass /collabgpt http://localhost:9487/collabgpt
  ProxyPassReverse /collabgpt http://localhost:9487/collabgpt
  ProxyPass /gpt ws://localhost:9487/gpt
  ProxyPassReverse /gpt ws://localhost:9487/gpt

pour benchmarker le code, utiliser lm-evaluation-harness

"""

# pour loader les shards de GPT-NeoX il faut voir ligne 400 du fichier
# /home/xtof/git/github/transformers/src/transformers/modeling_utils.py

app = Flask(__name__)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

NLAYERS = 44
NLAYERS_ON_SERVER = 15 # OK on lully
NLAYERS_ON_SERVER = 0

layer2clients = [[] for i in range(NLAYERS)]
stopit=False
colay=0
jobq = Queue()

@app.route('/status')
def getStatus():
    s='_'.join([str(len(layer2clients[i])) for i in range(len(layer2clients))])
    return s

@app.route('/collabgpt')
def index():
    # pour afficher le status des noeuds dans une page web
    print("index called",[len(layer2clients[i]) for i in range(NLAYERS)])
    lays =[{'num':i,'co':len(layer2clients[i])} for i in range(len(layer2clients))]
    return render_template('index.html',layers=lays)

@app.route("/shutdown", methods=['GET'])
def shutdown():
    global stopit
    stopit=True
    shutdown_func = request.environ.get('werkzeug.server.shutdown')
    if shutdown_func is None:
        raise RuntimeError('Not running werkzeug')
    shutdown_func()
    return "Shutting down..."

def stop():
    resp = requests.get('http://localhost:5000/shutdown')

def rmClient(layer,o):
    layer2clients[layer].remove(o)
def delClient(layer,idx):
    del layer2clients[layer][idx]
    # TODO: si on supprime le client 0 (donc l'actif) il faut stopper le "wait reponse" en cours et relancer la requete sur le client suivant
    # TODO: supprimer toutes les requetes du client en cours

# cette methode est appelee des qu'un client se connecte
# avec chaque client on etablit et maintient une WS bi-dir permanente pour envoyer les (X,Y)
@app.route('/gpt', websocket=True)
def newClient():
    global layer2clients

    # cree un websocket avec ce client particulier
    ws = simple_websocket.Server(request.environ)
    num = -1
    try:
        # new client tries to register
        num = wait4Int(ws)
        lien = Comm(ws)
        layer2clients[num].append(lien)
        print("GOT CLIENT",num)
        ws.send("OK")
        if all([len(cs)>0 for cs in layer2clients]): print("all layers served !")
        # the Comm link takes care of listening to messages from the client
        # so we create a thread to check for requests from this client
        threading.Thread(target=wait4requests,args=[lien]).start()
        # here, we only wait until the end
        while not stopit:
            time.sleep(6)
        print("quiting client",num)

    except simple_websocket.ConnectionClosed:
        if num>=0: rmClient(num,lien)
    return ''

def wait4requests(lien):
    global jobq
    while True:
        msg=lien.reqQueue.get()
        msg.lien=lien
        jobq.put(msg)
        # TODO: prioritize requests

class GPTNeotiXServer(nn.Module):
    def __init__(self,i):
        super().__init__()
        self.num=i

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        layer_past=None,
        output_attentions=False,
    ):
        print("sending layer")
        print("pseudolayer hidd",hidden_states.size())
        print("pseudolayer attm",attention_mask)
        print("pseudolayer hedm",head_mask)
        print("pseudolayer cach",use_cache)
        print("pseudolayer past",layer_past)
        print("pseudolayer outp",output_attentions)

        lien = layer2clients[self.num][0]
        print("pseudo-layer creating X msg",self.num,lien)
        msg = XNeoXMsg(hidden_states,attention_mask,head_mask,layer_past,use_cache,output_attentions)
        lien.send(msg)
        res = msg.getAnswer()
        for i in range(len(res)):
            if not res[i]==None: res[i] = res[i].type(torch.float32)
        return res

class GPTNeoXLocal(nn.Module):
    def __init__(self,i):
        super().__init__()
        self.num=i
        self.load()

    def load(self):
        cfg = {
                "architectures": [ "GPTNeoXForCausalLM" ],
                "attention_probs_dropout_prob": 0.1,
                "bos_token_id": 0,
                "eos_token_id": 2,
                "hidden_act": "gelu_fast",
                "hidden_dropout_prob": 0.1,
                "hidden_size": 6144,
                "initializer_range": 0.02,
                "intermediate_size": 24576,
                "layer_norm_eps": 1e-05,
                "max_position_embeddings": 2048,
                "model_type": "gpt_neox",
                "num_attention_heads": 64,
                "num_hidden_layers": 1, # 44,
                "rotary_emb_base": 10000,
                "rotary_pct": 0.25,
                "tie_word_embeddings": False,
                "torch_dtype": "float32",
                "transformers_version": "4.19.0.dev0",
                "use_cache": True,
                "vocab_size": 50432
                }
        print("creating layer structure...")
        c = transformers.PretrainedConfig(**cfg)
        mod = transformers.models.gpt_neox.GPTNeoXLayer(c)
        print("layer structure created; now loading layer weights...",self.num)
        with open("pytorch_model.bin.index.json","r") as f: fmap = json.loads(f.read())['weight_map']
        fs = set([fmap["gpt_neox.layers."+str(self.num)+"."+n] for n,_ in mod.named_parameters()])
        for sf in fs:
            o = torch.load("./"+sf,map_location="cpu")
            for n,p in mod.named_parameters():
                ff = fmap["gpt_neox.layers."+str(self.num)+"."+n]
                if ff==sf:
                    p.data = o["gpt_neox.layers."+str(self.num)+"."+n]
                    # conversion en float32 car fp16 ne marche pas sur CPU ?
                    p.data = p.data.type(torch.float32)
        self.b = mod
        print("layer loaded")

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        layer_past=None,
        output_attentions=False,
    ):
        global colay
        if self.firstLayer:
            newhidden_states  = torch.load("laysin/laysout1."+str(colay))
            newattention_mask = torch.load("laysin/laysout2."+str(colay))
            assert newhidden_states.shape==hidden_states.shape
            hidden_states=newhidden_states
            attention_mask=newattention_mask
        res= self.b.forward(hidden_states, attention_mask, head_mask, use_cache, layer_past, output_attentions)
        if self.lastLayer:
            torch.save(res[0],"laysout1."+str(colay))
            torch.save(res[1],"laysout2."+str(colay))
            colay += 1
        return res

class GPTNeoXEmpty(nn.Module):
    def __init__(self,i):
        super().__init__()
        self.num=i

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        layer_past=None,
        output_attentions=False,
    ):
        print("pseudo-layer creating X msg",self.num)
        print("pseudolayer hidd",hidden_states.size())
        print("pseudolayer attm",attention_mask)
        print("pseudolayer hedm",head_mask)
        print("pseudolayer cach",use_cache)
        print("pseudolayer past",layer_past)
        print("pseudolayer outp",output_attentions)

        res = (hidden_states,attention_mask)
        return res

class GPTNeoXClient(nn.Module):
    def __init__(self,i):
        super().__init__()
        self.num=i
        self.load()

    def load(self):
        cfg = {
                "architectures": [ "GPTNeoXForCausalLM" ],
                "attention_probs_dropout_prob": 0.1,
                "bos_token_id": 0,
                "eos_token_id": 2,
                "hidden_act": "gelu_fast",
                "hidden_dropout_prob": 0.1,
                "hidden_size": 6144,
                "initializer_range": 0.02,
                "intermediate_size": 24576,
                "layer_norm_eps": 1e-05,
                "max_position_embeddings": 2048,
                "model_type": "gpt_neox",
                "num_attention_heads": 64,
                "num_hidden_layers": 1, # 44,
                "rotary_emb_base": 10000,
                "rotary_pct": 0.25,
                "tie_word_embeddings": False,
                "torch_dtype": "float32",
                "transformers_version": "4.19.0.dev0",
                "use_cache": True,
                "vocab_size": 50432
                }
        print("creating layer structure...")
        c = transformers.PretrainedConfig(**cfg)
        mod = transformers.models.gpt_neox.GPTNeoXLayer(c)
        print("layer structure created; now loading layer weights...")
        with open("pytorch_model.bin.index.json","r") as f: fmap = json.loads(f.read())['weight_map']
        fs = set([fmap["gpt_neox.layers."+str(self.num)+"."+n] for n,_ in mod.named_parameters()])
        for sf in fs:
            # o = torch.load("/home/xtof/corpus/gpt-neox-20g/huggingface/gpt-neox-20b/"+sf,map_location="cpu")
            o = torch.load("./"+sf,map_location="cpu")
            for n,p in mod.named_parameters():
                ff = fmap["gpt_neox.layers."+str(self.num)+"."+n]
                if ff==sf:
                    p.data = o["gpt_neox.layers."+str(self.num)+"."+n]
                    # conversion en float32 car fp16 ne marche pas sur CPU ?
                    p.data = p.data.type(torch.float32)
        self.b = mod
        print("layer loaded")

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        layer_past=None,
        output_attentions=False,
    ):
        print("pseudo-layer creating X msg",self.num)
        print("pseudolayer hidd",hs.size())
        print("pseudolayer attm",attention_mask)
        print("pseudolayer hedm",head_mask)
        print("pseudolayer cach",use_cache)
        print("pseudolayer past",layer_past)
        print("pseudolayer outp",output_attentions)

        # TODO bugfix gestion du cache !
        hs = hidden_states
        am = attention_mask
        res= self.b.forward(hs, am, head_mask, use_cache, layer_past, output_attentions)
        return res

def initmod(layerclass=GPTNeotiXServer,offset=0):
    print("creating model...")

    cfg = {
            "architectures": [ "GPTNeoXForCausalLM" ],
            "attention_probs_dropout_prob": 0.1,
            "bos_token_id": 0,
            "eos_token_id": 2,
            "hidden_act": "gelu_fast",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 6144,
            "initializer_range": 0.02,
            "intermediate_size": 24576,
            "layer_norm_eps": 1e-05,
            "max_position_embeddings": 2048,
            "model_type": "gpt_neox",
            "num_attention_heads": 64,
            "num_hidden_layers": 0, # 44,
            "rotary_emb_base": 10000,
            "rotary_pct": 0.25,
            "tie_word_embeddings": False,
            "torch_dtype": "float32", # float16
            "transformers_version": "4.19.0.dev0",
            "use_cache": True,
            "vocab_size": 50432
            }

    # load embed layer from file
    with open("pytorch_model.bin.index.json","r") as f:
        fmap = json.loads(f.read())['weight_map']
    c = transformers.PretrainedConfig(**cfg)
    mod = transformers.models.gpt_neox.GPTNeoXForCausalLM(c)
    print("model created without any layer")
    fs = set()
    for n,p in mod.named_parameters():
        ff = fmap[n]
        fs.add(ff)
    # this loop is only on non-attention layers, because we have set num_layers=0 in config
    for sf in fs:
        o = torch.load("/home/xtof/corpus/gpt-neox-20g/huggingface/gpt-neox-20b/"+sf,map_location="cpu")
        for n,p in mod.named_parameters():
            ff = fmap[n]
            if ff==sf: p.data = o[n]
            # conversion en float32 car fp16 ne marche pas sur CPU ?
            p.data = p.data.type(torch.float32)
    print("embeddings loaded")
    mod.config.num_hidden_layers = NLAYERS
    for i in range(NLAYERS):
        # lay = GPT2BlockixServer(i)
        # lay = transformers.models.gpt_neox.GPTNeoXLayer(c)
        lay = layerclass(offset+i)
        mod.gpt_neox.layers.append(lay)
        lay.lastLayer=False
        lay.firstLayer=False
        if i==NLAYERS-1: lay.lastLayer=True
        elif i==0 and offset>0: lay.firstLayer=True
    print(str(NLAYERS)+" pseudo-layers created")
    for n,p in mod.named_parameters():
        print(n,p.size())
    printParms(mod)
    return mod

def runeval(mod):
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # s = 'yes no'
    # toks = tokenizer(s)["input_ids"]
    # print("yes no",toks)
    # yes no [9820, 642]
    tokyes, tokno = 9820, 642

    s = '"Is rain wet?" is a rhetorical question, yes or no?'
    toks = tokenizer(s)["input_ids"]
    x = torch.LongTensor(toks).view(1,-1)
    # out = mod.generate(x,max_length=20)
    out = mod(x)['logits']
    print(out.size(),flush=True)
    print("yes=",out[0,-1,tokyes],"no=",out[0,-1,tokno])

def checkClients():
    global layer2clients
    while not stopit:
        for i in range(len(layer2clients)):
            todel = []
            for j in range(len(layer2clients[i])):
                lien = layer2clients[i][j]
                if lien==None:
                    # this is the main server
                    continue
                t = time.time()
                if t-lien.lastTime > 9: todel.append(lien)
            for lien in todel: rmClient(i,lien)
        time.sleep(3)


def runServerWS():
    # je lance la Flask app dans un thread separe car je dois lancer le runeval() de Eleuther-AI dans le main
    # thread, car il utilise signal.signal()
    try:
        print("launching server")
        threading.Thread(target=lambda: app.run(host="0.0.0.0", debug=True, use_reloader=False)).start()
        threading.Thread(target=checkClients).start()
        print("the main thread is waiting for all clients to be available...")
    except KeyboardInterrupt:
        for ws in layer2clients:
            if not ws==None: ws.quit()

def runServerLocal():
    global colay
    os.system("rm -rf laysin")
    os.system("mkdir laysin")
    for i in range(int(44/NLAYERS)):
        print("running XP",i)
        mod=initmod(GPTNeoXLocal,offset=i*NLAYERS)
        colay=0
        xtof.runeval(mod)
        os.system("mv laysout* laysin/")

def launchGenerateTask():
    global jobq
    mod=initmodfull()
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    s = 'The main event that triggered World War I is'
    s = 'GPTNeoX20B is a 20B-parameter autoregressive Transformer model developed by EleutherAI.'
    while True:
        msg = jobq.get()
        if msg.s.strip() == "Q": break
        tgtlen = len(msg.s.split(" "))+20
        toks = tokenizer(msg.s, return_tensors="pt").input_ids
        # TODO: send the next job as soon as layer1 processing is done
        y = mod.generate(toks, do_sample=True, temperature=0.9, max_length=tgtlen)
        rep = tokenizer.batch_decode(y)[0]
        print(rep)
        ans = RepMsg(msg.s,rep)
        msg.lien.send(ans)
    stopit=True
    exit(1)

class MyGPTNeoLayer(transformers.models.gpt_neox.GPTNeoXLayer):
    def __init__(self,idx):
        super(transformers.models.gpt_neox.GPTNeoXLayer,self).__init__()
        self.idx=idx

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            use_cache=False,
            layer_past=None,
            output_attentions=False,
            ):  
        # if there's a client serving this layer, use it, otherwise, wait for one
        while len(layer2clients[self.idx])==0:
            time.sleep(3)
            if stopit: return None
        
        lien = layer2clients[self.idx][0]
        print("pseudo-layer creating X msg",self.idx,lien)
        msg = XNeoXMsg(hidden_states,attention_mask,head_mask,layer_past,use_cache,output_attentions)
        lien.send(msg)
        res = msg.getAnswer()
        print("pseudo-layer got Y msg",self.idx,lien)
        for i in range(len(res)):
            if not res[i]==None: res[i] = res[i].type(torch.float32)
        return res
        # return (hidden_states,attention_mask) # no-op

def initmodfull():
    # cfg = {
    #         "architectures": [ "GPTNeoXForCausalLM" ],
    #         "model_type": "gpt_neox",
    #         "num_hidden_layers": 15,
    #         "torch_dtype": "float32", # float16
    #         "use_cache": True,
    #         }
    # c = transformers.PretrainedConfig(**cfg)
    # c = transformers.models.gpt_neox.GPTNeoXConfig.from_pretrained("gpt-neox-20b")
    # c.is_decoder = True

    # mod = transformers.models.gpt_neox.GPTNeoXForCausalLM(c)
    numlayers = NLAYERS_ON_SERVER
    print("loading GPT-NeoX with nlayers",numlayers)
    mod = transformers.models.gpt_neox.GPTNeoXForCausalLM.from_pretrained("./", num_hidden_layers=numlayers,use_cache=False)
    print("completing GPT-NeoX with pseudo-layers up to 44")
    for i in range(numlayers): layer2clients[i].append(None)
    mod.config.num_hidden_layers = NLAYERS
    for i in range(numlayers,NLAYERS):
        lay = MyGPTNeoLayer(i)
        mod.gpt_neox.layers.append(lay)
    print("model created with all pseudo-layers")
    return mod

if __name__ == '__main__':
    taskthread = threading.Thread(target=launchGenerateTask).start()
    runServerWS()
    # runServerLocal()

