import torch.nn as nn
import torch
import numpy as np
import pickle
from utils import *
import sys
import transformers
import json
import simple_websocket
import time
from queue import Queue
import requests
import os.path
import urllib.request
import errno

# ce fichier doit etre lance sur chaque client

# module charge chez le client
class GPTNeoiXClient(nn.Module):
    def __init__(self,i):
        super().__init__()
        self.num=i
        # self.load()

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

    def _layerlist():
        with open("pytorch_model.bin.index.json","r") as f: fmap = json.loads(f.read())['weight_map']
        lay2fichs = [[] for i in range(44)]
        for k in fmap:
            if '.layer' in k:
                lay=int(k.split('.')[2])
                lay2fichs[lay].append(fmap[k])
        for i in range(44):
            fichs = list(set(lay2fichs[i]))
            print('<li>Layer '+str(i)+': <a href="http://fb.cerisara.fr/static/neoxweights/'+fichs[0]+'">Fich 1</a> and <a href="http://fb.cerisara.fr/static/neoxweights/'+fichs[1]+'">Fich 2</a></li>')

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        layer_past=None,
        output_attentions=False,
    ):
        print("client got input",self.num,hidden_states.size())
        print("pseudo-layer creating X msg",self.num)
        print("pseudolayer hidd",hidden_states.size())
        print("pseudolayer attm",attention_mask)
        print("pseudolayer hedm",head_mask)
        print("pseudolayer cach",use_cache)
        print("pseudolayer past",layer_past)
        print("pseudolayer outp",output_attentions)

        res= self.b.forward(hidden_states, attention_mask, head_mask, use_cache, layer_past, output_attentions)
        return res

    def terminateClient(self):
        self.active=False
        self.sendQueue.put("Q")

    def sendthread(self):
        try:
            while self.active:
                msg = self.sendQueue.get()
                if msg=="Q": break
                msg.send(self.ws)
        except:
            print("connection problem detected")
            self.terminateClient()

    def pingpong(self):
        while self.active:
            msg = PingMsg()
            self.sendQueue.put(msg)
            time.sleep(2)

    def sendRequest(self,s):
        # called from somewhere externally
        msg=ReqMsg(s)
        print("sending prompt",s)
        self.sendQueue.put(msg)

    def regClient(self):
        ws = simple_websocket.Client('wss://fb.cerisara.fr/gpt')
        self.ws=ws
        try:
            # des que le client est connecte, il envoit au server son numero de couche
            sendInt(ws,self.num)
            # il recupere un "OK"
            data = ws.receive()
            if not data.startswith("OK"):
                print("could not communicate correctly with server, please relaunch. Exiting.",self.num)
                exit(1)
            print("Connection to server fine. Waiting for server input...",self.num)
            self.active=True
            self.sendQueue = Queue()
            self.sendth = threading.Thread(target=self.sendthread)
            self.sendth.start()
            self.pingth = threading.Thread(target=self.pingpong)
            self.pingth.start()
            while self.active:
                # puis il attend que le server lui envoie des tensors a traiter
                d = ws.receive()
                if d.startswith("X"):
                    hs = wait4Tensor(ws)
                    if not hs==None: hs = hs.type(torch.float32)
                    am = wait4Tensor(ws)
                    if not am==None: am = am.type(torch.float32)
                    hm = wait4Tensor(ws)
                    if not hm==None: hm = hm.type(torch.float32)
                    lp = wait4Tensor(ws)
                    if not lp==None: lp = lp.type(torch.float32)
                    uc = wait4Boolean(ws)
                    oa = wait4Boolean(ws)
                    print("client: got some input from server, processing the data...",self.num,time.ctime())
                    res=self.forward(hs,am,hm,use_cache=uc,layer_past=lp,output_attentions=oa)
                    # debug
                    # res = (hs,am)
                    print("client: processing done; sending result",self.num,time.ctime())
                    msg = YNeoXMsg(res,self.num)
                    self.sendQueue.put(msg)
                elif d.startswith("R"):
                    dq = ws.receive()
                    da = ws.receive()
                    with open("reqreps.txt","a") as f:
                        f.write("REQ "+dq+"\n")
                        f.write("REP "+da+"\n")
                    print("client: got request response: saved in reqreps.txt")
                elif d.startswith("Q"):
                    print("quit client")
                    ws.close()
                    self.active=False
                    break
                else:
                    print("ERROR message recu",d,self.num)
                    exit()
        except (KeyboardInterrupt, EOFError, simple_websocket.ConnectionClosed):
            ws.close()
        self.terminateClient()

def showStatus():
    try:
        resp = requests.get('https://fb.cerisara.fr/status').text
        s=resp.split('_')
        ss=' '.join(['L'+str(i)+':'+s[i] for i in range(len(s))])
        print("STATUS:",ss)
    except:
        print("Error, no or unexpected response from the server? see https://fb.cerisara.fr/status")

def userReqs(mod):
    pipenom = "prompts.fifo."+str(mod.num)
    try:
        os.mkfifo(pipenom)
    except OSError as oe: 
        if oe.errno != errno.EEXIST: raise
    print('The named pipe',pipenom,'has been created to control the client: you may')
    print('- query the server status: echo "S" >',pipenom)
    print('- end the client: echo "Q" >',pipenom)
    print('- send a prompt that you want to be completed by GPT-NeoX-20G. For instance: echo "GPTNeoX20B is a 20B-parameter autoregressive Transformer model developed by EleutherAI." >',pipenom)

    while True:
        print("Opening FIFO...")
        with open(pipenom,"r") as f:
            print("FIFO opened")
            while True:
                data = f.read()
                if len(data) == 0:
                    print("Writer closed")
                    break
                s=data.strip()
                if len(s)>5: mod.sendRequest(s)
                elif s=="S": showStatus()
                elif s=="Q": mod.terminateClient()
                else: print("input too short")

def checkFiles(num):
    with open("pytorch_model.bin.index.json","r") as f: fmap = json.loads(f.read())['weight_map']
    k = ".layers."+str(num)+"."
    fichs = set()
    for fk in fmap.keys():
        if k in fk: fichs.add(fmap[fk])
    missing = []
    for f in fichs:
        fname = "./"+f
        if os.path.isfile(fname): print("fich found:",fname)
        else: missing.append(fname)
    if len(missing)==0: print("all layer files found...")
    else:
        print("nb layer files not found",len(missing),"Downloading them...")
        for fnom in missing:
            u = "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/"+fnom
            urllib.request.urlretrieve(u,fnom)

if __name__ == '__main__':
    # GPTNeoiXClient._layerlist()
    # exit()
    if len(sys.argv)<2:
        print("without arguments, client.py just queries the status of the server; here it is:")
        showStatus()
        print("\nThe main usage of client.py is for serving a layer; run it with one arg: client.py num_layer")
        exit()
    num = int(sys.argv[1])
    checkFiles(num)
    locmod = GPTNeoiXClient(num)
    locmod.load()
    ui = threading.Thread(target=userReqs,args=[locmod])
    ui.start()
    locmod.regClient()

