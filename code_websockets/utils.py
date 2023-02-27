import torch
import numpy as np
import threading
import time
from queue import Queue

def _async_raise(tid, exctype):
    '''Raises an exception in the threads with id tid'''
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid),
                                                     ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # "if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

class ThreadWithExc(threading.Thread):
    '''A thread class that supports raising an exception in the thread from
       another thread.
    '''
    def _get_my_tid(self):
        """determines this (self's) thread id

        CAREFUL: this function is executed in the context of the caller
        thread, to get the identity of the thread represented by this
        instance.
        """
        if not self.isAlive():
            raise threading.ThreadError("the thread is not active")

        # do we have it cached?
        if hasattr(self, "_thread_id"):
            return self._thread_id

        # no, look for it in the _active dict
        for tid, tobj in threading._active.items():
            if tobj is self:
                self._thread_id = tid
                return tid

        # TODO: in python 2.6, there's a simpler way to do: self.ident

        raise AssertionError("could not determine the thread's id")

    def raiseExc(self, exctype):
        """Raises the given exception type in the context of this thread.

        If the thread is busy in a system call (time.sleep(),
        socket.accept(), ...), the exception is simply ignored.

        If you are sure that your exception should terminate the thread,
        one way to ensure that it works is:

            t = ThreadWithExc( ... )
            ...
            t.raiseExc( SomeException )
            while t.isAlive():
                time.sleep( 0.1 )
                t.raiseExc( SomeException )

        If the exception is to be caught by the thread, you need a way to
        check that your thread has caught it.

        CAREFUL: this function is executed in the context of the
        caller thread, to raise an exception in the context of the
        thread represented by this instance.
        """
        _async_raise( self._get_my_tid(), exctype )

class DetPrint:
    prthr = None
    prq = Queue()

    @classmethod
    def print(cls,*s):
        if cls.prthr==None:
            cls.prthr = threading.Thread(target=lambda: cls.qprint()).start()
        cls.prq.put(s)

    @classmethod
    def qprint(cls):
        while True:
            s=cls.prq.get()
            if s==None: break
            print(*s,flush=True)

def phprint(t):
    v=t.view(-1,)
    x=str(v[0:5])
    return x

def printParms(mod):
    ntot=0
    for n,p in mod.named_parameters():
        sh = p.shape
        if len(sh)>2: print("BUG")
        elif len(sh)==2: ss = sh[0]*sh[1]
        elif len(sh)==1: ss = sh[0]
        ntot+=ss
        print("PSIZE",ss,n,p.size())
    print("TOTALSIZE",ntot)

def sendBoolean(ws,b):
    if b: ws.send("BT")
    else: ws.send("BF")

def sendTensor(ws,t):
    pref = ""
    if t==None:
        ws.send("N")
        return
    elif t.dtype==torch.float64: pref="TD"
    elif t.dtype==torch.float32: pref="TS"
    elif t.dtype==torch.float16: pref="TH"
    else:
        print("ERROR: unsuported tensor",t)
        assert False
    x = t.cpu().detach().numpy()
    sh = pref+' '.join([str(v) for v in x.shape])
    ws.send(sh)
    data = x.tobytes()
    ws.send(data)

def sendInt(ws,i):
    ws.send('I'+str(i))

def sendTensorList(ws,l):
    print("sendlist",l)
    if l==None: ws.send("N")
    else:
        ws.send('L'+str(len(l)))
        for i in range(len(l)):
            if type(l[i])==tuple:
                ws.send("L")
                sendTensorList(ws,l[i])
            else: sendTensor(ws,l[i])

def wait4TensorList(ws):
    data = ws.receive()
    if data[0]=='N': return None
    assert data[0]=='L'
    n = int(data[1:])
    res = []
    for i in range(n):
        t = wait4Tensor(ws)
        res.append(t)
    return res

def wait4Pong(ws):
    data = ws.receive(timeout=3)
    if data==None: return False
    if not data[0]=='O': return False
    return True

def wait4Int(ws):
    data = ws.receive()
    assert data[0]=='I'
    return int(data[1:])

def wait4Boolean(ws):
    data = ws.receive()
    assert data[0]=='B'
    return data[1]=='T'

def wait4Tensor(ws):
    # this function may return a list !
    data = ws.receive()
    if data[0]=='N': return None
    elif data[0]=='L': return wait4TensorList(ws)
    assert data[0]=='T'
    prec = data[1]
    sh = [int(v) for v in data[2:].split(' ')]
    data = ws.receive()
    if prec=='D':   y = np.frombuffer(data,dtype='float64')
    elif prec=='S': y = np.frombuffer(data,dtype='float32')
    elif prec=='H': y = np.frombuffer(data,dtype='float16')
    else: assert False
    y = np.copy(y)
    y = y.reshape(*sh)
    res = torch.FloatTensor(y)
    return res 

class Message:
    def send(ws):
        pass

class QuitMsg(Message):
    def __init__(self):
        pass

class PingMsg(Message):
    def __init__(self):
        # message sent by clients
        self.reqAnswer=False

    def send(self,ws):
        ws.send("P")

class YNeoXMsg(Message):
    def __init__(self,res,num):
        # message sent by clients
        self.res=res
        self.num=num
        self.reqAnswer=False

    def send(self,ws):
        # called by the client sending queue
        ws.send("Y")
        sendTensorList(ws,self.res)
        print("results correctly sent. Waiting for next input from server...",self.num)

class ReqMsg(Message):
    def __init__(self,s):
        self.s=s
        self.reqAnswer=False

    def send(self,ws):
        # called by the client sending queue
        ws.send("U")
        ws.send(self.s)
        print("sending UTT request message")

class RepMsg(Message):
    def __init__(self,q,a):
        self.q=q
        self.a=a
        self.reqAnswer=False

    def send(self,ws):
        # called by the client sending queue
        ws.send("R")
        ws.send(self.q)
        ws.send(self.a)
        print("sending UTT response message")

class XNeoXMsg(Message):
    # vecteur X envoye au client pour qu'il le calcule
    def __init__(self,hidden_states,attention_mask,head_mask,layer_past,use_cache,output_attentions):
        self.reqAnswer=True
        self.hidden_states,self.attention_mask,self.head_mask,self.layer_past,self.use_cache,self.output_attentions=hidden_states,attention_mask,head_mask,layer_past,use_cache,output_attentions
        self.ansQueue = Queue()

    def getAnswer(self):
        # called by the pseudo-layer
        res = self.ansQueue.get()
        return res

    def send(self,ws):
        # called by the Comm link
        ws.send("X")
        sendTensor(ws, self.hidden_states)
        sendTensor(ws, self.attention_mask)
        sendTensor(ws, self.head_mask)
        sendTensor(ws, self.layer_past)
        sendBoolean(ws, self.use_cache)
        sendBoolean(ws, self.output_attentions)

    def recv(self,ws):
        # called by the Comm link
        # le char "Y" a deja ete parse
        res = wait4TensorList(ws)
        self.ansQueue.put(res)
        if res==None: return False
        return True

class Comm:
    """
    on n'a qu'un seul canal de comm entre le server et le client
    - les echanges X <-> Y sont synchrones (le server n'envoit pas de nouveau X tant qu'il n'a pas recu le Y precedent)
    - mais le client peut envoyer au server une nouvelle phrase UTT a traiter a tout moment, mais une seule a la fois (pas 2 phrases en meme temps):
        - est-ce OK si cette phrase est envoyee pendant que le server envoit un X ?
        - quid si UTT est envoyee pendant que le client envoit un Y ? = Websockets ne sont pas thread-safe, donc plusieurs threads ne peuvent pas write en meme temps !
        donc il faut que le client gere un LOCK pour l'ecriture dans la websocket,
        et que le server jette tout message non conforme (si clients malveillants)
    """
    def __init__(self,ws):
        # websocket vers le client
        self.ws = ws
        self.newid = 0
        # we use a queue to send all msgs to the client to be thread-safe
        self.sendQueue = Queue()
        self.active = True
        self.lastTime = 0
        # this link stores internally all UTT requests from the client; the server shall pick them from the queue
        self.reqQueue = Queue()

        self.sendloop = threading.Thread(target=self._sendloop)
        self.sendloop.start()
        # pas besoin de ping msg: tout client doit envoyer un pong toutes les 3 secondes
        # TODO: et inversement ??
        self.recvloop = ThreadWithExc(target=self._recvloop)
        self.recvloop.start()

    def send(self,msg):
        # called by the pseudolayer
        # interface appelable par d'autres programmes: thread safe !
        self.sendQueue.put(msg)

    def quit(self):
        self.active=False
        self.sendQueue.clear()
        self.send(QuitMsg())
        self.recvloop.raiseExc(Exception())
        while self.recvloop.isAlive():
                time.sleep(0.1)
                self.recvloop.raiseExc(Exception())

    def _pongcallbackOK(self):
        self.lastTime = time.time()
    def _pongcallbackKO(self):
        self.active=False

    def _recvloop(self):
        # loop that listens to the client
        try:
            while self.active:
                data = self.ws.receive()
                # soit c'est un Y = reponse a X
                if data=='Y':
                    # wait4Answer contient le message que le server a envoye au client et qui attend une reponse
                    if self.wait4Answer == None:
                        print("ERROR Y without X")
                        self.active=False
                    else:
                        if self.wait4Answer.recv(self.ws):
                            # reponse bien recue
                            self.lastTime = time.time()
                        else:
                            print("ERROR message recu du client ne correspond pas au message envoye")
                # soit c'est un pong
                elif data=='P':
                    self.lastTime = time.time()
                # soit c'est une nouvelle requete UTT
                elif data=='U':
                    s = self.ws.receive()
                    self.lastTime = time.time()
                    self.reqQueue.put(ReqMsg(s))
                # soit c'est un QUIT
                elif data=='Q':
                    print("quiting comm")
                    self.active=False
                    break
                else:
                    print("ERROR unknown message from client")
                    self.active=False
        except Exception as e:
            print("Exception: get out of server receive loop")
            print(e)

        self.active=False
        self.ws.send("Q")

    def _sendloop(self):
        try:
            while self.active:
                msg = self.sendQueue.get()
                if type(msg)==QuitMsg: 
                    self.active=False
                    break
                # TODO ne pas envoyer le message si on attend une reponse !
                msg.id = self.newid
                self.newid += 1
                msg.send(self.ws)
                if msg.reqAnswer:
                    self.wait4Answer = msg
                else:
                    self.wait4Answer = None
        except:
            pass
        self.active=False


