
# coding: utf-8

# In[1]:


import random
import threading
import numpy as np
import tensorflow as tf
import DataReader
import Queue


# In[2]:


class example(object):
    def __init__(self, article, absent, vocab, hps):
        self.hps = hps
        start = vocab.makeid(data.decodestart)
        stop = vocab.makeid(data.decodestop)


# In[4]:


words = article.split()
if len(words) > hps.encodingmaximum:
    words = words[:hps.encodingmaximum]
self.length = len(words) 
self.encodein = [vocab.makeid(w) for w in words]


# In[5]:


abstract = ' '.join(absent)
absword = abstract.split()
abs_ids = [vocab.makeid(w) for w in absword]


# In[6]:


self.decodeip, self.target = self.gettarget(abs_ids, hps.MaximumDecode, decodestart, decodestop)
self.dec_len = len(self.decodeip)


# In[7]:


if hps.Pgenerator:
    self.encodeinadd, self.articleoutofvocab = data.articletoid(words, vocab)
    extendabstract = data.abstoid(absword, vocab, self.articleoutofvocab)
    _, self.target = self.gettarget(extendabstract, hps.MaximumDecode, decodestart, decodestop)
self.article = article
self.abstract = abstract
self.sentence = absent


# In[8]:


def seektarget(self, sequence, max_len, start_id, stop_id):
    inp = [start_id] + sequence[:]
    target = sequence[:]
    if len(inp) > max_len:
        inp = inp[:max_len]
        target = target[:max_len]
    else:
        target.append(stop_id)
    assert len(inp) == len(target)
    return inp, target


# In[ ]:


def padinput(self, max_len, padidx):
    while len(self.encodein) < max_len:
        self.encodein.append(padidx)
    if self.hps.Pgenerator:
        while len(self.encodeinadd) < max_len:
            self.encodeinadd.append(padidx)  


# In[ ]:


def padinput(self, max_len, padidx):
    while len(self.decodeip) < max_len:
        self.decodeip.append(padidx)
    while len(self.target) < max_len:
        self.target.append(padidx)


# In[ ]:


class batcher(object):
    MAXBATCH = 50
    def __init__(self, path, vocab, hps, path):
        self.dataPath = path
        self._vocab = vocab
        self._hps = hps
        self.pathone = path
        self.batchlist = Queue.Queue(self.MAXBATCH)
        self.equeue = Queue.Queue(self.MAXBATCH * self._hps.bsize)
        self._numegthread = 1
        self._numbatcht = 1
        self.batchingcache = 1
        self.finish = False
        self.egthread = []
        for _ in xrange(self._numegthread):
            self.egthread.append(Thread(target=self.filleg))
            self.egthread[-1].daemon = True
            self.egthread[-1].start()
        self.batcht = []
        for _ in xrange(self._numbatcht):
            self.batcht.append(Thread(target=self.fillqueue))
            self.batcht[-1].daemon = True
            self.batcht[-1].start()
        if not path:
            self.threadseek = Thread(target=self.watch)
            self.threadseek.daemon = True
            self.threadseek.start()


# In[ ]:


def batchnew(self):
    if self.batchlist.qsize() == 0:
        tf.logging.warning('Bucket input queue empty calling batchnew.', self.batchlist.qsize(), self.equeue.qsize())
        if self.pathone and self.finish:
            tf.logging.info("Finished reading dataset")
            return None
    batch = self.batchlist.get() # get the next Batch
    return batch
def filleg(self):
    generatorin = self.tgen(data.egen(self.dataPath, self.pathone))
    while True:
        try:
            (article, abstract) = generatorin.next() 
        except StopIteration:
            tf.logging.info("exhausted data.")
            if self.pathone:
                tf.logging.info("This thread is stopping.")
                self.finish = True
                break
            else:
                raise Exception("path mode is off")


# In[ ]:


absent = [sent.strip() for sent in data.abtose(abstract)]
example = Example(article, absent, self._vocab, self._hps)
self.equeue.put(example)

    def fillqueue(self):
        while True:
if self._hps.mode != 'decode':
    inputs = []
    for _ in xrange(self._hps.bsize * self.batchingcache):
        inputs.append(self.equeue.get())
    inputs = sorted(inputs, key=lambda inp: inp.length) 
    batches = []
    for i in xrange(0, len(inputs), self._hps.bsize):
        batches.append(inputs[i:i + self._hps.bsize])
    if not self.pathone:
        shuffle(batches)
    for b in batches:
        self.batchlist.put(Batch(b, self._hps, self._vocab))
else:
    ex = self.equeue.get()
    b = [ex for _ in xrange(self._hps.bsize)]
    self.batchlist.put(Batch(b, self._hps, self._vocab))


# In[ ]:


def watch(self):
    while True:
        time.sleep(60)
        for idx,t in enumerate(self.egthread):
            if not t.is_alive(): 
                tf.logging.error(' Restarting.')
                newthread = Thread(target=self.filleg)
                self.egthread[idx] = newthread
                newthread.daemon = True
                newthread.start()
                for idx,t in enumerate(self.batcht):
                    if not t.is_alive():
                        tf.logging.error('Restarting.')
                        newthread = Thread(target=self.fillqueue)
                        self.batcht[idx] = newthread
                        newthread.daemon = True
                        newthread.start()
def tgen(self, egen):
    while True:
        e = egen.next()
        try:
            text = e.features.feature['article'].blist.value[0]
            abstract = e.features.feature['abstract'].blist.value[0]
        except ValueError:
            tf.logging.error('Failed to get article or abstract')
            continue
        if len(text)==0:
            tf.logging.warning('Found an empty article text')
        else:
            yield (text, abstract


# In[ ]:


class batch(object):
    def __init__(self, exlist, hps, vocab):
        self.padidx = vocab.makeid(data.PAD_TOKEN)
        self.initialize(exlist, hps)
        self.initializeDec(exlist, hps)
        self.store(exlist)
    
    def initialize(self, exlist, hps):
        sequencelengthmax = max([ex.length for ex in exlist])
        for ex in exlist:
            ex.padinput(sequencelengthmax, self.padidx)
        self.batchencoding = np.zeros((hps.bsize, sequencelengthmax), dtype=np.int32)
        self.lengths = np.zeros((hps.bsize), dtype=np.int32)
        self.pad = np.zeros((hps.bsize, sequencelengthmax), dtype=np.float32)
        
        for i, ex in enumerate(exlist):
            self.batchencoding[i, :] = ex.encodein[:]
            self.lengths[i] = ex.length
            for j in xrange(ex.length):
                self.pad[i][j] = 1
                
        if hps.Pgenerator:
            self.maximumoutofvocab = max([len(ex.articleoutofvocab) for ex in exlist])
            self.articleoutofvocabplus = [ex.articleoutofvocab for ex in exlist]
            self.batchencodingadd = np.zeros((hps.bsize, sequencelengthmax), dtype=np.int32)
            for i, ex in enumerate(exlist):
                self.batchencodingadd[i, :] = ex.encodeinadd[:]


# In[ ]:


def initializeDec(self, exlist, hps):
    for ex in exlist:
        ex.padInput(hps.MaximumDecode, self.padidx)
    self.Dbatch = np.zeros((hps.bsize, hps.MaximumDecode), dtype=np.int32)
    self.tbatch = np.zeros((hps.bsize, hps.MaximumDecode), dtype=np.int32)
    self.decoderpad = np.zeros((hps.bsize, hps.MaximumDecode), dtype=np.float32)
    
    for i, ex in enumerate(exlist):
        self.Dbatch[i, :] = ex.decodeip[:]
        self.tbatch[i, :] = ex.target[:]
        for j in xrange(ex.dec_len):
            self.decoderpad[i][j] = 1
        
def store(self, exlist):
    self.articles = [ex.article for ex in exlist]
    self.abstracts = [ex.abstract for ex in exlist]
    self.sentences = [ex.sentence for ex in exlist]

