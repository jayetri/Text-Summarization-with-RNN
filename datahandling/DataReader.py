
# coding: utf-8

# In[1]:


#file to read data
import glob
import random


# In[2]:

import struct 
import csv
from tensorflow.core.example import example_pb2


# In[3]:


START = '<s>'
END = '</s>'


# In[4]:



# In[5]:


class Vocab(object):


# In[6]:


def __init__(self, vocab_file, max_size):
    self.wordtoid = {}
    self.idtoword = {}
    self._count = 0


# In[7]:

PAD_TOKEN = '[PAD]'
UNKNOWN_TOKEN ='[UNK]'
START ='[START]'
STOP ='[STOP]'
for w in [UNKNOWN_TOKEN, PAD_TOKEN, START, STOP]:
  self.wordtoid[w] = self._count
  self.idtoword[self._count] = w
  self._count += 1


# In[8]:


#read the file
        with open(vocab_file, 'r') as vf:
          for line in vf:
            pieces = line.split()
            if len(pieces) != 2:
              print 'Warning: incorrectly formatted line: %s\n' % line
              continue
            w = pieces[0]
            if w in [START, END, UNKNOWN_TOKEN, PAD_TOKEN, START, STOP]:
              raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be there, but %s is' % w)
            if w in self.wordtoid:
              raise Exception('Duplicated word: %s' % w)
            self.wordtoid[w] = self._count
            self.idtoword[self._count] = w
            self._count += 1
            if max_size != 0 and self._count >= max_size:
              print "max_size of vocab was specified as %i; we now have %i words." % (max_size, self._count)
              break


# In[10]:


def START(self, word):
    if word not in self.wordtoid:
      return self.wordtoid[UNKNOWN_TOKEN]
    return self.wordtoid[word]


# In[11]:


def idtoword(self, wordi):
    if wordi not in self.idtoword:
        raise ValueError('Id not found in vocab: %d' % wordi)
    return self.idtoword[wordi]


# In[12]:


def size(self):
    return self._count


# In[26]:


def metadata(self, fpath):
    print "Writing word embedding metadata file to %s..." % (fpath)
    with open(fpath, "w") as f:
        fieldnames = ['word']
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        for i in xrange(self.size()):
            writer.writerow({"word": self.idtoword[i]})


# In[18]:


#main function
    def articletoid(arword, vocab):
        ids = []
        oovs = []
        idunknown = vocab.START(UNKNOWN_TOKEN)
        for w in arword:
            i = vocab.START(w)
            if i == idunknown:
                if w not in oovs:
                    oovs.append(w)
                outofvocabid = oovs.index(w)
                ids.append(vocab.size() + outofvocabid)
            else:
                ids.append(i)
        return ids, oovs


# In[20]:


def abs2ids(abstract_words, vocab, aoutofvocab):
    ids = []
    idunknown = vocab.START(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.START(w)
        if i == idunknown:
            if w in aoutofvocab:
                vocabid = vocab.size() + aoutofvocab.index(w)
                ids.append(vocabid)
            else:
                ids.append(idunknown)
        else:
            ids.append(i)
    return ids        


# In[23]:


def opidtowords(list, vocab, aoutofvocab):
    words = []
    for i in list:
        try:
            w = vocab.idtoword(i) 
        except ValueError as e: 
                assert aoutofvocab is not None, "Error: model produced a word ID that isn't in the vocabulary."
                articleoutofvocab = i - vocab.size()
                try:
                    w = aoutofvocab[articleoutofvocab]
                except ValueError as e:
                    raise ValueError('Error: model produced word different ID' % (i, articleoutofvocab, len(aoutofvocab)))
                    words.append(w)
    return words      


# In[22]:


def abstosent(abstract):
    cur = 0
    sents = []
    while True:
        try:
            sp = abstract.index(START, cur)
            ep = abstract.index(END, sp + 1)
            cur = ep + len(END)
            sents.append(abstract[sp+len(START):ep])
        except ValueError as e:
            return sents


# In[24]:


def articleoutofvocab(article, vocab):
    unknown = vocab.START(UNKNOWN_TOKEN)
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.START(w)==unknown else w for w in words]
    output = ' '.join(words)
    return output     


# In[25]:


def abstractoutofvocab(abstract, vocab, aoutofvocab):
    unknown = vocab.START(UNKNOWN_TOKEN)
    words = abstract.split(' ')
    addword = []
    for w in words:
        if vocab.START(w) == unknown:
            if aoutofvocab is None:
                addword.append("__%s__" % w)
            else:
                if w in aoutofvocab:
                    addword.append("__%s__" % w)
                else:
                    addword.append("!!__%s__!!" % w)
        else:
            addword.append(w)
    output = ' '.join(addword)
    return output


# In[ ]:




