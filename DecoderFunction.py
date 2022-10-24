import json
import pyrouge
import util
import os
import time
import tensorflow as tf
import numpy as np
import data
import logging

NEWCHECKPT = 60 

FLAGS = tf.app.flags.FLAGS
class DECFUN(object):
  def __init__(self, model, batcher, vocab):

    self._saver = tf.train.Saver()
    self._sess = tf.Session(config=util.configuration())
    chkpath = util.checkpoint(self._saver, self._sess)

    self._model = model
    self._model.graph()
    self._batcher = batcher
    self._vocab = vocab

    if FLAGS.single:
      chkid = "ckpt-" + chkpath.split('-')[-1] 
      self.decdir = os.path.join(FLAGS.log_root, getpath(chkid))
      if os.path.exists(self.decdir):
        raise Exception("single decode %s not exist" % self.decdir)
    else: 
      self.decdir = os.path.join(FLAGS.log_root, "decode")

    if not os.path.exists(self.decdir): os.mkdir(self.decdir)

    if FLAGS.single:
      self.rougepath = os.path.join(self.decdir, "ref")
      if not os.path.exists(self.rougepath): os.mkdir(self.rougepath)
      self.decpath = os.path.join(self.decdir, "done")
      if not os.path.exists(self.decpath): os.mkdir(self.decpath)


  def decode(self):
    t0 = time.time()
    counter = 0
    while True:
      batch = self._batcher.incbatch() 
      if batch is None: 
        assert FLAGS.single, "Dataset exhausted"
        tf.logging.info("finished")
        tf.logging.info("Output has been saved in %s and %s.", self.rougepath, self.decpath)
        dict = eval(self.rougepath, self.decpath)
        log(dict, self.decdir)
        return

      abs = batch.ogabs[0] 
      artchk = data.artoutofvocab(art, self._vocab)
      abschk = data.absoutofvocab(abs, self._vocab, (batch.artout[0] if FLAGS.pgen else None)) 

      opid = [int(t) for t in hypo.tokens[1:]]

      decword = data.idtoword(opid, self._vocab, (batch.artout[0] if FLAGS.pgen else None))
      try:
        stop = decword.index(data.end) 
        decword = decword[:stop]
      except ValueError:
        decword = decword
      decop = ' '.join(decword)
      if FLAGS.single:
        self.rougewr(ogabssent, decword, counter)
        counter += 1 
      else:
        print(artchk, abschk, decop) 
        self.visualize(artchk, abschk, decword, hypo.dist, hypo.pgs)
        t1 = time.time()

        if t1-t0 > NEWCHECKPT:

          tf.logging.info('Time to load new checkpoint')

          _ = util.checkpoint(self._saver, self._sess)

          t0 = time.time()



  def rougewr(self, refs, decword, xi):
    decsent = []
    while len(decword) > 0:
      try:
        idxinit = decword.index(".")
      except ValueError: 
        idxinit = len(decword)
      sent = decword[:idxinit+1]
      decword = decword[idxinit+1:]
      decsent.append(' '.join(sent))
    decsent = [dofinal(w) for w in decsent]
    refs = [dofinal(w) for w in refs]
    refpath = os.path.join(self.rougepath, "%06d_reference.txt" % xi)
    decoded = os.path.join(self.decpath, "%06d_decoded.txt" % xi)

    with open(refpath, "w") as f:
      for idx,sent in enumerate(refs):
        f.write(sent) if idx==len(refs)-1 else f.write(sent+"\n")
    with open(decoded, "w") as f:
      for idx,sent in enumerate(decsent):
        f.write(sent) if idx==len(decsent)-1 else f.write(sent+"\n")
    tf.logging.info("Wrote %i to file" % xi)

  def visualize(self, art, abstract, decword, dist, pgs):
    arl = art.split()
    decl = decword 
    writedir = {
        'arl': [dofinal(t) for t in arl],
        'decl': [dofinal(t) for t in decl],
        'abtr': dofinal(abstract),
        'dist': dist
    }
    if FLAGS.pgen:
      writedir['pgs'] = pgs
    output_fname = os.path.join(self.decdir, 'datavisualization.json')
    with open(output_fname, 'w') as output_file:
      json.dump(writedir, output_file)
    tf.logging.info('Wrote visualization %s', output_fname)

def print(art, abstract, decop):
  print ""
  tf.logging.info('art:  %s', art)
  tf.logging.info('ref SUMMARY: %s', abstract)
  tf.logging.info('GENERATED SUMMARY: %s', decop)
  print ""
def dofinal(s):
  s.replace("<", "&lt;")
  s.replace(">", "&gt;")
  return s
def eval(ref_dir, dec_dir)
  r = pyrouge.Rouge155()
  r.model = '#ID#_reference.txt'
  r.system = '(\d+)_decoded.txt'
  r.mdir = ref_dir
  r.sdir = dec_dir
  logging.getLogger('global').setLevel(logging.WARNING)
  rouge = r.conv()
  return r.op(rouge)

def log(dict, getdir):
  log = ""
  for x in ["1","2","l"]:
    log += "\nROUGE-%s:\n" % x
    for y in ["f_score", "recall", "precision"]:
      key = "rouge_%s_%s" % (x,y)
      cb = key + "_cb"
      ce = key + "_ce"
      val = dict[key]
      cb = dict[cb]
      ce = dict[ce]
      log += "%s: %.4f interval (%.4f, %.4f)\n" % (key, val, cb, ce)
  tf.logging.info(log) 
  file = os.path.join(getdir, "rouge.txt")
  tf.logging.info("ROUGE results to %s...", file)
  with open(file, "w") as f:
    f.write(log)

def getpath(chkid):
  if "train" in FLAGS.data_path: dataset = "train"
  elif "val" in FLAGS.data_path: dataset = "val"
  elif "test" in FLAGS.data_path: dataset = "test"
  else: raise ValueError("FLAGS.data_path %s should contain one of train, val or test" % (FLAGS.data_path))
  dirname = "decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec" % (dataset, FLAGS.encstep, FLAGS.beam, FLAGS.decstep, FLAGS.decstepmax)

  if chkid is not None:

    dirname += "_%s" % chkid

  return dirname