import sys
import time
import os
import tensorflow as tf
imort DecoderFunction
import util
import numpy as np
import collections
from data import Vocab
import Batcher
import model
from tensorflow.python import debug as tf_debug

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_string('exp_name', '', 'Name for experiment')
tf.app.flags.DEFINE_string('datadir', '', 'Expression to tf.Example datafiles.')
tf.app.flags.DEFINE_boolean('pass', False, 'For decode mode only.')
tf.app.flags.DEFINE_string('log', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('vocabdir', '', 'Path expression to text vocabulary file.')


# Hyperparameters
tf.app.flags.DEFINE_integer('encodemaxsteps', 300, 'max timesteps of encoder (max source text tokens)')

tf.app.flags.DEFINE_integer('decodemaxsteps', 100, 'max timesteps of decoder (max summary tokens)')

#tf.app.flags.DEFINE_integer('beam', 2, 'beam size for beam search decoding.')

tf.app.flags.DEFINE_integer('decodeminsteps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')

tf.app.flags.DEFINE_integer('vocab_size', 50000, 'Size of vocabulary.')

tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')

tf.app.flags.DEFINE_float('adagrad', 0.1, 'initial accumulator value for Adagrad')

tf.app.flags.DEFINE_float('randomINIT', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_integer('hidden', 128, 'dimension of RNN hidden states')

tf.app.flags.DEFINE_integer('embed', 64, 'dimension of word embeddings')

tf.app.flags.DEFINE_integer('batch', 20, 'minibatch size')

tf.app.flags.DEFINE_float('truncinit', 1e-4, 'std of trunc norm init, used for initializing everything else')

tf.app.flags.DEFINE_float('maxgrad', 2.0, 'for gradient clipping')


tf.app.flags.DEFINE_boolean('pgen', True, 'for pointer generator')



def losscalculate(loss, average, summary, step, decay=0.99):
  if average == 0:  
    average = loss
  else:
    average = average * decay + (1 - decay) * loss
  average = min(average, 12) 
  lsum = tf.Summary()
  tag = 'average/decay=%f' % (decay)
  lsum.value.add(tag=tag, simple_value=average)
  summary.add(lsum, step)
  tf.logging.info('average: %f', average)

  return average
def restoretobest():
  tf.logging.info("Restoring bestmodel...")
  sess = tf.Session(config=util.config())
  print "Initializing all variables..."
  sess.run(tf.varinit())

  saver = tf.train.Saver([v for v in tf.all_variables() if "Adagrad" not in v.name])

  print "Restoring all non-adagrad variables"
  chkpt = util.load(saver, sess, "eval")
  print "Restored %s." % chkpt

  newmodel = chkpt.split("/")[-1].replace("bestmodel", "model")
  fname = os.path.join(FLAGS.log, "train", newmodel)
  print "Saving model to %s..." % (fname)
  new_saver = tf.train.Saver() 
  new_saver.save(sess, fname)
  print "Saved."
  exit()

def settrain(model, batcher):
  traindir = os.path.join(FLAGS.log, "train")
  if not os.path.exists(traindir): os.makedirs(traindir)
  model.graph()
  
  if FLAGS.restoretobest:
    restoretobest()
  saver = tf.train.Saver(maxkeep=3)
  sv = tf.train.Supervisor(logdir=traindir, main=True, saver=saver, summary_op=None, saveafter=60, setCHKPT=60, step=model.step)
  summary = sv.summary
  tf.logging.info("session")

  managecontext = sv.seeksession(config=util.config())
  tf.logging.info("Create session.")
  try:
    trainrun(model, batcher, managecontext, sv, summary)
  except KeyboardInterrupt:
    tf.logging.info("Stopping supervisor...")
    sv.stop()

def trainrun(model, batcher, managecontext, sv, summary):

  tf.logging.info("starting trainrun")
  with managecontext as sess:
    if FLAGS.debug:
      sess = tf_debug.LocalCLIDebugWrapperSession(sess)
      sess.filter("nan", tf_debug.nan)
    while True: 
      batch = batcher.nextb()
      tf.logging.info('running training')
      t0=time.time()
      results = model.trainrun(sess, batch)
      t1=time.time()
      tf.logging.info('seconds for training step: %.3f', t1-t0)
      loss = results['loss']
      tf.logging.info('loss: %f', loss)
      if not np.isfinite(loss):
        raise Exception("Loss is not finite. Stopping.")
      if FLAGS.coverage:
      summaries = results['summaries'] 
      trainstep = results['step']
      summary.add(summaries, trainstep)
      if trainstep % 100 == 0:
        summary.flush()

def evalrun(model, batcher, vocab):

  model.graph()
  saver = tf.train.Saver(maxkeep=3) 
  sess = tf.Session(config=util.config())
  evaldir = os.path.join(FLAGS.log, "eval")
  bestdir = os.path.join(evaldir, 'bestmodel') 
  summary = tf.summary.FileWriter(evaldir)
  average = 0 
  lossb = None 

  while True:
    CHKPT_ = util.load(saver, sess)
    batch = batcher.nextb()
    t0=time.time()
    results = model.evalrun(sess, batch)
    t1=time.time()
    tf.logging.info('seconds for batch: %.2f', t1-t0)
    loss = results['loss']
    tf.logging.info('loss: %f', loss)

    summaries = results['summaries']
    trainstep = results['step']
    summary.add(summaries, trainstep)


    average = losscalculate(np.asscalar(loss), average, summary, trainstep)
    if lossb is None or average < lossb:
      tf.logging.info('Found new best model with %.3f average. Saving to %s', average, bestdir)
      saver.save(sess, bestdir, step=trainstep, newfile='checkpoint_best')
      lossb = average

    if trainstep % 100 == 0:

      summary.flush()

def main(argsv):
  if len(argsv) != 1:
    raise Exception("Problem with flags: %s" % argsv)
  tf.logging.verbosity(tf.logging.INFO) 
  tf.logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))
  FLAGS.log = os.path.join(FLAGS.log, FLAGS.exp_name)
  if not os.path.exists(FLAGS.log):
    if FLAGS.mode=="train":
      os.makedirs(FLAGS.log)
    else:
      raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log))

  vocab = Vocab(FLAGS.vocabdir, FLAGS.vocab_size) # create a vocabulary
  if FLAGS.mode == 'decode':
    FLAGS.batch = FLAGS.beam
  if FLAGS.pass and FLAGS.mode!='decode':
    raise Exception("The pass flag should not be true")
  hyperlist = ['mode', 'lr', 'adagrad', 'randomINIT', 'truncinit', 'maxgrad', 'hidden', 'embed', 'batch', 'decodemaxsteps', 'encodemaxsteps', 'coverage', 'cov_loss_wt', 'pgen']
  dict = {}
  for key,val in FLAGS.__flags.iteritems():
    if key in hyperlist:
      dict[key] = val 
  hps = namedtuple("HParams", dict.keys())(**dict)
  batcher = Batcher(FLAGS.datadir, vocab, hps, pass=FLAGS.pass)
  tf.rand(111) 
  if hps.mode == 'train':
    print "creating model..."
    model = SummarizationModel(hps, vocab)
    settrain(model, batcher)
  elif hps.mode == 'eval':
    model = SummarizationModel(hps, vocab)
    evalrun(model, batcher, vocab)
  elif hps.mode == 'decode':
    decmodelhyp = hps
    decmodelhyp = hps._replace(decodemaxsteps=1) 
    model = SummarizationModel(decmodelhyp, vocab)
    decoder = BeamSearchDecoder(model, batcher, vocab)
    decoder.decode() 
  else:
    raise ValueError("flag must be one of train/eval/decode")
if __name__ == '__main__':
  tf.app.run()