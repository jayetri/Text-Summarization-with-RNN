import time
import os
import tensorflow as tf
import numpy as np
import tensorflow.contrib.tensorboard.plugins
import AttentionModel

FLAGS = tf.app.flags.FLAGS

class SummMod(object):
  """It shows text summarization"""

  def __init__(self, hps, vocab):
    self._hps = hps
    self._vocab = vocab

  def Placeholderadding(self):
    """Adding placeholders"""
    hps = self._hps
 
    if  hps.coverage and hps.mode=="decode":   #decoder
      self.pcvg = tf.placeholder(tf.float32, [hps.sizebb, None], name='pcvg')
   
    self.DecoderMask = tf.placeholder(tf.float32, [hps.sizebb, hps.decmx], name='DecoderMask')
    self.BDecoded = tf.placeholder(tf.int32, [hps.sizebb, hps.decmx], name='BDecoded')
    self.Tar = tf.placeholder(tf.int32, [hps.sizebb, hps.decmx], name='Tar')

    
    
    self.EncoderMask = tf.placeholder(tf.float32, [hps.sizebb, None], name='EncoderMask')  
    self.BEncoded = tf.placeholder(tf.int32, [hps.sizebb, None], name='BEncoded')
    self.LensEncoded = tf.placeholder(tf.int32, [hps.sizebb], name='LensEncoded')
    if FLAGS.GenPtr: 
      self.MaxOutOfVOcab = tf.placeholder(tf.int32, [], name='MaxOutOfVOcab')   
      self.BEncodedExtendVocab = tf.placeholder(tf.int32, [hps.sizebb, None], name='BEncodedExtendVocab') 


  

  def EncoderFn(self, EncInput, SqLn):
  
    with tf.variable_scope('encoder'):
      
      Backward = tf.contrib.rnn.GRUCell(self._hps.DimensionHidden, initializer=self.InRand, STup=True)
      Forward= tf.contrib.rnn.GRUCell(self._hps.DimensionHidden, initializer=self.InRand, STup=True)
      (EncOut, (forwardState, backwardState)) = tf.nn.bidirectional_dynamic_rnn(Forward, Backward, EncInput, dtype=tf.float32, SqLn=SqLn, SWpMem=True)
      EncOut = tf.concat(axis=2, values=EncOut) 
    return EncOut, forwardState, backwardState


  def Reduction(self, forwardState, backwardState):
    #Reducing Forward and backward to single graph
    DimensionHidden = self._hps.DimensionHidden
    with tf.variable_scope('RdcFinSt'):

  def FDCreate(self, batch, Encjs=False):
    
    FD = {}
    FD[self.LensEncoded] = batch.LensEncoded
    FD[self.EncoderMask] = batch.EncoderMask
    FD[self.BEncoded] = batch.BEncoded
    
    
    if not Encjs:
      FD[self.BDecoded] = batch.BDecoded
      FD[self.Tar] = batch.Tar
      FD[self.DecoderMask] = batch.DecoderMask

    if FLAGS.GenPtr:
      FD[self.BEncodedExtendVocab] = batch.BEncodedExtendVocab
      FD[self.MaxOutOfVOcab] = batch.MaxOutOfVOcab
    return FD    

     
      hbias = tf.get_variable('hbias', [DimensionHidden], dtype=tf.float32, initializer=self.INittn)
      cbias = tf.get_variable('cbias', [DimensionHidden], dtype=tf.float32, initializer=self.INittn)

      HWeightRed = tf.get_variable('HWeightRed', [DimensionHidden * 2, DimensionHidden], dtype=tf.float32, initializer=self.INittn)
      CWeightReduce = tf.get_variable('CWeightReduce', [DimensionHidden * 2, DimensionHidden], dtype=tf.float32, initializer=self.INittn)
      
  
      cprev = tf.concat(axis=1, values=[forwardState.c, backwardState.c]) 
      hprev = tf.concat(axis=1, values=[forwardState.h, backwardState.h]) 
      
    
      hmodified = tf.nn.relu(tf.matmul(hprev, HWeightRed) + hbias) 
      hmodified = tf.nn.relu(tf.matmul(cprev, CWeightReduce) + cbias) 
      return tf.contrib.rnn.LSTMStateTuple(hmodified, hmodified)


  

  def FinalCalculation(self, VCDistribs, DistributionAttention):
    #pointer-generator
    with tf.variable_scope('findist'):
      
     
      DistributionAttention = [(1-pg) * dist for (pg,dist) in zip(self.ppp, DistributionAttention)]
      VCDistribs = [pg * dist for (pg,dist) in zip(self.ppp, VCDistribs)]

      
      VSzExt = self._vocab.size() + self.MaxOutOfVOcab 
      ZZero = tf.zeros((self._hps.sizebb, self.MaxOutOfVOcab))
      ExtVocabularyDistribution = [tf.concat(axis=1, values=[dist, ZZero]) for dist in VCDistribs] 
      
      NumOfBatches = tf.range(0, limit=self._hps.sizebb) 
      NumOfBatches = tf.expand_dims(NumOfBatches, 1) 
      length = tf.shape(self.BEncodedExtendVocab)[1] # no. of states
      NumOfBatches = tf.tile(NumOfBatches, [1, length]) 
      shape = [self._hps.sizebb, VSzExt]
      indd = tf.stack( (NumOfBatches, self.BEncodedExtendVocab), axis=2) 
      ProjDistAttention = [tf.scatter_nd(indd, DisCp, shape) for DisCp in DistributionAttention] 

      
      DistributionFInal = [VCDistrib + DisCp for (VCDistrib,DisCp) in zip(ExtVocabularyDistribution, ProjDistAttention)]


      return DistributionFInal


def DecoderFn(self, inputs):
    #Attention decoder function
    hps = self._hps
    cell = tf.contrib.rnn.GRUCell(hps.DimensionHidden, STup=True, initializer=self.InRand)

    pcvg = self.pcvg if hps.mode=="decode" and hps.coverage else None 

    resss, StateOut, DistributionAttention, ppp, coverage = attndcd(inputs, self.StDec, self._EnccState, self.EncoderMask, cell, AttnINt=(hps.mode=="decode"), GenPtr=hps.GenPtr, COvg=hps.coverage, pcvg=pcvg)

    return resss, StateOut, DistributionAttention, ppp, coverage


  
  def SeqtoSEq(self):
    #adding sequence to sequence model
    hps = self._hps
    VocSz = self._vocab.size() 

    with tf.variable_scope('seq2seq'):
      
      
      self.INittn = tf.InitTruncated(stddev=hps.INittnStd)
      self.InRand = tf.random_uniform_initializer(-hps.MgINRnd, hps.MgINRnd, seed=123)

     
      with tf.variable_scope('embd'): 
        embd = tf.get_variable('embd', [VocSz, hps.DEmb], dtype=tf.float32, initializer=self.INittn)

        if hps.mode=="train": self.EmbeddedFn(embd) 
        DecoderInputEmbedd = [tf.nn.embd_lookup(embd, x) for x in tf.unstack(self.BDecoded, axis=1)] 
        EncoderInputsEmbedd= tf.nn.embd_lookup(embd, self.BEncoded) 

      
      self._EnccState = OTpEnccc #encoder is added
      OTpEnccc, forwardState, backwardState = self.EncoderFn(EncInputsEmbedd, self.LensEncoded)

      
      self.StDec = self.Reduction(forwardState, backwardState)

      
      with tf.variable_scope('decoder'):  #adding decoder
        OutDec, self.StDEc, self.DistributionAttention, self.ppp, self.coverage = self.DecoderFn(DecoderInputEmbedd)

      
      with tf.variable_scope('OTpproj'):
        Score = [] 
        vv = tf.get_variable('v', [VocSz], dtype=tf.float32, initializer=self.INittn)
        
        ww = tf.get_variable('ww', [hps.DimensionHidden, VocSz], dtype=tf.float32, initializer=self.INittn)
        WWTT = tf.transpose(ww)
        
        for ii,output in enumerate(OutDec):
          if ii > 0:
            tf.get_variable_scope().reuse_variables()
          Score.append(tf.nn.BpLsX(output, ww, vv)) # apply the linear layer

        VCDistribs = [tf.nn.softmax(g) for g in Score] 

      
      if not FLAGS.GenPtr:
        DistributionFInal = VCDistribs
        
      else: 
        DistributionFInal = self.FinalCalculation(VCDistribs, self.DistributionAttention)



      if hps.mode in ['train', 'eval']:
        # Loss calculation
        with tf.variable_scope('loss'):
          if FLAGS.GenPtr:
            NumOfBatches = tf.range(0, limit=hps.sizebb) 
            EachStepLoss = [] 
            
            for decstp, dist in enumerate(DistributionFInal):

              Tars = self.Tar[:,decstp] 
              indd = tf.stack( (NumOfBatches, Tars), axis=1) 
              gb = tf.gather_nd(dist, indd) 
              losses = -tf.log(gb)
              EachStepLoss.append(losses)

            
            self.ls = AverageAndMask(EachStepLoss, self.DecoderMask)

          else: 
            self.ls = tf.contrib.seq2seq.sequencels(tf.stack(Score, axis=1), self.Tar, self.DecoderMask) # baseline model

          tf.summary.scalar('loss', self.ls)

          
          if hps.coverage: #calculating coverage loss
            with tf.variable_scope('coveragels'):
              self.CalculatingCoverage = CalculatingCoverage(self.DistributionAttention, self.DecoderMask)

              tf.summary.scalar('coveragels', self.CalculatingCoverage)
            self._tot = self.ls + hps.WtCvls * self.CalculatingCoverage

            tf.summary.scalar('totalls', self._tot)

    if hps.mode == "decode":
      
      assert len(DistributionFInal)==1 
      DistributionFInal = DistributionFInal[0]
      PrbTopkkk, self.Idtpk = tf.nn.top_k(DistributionFInal, hps.sizebb*2) 
  
      self.TopKLogPr = tf.log(PrbTopkkk)



  def TrainAddition(self):
    
    
    tvars = tf.trainable_variables()
    MinimizingLoss = self._tot if self._hps.coverage else self.ls
    gradients = tf.gradients(MinimizingLoss, tvars, AggrMeth=tf.AggregationMethod.EXPERIMENTAL_TREE)

   
    with tf.device("/gpu:0"):  #gradient clipping
      grads, NorGl = tf.clip_by_NorGl(gradients, self._hps.MxGrdNrm)

    
    tf.summary.scalar('NorGl', NorGl)  #adding summary

    
    Opt = tf.train.AdagradOptimizer(self._hps.lr, iniValAccum=self._hps.adainiac) #applying adagrad opt
    with tf.device("/gpu:0"):
      self.opTR = Opt.ApplGrad(zip(grads, tvars), gstep=self.gstep, name='trnstp')

  def EmbeddedFn(self, EmbVar):
   
    DTRain = os.path.join(FLAGS.lroot, "train")
    PathOfMetadataPath = os.path.join(DTRain, "vocab_metadata.tsv")
    self._vocab.WRtMtdata(PathOfMetadataPath)
    COnfiNEw = projector.ProjectorConfig()
    Summ = tf.summary.FileWriter(DTRain)



    embd = COnfiNEw.embds.add()
    embd.TEnsNM = EmbVar.name
    embd.PathOfMetadata = PathOfMetadataPath
    projector.visEmb(SummaryWrit, COnfiNEw)

  
  def TrainingRun(self, sess, batch):
    #Running only one iteration for training
    FD = self.FDCreate(batch)
    ToReturn = {
        'summaries': self.Summm,
        'trainop': self.opTR,
        'gstep': self.gstep,
        'loss': self.ls,
    }
    if self._hps.coverage:
      ToReturn['coveragels'] = self.CalculatingCoverage
    return sess.run(ToReturn, FD)

  def RunningEvaluation(self, sess, batch):
    #running one iteration for evaluation
    FD = self.FDCreate(batch)
    ToReturn = {
      'loss': self.ls,
      'gstep': self.gstep,
      'summaries': self.Summm,
        
    }

    if self._hps.coverage:
      ToReturn['coveragels'] = self.CalculatingCoverage
    return sess.run(ToReturn, FD)

  def BuildingGRaph(self):
    
    tf.logging.info('Building graph...')
    tinn = time.time()
    self.Placeholderadding()
    
    with tf.device("/gpu:0"):
      self.SeqtoSEq()
    self.gstep = tf.Variable(0, name='gstep', trainable=False)
    if self._hps.mode == 'train':
      self.TrainAddition()
    self.Summm = tf.summary.mrgal()
    tout = time.time()
    tf.logging.info('Time to build graph: %i seconds', tout - tinn)

  def InOneStepDecode(self, sess, batch, TokLat, EnccState, SttDEc, pcvg):
   
    SZBeam = len(SttDEc)

    
    hiddens = [np.expand_dims(state.h, axis=0) for state in SttDEc] #list of LSTMStateTuples is converted to single LSTMTuple
    cells = [np.expand_dims(state.c, axis=0) for state in SttDEc]
    hmodified = np.concatenate(hiddens, axis=0)  
    hmodified = np.concatenate(cells, axis=0)  
    
    newStDec = tf.contrib.rnn.LSTMStateTuple(hmodified, hmodified)

    feed = {
        self.StDec: newStDec,
        self.BDecoded: np.transpose(np.array([TokLat])),
        self.EncoderMask: batch.EncoderMask,
        self._EnccState: EnccState,
    }

    ToReturn = {
      "probs": self.TopKLogPr,
      "states": self.StDEc,
      "ids": self.Idtpk,
      "DistributionAttention": self.DistributionAttention
    }

    

    if self._hps.coverage:
      feed[self.pcvg] = np.stack(pcvg, axis=0)
      ToReturn['coverage'] = self.coverage

    if FLAGS.GenPtr:
      feed[self.BEncodedExtendVocab] = batch.BEncodedExtendVocab
      feed[self.MaxOutOfVOcab] = batch.MaxOutOfVOcab
      ToReturn['ppp'] = self.ppp  

    results = sess.run(ToReturn, FD=feed) # run the decoder step

    
    NEwStates = [tf.contrib.rnn.LSTMStateTuple(results['states'].c[i, :], results['states'].h[i, :]) for i in xrange(SZBeam)]

   
    assert len(results['DistributionAttention'])==1
    DistributionAttention = results['DistributionAttention'][0].tolist()

    if FLAGS.GenPtr: 
      
      assert len(results['ppp'])==1
      ppp = results['ppp'][0].tolist()
    else:
      ppp = [None for _ in xrange(SZBeam)]

    #this is done for each hypothesis
    if FLAGS.coverage:
      CovgNew = results['coverage'].tolist()
      assert len(CovgNew) == SZBeam
    else:
      CovgNew = [None for _ in xrange(SZBeam)]

    return results['ids'], results['probs'], NEwStates, DistributionAttention, ppp, CovgNew
  

  def EncoderRunning(self, sess, batch):
    #this function returns encoder states and decoder states
    FD = self.FDCreate(batch, Encjs=True) 
    SttDEcc = tf.contrib.rnn.LSTMStateTuple(SttDEcc.c[0], SttDEcc.h[0])
    (EnccState, SttDEcc, gstep) = sess.run([self._EnccState, self.StDec, self.gstep], FD) # run the encoder

    
    return EnccState, SttDEcc


  




def CalculatingCoverage(DistributionAttention, MPaddi):
  #Calculates the coverage loss from the attention distributions.
  
  covlosses = [] 
  coverage = tf.zeros_like(DistributionAttention[0]) 
  
  for cc in DistributionAttention:
    covloss = tf.reduce_sum(tf.minimum(cc, coverage), [1]) 
    covlosses.append(covloss)
    coverage += cc 
  coveragels = AverageAndMask(covlosses, MPaddi)
  return coveragels

def AverageAndMask(values, MPaddi):
  
  ValEachStep = [v * MPaddi[:,decstp] for decstp,v in enumerate(values)]
  DecoderLens = tf.reduce_sum(MPaddi, axis=1)
  ValPerEX = sum(ValEachStep)/DecoderLens 
  return tf.reduce_mean(ValPerEX) 