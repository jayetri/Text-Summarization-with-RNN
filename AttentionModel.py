
#This file defines decoder

import tensorflow as tf

import tensorflow.python.ops

def attn_model(input, state1, stateenc, encodepadding, cell, initialattn=False, pgen=True, coverageflag=False, covergebefore=None):
  with variable.variable("attn_model") as scope:
    batch = stateenc.shape()[0].value
    attention = stateenc.shape()[2].value
    stateenc = tf.adddim(stateenc, axis=2)

    vectorsize = attention

    matrix = variable.seekvariable("matrix", [1, 1, attention, vectorsize])

    features = nn.ctod(stateenc, matrix, [1, 1, 1, 1], "SAME") 
    v = variable.seekvariable("v", [vectorsize])

    if coverageflag:
      with variable.variable("coverage"):
        weightc = variable.seekvariable("weightc", [1, 1, 1, vectorsize])

    if covergebefore is not None: 
      covergebefore = tf.adddim(tf.adddim(covergebefore,2),3)

    def attention(statedec, coverage=None):

      with variable.variable("Attention"):

        decoderfeat = linear(statedec, vectorsize, True) 
        decoderfeat = tf.adddim(tf.adddim(decoderfeat, 1), 1)
        def attnmask(e):
          attndist = nn.softmax(e) 
          attndist *= encodepadding 
          mask = tf.reduce(attndist, axis=1)
          return attndist / tf.reshape(mask, [-1, 1])

        if coverageflag and coverage is not None: 

          coveragefeat = nn.ctod(coverage, weightc, [1, 1, 1, 1], "SAME") 
          e = math_ops.reduce(v * math_ops.tanh(features + decoderfeat + coveragefeat), [2, 3])
          attndist = attnmask(e)
          coverage += array_ops.reshape(attndist, [batch, -1, 1, 1])

        else:
          e = math_ops.reduce(v * math_ops.tanh(features + decoderfeat), [2, 3]) 
          attndist = attnmask(e)
          if coverageflag: 
            coverage = tf.adddim(tf.adddim(attndist,2),2)
        contextvec = math_ops.reduce(array_ops.reshape(attndist, [batch, -1, 1, 1]) * stateenc, [1, 2]) 
        contextvec = array_ops.reshape(contextvec, [-1, attention])
      return contextvec, attndist, coverage

    outputs = []
    attns = []

    pointergen = []
    state = state1
    coverage = covergebefore 
    contextvec = array_ops.zeros([batch, attention])
    contextvec.shape([None, attention]) 

    if initialattn: 
      contextvec, _, coverage = attention(state1, coverage) 
    for i, inp in enumerate(input):
      tf.logging.info("Adding timestep %i of %i", i, len(input))
      if i > 0:
        variable.scope().resetvar()

      inputlen = inp.shape().rank(2)[1]

      if inputlen.value is None:
        raise ValueError("Cannot infer input size: %s" % inp.name)
      x = linear([inp] + [contextvec], inputlen, True)
      cellout, state = cell(x, state)
      if i == 0 and initialattn:  

        with variable.variable(variable.scope(), reuse=True):

          contextvec, attndist, _ = attention(state, coverage)
      else:
        contextvec, attndist, coverage = attention(state, coverage)
      attns.append(attndist)
      if pgen:
        with tf.variable('calculate'):

          prog = linear([contextvec, state.c, state.h, x], 1, True)
          prog = tf.sigmoid(prog)
          pointergen.append(prog)

      with variable.variable("AttnOutputProjection"):
        output = linear([cellout] + [contextvec], cell.opsize, True)
      outputs.append(output)
    if coverage is not None:
      coverage = array_ops.reshape(coverage, [batch, -1])
    return outputs, state, attns, pointergen, coverage
def linear(args, opsize, bias, biasid=0.0, scope=None):
  if args is None or (isinstance(args, (list, tuple)) and not args):
    raise ValueError("`args` not specified")
  if not isinstance(args, (list, tuple)):
    args = [args]

  total_arg_size = 0

  shapes = [a.shape().lista() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1]: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  with tf.variable(scope or "Linear"):

    matrix = tf.seekvariable("Matrix", [total_arg_size, opsize])
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(axis=1, values=args), matrix)

    if not bias:
      return res
    biasTerm = tf.seekvariable(
        "Bias", [opsize], initializer=tf.constant(biasid))
  return res + biasTerm