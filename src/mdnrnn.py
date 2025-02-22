import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import json
from collections import namedtuple


#I did implement that (Kerolous)

class MDNRNN(object):
    def __init__(self, hps, reuse= False, gpu_mode= False):
        self.hps = hps
        self.reuse = reuse
        with tf.compat.v1.variable_scope(name_or_scope='mdn_rnn',reuse=self.reuse):
            if not gpu_mode:
                with tf.device("/cpu:0"):
                    tf.compat.v1.logging.info("model using cpu.")
                    self.g = tf.Graph()
                    with self.g.as_default():
                        self.build_model(hps)
            else:
                tf.compat.v1.logging.info("model using gpu.")
                self.g = tf.Graph()
                with self.g.as_default():
                    self.build_model(hps)
        self._init_session()
        
    def build_model(self,hps):
        self.num_mixture = self.hps.num_mixture
        KMIX = self.num_mixture
        LENGTH = self.hps.max_seq_len
        INWIDTH = self.hps.input_seq_width
        OUTWIDTH = self.hps.output_seq_width
        if self.hps.is_training:
            self.global_step = tf.compat.v1.Variable(0, name='global_step', trainable=False)
        cell_fn = tfa.rnn.LayerNormLSTMCell
        use_recurrent_dropout = False if self.hps.use_recurrent_dropout == 0 else True
        use_input_dropout = False if self.hps.use_input_dropout == 0 else True
        use_output_dropout = False if self.hps.use_output_dropout == 0 else True
        use_layer_norm = False if self.hps.use_layer_norm == 0 else True
        if use_recurrent_dropout:
            cell = cell_fn(self.hps.rnn_size, layer_norm= use_layer_norm, recurrent_dropout=self.hps.recurrent_dropout_prob)
        else:
            cell = cell_fn(self.hps.rnn_size, layer_norm= use_layer_norm)
        if use_input_dropout:
            cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.hps.input_dropout_prob)
        if use_output_dropout:
            cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.hps.output_keep_prob)
        self.cell = cell
        self.sequence_length = LENGTH
        self.input_x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.sequence_length, INWIDTH])
        self.output_x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.hps.batch_size, self.sequence_length, OUTWIDTH])
        actual_input_x = self.input_x
        self.initial_state = cell.zero_state(batch_size=self.hps.batch_size, dtype=tf.float32)
        NOUT = OUTWIDTH * KMIX * 3
        with tf.compat.v1.variable_scope("RNN"):
            output_w = tf.compat.v1.get_variable(name="output_w",shape=[self.hps.rnn_size, NOUT])
            output_b = tf.compat.v1.get_variable(name="output_b",shape=[NOUT])
        output, last_state = tf.compat.v1.nn.dynamic_rnn(cell=cell,
                                                        inputs=actual_input_x,
                                                        sequence_length=self.sequence_length,
                                                        initial_state=self.initial_state,
                                                        dtype=tf.float32,
                                                        swap_memory=True,
                                                        scope="RNN")
        output = tf.reshape(output, [-1, self.hps.rnn_size])
        output = tf.compat.v1.nn.xw_plus_b(output, output_w, output_b)
        output = tf.reshape(output, [-1, KMIX * 3])
        self.final_state = last_state
        def get_mdn_coef(output):
            logmix , logmean , logstd = tf.split(output ,3 , 1)
            logmix -= tf.reduce_logsumexp(logmix ,1 ,keepdims=True)
            return logmix , logmean , logstd
        out_logmix , out_logmean , out_logstd = get_mdn_coef(output)
        self.out_logmix = out_logmix
        self.out_logmean = out_logmean
        self.out_logstd = out_logstd
        LogSqrtTwoPi = np.log(np.sqrtrt(2.0 * np.pi))
        def tf_lognormally(y, mean, std):
            return -0.5 *((y - mean)/ tf.exp(logstd) ** 2) - std - LogSqrtTwoPi
        def loss_fucn(mix, mean, std, y):
            v = mix + tf_lognormally(y, mean, std)
            v = tf.reduce_logsumexp(v ,1 ,keepdims=True)
            return -tf.reduce_mean(v)
        flat_target_data = tf.reshape(self.output_x, [-1,1])
        loss = loss_fucn(out_logmix, out_logmean, out_logstd, flat_target_data)
        self.cost = tf.reduce_mean(loss)
        if self.hps.is_training == 1:
            self.lr = tf.compat.v1.Variable(self.hps.learning_rate, trainable=False)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
            gvs = self.optimizer.compute_gradients(self.cost)
            capped_gvs = [tf.clip_by_value(grad, -self.hps.grad_clip, self.hps.grad_clip) for grad , var in gvs]
            self.train_op = self.optimizer.apply_gradients(capped_gvs, global_step=self.global_step, name='train_step')
        self.init = tf.compat.v1.global_variables_initializer()

  #From here I didn't implement that it was already done by the course      
  
  # Making a method that initializes the tensorflow graph session, used to run the MDN-RNN model inference or training
    def _init_session(self):
      self.sess = tf.Session(graph=self.g)
      self.sess.run(self.init)

    # Making a method that closes the tensorflow graph session currently running (closing a session is necessary to overcome nested graphs)
    def close_sess(self):
      self.sess.close()

    # Making a method that extracts all trainable variables from the RNN graph into a python list
    def get_model_params(self):
      model_names = []
      model_params = []
      model_shapes = []
      with self.g.as_default():
        t_vars = tf.trainable_variables()
        for var in t_vars:
          param_name = var.name
          p = self.sess.run(var)
          model_names.append(param_name)
          params = np.round(p*10000).astype(np.int).tolist()
          model_params.append(params)
          model_shapes.append(p.shape)
      return model_params, model_shapes, model_names

    # Making a method that randomly initializes the RNN parameters
    def get_random_model_params(self, stdev=0.5):
      _, mshape, _ = self.get_model_params()
      rparam = []
      for s in mshape:
        rparam.append(np.random.standard_cauchy(s)*stdev)
      return rparam

    # Making a method that sets some parameters to random values in the RNN model (this is usually done at the beginning of the training process)
    def set_random_params(self, stdev=0.5):
      rparam = self.get_random_model_params(stdev)
      self.set_model_params(rparam)

    # Making a method that sets specific weights to chosen values in the RNN model
    def set_model_params(self, params):
      with self.g.as_default():
        t_vars = tf.trainable_variables()
        idx = 0
        for var in t_vars:
          pshape = self.sess.run(var).shape
          p = np.array(params[idx])
          assert pshape == p.shape, "inconsistent shape"
          assign_op = var.assign(p.astype(np.float)/10000.)
          self.sess.run(assign_op)
          idx += 1

    # Making a method that loads saved RNN weights from a JSON file
    def load_json(self, jsonfile='Weights/rnn_weights.json'):
      with open(jsonfile, 'r') as f:
        params = json.load(f)
      self.set_model_params(params)

    # Making a method that saves trained RNN weights into a JSON file
    def save_json(self, jsonfile='Weights/rnn_weights.json'):
      model_params, model_shapes, model_names = self.get_model_params()
      qparams = []
      for p in model_params:
        qparams.append(p)
      with open(jsonfile, 'wt') as outfile:
        json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))

  # Setting the Hyperparameters

  MODE_ZCH = 0
  MODE_ZC = 1
  MODE_Z = 2
  MODE_Z_HIDDEN = 3
  MODE_ZH = 4
  HyperParams = namedtuple('HyperParams', ['num_steps',
                                           'max_seq_len',
                                           'input_seq_width',
                                           'output_seq_width',
                                           'rnn_size',
                                           'batch_size',
                                           'grad_clip',
                                           'num_mixture',
                                           'learning_rate',
                                           'decay_rate',
                                           'min_learning_rate',
                                           'use_layer_norm',
                                           'use_recurrent_dropout',
                                           'recurrent_dropout_prob',
                                           'use_input_dropout',
                                           'input_dropout_prob',
                                           'use_output_dropout',
                                           'output_dropout_prob',
                                           'is_training',
                                          ])

  # Making a function that returns all the default hyperparameters of the MDN-RNN model

  def default_hps():
    return HyperParams(num_steps=2000,
                       max_seq_len=1000,
                       input_seq_width=35,
                       output_seq_width=32,
                       rnn_size=256,
                       batch_size=100,
                       grad_clip=1.0,
                       num_mixture=5,
                       learning_rate=0.001,
                       decay_rate=1.0,
                       min_learning_rate=0.00001,
                       use_layer_norm=0,
                       use_recurrent_dropout=0,
                       recurrent_dropout_prob=0.90,
                       use_input_dropout=0,
                       input_dropout_prob=0.90,
                       use_output_dropout=0,
                       output_dropout_prob=0.90,
                       is_training=1)

  # Getting and sampling these default hyperparameters

  hps_model = default_hps()
  hps_sample = hps_model._replace(batch_size=1, max_seq_len=1, use_recurrent_dropout=0, is_training=0)

  # Making a function that samples the index of a probability distribution function (pdf)
  def get_pi_idx(x, pdf):
    N = pdf.size
    accumulate = 0
    for i in range(0, N):
      accumulate += pdf[i]
      if (accumulate >= x):
        return i
    print('error with sampling ensemble')
    return -1

  # Making a function that samples sequences of inputs for the RNN model using a pre-trained VAE model

  def sample_sequence(sess, s_model, hps, init_z, actions, temperature=1.0, seq_len=1000):  
    OUTWIDTH = hps.output_seq_width
    prev_x = np.zeros((1, 1, OUTWIDTH))
    prev_x[0][0] = init_z
    prev_state = sess.run(s_model.initial_state)
    strokes = np.zeros((seq_len, OUTWIDTH), dtype=np.float32)
    for i in range(seq_len):
      input_x = np.concatenate((prev_x, actions[i].reshape((1, 1, 3))), axis=2)
      feed = {s_model.input_x: input_x, s_model.initial_state:prev_state}
      [logmix, mean, logstd, next_state] = sess.run([s_model.out_logmix, s_model.out_mean, s_model.out_logstd, s_model.final_state], feed)
      logmix2 = np.copy(logmix)/temperature
      logmix2 -= logmix2.max()
      logmix2 = np.exp(logmix2)
      logmix2 /= logmix2.sum(axis=1).reshape(OUTWIDTH, 1)
      mixture_idx = np.zeros(OUTWIDTH)
      chosen_mean = np.zeros(OUTWIDTH)
      chosen_logstd = np.zeros(OUTWIDTH)
      for j in range(OUTWIDTH):
        idx = get_pi_idx(np.random.rand(), logmix2[j])
        mixture_idx[j] = idx
        chosen_mean[j] = mean[j][idx]
        chosen_logstd[j] = logstd[j][idx]
      rand_gaussian = np.random.randn(OUTWIDTH)*np.sqrt(temperature)
      next_x = chosen_mean+np.exp(chosen_logstd)*rand_gaussian
      strokes[i,:] = next_x
      prev_x[0][0] = next_x
      prev_state = next_state
    return strokes

  # Making a function that returns the initial state of the RNN model

  def rnn_init_state(rnn):
    return rnn.sess.run(rnn.initial_state)

  # Making a function that returns the final state of the RNN model

  def rnn_next_state(rnn, z, a, prev_state):
    input_x = np.concatenate((z.reshape((1, 1, 32)), a.reshape((1, 1, 3))), axis=2)
    feed = {rnn.input_x: input_x, rnn.initial_state:prev_state}
    return rnn.sess.run(rnn.final_state, feed)

  # Making a function that returns the size of the RNN output depending on the mode

  def rnn_output_size(mode):
    if mode == MODE_ZCH:
      return (32+256+256)
    if (mode == MODE_ZC) or (mode == MODE_ZH):
      return (32+256)
    return 32

  # Making a function that returns the RNN output depending on the mode

  def rnn_output(state, z, mode):
    if mode == MODE_ZCH:
      return np.concatenate([z, np.concatenate((state.c,state.h), axis=1)[0]])
    if mode == MODE_ZC:
      return np.concatenate([z, state.c[0]])
    if mode == MODE_ZH:
      return np.concatenate([z, state.h[0]])
    return z
