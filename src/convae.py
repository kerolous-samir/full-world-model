#Importing Libraries
import numpy as np
import tensorflow as tf
import json
import os

#I did implement that (Kerolous)
class CNNVAE(object):
    def __init__(self, z_size=32, batch_size=1, learning_rate=0.0001, kl_tolerance=0.5, is_training=False, reuse= False, gpu_mode= False):
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance
        self.is_learning = is_learning
        self.reuse = reuse
        with tf.compat.v1.variable_scope(name_or_scope='conv_vae',reuse=self.reuse):
            if not gpu_mode:
                with tf.device("/cpu:0"):
                    tf.compat.v1.logging.info("model using cpu.")
                    self._build_graph()
            else:
                tf.compat.v1.logging.info("model using gpu.")
                self._build_graph()
        self._init_session()
    
    def _build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            #Encode
            self.x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None,64,64,3])
            h = tf.compat.v1.layers.conv2d(self.x, 32, 4, 2, activation=tf.nn.relu, name='enc_conv1')
            h = tf.compat.v1.layers.conv2d(h, 64, 4, 2, activation=tf.nn.relu, name='enc_conv2')
            h = tf.compat.v1.layers.conv2d(h, 128, 4, 2, activation=tf.nn.relu, name='enc_conv3')
            h = tf.compat.v1.layers.conv2d(h, 256, 4, 2, activation=tf.nn.relu, name='enc_conv4')
            h = tf.compat.v1.reshape(h, [-1,2*2*256])
            self.mu = tf.compat.v1.layers.dense(h, self.z_size, name='enc_fc_mu')
            self.log_var = tf.compat.v1.layers.dense(h, self.z_size, name='enc_fc_log_var')
            self.sigma = tf.compat.v1.exp(self.log_var / 2.0)
            self.epsilon = tf.compat.v1.random_normal([self.batch_size,self.z_size])
            self.z = self.mu + self.sigma * self.epsilon
            #Decode
            h = tf.compat.v1.layers.dense(self.z, 1024, name='dec_fc')
            h = tf.compat.v1.reshape([-1,1,1,1024])
            h = tf.compat.v1.layers.conv2d_transpose(h, 128, 5, 2, activation=tf.nn.relu, name='dec_conv1')
            h = tf.compat.v1.layers.conv2d_transpose(h, 64, 5, 2, activation=tf.nn.relu, name='dec_conv2')
            h = tf.compat.v1.layers.conv2d_transpose(h, 32, 6, 2, activation=tf.nn.relu, name='dec_conv3')
            self.y = tf.compat.v1.layers.conv2d_transpose(h, 3, 6, 2, activation=tf.nn.sigmoid, name='dec_conv4')
            #Train
        if is_training:
            self.global_step = tf.compat.v1.Variable(0, name='global_step', trainable=False)
            self.mse_loss = tf.compat.v1.reduce_sum(tf.compat.v1.square(self.x - self.y),reduction_indices=[1,2,3])
            self.mse_loss = tf.compat.v1.reduce_mean(self.mse_loss)
            self.kl_loss = -0.5 * tf.compat.v1.reduce_sum((1 + self.log_var - tf.compat.v1.square(self.mu) - tf.compat.v1.exp(self.log_var)), reduction_indices= 1 )
            self.kl_loss =  tf.compat.v1.reduce_mean(self.kl_loss)
            self.kl_loss = tf.compat.v1.maximum(self.kl_loss, self.kl_tolerance * self.z_size)
            self.loss = self.mse_loss + self.kl_loss
            self.lr = tf.compat.v1.Variable(self.learning_rate, trainable=False)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
            grads = self.optimizer.compute_gradients(self.loss)
            self.train_op = self.optimizer.apply_gradients(grads, global_step=self.global_step, name='train_step')
        self.init = tf.compat.v1.global_variables_initializer()
    
    #From here I didn't implement that it was already done by the course    
   
    def _init_session(self):
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

      # Making a method that closes a TensorFlow session
      def close_sess(self):
        self.sess.close()

      # Making a method that encodes a raw frame x into the latent space representation
      def encode(self, x):
        return self.sess.run(self.z, feed_dict={self.x: x})

      # Making a method that encodes a raw frame x into the mean and logvariance parts of the latent vectors space
      def encode_mu_logvar(self, x):
        (mu, logvar) = self.sess.run([self.mu, self.logvar], feed_dict={self.x: x})
        return mu, logvar

      # Making a method that decodes a latent vector z into the reconstructed frame
      def decode(self, z):
        return self.sess.run(self.y, feed_dict={self.z: z})

      # Making a method that gets the training parameters of the VAE model
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

      # Making a method that gets the random parameters of the VAE model
      def get_random_model_params(self, stdev=0.5):
        _, mshape, _ = self.get_model_params()
        rparam = []
        for s in mshape:
          rparam.append(np.random.standard_cauchy(s)*stdev)
        return rparam

      # Making a method that sets specific weights to chosen values in the VAE model
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

      # Making a method that loads saved VAE weights from a JSON file
      def load_json(self, jsonfile='Weights/vae_weights.json'):
        with open(jsonfile, 'r') as f:
          params = json.load(f)
        self.set_model_params(params)

      # Making a method that saves trained VAE weights into a JSON file
      def save_json(self, jsonfile='Weights/vae_weights.json'):
        model_params, model_shapes, model_names = self.get_model_params()
        qparams = []
        for p in model_params:
          qparams.append(p)
        with open(jsonfile, 'wt') as outfile:
          json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))

      # Making a method that sets some parameters to random values in the VAE model (this is usually done at the beginning of the training process)
      def set_random_params(self, stdev=0.5):
        rparam = self.get_random_model_params(stdev)
        self.set_model_params(rparam)

      # Making a method that saves the model into a chosen directory
      def save_model(self, model_save_path):
        sess = self.sess
        with self.g.as_default():
          saver = tf.train.Saver(tf.global_variables())
        checkpoint_path = os.path.join(model_save_path, 'vae')
        tf.logging.info('saving model %s.', checkpoint_path)
        saver.save(sess, checkpoint_path, 0)

      # Making a method that loads a saved checkpoint that restores all saved trained VAE weights
      def load_checkpoint(self, checkpoint_path):
        sess = self.sess
        with self.g.as_default():
          saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        print('loading model', ckpt.model_checkpoint_path)
        tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
