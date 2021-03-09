import os,sys
import json
import tensorflow as tf
from utils import *
from arguments import *

a3m = parse_a3m(msa_file)
#
# network
#
config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
)
activation = tf.nn.elu
conv1d = tf.layers.conv1d
conv2d = tf.layers.conv2d
with tf.Graph().as_default():

    with tf.name_scope('input'):
        ncol = tf.placeholder(dtype=tf.int32, shape=())
        nrow = tf.placeholder(dtype=tf.int32, shape=())
        msa = tf.placeholder(dtype=tf.uint8, shape=(None,None))
        is_train = tf.placeholder(tf.bool, name='is_train')

    #
    # collect features
    #
    msa1hot  = tf.one_hot(msa, ns, dtype=tf.float32) #NxLx21
    w = reweight(msa1hot, wmin)

    # 1D features
    f1d_seq = msa1hot[0,:,:20] #Lx20 Taking the first sequence
    f1d_pssm = msa2pssm(msa1hot, w) # Getting pssm Lx(21+1)

    f1d = tf.concat(values=[f1d_seq, f1d_pssm], axis=1) #Lx42
    f1d = tf.expand_dims(f1d, axis=0)
    f1d = tf.reshape(f1d, [1,ncol,42])

    # 2D features
    f2d_dca = tf.cond(nrow>1, lambda: fast_dca(msa1hot, w), lambda: tf.zeros([ncol,ncol,442], tf.float32))
    #ncolxncolx442
    f2d_dca = tf.expand_dims(f2d_dca, axis=0)

    f2d = tf.concat([tf.tile(f1d[:,:,None,:], [1,1,ncol,1]), 
                    tf.tile(f1d[:,None,:,:], [1,ncol,1,1]),
                    f2d_dca], axis=-1)
    f2d = tf.reshape(f2d, [1,ncol,ncol,442+2*42])

    with tf.Session(config=config) as sess:
        #saver.restore(sess, ckpt)
        out = sess.run(f2d, feed_dict = {msa : a3m, ncol : a3m.shape[1], nrow : a3m.shape[0], is_train : 0})
