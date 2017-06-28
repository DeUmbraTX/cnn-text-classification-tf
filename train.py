#! /usr/bin/env python
import os
import sys
import time
import logging
import datetime
from os.path import join

import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

import data_helpers
from text_cnn import TextCNN

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
tf.flags.DEFINE_string('data_dir', '/storage/twitter/streams/french-elections/cnn-clf/v3', "Directory with all the data.")
tf.flags.DEFINE_boolean("combine_langs", False, "Combine english and french language files.")
tf.flags.DEFINE_boolean("balance_two_class", False, "")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("use_orig", False, "Use original pre-fork setup for MR polarity data.")
tf.flags.DEFINE_string("run_id", "", "identifier for this run")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
if FLAGS.data_dir:
    FLAGS.positive_data_file = ''
    FLAGS.negative_data_file = ''

logging.info("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    logging.info("{}={}".format(attr.upper(), value))
logging.info("")

"""
# orignal MR data
2017-04-08 16:48:55,601 : INFO : Vocabulary Size: 18758
2017-04-08 16:48:55,602 : INFO : Train/Dev split: 9596/1066

# tweets (mx toks 512)
2017-04-08 18:37:27,370 : INFO : Vocabulary Size: 45308
2017-04-08 18:37:27,370 : INFO : Train/Dev split: 7182/798

# tweets (mx toks 1000)
2017-04-08 23:40:42,377 : INFO : Vocabulary Size: 49698
2017-04-08 23:40:42,377 : INFO : Train/Dev split: 7182/798
"""

# Data Preparation
# ==================================================

# Load data
logging.info("Loading data...")
if FLAGS.use_orig:
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
else:
    if FLAGS.balance_two_class:
        x_path = join(FLAGS.data_dir, 'x.txt')
        y_path = join(FLAGS.data_dir, 'y.txt')
        x_text, y = data_helpers.load_data_and_labels_twoclass(x_path, y_path)
    elif not FLAGS.combine_langs:
        x_path = join(FLAGS.data_dir, 'x.txt')
        y_path = join(FLAGS.data_dir, 'y.txt')
        x_text, y = data_helpers.load_data_and_labels_v2(x_path, y_path)
    else:
        x_text, y = data_helpers.load_data_and_labels_v3(FLAGS.data_dir)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

logging.info("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
logging.info("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
logging.info("max doc length:  {:d}".format(max_document_length))

#sys.exit(0)

# t0 = x_text[0]
# t1 = list(vocab_processor.reverse([x[0]]))[0]

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        if not FLAGS.run_id:
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        else:
            out_dir = os.path.abspath(os.path.join(FLAGS.data_dir, 'runs', FLAGS.run_id))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        logging.info("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)

            logging.info("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)

            logging.info("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))

            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                logging.info("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                logging.info("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                logging.info("Saved model checkpoint to {}\n".format(path))
