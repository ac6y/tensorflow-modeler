#!/usr/bin/env python
"""
This is a TensorFlow rapid model prototyping tool intended to be relatively
general-purpose and very easy to set up.
"""
# 2016-07-19 M. Okun
# Copyright 2016 M. Okun
# All Rights Reserved

# License follows for original TensorFlow Wide & Deep Tutorial, from which this
# code was adapted, and which is described at
# https://www.tensorflow.org/versions/r0.9/tutorials/wide_and_deep/index.html

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# future
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# system
# import subprocess
import os
import tempfile

# utils
# import webbrowser
import logging
import yaml

# pandas
import pandas as pd

# tensorflow
import tensorflow as tf
from tensorflow.contrib.learn import monitors
from tensorflow.contrib.layers import (
    real_valued_column,
    sparse_column_with_hash_bucket,
    sparse_column_with_integerized_feature,
    # sparse_column_with_keys,
    # sparse_feature_cross,
    embedding_column,
    # crossed_column,
)


# sklearn
# from sklearn import metrics
# from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import cross_val_score
# from sklearn.metrics import roc_curve, auc, roc_auc_score


####################
# FLAGS
####################
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_bool(
    "verbose",
    True,
    "Show INFO and WARN messages")

# --- Input and config
tf.app.flags.DEFINE_string(
    "model_conf_file",
    "MODEL_CONF.yaml",
    "Path to the Model Config Yaml file, default:{}")

tf.app.flags.DEFINE_string(
    "data_conf_file",
    "DATA_CONF.yaml",
    "Path to the Data Config Yaml file, default:{}")

tf.app.flags.DEFINE_string(
    "train_n_test_file",
    None,
    "Path to the combined training and testing data (will be split).")

# --- Output
tf.app.flags.DEFINE_string(
    "model_dir",
    None,
    "Base directory for output models.")

tf.app.flags.DEFINE_string(
    "weights_file",
    None,
    "File path for serialized model weights output.")

# --- Model
tf.app.flags.DEFINE_string(
    "model_type",
    None,
    "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")

tf.app.flags.DEFINE_integer(
    "train_steps",
    None,
    "Number of training steps.")

tf.app.flags.DEFINE_integer(
    "learn_rate",
    None,
    "Learning Rate")


########################
# Load Model Config Yaml
########################
with open(FLAGS.model_conf_file) as modelfile:
    MODEL_CONF = yaml.load(modelfile)

##########################
# Load Feature Config Yaml
##########################
with open(FLAGS.data_conf_file) as datafile:
    DATA_CONF = yaml.load(datafile)

##############################################################
# Get settings from conf files if not overidden with cli flags
##############################################################

# Get a model directory
MODEL_DIR = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir

TRAIN_N_TEST_FILE = os.path.abspath(DATA_CONF['train_n_test_file']) \
    if not FLAGS.train_n_test_file else FLAGS.train_n_test_file
MODEL_TYPE = MODEL_CONF['model_type']\
    if not FLAGS.model_type else FLAGS.model_type
WEIGHTS_FILE = MODEL_CONF['weights_file']\
    if not FLAGS.weights_file else FLAGS.weights_file
TRAIN_STEPS = MODEL_CONF['train_steps']\
    if not FLAGS.train_steps else FLAGS.train_steps
LEARN_RATE = MODEL_CONF['learn_rate']\
    if not FLAGS.learn_rate else FLAGS.learn_rate


# deep params
HIDDEN_UNITS = MODEL_CONF['hidden_units']
EMBEDDING_DIMENSION = MODEL_CONF['embedding_dimension']


#################################################################
# Features and labels
#################################################################
LABEL_COLUMN = DATA_CONF['LABEL_COLUMN']
MULTI_CATEGORY_COLUMNS = DATA_CONF['MULTI_CATEGORY_COLUMNS']
BINARY_COLUMNS = DATA_CONF['BINARY_COLUMNS']
CONTINUOUS_COLUMNS = DATA_CONF['CONTINUOUS_COLUMNS']

# setup exponential decay function
# (from https://terrytangyuan.github.io/2016/03/14/scikit-flow-intro/)
GLOBAL_STEP = tf.Variable(0, name="global_step", trainable=False)


def exp_decay(global_step):
    return tf.train.exponential_decay(
        learning_rate=LEARN_RATE, global_step=global_step,
        decay_steps=100, decay_rate=0.001)


#################################################################
# General-purpose code
#################################################################


def build_estimator(model_dir=MODEL_DIR):
    """
    Build an estimator using
    CONTINTUOUS_COLUMNS, BINARY_COLUMNS and MULTI_CATEGORY_COLUMNS.
    """
    bucketized_columns = \
        [sparse_column_with_hash_bucket(col, 1000)
         for col in MULTI_CATEGORY_COLUMNS] + \
        [sparse_column_with_integerized_feature(col, bucket_size=2)
         for col in BINARY_COLUMNS]

    real_valued_columns = \
        [real_valued_column(col) for col in CONTINUOUS_COLUMNS]

    crossed_columns = \
        []

    # Wide columns and deep columns.
    wide_columns = \
        bucketized_columns + \
        real_valued_columns + \
        crossed_columns

    # embedding columns for hash_bucket columns
    deep_columns = \
        [embedding_column(col, dimension=EMBEDDING_DIMENSION)
         for col in bucketized_columns] + \
        real_valued_columns + \
        crossed_columns

    if MODEL_TYPE == "wide":
        print('Creating wide LinearClassifier model...\n')
        model = tf.contrib.learn.LinearClassifier(
            model_dir=model_dir,
            n_classes=2,
            feature_columns=wide_columns,
            # optimizer=tf.train.GradientDescentOptimizer(
            #     learning_rate=FLAGS.learn_rate)
            # optimizer=tf.train.FtrlOptimizer(
            #     learning_rate=LEARN_RATE,
            #     l1_regularization_strength=0.0,
            #     l2_regularization_strength=0.0),
        )

    elif MODEL_TYPE == "deep":
        print('Creating deep DNNClassifier model...\n')
        model = tf.contrib.learn.DNNClassifier(
            model_dir=model_dir,
            n_classes=2,
            feature_columns=deep_columns,
            hidden_units=HIDDEN_UNITS,
            # optimizer=tf.train.FtrlOptimizer(
            #     learning_rate=LEARN_RATE,
            #     l1_regularization_strength=0.0,
            #     l2_regularization_strength=0.0),
        )
    else:
        print('Creating deepNwide DNNLinearCombinedClassifier model...\n')
        model = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            n_classes=2,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=HIDDEN_UNITS,
            # optimizer=tf.train.FtrlOptimizer(
            #     learning_rate=LEARN_RATE,
            #     l1_regularization_strength=0.0,
            #     l2_regularization_strength=0.0),
        )

    return model


def input_fn(df):
    """Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name
    # (k) to the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values.astype(float))
                       for k in CONTINUOUS_COLUMNS}

    # Creates a dictionary mapping from each categorical feature column name
    # (k) to the values of that column stored in a tf.SparseTensor.
    multi_category_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values.astype(str),
        shape=[df[k].size, 1])
                           for k in MULTI_CATEGORY_COLUMNS}

    binary_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values.astype(int),
        shape=[df[k].size, 1])
                   for k in BINARY_COLUMNS}
    # DEBUG
    print('multi_category_cols:')
    print({k: df[k].dtype for k in multi_category_cols})
    print('binary_cols:')
    print({k: df[k].dtype for k in binary_cols})
    print('continuous_cols:')
    print({k: df[k].dtype for k in continuous_cols})

    feature_cols = dict(continuous_cols)
    feature_cols.update(binary_cols)
    feature_cols.update(multi_category_cols)

    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].astype(int).values)

    # Returns the feature columns and the label.
    return feature_cols, label


def train_and_eval():
    """Train and evaluate the model"""
    # Read data
    df_train_test = pd.read_csv(
        tf.gfile.Open(TRAIN_N_TEST_FILE),
        low_memory=False)

    # Split training and testing data
    (df_train, df_test) = train_test_split(df_train_test, test_size=0.25)

    # Build the estimator
    print("\nBuilding Estimator...")
    model = build_estimator(MODEL_DIR)

    # Create a validation monitor
    val_mon = monitors.ValidationMonitor(
        input_fn=lambda: input_fn(df_test),
        every_n_steps=10, early_stopping_rounds=100)

    # Fit and evaluate
    print("\nFitting with {} steps".format(TRAIN_STEPS))
    model.fit(
        input_fn=lambda: input_fn(df_train),
        steps=TRAIN_STEPS,
        # monitors=[val_mon]) # why doesn't this work when it's passed in?
        )

    print("\nEvaluating...")
    results = model.evaluate(
        input_fn=lambda: input_fn(df_test),
        steps=1)

    return model, results


# --- Helpers
def print_info(results):
    """
    Print some helpful output, given a results objects
    """
    print("model directory = {}".format(MODEL_DIR))
    print("\n***************")
    print("Model Type: {}".format(MODEL_TYPE))
    print("Steps: {}".format(TRAIN_STEPS))
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))
    print("***************\n")

    print("\n#####################################")
    print("To open Tensorboard with this model:")
    print("python -m webbrowser http://0.0.0.0:6006")
    print("tensorboard --logdir {}".format(MODEL_DIR))
    print("#####################################\n")


def main(_):
    """ Main Functionality """

    config = tf.contrib.learn.RunConfig()

    print("model directory = {}".format(MODEL_DIR))

    # Set verbosity
    if FLAGS.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.ERROR)

    # --- Run
    model, results = train_and_eval()

    # --- Final Output
    print_info(results)

    # DNNLinearCombinedClassifier doesn't have m.weights_
#    if MODEL_TYPE in ["deep", "wide"]:
#        print("weights:\n{}".format(model.weights_))

#        if WEIGHTS_FILE:
#            with open(_WEIGHTS_FILE, 'wb') as weightfile:
#                # todo we can make a nicer weights file
#                weightfile.write(str(model.weights_))

if __name__ == "__main__":
    # tf.app.run()
    # init = tf.initialize_all_variables()
    tf.app.run()
