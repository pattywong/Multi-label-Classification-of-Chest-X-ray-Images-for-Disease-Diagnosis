
# python main.py mode=<mode> method=<method> weights=<weights>

import numpy as np
import pandas as pd
import os, sys, math, datetime
from model.densenet import *
from dataset.dataset import *
from utils.utils import *
from utils.visualize import *
from eval.eval import *
from custom_loss.custom_loss1 import *

import keras
from keras import backend as K
from keras.models import model_from_json
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# GPU Settings
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
        
if __name__ == "__main__":
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(
                        description='Multi-label Classification of Chest X-ray Images for Disease Diagnosis')
    parser.add_argument("--mode",
                        metavar="<mode>", 
                        help="'train' or 'test'", 
                        required=True)
    parser.add_argument("--method",
                        metavar="<method>", 
                        help="'baseline', 'labelpowerset', 'classifierchains' or 'customloss'", 
                        required=True)
    parser.add_argument("--weights",
                        metavar="<weights>", 
                        help="'latest', 'imagenet' or /path/to/weights", 
                        required=True)
    args = parser.parse_args()
    
    # method
    if args.method == 'baseline' or args.method == 'customloss':
        method = Baseline()
    elif args.method == 'labelpowerset':
        method = Labelpowerset()
    elif args.method == 'classifierchains':
        method = Classifierchains()
    # mode
    if args.mode == 'train':
        method.train()
    elif args.mode == 'test':
        method.test()
