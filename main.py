''' 
    Title: Multi-label learning algorithms for multi-label classification of chest X-ray images 
    Author: Patsaya Wongchaisirikul (Patty)
    Github: https://github.com/pattywong
    Linkedin: https://www.linkedin.com/in/patsaya/
    Commands
        : python main.py --mode=train --method=baseline
        : python main.py --mode=train --method=labelpowerset
        : python main.py --mode=train --method=customloss
        : python main.py --mode=test --method=baseline --weights=weights_file_name
        : python main.py --mode=test --method=labelpowerset --weights=weights_file_name
        : python main.py --mode=test --method=customloss --weights=weights_file_name
'''

import numpy as np
import pandas as pd
import os, sys, math, datetime
from model.densenet121 import *
from utils.dataset import *
from utils.utils import *
from utils.visualize import *
from utils.eval import *
from baseline.custom_loss1 import *

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
                        help="'baseline', 'labelpowerset', or 'customloss'", 
                        required=True)
    parser.add_argument("--weights",
                        metavar="<weights>", 
                        help="None or /path/to/weights")
    args = parser.parse_args()
    
    # method
    if args.method == 'baseline':
        method = Baseline()
        method.LOSS_FUNC = 'default'
    
    elif args.method == 'customloss':
        method = Baseline()
        method.LOSS_FUNC = 'customloss1'
        
    elif args.method == 'labelpowerset':
        method = Labelpowerset()
        
    # base weights / trained weights
    if args.weights == None:
        method.WEIGHTS_FILE = None
    else:
        method.WEIGHTS_FILE = args.weights

    # mode
    if args.mode == 'train':
        method.train()
    elif args.mode == 'test':
        method.test()