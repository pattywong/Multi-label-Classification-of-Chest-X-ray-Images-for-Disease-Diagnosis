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

# Commands
# python baseline.py train
# python baseline.py test

class Baseline:
    
    # Weight Initialization
    # "None": Imagenet weights
    # "filename": filename.h5 from the weights folder
    WEIGHTS_FILE = None
    
    # Naming the folder 
    # For training mode --> logs
    # For test mode --> results
    FILENAME = "baseline-test-202"
    
    # "default": Binary Cross Entropy Loss (BCE)
    # For Custom loss Method
    # "customloss1": (BCE+MSML+CorL) 
    LOSS_FUNC = "default"

    # Use-class-weights mode
    USE_CLASS_WEIGHTS = True
    
    # Learning rate
    LEARNING_RATE = 0.00001
    
    # Number of epochs 
    N_EPOCHS = 20
    
    # Select an optimizer 
    # "adam" / "sgd"
    OPTIMIZER = "adam"
    
    # For Binary Relevance Method
    # None: Disable 
    # [0,4]: Classifier (x) 
    BR_MODE = None
    
    # For Hierarchy learning / Progressive learning 
    # Train-only-last-layer mode
    ONLY_LAST_LAYER = False
    
    # Extract features (1024,1)
    EXTRACT_FEATURES = True
    
    def __init__(self):
        
        self.activation = "sigmoid"   
        self.n_labels = 5
        
        # Dataset
        self.dataset = Dataset()
        
        if self.BR_MODE != None:
            self.n_labels = 1
            self.dataset.y_set_train = self.dataset.y_set_train[:,self.BR_MODE]
            self.dataset.y_set_val = self.dataset.y_set_val[:,self.BR_MODE]
            self.dataset.y_set_test = self.dataset.y_set_test[:,self.BR_MODE]
        
        self.dataset.build_generators()
        self.n_classes = self.n_labels
    
        # Model
        if self.WEIGHTS_FILE == None:
            self.dn = Densenet121()
            self.model = self.dn.get_model(self.n_labels, self.activation)

#             serialize model to JSON
#             model_json = self.model.to_json()
#             with open("br-model.json", "w") as json_file:
#                 json_file.write(model_json)
        
        else:
            # Model path
            if self.BR_MODE == None:
                self.model_path = "model/baseline.json"
            else:
                self.model_path = "model/br-model.json"
            
            # Load Model from path
            json_file = open(self.model_path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
                
            # Load weights
            self.weights_path = f"weights/{self.WEIGHTS_FILE}.h5"
            self.model.load_weights(self.weights_path)  
        
        # Extractor
        self.extractor = keras.Model(self.model.input, self.model.layers[-2].output) 
        
        # Loss funciton
        if self.LOSS_FUNC == "default":
            self.loss_function = 'binary_crossentropy'
        elif self.LOSS_FUNC == "customloss1":
            self.loss_function = custom_loss1
        
        # Optimizer
        if self.OPTIMIZER == "adam":
            opt = keras.optimizers.Adam(learning_rate=self.LEARNING_RATE)
        elif self.OPTIMIZER == "sgd":
            opt = keras.optimizers.SGD(lr=self.LEARNING_RATE)
        
        # Metrics 
        auc = tf.keras.metrics.AUC(num_thresholds=200, curve='ROC', summation_method='interpolation', multi_label=True)
        self.metrics = ["binary_accuracy", self.subset_accuracy, auc]
        
        # Train-only-last-layer mode
        if self.ONLY_LAST_LAYER == True:
            # Freeze all layers in the model except the last one
            for layer in self.model.layers[:-1]:
                layer.trainable = False
            self.model.layers[-1].trainable = True
        
        # Compile model
        self.model.compile(loss= self.loss_function, optimizer=opt, metrics=self.metrics)
            
    def display_config(self):

        #This function displays all attributes existing in this class.
        cprint('Configuration', 'yellow', attrs=['reverse'])
        cprint(''.join("%s: %s\n" % item for item in vars(self).items()), "yellow")
    
    def get_model(self): 
        return self.model
    
    def get_extractor(self): 
        return self.extractor
    
    def get_n_layers(self): 
        return len(self.model.layers)
    
    def get_weights(self, layer_index):
        
        layer = self.model.layers[layer_index]
        layer_biases = layer.get_weights()[1]
        layer_weights = np.array(layer.get_weights()[0]).transpose()
        return layer_biases, layer_weights
    
    def train(self):
        
        # Class weights 
        if self.USE_CLASS_WEIGHTS == True:
            self.class_weights = get_class_weights(self.dataset.ANNOTATION_DIR, self.dataset.ANNOTATION_TRAIN_FILE, self.dataset.LABELS, self.dataset.y_set_train, "default")
        else: 
            self.class_weights = None
        
        # Callbacks
        self.logs_file = f"logs/{self.FILENAME}"
        
        if not os.path.exists(self.logs_file):
            os.makedirs(self.logs_file)
            
        self.LOGS_PATH = os.path.join(os.getcwd(), self.logs_file)
        
        self.CHECKPOINT_DIR = os.path.join(self.LOGS_PATH, "w-{epoch:02d}-{val_loss:.2f}-{val_subset_accuracy:.2f}.h5" )
        
        # Learning rate
        def scheduler(epoch, lr):
            if epoch < 5: return (lr)
            else: return (lr * tf.math.exp(-0.1))
            
        lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
        reduced_lr = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=.05, patience=2,verbose=1,
                    mode='min', cooldown=0, min_lr=1e-6)
        
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath = self.CHECKPOINT_DIR,
                        verbose=2, save_weights_only=False, save_freq="epoch")
        
        self.callbacks = [model_checkpoint, lr_scheduler, reduced_lr]
        
        # Fitting model
        history = self.model.fit(
                x = self.dataset.generator_train, 
                validation_data=self.dataset.generator_val,
                epochs=self.N_EPOCHS,
                steps_per_epoch=self.dataset.TRAIN_STEPS,
                validation_steps=self.dataset.VAL_STEPS,
                verbose=2,
                callbacks=[self.callbacks],
                class_weight=self.class_weights)
        
        # Training history
        hist_csv = [[f"lr: {self.LEARNING_RATE}, epochs: {self.N_EPOCHS}"]]
        hist_data = dict()
        keys = history.history.keys()
        epochs_col = [i for i in range(self.N_EPOCHS)]
        hist_csv = pd.DataFrame()
        hist_csv["Epoch"] = epochs_col
        for key in keys:
            hist_csv[f"{key}"] = history.history[key]
        pd.DataFrame(hist_csv).to_csv(f"{self.LOGS_PATH}/history.csv")
        
    def evaluate(self):
        
        # y_scores
        self.y_test_pred = self.model.predict(x=self.dataset.generator_test)
        
        # y_true
        self.y_test_true = self.dataset.y_set_test
        
        # AUC scores
        if self.BR_MODE == None:
            self.auc_scores = [score for score in calculate_auc_bl(self.y_test_pred, self.y_test_true, self.n_classes)]
            self.auc_scores = self.auc_scores + [np.mean(self.auc_scores)]
            
            # Thresholds 
            self.thresholds = get_thresholds(self.y_test_true, self.y_test_pred, self.dataset.LABELS)
            
            # Round_y
            self.round_y_pred = round_scores(self.y_test_pred, self.thresholds, self.dataset.LABELS, self.n_labels) 
        
            # Corr coefs
            self.corr_gt_arr = get_corr_values(pd.DataFrame(self.y_test_true))
            self.corr_pred_arr = get_corr_values(pd.DataFrame(self.round_y_pred))

            # RMSE 
            self.corr_gt, self.corr_pred, self.rmse = cal_corr_rmse(self.corr_gt_arr, self.corr_pred_arr)

            self.corr = pd.DataFrame([self.corr_gt + [self.rmse], self.corr_pred + [self.rmse]])
            self.corr.columns = ['CL', 'CE', 'CA', 'CP', 'LE', 'LA', 'LP', 'EA', 'EP', 'AP', 'RMSE']
        
            # df
            self.round_y_pred_df = pd.DataFrame(self.round_y_pred)
            self.round_y_pred_df.columns = self.dataset.LABELS
            self.y_set_test_df = pd.DataFrame(self.dataset.y_set_test)
            self.y_set_test_df.columns = self.dataset.LABELS
            self.df_bl = self.y_set_test_df.join(self.round_y_pred_df, lsuffix='-act', rsuffix='-pred')
            
        else:
            self.auc_scores = calculate_auc_br(self.y_test_pred, self.y_test_true)
            print("*****")
            print(self.auc_scores)
            print("*****")

        # 1024-features
        if self.EXTRACT_FEATURES == True:
            self.y_train_pred = self.model.predict(x=self.dataset.generator_train)
            self.bl_1024_train = self.extractor.predict(x=self.dataset.generator_train)
            self.bl_1024_test = self.extractor.predict(x=self.dataset.generator_test)
        
    def generate_csv(self):
        
        if not os.path.exists("results/csv"):
            os.makedirs("results/csv")
        
        to_csv(self.y_test_pred, "y_test_pred")
        to_csv(self.y_test_true, "y_test_true")
        
        if self.BR_MODE == None:
            to_csv(self.auc_scores, "auc_scores", self.dataset.LABELS + ["mean"] )
            to_csv(self.thresholds, "thresholds", self.dataset.LABELS)
            to_csv(self.round_y_pred, "round_y_pred")
            to_csv(self.corr_gt_arr, "corr_gt_arr")
            to_csv(self.corr_pred_arr, "corr_pred_arr", self.dataset.LABELS, self.dataset.LABELS)
            to_csv(self.df_bl, "df_bl")
            gen_raw_cm(self.df_bl, f"{self.FILENAME}")
        else:
            to_csv([self.auc_scores], "auc_scores")
        
        if self.EXTRACT_FEATURES == True:
            to_csv(self.bl_1024_train, "bl_1024_train")
            to_csv(self.bl_1024_test, "bl_1024_test")
            to_csv(self.y_train_pred, "y_train_pred")

        os.rename("results/csv", f"results/{self.FILENAME}")
    
    # Custom metrics to log 
    def subset_accuracy(self, y_true, y_pred):
        return K.cast(K.equal(K.argmax(y_true, axis=-1),
                    K.argmax(y_pred, axis=-1)),K.floatx())

        
if __name__ == "__main__":
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='chexpa')
    parser.add_argument("command", metavar="<command>", help="'train' or 'test'")
    args = parser.parse_args()
    
    # Command 
    BL = Baseline()
    if args.command == "train": BL.train()
    elif args.command == "test":
        BL.evaluate()
        BL.generate_csv()