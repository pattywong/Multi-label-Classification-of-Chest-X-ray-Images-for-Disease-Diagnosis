import numpy as np
import pandas as pd
import os, sys, math, datetime
import keras
from keras.models import model_from_json
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from keras.applications.densenet import DenseNet121

sys.path.append('/home/patsaya/main-chexpa/')
from model.densenet121 import *
from utils.dataset import *
from utils.utils import *
from utils.visualize import *
from utils.eval import *

class LabelPowerset:
    
    # Weight Initialization
    # "None": Imagenet weights
    # "filename": filename.h5 from the weights folder
    WEIGHTS_FILE = "w-labelpowerset"
    
    # Naming the folder 
    # For training mode --> logs
    # For test mode --> results
    FILENAME = "labelpowerset"
    
    # Use-class-weights mode
    USE_CLASS_WEIGHTS = True
    
    # Learning rate
    LEARNING_RATE = 0.00001
    
    # Number of epochs 
    N_EPOCHS = 20
    
    # Select an optimizer 
    # "adam" / "sgd"
    OPTIMIZER = "adam"

    # Train-only-last-layer mode
    ONLY_LAST_LAYER = True
    
    # Extract features (1024,1)
    EXTRACT_FEATURES = False
    
    def __init__(self):
        
        # Dataset
        self.dataset = Dataset()
        self.n_classes = 2 ** len(self.dataset.LABELS)
        self.classes = [i for i in range(32)]
        self.n_labels = 5
        self.y_test_true_label = self.dataset.y_set_test
        
        # Transforming disease classes (5 classes) to subsets of disease classes (32 classes)
        self.dataset.y_set_train = self.class_label_generator(self.dataset.y_set_train)['Classes'] 
        self.dataset.y_set_val = self.class_label_generator(self.dataset.y_set_val)['Classes'] 
        self.dataset.y_set_test = self.class_label_generator(self.dataset.y_set_test)['Classes'] 
        
        # Build generators
        self.dataset.build_generators()
        
        # Activation funciton
        self.activation = "softmax"
        
        # Loss function
        self.loss_function = 'sparse_categorical_crossentropy'
        
        # Model
        if self.WEIGHTS_FILE == None:
            self.dn = Densenet121()
            self.model = self.dn.get_model(self.n_classes, self.activation)
        else:
            # Load Model from path
            self.model_path = "model/labelpowerset.json"
            json_file = open(self.model_path, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)

            # Load weights
            self.weights_path = f"weights/{self.WEIGHTS_FILE}.h5"
            self.model.load_weights(self.weights_path)
        
        # Extractor 
        self.extractor = keras.Model(self.model.input, self.model.layers[-2].output)
        
        # Optimizer
        if self.OPTIMIZER == "adam":
            opt = keras.optimizers.Adam(learning_rate=self.LEARNING_RATE)
        elif self.OPTIMIZER == "sgd":
            opt = keras.optimizers.SGD(lr=self.LEARNING_RATE)
        
        # Metrics
        auc = tf.keras.metrics.AUC(num_thresholds=200, curve='ROC', summation_method='interpolation', multi_label=False)
        self.metrics = ['accuracy']

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
            self.class_weights = get_class_weights(self.dataset.ANNOTATION_DIR, self.dataset.ANNOTATION_TRAIN_FILE, self.dataset.LABELS, self.dataset.y_set_train, "labelpowerset")
        else: 
            self.class_weights = None
        
        # callbacks
        self.logs_file = f"logs/{self.FILENAME}"
        
        if not os.path.exists(self.logs_file):
            os.makedirs(self.logs_file)
            
        self.LOGS_PATH = os.path.join(os.getcwd(), self.logs_file)
        
        self.CHECKPOINT_DIR = os.path.join(self.LOGS_PATH, "w-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.h5" )
 
        # Learning rate
        def scheduler(epoch, lr):
            if epoch < 5: return (lr)
            else: return (lr * tf.math.exp(-0.1))
            
        lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath = self.CHECKPOINT_DIR,
                        verbose=2, save_weights_only=False, save_freq="epoch")
        reduced_lr = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=.05, patience=2,verbose=1,
                    mode='min', cooldown=0, min_lr=1e-6)
        self.callbacks = [model_checkpoint, lr_scheduler, reduced_lr]

        history = self.model.fit(x = self.dataset.generator_train, 
                    validation_data=self.dataset.generator_val,
                    epochs=self.N_EPOCHS,
                    steps_per_epoch=self.dataset.TRAIN_STEPS,
                    validation_steps=self.dataset.VAL_STEPS,
                    verbose=2,
                    callbacks=[self.callbacks],
                    class_weight=self.class_weights)
        
        # History csv
        cprint("\n**History**\n", "cyan", attrs = ['reverse'])
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
        
        print("**Evaluating model...**")
        
        # y_test
        self.y_test_pred = self.model.predict(x=self.dataset.generator_test)
        
        self.y_test_pred_label = [to_label_scores(score) for score in self.y_test_pred]
        self.y_test_true = to_categorical(self.dataset.y_set_test, num_classes=32)
        
        # AUC scores
        self.label_auc_scores = calculate_auc_label(self.y_test_pred, self.y_test_true, self.n_classes, len(self.dataset.LABELS))
        self.label_auc_scores = [score for score in self.label_auc_scores]
        self.label_auc_scores = self.label_auc_scores + [np.mean(self.label_auc_scores)]
        
        if len(self.dataset.y_set_test) > 500:
            self.class_auc_scores, self.label_auc_scores = calculate_auc_lp(self.y_test_pred, self.y_test_true, self.n_classes, len(self.dataset.LABELS))
            self.class_auc_scores = [score for score in self.class_auc_scores]
            self.class_auc_scores = self.class_auc_scores + [np.mean(self.class_auc_scores)]
        
        # Thresholds - No thresholds
        self.thresholds = get_thresholds(self.y_test_true, self.y_test_pred, self.classes)
        self.thresholds_label = get_thresholds(self.y_test_true_label, self.y_test_pred_label, self.dataset.LABELS)

        # Round_y
        self.round_y_pred = round_scores(self.y_test_pred, self.thresholds, self.dataset.LABELS, self.n_classes) 
        self.round_y_pred_label = round_scores(self.y_test_pred_label, self.thresholds_label, self.dataset.LABELS, self.n_labels) 
        
        # Corr coefs
        self.corr_gt_arr = get_corr_values(pd.DataFrame(self.y_test_true_label))
        self.corr_pred_arr = get_corr_values(pd.DataFrame(self.round_y_pred_label))
        
        # RMSE 
        self.corr_gt, self.corr_pred, self.rmse = cal_corr_rmse(self.corr_gt_arr, self.corr_pred_arr)
        self.corr = pd.DataFrame([self.corr_gt + [self.rmse], self.corr_pred + [self.rmse]])
        self.corr.columns = ['CL', 'CE', 'CA', 'CP', 'LE', 'LA', 'LP', 'EA', 'EP', 'AP', 'RMSE']
        
        # df
        self.round_y_pred_df = pd.DataFrame(self.round_y_pred_label)
        self.round_y_pred_df.columns = self.dataset.LABELS
        self.y_set_test_df = pd.DataFrame(self.y_test_true_label)
        self.y_set_test_df.columns = self.dataset.LABELS 
        self.df_lp = self.y_set_test_df.join(self.round_y_pred_df, lsuffix='-act', rsuffix='-pred')
        
        # 1024-features
        if self.EXTRACT_FEATURES == True:
            self.lp_1024_train = self.extractor.predict(x=self.dataset.generator_train)
            self.lp_1024_test = self.extractor.predict(x=self.dataset.generator_test)

    def generate_csv(self):
        
        if not os.path.exists("results/csv"):
            os.makedirs("results/csv")

        to_csv(self.y_test_pred, "y_test_pred_class")
        to_csv(self.y_test_pred_label, "y_test_pred")
        to_csv(self.y_test_true, "y_test_true")
        to_csv(self.thresholds_label, "thresholds")
        to_csv(self.round_y_pred, "round_y_pred_class")
        to_csv(self.round_y_pred_label, "round_y_pred")
        to_csv(self.corr_gt_arr, "corr_gt_arr")
        to_csv(self.corr_pred_arr, "corr_pred_arr", self.dataset.LABELS, self.dataset.LABELS)
        to_csv(self.df_lp, "df_lp")
        gen_raw_cm(self.df_lp, f"{self.WEIGHTS_FILE[2:]}_{self.FILENAME}")
        
        to_csv(self.label_auc_scores, "label_auc_scores", self.dataset.LABELS + ["mean"] )
        if len(self.dataset.y_set_test) > 500:
            to_csv(self.class_auc_scores, "class_auc_scores", [i for i in range(0,32)] + ["mean"] )
            
        if self.EXTRACT_FEATURES == True:
            to_csv(self.lp_1024_train, "lp_1024_train")
            to_csv(self.lp_1024_test, "lp_1024_test")
        
        os.rename("results/csv", f"results/{self.FILENAME}")
    
    
    def class_label_generator(self, df_y):
        '''
        function description:
        df_y: dataframe y
        create "Classes" column in the DataFrame to generate classes of all subsets of labels.
        '''
        df_y = pd.DataFrame(df_y)
        classes_array = []
        for index in range(len(df_y)):
            label_set = df_y.iloc[index,:].values
            class_label = get_class_label(label_set)
            classes_array.append(class_label)
        df_y['Classes'] = classes_array
        return df_y

    def test(self):
        self.evaluate()
        self.generate_csv()