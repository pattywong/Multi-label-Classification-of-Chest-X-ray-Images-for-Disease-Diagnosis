from keras.layers import Dense, Input
from keras.models import Model
from keras.applications.densenet import DenseNet121

class Densenet121:
    
    def __init__(self):
        self.module_name = "DenseNet121",
        self.input_shape = (224,224,3)
        
    def get_input_size(self):
        return self.input_shape[:2]
        
    def get_method(self):
        return self.method
    
    def get_model(self, n_classes, activation, use_base_weights=True, weights_path=None, input_shape=None):
        
        if use_base_weights is True:
            base_weights = "imagenet"
        else:
            base_weights = None
            
        if input_shape is None:
            input_shape = self.input_shape
            
        img_input = Input(shape = input_shape)
        
        # Modify here
        # DenseNet121 with sigmoid activation
        base_model = DenseNet121(
                    include_top=False,
                    input_tensor=img_input,
                    input_shape=input_shape,
                    weights=base_weights,
                    pooling="avg")

        x = base_model.output
        predictions = Dense(n_classes, activation=activation, name="predictions")(x)
        model = Model(img_input, predictions)
            
        if weights_path is not None:
            print(f"load model weights_path: {weights_path}")
            model.load_weights(weights_path)

        return model

if __name__ == '__main__':
    pass