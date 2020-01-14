import examples_common as common
import os
import logging
import tflite_utils
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
import tflite2xcore_conv as xcore_conv
from tflite2xcore import read_flatbuffer
__version__ = '1.2.0'
__author__ = 'Luis Mata'


# Abstract parent class
class Model(ABC):

    def __init__(self, name, path, input_dim, output_dim):
        '''
        Initialization function of the class Model. Parameters needed are:
        \t- name: (string) name of the model, models directory name
        \t- data_dir: (path) where the data directory is located
        \t- models_dir: (path) where the models directory is located
        \t- input_dim: (int) input dimension, must be multiple of 32
        \t- output_dim: (int) the number of classes to train
        \t- path: (Path) working directory to store everything
        '''
        self.models = {}  # paths included data_dir : Path ; models_dir : Path
        self.data = {}  # For storing data
        self.converters = {}  # For storing the converters float and quant
        self.test_data = np.empty([output_dim, input_dim, 1, 1])
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.__name = name
        if not os.path.exists(path):
            path.mkdir()
        self.models['data_dir'] = path / 'test_data'
        if not os.path.exists(self.models['data_dir']):
            self.models['data_dir'].mkdir()
        self.models['models_dir'] = path / 'models'
        if not os.path.exists(self.models['models_dir']):
            self.models['models_dir'].mkdir()

    @property
    def name(self):
        return self.__name

    @name.setter
    def setname(self, name):
        self.__name = name

    @name.getter
    def getname(self):
        return self.__name

    @abstractmethod
    def build(self):
        # kb arguments for the compiler?
        # instantiate model object
        # polimorphism argmax
        '''
        Here should be the model definition to be built,
        compiled and summarized.The model should be stored
        in the dictionary with its name: self.models[self.name]=model
        '''
        pass

    @abstractmethod
    def save_core_model():
        '''
        Function to store training and original model files in the
        corresponding format.
        '''
        pass

    @abstractmethod
    def load_core_model(self, load_path):
        # restore Models state with submode
        '''
        If we don't want to build our model from scratch and
        we have it stored somewhere, we can load it with this function.
        - load_path: path where the model is stored
        '''
        pass

    @abstractmethod
    def prep_data(self):  # Loading and preprocessing
        # everything that doesn't happend on the fly
        '''
        To prepare or download the training and test data.
        Should return a dictionary:
        {'x_train':xt, 'y_train':yt, 'x_test':xtt, 'y_test':ytt}
        '''
        pass

    @abstractmethod
    def train(self):
        '''
        GPU and CPU usage should be differentiated.
        Fit with hyperparams and if we want to save
        original model and training data,
        that should be done here.
        '''
        pass

    @abstractmethod
    def gen_test_data(self):  # naming
        '''
        Select the test data examples for storing
        along with the converted models.
        Must fill the data dictionary with an entry called 'export_data'
        '''
        pass

    @abstractmethod
    def to_tf_float(self):  # polymorphism affects here
        '''
        Create converter from original model
        to TensorFlow Lite Float.
        Converter stored with the key 'model_float' in
        self.converters. Also the path of the model is saved
        using this function.
        '''
        assert self.name in self.models

    @abstractmethod
    def to_tf_quant(self):
        '''
        Create converter from original model
        to TensorFlow Lite Float.
        Converter stored with the key 'model_quant' in
        self.converters. Also the path of the model is saved
        using this function.
        '''
        assert self.name in self.models

    @abstractmethod
    def to_tf_stripped(self):
        '''
        Create converter from original model
        to TensorFlow Lite Float.
        Converter stored with the key 'model_stripped' in
        self.converters. Also the path of the model is saved
        using this function.
        '''
        assert 'model_quant' in self.models

    @abstractmethod
    def to_tf_xcore(self):
        '''
        Create converter from original model
        to TensorFlow Lite Float.
        Converter stored with the key 'model_xcore' in
        self.converters. Also the path of the model is saved
        using this function.
        '''
        assert 'model_quant' in self.models

    def populate_converters(self):  # Actually, data it's being saved here too
        # Naming
        # TODO
        '''
        Create all the converters in a row in the logical order.
        The only thing needed is the presence
        of the original model in the models dictionary:
        self.models[self.name] must exist.
        '''
        assert self.name in self.models
        self.to_tf_float()
        self.to_tf_quant()
        # self.to_tf_stripped()
        self.to_tf_xcore()

    def convert_and_save_model(self, name):
        test_data = self.data['export_data']
        # float or quant models are converted and saved in the same way
        if name == 'model_float' or name == 'model_quant':
            model_file = common.save_from_tflite_converter(
                self.models[name],
                self.models['models_dir'],
                name)
            common.save_test_data_for_regular_model(
                model_file,
                test_data,
                data_dir=self.models['data_dir'],
                base_file_name=name)
        else:
            # only stripped models are saved from json
            # and the test_data has a special format for each
            if name == 'model_stripped':
                common.save_from_json(
                    self.models[name],
                    self.models['models_dir'],
                    name)
                common.save_test_data_for_stripped_model(
                    self.models[name],
                    test_data,
                    data_dir=self.models['data_dir'])
            else:
                '''
                common.save_test_data_for_xcore_model(
                    self.models[name],
                    test_data,
                    data_dir=self.models['data_dir'])
                    '''

    @abstractmethod
    def convert_and_save(self):
        '''
        Will save all the models in the self.models dictionary along with the
        test data provided as parameter.
        The models to be saved are:
        \t- tflite float
        \t- tflite quant
        \t- tflite stripped
        \t- tflite xcore
        '''
        # test_data = self.data['export_data']
        # float
        # self.convert_from_tflite(test_data, 'model_float')
        self.convert_and_save_model('model_float')
        # quant
        # self.convert_from_tflite(test_data, 'model_quant')
        self.convert_and_save_model('model_quant')
        # stripped
        # self.converter_from_json_stripped(test_data, 'model_stripped')
        # self.convert_and_save_model('model_stripped')  # as it breaks for now
        # xcore
        # self.conver_from_json_xcore(test_data, 'model_xcore')
        self.convert_and_save_model('model_xcore')


# Polymorphism: Keras
class KerasModel(Model):

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def prep_data(self):
        pass

    @abstractmethod
    def train(self, BS, EPOCHS):
        assert self.data
        self.models[self.name].fit(
            self.data['x_train'],
            self.data['y_train'],
            epochs=EPOCHS,
            batch_size=BS,
            validation_data=(self.data['x_test'], self.data['y_test']))

    @abstractmethod
    def gen_test_data(self):
        '''
        self.data['export_data'] =
        '''
        pass

    def save_core_model(self):
        print('Saving the following data keys:', self.data.keys())
        np.savez(self.models['data_dir'] / 'data', **self.data)
        self.models[self.name].save(str(self.models['models_dir']/'model.h5'))

    def load_core_model(self):
        data_path = self.models['data_dir']/'data.npz'
        model_path = self.models['models_dir']/'model.h5'
        try:
            logging.info(f"Loading data from {data_path}")
            self.data = dict(np.load(data_path))
            logging.info(f"Loading keras model from {model_path}")
            self.models[self.name] = tf.keras.models.load_model(model_path)
        except FileNotFoundError as e:
            logging.error(f"{e} (Hint: use the --train_model flag)")
            return
        out_shape = self.models[self.name].output_shape[1]
        if out_shape != self.output_dim:
            raise ValueError(f"number of specified classes ({self.output_dim})"
                             f"does not match model output shape ({out_shape})"
                             )

    def to_tf_float(self):  # affected by poly
        assert self.name in self.models
        self.models['model_float'] = tf.lite.TFLiteConverter.from_keras_model(
            self.models[self.name])

    def to_tf_quant(self):  # affected by poly
        assert self.name in self.models
        assert 'quant' in self.data
        self.models['model_quant'] = tf.lite.TFLiteConverter.from_keras_model(
            self.models[self.name])
        common.quantize_converter(
            self.models['model_quant'], self.data['quant'])

    def to_tf_stripped(self):  # not really affected by poly
        assert 'model_quant' in self.models
        desc = "TOCO Converted and stripped."
        model_quant_file = common.save_from_tflite_converter(
            self.models['model_quant'],
            self.models['models_dir'],
            "model_quant")
        model_quant = tflite_utils.load_tflite_as_json(model_quant_file)
        self.models['model_stripped'] = common.strip_model_quant(model_quant)
        self.models['model_stripped']['description'] = desc

    def to_tf_xcore(self):  # not really affected by poly
        assert 'model_quant' in self.models
        output_path = str(self.models['models_dir']/'model_xcore.tflite')
        html_path = output_path[:-6]+'html'
        model_quant_file = common.save_from_tflite_converter(
            self.models['model_quant'],
            self.models['models_dir'],
            "model_quant", visualize=False)
        # xcore conversion happens here
        xcore_conv.main(str(model_quant_file), output_path, is_classifier=True)  # TODO: change this later
        self.models['model_xcore'] = read_flatbuffer(output_path)
        # Viz
        import tflite_visualize
        tflite_visualize.main(output_path, html_path)

    def convert_and_save(self):
        super().convert_and_save()


# Polymorphism: FunctionModel
class FunctionModel(Model):

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def load(self, load_path):
        pass

    @abstractmethod
    def prep_data(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def gen_test_data(self):
        pass


# Polymorphism: Saved Model
class SavedModel(Model):

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def load(self, load_path):
        pass

    @abstractmethod
    def prep_data(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def gen_test_data(self):
        pass
