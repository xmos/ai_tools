# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import examples_common as common
import os
import logging
import pathlib
import tempfile
import tflite_utils
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
import tflite2xcore_conv as xcore_conv
import tflite_visualize
from tflite2xcore import read_flatbuffer, write_flatbuffer
__version__ = '1.4.1'
__author__ = 'Luis Mata'


# Abstract parent class
class Model(ABC):

    def __init__(self, name, path, input_dim, output_dim=1):
        '''
        Initialization function of the class Model. Parameters needed are:
        \t- name     (string): Name of the model and model directory name
        \t- path       (Path): Working directory where everything is stored
        \t- input_dim   (int): input dimension, must be multiple of 32
        \t- output_dim  (int): the number of classes to train
        Other properties derived:
        \t- core_model(Model): The main model from which others derive
        \t- models     (dict): To store model, and dir paths
        \t\t- keys: 'model_float', 'model_quant', 'data_dir', 'models_dir'
        \t- data       (dict): To store all kinds of data
        \t\t- keys: 'quant', 'export', are necessary
        \t- converters (dict): To store all converter objects
        '''
        self.__name = name
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.core_model = None
        self.models = {}
        self.data = {}
        self.converters = {}
        self.test_data = np.empty([output_dim, input_dim, 1, 1])

        if not os.path.exists(path):  # Path generation
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
        '''
        Here should be the model definition to be built,
        compiled and summarized. The model should be stored in self.core_model.
        '''
        pass

    @abstractmethod
    def save_core_model(self):
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
    def gen_test_data(self):
        '''
        Select the test data examples for storing
        along with the converted models.
        Must fill the data dictionary with an entry called 'export_data'
        '''
        pass

    @abstractmethod
    def to_tf_float(self):  # polymorphism affects here
        '''
        Create converter from original model to TensorFlow Lite Float.
        Converter stored with the key 'model_float' in self.converters.
        Model is saved to disk and the path of the model is
        stored in self.models with the key 'model_float'.
        '''
        assert self.core_model, "core model does not exist"

    @abstractmethod
    def to_tf_quant(self):
        '''
        Create converter from original model to TensorFlow Lite Float.
        Converter stored with the key 'model_quant' in self.converters.
        Model is saved to disk and the path of the model is
        stored in self.models with the key 'model_quant'.
        '''
        assert self.core_model, "core model has not been initialized"
        assert 'quant' in self.data, "representative dataset has not been prepared"

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
        model = read_flatbuffer(str(self.models['model_quant']))
        xcore_conv.strip_model(model)
        self.models['model_stripped'] = self.models['models_dir'] / "model_stripped.tflite"
        write_flatbuffer(model, str(self.models['model_stripped']))

        # TODO: refactor this
        base_file_name = 'model_stripped'
        if True:  # visualize:
            model_html = self.models['models_dir'] / f"{base_file_name}.html"
            tflite_visualize.main(self.models[base_file_name], model_html)
            logging.info(f"{base_file_name} visualization saved to {os.path.realpath(model_html)}")

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
        self.models['model_xcore'] = str(self.models['models_dir'] / 'model_xcore.tflite')
        xcore_conv.main(str(self.models['model_quant']),
                        str(self.models['model_xcore']),
                        is_classifier=True)  # TODO: change this later

        # TODO: refactor this
        base_file_name = 'model_xcore'
        if True:  # visualize:
            model_html = self.models['models_dir'] / f"{base_file_name}.html"
            tflite_visualize.main(self.models[base_file_name], model_html)
            logging.info(f"{base_file_name} visualization saved to {os.path.realpath(model_html)}")

    def _save_data_for_canonical_model(self, model_key):
        # create interpreter
        interpreter = tf.lite.Interpreter(model_path=str(self.models[model_key]))

        # extract reference labels for the test examples
        logging.info(f"Extracting examples for {model_key}...")
        x_test = self.data['export_data']
        data = {'x_test': x_test,
                'y_test': common.apply_interpreter_to_examples(interpreter, x_test)}

        # save data
        common.save_test_data(data, self.models['data_dir'], model_key)

    def save_tf_float_data(self):
        assert 'model_float' in self.models
        self._save_data_for_canonical_model('model_float')

    def save_tf_quant_data(self):
        assert 'model_quant' in self.models
        self._save_data_for_canonical_model('model_quant')

    def save_tf_stripped_data(self, add_float_outputs=True):

        model = read_flatbuffer(str(self.models['model_stripped']))
        output_quant = model.subgraphs[0].outputs[0].quantization
        input_quant = model.subgraphs[0].inputs[0].quantization
        xcore_conv.add_float_input_output(model)
        x_test = common.quantize(
            self.data['export_data'],
            input_quant['scale'][0],
            input_quant['zero_point'][0])

        base_file_name = 'model_stripped'
        with tempfile.TemporaryDirectory() as tmp_dir:
            # create interpreter
            model_tmp_file = os.path.join(tmp_dir, 'model_tmp.tflite')
            write_flatbuffer(model, model_tmp_file)
            interpreter = tf.lite.Interpreter(model_path=model_tmp_file)

            # extract and quantize reference labels for the test examples
            logging.info(f"Extracting examples for {base_file_name}...")
            y_test = common.apply_interpreter_to_examples(
                interpreter, self.data['export_data'])
            if add_float_outputs:
                y_test = map(
                    lambda y: common.quantize(
                        y,
                        output_quant['scale'][0],
                        output_quant['zero_point'][0]),
                    y_test)
            data = {'x_test': x_test, 'y_test': np.vstack(list(y_test))}

        common.save_test_data(data, self.models['data_dir'], base_file_name)

    def save_tf_xcore_data(self):
        model = read_flatbuffer(str(self.models['model_xcore']))
        input_quant = model.subgraphs[0].inputs[0].quantization

        input_tensor_type = model.subgraphs[0].inputs[0].type

        if str(input_tensor_type) == 'TensorType.INT16':
            dtype = np.int16
        elif str(input_tensor_type) == 'TensorType.INT8':
            dtype = np.int8
        else:
            raise NotImplementedError(f"input tensor type {input_tensor_type} "
                                      "not supported in save_tf_xcore_data")
        # TODO: and this?
        # pad: from common.save_test_data_for_xcore_model
        '''
        if pad_input_channel_dim:
            old_shape = x_test_float.shape
            pad_shape = list(old_shape[:-1]) + [input_tensor['shape'][-1] - old_shape[-1]]
            pads = np.zeros(pad_shape, dtype=x_test_float.dtype)
            x_test_float = np.concatenate([x_test_float, pads], axis=3)
        '''
        # quantize
        x_test = common.quantize(
            self.data['export_data'],
            input_quant['scale'][0],
            input_quant['zero_point'][0],
            dtype)
        # save data
        base_file_name = 'model_xcore'
        common.save_test_data(
            {'x_test': x_test}, self.models['data_dir'], base_file_name)

    def populate_converters(self):  # Actually, data it's being saved here too
        '''
        Create all the converters in a row in the logical order.
        The only thing needed is the presence
        of the original model in the models dictionary:
        self.core_model must exist.
        '''
        self.to_tf_float()
        self.save_tf_float_data()

        self.to_tf_quant()
        self.save_tf_quant_data()

        self.to_tf_stripped()
        self.save_tf_stripped_data()

        self.to_tf_xcore()
        self.save_tf_xcore_data()


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
        self.core_model.fit(
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
        self.core_model.save(str(self.models['models_dir']/'model.h5'))

    def load_core_model(self):
        data_path = self.models['data_dir']/'data.npz'
        model_path = self.models['models_dir']/'model.h5'
        try:
            logging.info(f"Loading data from {data_path}")
            self.data = dict(np.load(data_path))
            logging.info(f"Loading keras model from {model_path}")
            self.core_model = tf.keras.models.load_model(model_path)
        except FileNotFoundError as e:
            logging.error(f"{e} (Hint: use the --train_model flag)")
            return
        out_shape = self.core_model.output_shape[1]
        if out_shape != self.output_dim:
            raise ValueError(f"number of specified classes ({self.output_dim})"
                             f"does not match model output shape ({out_shape})"
                             )

    def to_tf_float(self):
        super().to_tf_float()
        self.converters['model_float'] = tf.lite.TFLiteConverter.from_keras_model(
            self.core_model)
        self.models['model_float'] = common.save_from_tflite_converter(
            self.converters['model_float'],
            self.models['models_dir'],
            'model_float')

    def to_tf_quant(self):
        super().to_tf_quant()
        self.converters['model_quant'] = tf.lite.TFLiteConverter.from_keras_model(
            self.core_model)
        common.quantize_converter(
            self.converters['model_quant'], self.data['quant'])
        self.models['model_quant'] = common.save_from_tflite_converter(
            self.converters['model_quant'],
            self.models['models_dir'],
            'model_quant')

    def to_tf_stripped(self):
        super().to_tf_stripped()

    def to_tf_xcore(self):
        super().to_tf_xcore()


# Polymorphism: FunctionModel
class FunctionModel(Model):

    @abstractmethod
    def build(self):  # Implementation dependant
        pass

    @abstractmethod
    def prep_data(self):
        pass

    @abstractmethod
    def train(self, BS, EPOCHS):  # Nice default
        assert self.data
        self.core_model.fit(
            self.data['x_train'],
            self.data['y_train'],
            epochs=EPOCHS,
            batch_size=BS,
            validation_data=(self.data['x_test'], self.data['y_test']))

    @abstractmethod
    def gen_test_data(self):
        pass

    # Import and export core model
    def save_core_model(self):
        print('Saving the following data keys:', self.data.keys())
        np.savez(self.models['data_dir'] / 'data', **self.data)
        tf.saved_model.save(
            self.core_model, str(self.models['models_dir']/'model'))

    def load_core_model(self):
        data_path = self.models['data_dir']/'data.npz'
        model_path = self.models['models_dir']/'model'
        try:
            logging.info(f"Loading data from {data_path}")
            self.data = dict(np.load(data_path))
            logging.info(f"Loading keras model from {model_path}")
            self.core_model = tf.saved_model.load(str(model_path))
            # tf.keras.models.load_model(model_path)
        except FileNotFoundError as e:
            logging.error(f"{e} (Hint: use the --train_model flag)")
            return
        ''' What about this?
        out_shape = self.core_model.output_shape[1]
        if out_shape != self.output_dim:
            raise ValueError(f"number of specified classes ({self.output_dim})"
                             f"does not match model output shape ({out_shape})"
                             )
        '''

    # Conversions
    def to_tf_float(self):
        super().to_tf_float()
        self.converters['model_float'] = tf.lite.TFLiteConverter.from_concrete_functions(
            self.function_model)
        self.models['model_float'] = common.save_from_tflite_converter(
            self.converters['model_float'],
            self.models['models_dir'],
            'model_float')

    def to_tf_quant(self):
        super().to_tf_quant()
        self.converters['model_quant'] = tf.lite.TFLiteConverter.from_concrete_functions(
            self.function_model)
        common.quantize_converter(
            self.converters['model_quant'], self.data['quant'])
        self.models['model_quant'] = common.save_from_tflite_converter(
            self.converters['model_quant'],
            self.models['models_dir'],
            'model_quant')

    def to_tf_stripped(self):  # must change this to non abstract in parent class if this design is final
        super().to_tf_stripped()

    def to_tf_xcore(self):
        super().to_tf_xcore()


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
