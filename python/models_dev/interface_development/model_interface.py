# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import examples_common as common
import sys
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
from xcore_model import TensorType
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model_development')))
import model_tools as mt
__version__ = '1.6.0'
__author__ = 'Luis Mata'


# Abstract parent class
class Model(ABC):

    def __init__(self, name, path):
        '''
        Initialization function of the class Model. Parameters needed are:
        \t- name     (string): Name of the model and model directory name
        \t- path       (Path): Working directory where everything is stored
        Other properties derived:
        \t- core_model(Model): The main model from which others derive
        \t- models     (dict): To store model, and dir paths
        \t\t- keys: 'model_float', 'model_quant', 'data_dir', 'models_dir'
        \t- data       (dict): To store all kinds of data
        \t\t- keys: 'quant', 'export', are necessary
        \t- converters (dict): To store all converter objects
        '''
        self.__name = name
        self.core_model = None
        self.models = {}
        self.data = {}
        self.converters = {}

        if type(path) is pathlib.PosixPath:
            self._path = path
        elif type(path) is str:
            self._path = pathlib.Path(path)

        self._path.mkdir(parents=True, exist_ok=True)  # Path generation
        self.models['data_dir'] = self._path / 'test_data'
        self.models['data_dir'].mkdir(exist_ok=True)
        self.models['models_dir'] = self._path / 'models'
        self.models['models_dir'].mkdir(exist_ok=True)

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

    def to_tf_stripped(self, **converter_args):
        '''
        Create converter from original model
        to TensorFlow Lite Float.
        Converter stored with the key 'model_stripped' in
        self.converters. Also the path of the model is saved
        using this function.
        '''
        assert 'model_quant' in self.models
        model = read_flatbuffer(str(self.models['model_quant']))
        xcore_conv.strip_model(model, **converter_args)
        self.models['model_stripped'] = self.models['models_dir'] / "model_stripped.tflite"
        write_flatbuffer(model, str(self.models['model_stripped']))

        if True:  # visualize:
            self._save_visualization('model_stripped')

    def to_tf_xcore(self, **converter_args):
        '''
        Create converter from original model
        to TensorFlow Lite Float.
        Converter stored with the key 'model_xcore' in
        self.converters. Also the path of the model is saved
        using this function.
        '''
        assert 'model_stripped' in self.models
        self.models['model_xcore'] = str(self.models['models_dir'] / 'model_xcore.tflite')
        xcore_conv.main(str(self.models['model_stripped']),
                        str(self.models['model_xcore']),
                        **converter_args)

        if True:  # visualize:
            self._save_visualization('model_xcore')

    def _save_visualization(self, base_file_name):
        assert str(base_file_name) in self.models, 'Model need to exist to prepare visualization.'
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

    def save_tf_stripped_data(self):
        '''
         common.save_test_data_for_stripped_model(
        model_stripped, x_test_float, data_dir=DATA_DIR, add_float_outputs=False)
        '''
        assert 'model_stripped' in self.models

        model = read_flatbuffer(str(self.models['model_stripped']))
        output_quant = model.subgraphs[0].outputs[0].quantization
        input_quant = model.subgraphs[0].inputs[0].quantization
        xcore_conv.add_float_input_output(model)

        base_file_name = 'model_stripped'
        with tempfile.TemporaryDirectory() as tmp_dir:
            # create interpreter
            model_tmp_file = os.path.join(tmp_dir, 'model_tmp.tflite')
            write_flatbuffer(model, model_tmp_file)
            interpreter = tf.lite.Interpreter(model_path=model_tmp_file)

            # extract and quantize reference labels for the test examples
            logging.info(f"Extracting examples for {base_file_name}...")
            x_test = common.quantize(self.data['export_data'], input_quant['scale'][0], input_quant['zero_point'][0])
            y_test = common.apply_interpreter_to_examples(interpreter, self.data['export_data'])
            # The next line breaks in FunctionModels & Keras without ouput dimension
            y_test = map(
                lambda y: common.quantize(y, output_quant['scale'][0], output_quant['zero_point'][0]),
                y_test
            )
            data = {'x_test': x_test,
                    'y_test': np.vstack(list(y_test))}

        common.save_test_data(data, self.models['data_dir'], base_file_name)

    def save_tf_xcore_data(self):
        assert 'model_xcore' in self.models

        model = read_flatbuffer(str(self.models['model_xcore']))
        input_tensor = model.subgraphs[0].inputs[0]
        input_quant = input_tensor.quantization

        if input_tensor.type != TensorType.INT8:
            raise NotImplementedError(f"input tensor type {input_tensor.type} "
                                      "not supported in save_tf_xcore_data")

        # quantize test data
        x_test = common.quantize(self.data['export_data'], input_quant['scale'][0], input_quant['zero_point'][0])

        # we pad tensor dimensions other than the first (i.e. batch)
        assert len(input_tensor.shape) == len(x_test.shape)
        pad_width = [(0, input_tensor.shape[j] - x_test.shape[j] if j > 0 else 0)
                     for j in range(len(x_test.shape))]
        x_test = np.pad(x_test, pad_width)

        # save data
        common.save_test_data({'x_test': x_test}, self.models['data_dir'], 'model_xcore')

    def populate_converters(self):  # Actually, data it's being saved here too
        # TODO: find a better name for this
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


class KerasModel(Model):

    @abstractmethod
    def build(self):
        self._prep_backend()

    def _prep_backend(self):
        tf.keras.backend.clear_session()
        tflite_utils.set_all_seeds()

    @property
    def input_shape(self):
        return self.core_model.input_shape[1:]

    @property
    def output_shape(self):
        return self.core_model.output_shape[1:]

    def train(self, save_history=False, **kwargs):
        assert self.data
        self.history = self.core_model.fit(
            self.data['x_train'], self.data['y_train'],
            validation_data=(self.data['x_test'], self.data['y_test']),
            **kwargs)
        if save_history:
            self.save_training_history()

    def save_training_history(self): # TODO: generalize this idea to KerasModel
        logger = logging.getLogger()
        old_log_level = logger.level  # deal with matplotlib spam
        logger.setLevel(logging.INFO)
        mt.plot_history(
            self.history, title=self.name+' metrics',
            path=self.models['models_dir']/(self.name+'_history.png'))
        logger.setLevel(old_log_level)

    @abstractmethod
    def gen_test_data(self):
        '''
        self.data['export_data'] =
        '''
        pass

    def save_core_model(self):
        if not (len(self.data.keys()) == 0):
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
            raise FileNotFoundError(
                f"Model file not found (Hint: use the --train_model flag)") from e

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


class FunctionModel(Model):

    def __init__(self, name, path):
        super().__init__(name, path)
        self.loaded = False

    @abstractmethod
    def build(self):  # Implementation dependant
        pass

    @abstractmethod
    def gen_test_data(self):
        pass

    # Import and export core model
    def save_core_model(self):
        model_path = str(self.models['models_dir']/'model')
        if not (len(self.data.keys()) == 0):
            print('Saving the following data keys:', self.data.keys())
            np.savez(self.models['data_dir'] / 'data', **self.data)
        tf.saved_model.save(
            self.core_model, model_path, signatures=self.concrete_function
        )

    def load_core_model(self):
        data_path = self.models['data_dir']/'data.npz'
        model_path = str(self.models['models_dir']/'model')
        try:
            logging.info(f"Loading data from {data_path}")
            self.data = dict(np.load(data_path))
            logging.info(f"Loading keras model from {model_path}")
            self.core_model = tf.saved_model.load(model_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model file not found (Hint: use the --train_model flag)") from e

    @property
    @abstractmethod
    def concrete_function(self):
        pass

    # Conversions
    def to_tf_float(self):
        super().to_tf_float()
        self.converters['model_float'] = tf.lite.TFLiteConverter.from_concrete_functions(
            [self.concrete_function])
        self.models['model_float'] = common.save_from_tflite_converter(
            self.converters['model_float'],
            self.models['models_dir'],
            'model_float')

    def to_tf_quant(self):
        super().to_tf_quant()
        self.converters['model_quant'] = tf.lite.TFLiteConverter.from_concrete_functions(
            [self.concrete_function])
        common.quantize_converter(
            self.converters['model_quant'], self.data['quant'])
        self.models['model_quant'] = common.save_from_tflite_converter(
            self.converters['model_quant'],
            self.models['models_dir'],
            'model_quant')


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
