# Copyright (c) 2018-2019, XMOS Ltd, All rights reserved
import sys
import os
import logging
import pathlib
import tempfile
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
import tflite2xcore.converter as xcore_conv
from tflite2xcore.model_generation import utils
from tflite2xcore import read_flatbuffer, write_flatbuffer, tflite_visualize
from tflite2xcore.xcore_model import TensorType


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

        self.input_init = None  # for initializing input data wit a distribution

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
        assert self.core_model, "core model does not exist"

    @abstractmethod
    def to_tf_quant(self):
        assert self.core_model, "core model has not been initialized"
        assert 'quant' in self.data, "representative dataset has not been prepared"

    def to_tf_stripped(self, **converter_args):
        assert 'model_quant' in self.models
        model = read_flatbuffer(str(self.models['model_quant']))
        xcore_conv.strip_model(model, **converter_args)
        self.models['model_stripped'] = self.models['models_dir'] / "model_stripped.tflite"
        write_flatbuffer(model, str(self.models['model_stripped']))

    def to_tf_xcore(self, **converter_args):
        assert 'model_quant' in self.models
        self.models['model_xcore'] = str(self.models['models_dir'] / 'model_xcore.tflite')
        xcore_conv.convert(str(self.models['model_quant']),
                           str(self.models['model_xcore']),
                           **converter_args)

    def _save_visualization(self, base_file_name):
        assert str(base_file_name) in self.models, 'Model need to exist to prepare visualization.'
        model_html = self.models['models_dir'] / f"{base_file_name}.html"
        tflite_visualize.main(self.models[base_file_name], model_html)
        logging.info(f"{base_file_name} visualization saved to {os.path.realpath(model_html)}")

    def _save_data_dict(self, data, *, base_file_name):
        # save test data in numpy format
        test_data_dir = self.models['data_dir'] / base_file_name
        test_data_dir.mkdir(exist_ok=True, parents=True)
        np.savez(test_data_dir / f"{base_file_name}.npz", **data)

        # save individual binary files for easier low level access
        for key, test_set in data.items():
            for j, arr in enumerate(test_set):
                with open(test_data_dir / f"test_{j}.{key[0]}", 'wb') as f:
                    f.write(arr.flatten().tostring())

        logging.info(f"test examples for {base_file_name} saved to {test_data_dir}")

    def _save_data_for_canonical_model(self, model_key):
        # create interpreter
        interpreter = tf.lite.Interpreter(model_path=str(self.models[model_key]))

        # extract reference labels for the test examples
        logging.info(f"Extracting examples for {model_key}...")
        x_test = self.data['export_data']
        data = {'x_test': x_test,
                'y_test': utils.apply_interpreter_to_examples(interpreter, x_test)}

        # save data
        self._save_data_dict(data, base_file_name=model_key)

    def _save_from_tflite_converter(self, model_key):
        converter = self.converters[model_key]
        self.models[model_key] = self.models['models_dir'] / f"{model_key}.tflite"
        logging.info(f"Converting {model_key}...")
        size = self.models[model_key].write_bytes(converter.convert())
        logging.info(f"{self.models[model_key]} size: {size/1024:.0f} KB")

    def save_tf_float_data(self):
        assert 'model_float' in self.models
        self._save_data_for_canonical_model('model_float')

    def save_tf_quant_data(self):
        assert 'model_quant' in self.models
        self._save_data_for_canonical_model('model_quant')

    def save_tf_stripped_data(self):
        assert 'model_stripped' in self.models

        model = read_flatbuffer(str(self.models['model_stripped']))
        output_quant = model.subgraphs[0].outputs[0].quantization
        input_quant = model.subgraphs[0].inputs[0].quantization
        xcore_conv.add_float_input_output(model)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # create interpreter
            model_tmp_file = os.path.join(tmp_dir, 'model_tmp.tflite')
            write_flatbuffer(model, model_tmp_file)
            interpreter = tf.lite.Interpreter(model_path=model_tmp_file)

            # extract and quantize reference labels for the test examples
            logging.info("Extracting examples for model_stripped...")
            x_test = utils.quantize(self.data['export_data'], input_quant['scale'][0], input_quant['zero_point'][0])
            y_test = utils.apply_interpreter_to_examples(interpreter, self.data['export_data'])
            # The next line breaks in FunctionModels & Keras without ouput dimension
            y_test = map(
                lambda y: utils.quantize(y, output_quant['scale'][0], output_quant['zero_point'][0]),
                y_test
            )
            data = {'x_test': x_test,
                    'y_test': np.vstack(list(y_test))}

        self._save_data_dict(data, base_file_name='model_stripped')

    def save_tf_xcore_data(self):
        assert 'model_xcore' in self.models

        model = read_flatbuffer(str(self.models['model_xcore']))
        input_tensor = model.subgraphs[0].inputs[0]
        input_quant = input_tensor.quantization

        if input_tensor.type != TensorType.INT8:
            raise NotImplementedError(f"input tensor type {input_tensor.type} "
                                      "not supported in save_tf_xcore_data")

        # quantize test data
        x_test = utils.quantize(self.data['export_data'], input_quant['scale'][0], input_quant['zero_point'][0])

        # we pad tensor dimensions other than the first (i.e. batch)
        assert len(input_tensor.shape) == len(x_test.shape)
        pad_width = [(0, input_tensor.shape[j] - x_test.shape[j] if j > 0 else 0)
                     for j in range(len(x_test.shape))]
        x_test = np.pad(x_test, pad_width)

        # save data
        self._save_data_dict({'x_test': x_test}, base_file_name='model_xcore')

    def populate_converters(self, *, xcore_num_threads=None):
        # Actually, data it's being saved here too
        # TODO: find a better name for this
        self.to_tf_float()
        self.save_tf_float_data()
        self._save_visualization('model_float')

        self.to_tf_quant()
        self.save_tf_quant_data()
        self._save_visualization('model_quant')

        self.to_tf_stripped()
        self.save_tf_stripped_data()
        self._save_visualization('model_stripped')

        if xcore_num_threads:
            self.to_tf_xcore(num_threads=xcore_num_threads)
        else:
            self.to_tf_xcore()
        self.save_tf_xcore_data()
        self._save_visualization('model_xcore')


class KerasModel(Model):

    @abstractmethod
    def build(self):
        self._prep_backend()

    def _prep_backend(self):
        tf.keras.backend.clear_session()
        utils.set_all_seeds()

    @property
    def input_shape(self):
        return self.core_model.input_shape[1:]

    @property
    def output_shape(self):
        return self.core_model.output_shape[1:]

    def train(self, save_history=True, **kwargs):
        assert self.data
        self.history = self.core_model.fit(
            self.data['x_train'], self.data['y_train'],
            validation_data=(self.data['x_test'], self.data['y_test']),
            **kwargs)
        if save_history:
            self.save_training_history()

    def save_training_history(self):
        with utils.LoggingContext(logging.getLogger(), logging.INFO):
            utils.plot_history(
                self.history, title=self.name+' metrics',
                path=self.models['models_dir']/('training_history.png'))

    def save_core_model(self):
        if not (len(self.data.keys()) == 0):
            logging.info(f"Saving the following data keys: {self.data.keys()}")
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
        self._save_from_tflite_converter('model_float')

    def to_tf_quant(self):
        super().to_tf_quant()
        self.converters['model_quant'] = tf.lite.TFLiteConverter.from_keras_model(
            self.core_model)
        utils.quantize_converter(
            self.converters['model_quant'], self.data['quant'])
        self._save_from_tflite_converter('model_quant')


class KerasClassifier(KerasModel):
    def __init__(self, *args, opt_classifier=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._opt_classifier = opt_classifier

    def to_tf_xcore(self):
        super().to_tf_xcore(is_classifier=self._opt_classifier)


class FunctionModel(Model):

    def __init__(self, name, path):
        super().__init__(name, path)
        self.loaded = False

    @abstractmethod
    def build(self):  # Implementation dependant
        pass

    # Import and export core model
    def save_core_model(self):
        model_path = str(self.models['models_dir']/'model')
        if not (len(self.data.keys()) == 0):
            logging.info(f"Saving the following data keys: {self.data.keys()}")
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
        self._save_from_tflite_converter('model_float')

    def to_tf_quant(self):
        super().to_tf_quant()
        self.converters['model_quant'] = tf.lite.TFLiteConverter.from_concrete_functions(
            [self.concrete_function])
        utils.quantize_converter(
            self.converters['model_quant'], self.data['quant'])
        self._save_from_tflite_converter('model_quant')
