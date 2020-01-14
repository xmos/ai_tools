import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from termcolor import colored
import model_interface as mi
import tflite_utils


class FcDeepinShallowoutFinal(mi.KerasModel):

    def generate_fake_lin_sep_dataset(self, classes=2, dim=32, *,
                                      train_samples_per_class=5120,
                                      test_samples_per_class=1024):
        z = np.linspace(0, 2*np.pi, dim)

        # generate data and class labels
        x_train, x_test, y_train, y_test = [], [], [], []
        for j in range(classes):
            mean = np.sin(z) + 10*j/classes
            cov = 10 * np.diag(.5*np.cos(j * z) + 2) / (classes-1)
            x_train.append(
                np.random.multivariate_normal(
                    mean, cov, size=train_samples_per_class))
            x_test.append(
                np.random.multivariate_normal(
                    mean, cov, size=test_samples_per_class))
            y_train.append(j * np.ones((train_samples_per_class, 1)))
            y_test.append(j * np.ones((test_samples_per_class, 1)))

        # stack arrays
        x_train = np.vstack(x_train)
        y_train = np.vstack(y_train)
        x_test = np.vstack(x_test)
        y_test = np.vstack(y_test)

        # normalize
        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std

        # expand dimensions for TFLite compatibility
        def expand_array(arr):
            return np.reshape(arr, arr.shape + (1, 1))
        x_train = expand_array(x_train)
        x_test = expand_array(x_test)

        return {'x_train': np.float32(x_train), 'y_train': np.float32(y_train),
                'x_test': np.float32(x_test), 'y_test': np.float32(y_test)}

    # add keyboard optimizer, loss and metrics???
    def build(self, input_dim, out_dim=2):
        input_dim = self.input_dim
        output_dim = self.output_dim
        # Env
        tf.keras.backend.clear_session()
        tflite_utils.set_all_seeds()
        # Building
        model = tf.keras.Sequential(name=self.name)
        model.add(layers.Flatten(input_shape=(input_dim, 1, 1),
                                 name='input'))
        model.add(layers.Dense(output_dim, activation='softmax',
                               name='ouptut'))
        # Compilation
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        # Add to dict
        self.models[self.name] = model
        # Show summary
        model.summary()

    def prep_data(self):
        self.data = self.generate_fake_lin_sep_dataset(
            self.output_dim,
            self.input_dim,
            train_samples_per_class=51200//self.output_dim,
            test_samples_per_class=10240//self.output_dim)

    def gen_test_data(self):
        if not self.data:
            self.prep_data()
        subset_inds = np.searchsorted(self.data['y_test'].flatten(),
                                      np.arange(self.output_dim))
        self.data['export_data'] = self.data['x_test'][subset_inds]
        self.data['quant'] = self.data['x_train']

    def train(self):
        self.BS = 128
        self.EPOCHS = 5*(self.output_dim-1)
        super().train(self.BS, self.EPOCHS)


def printc(*s, c='green', back='on_grey'):
    if len(s) == 1:
        print(colored(str(s)[2:-3], c, back))
    else:
        print(colored(s[0], c, back), str(s[1:])[1:-2])


shutil.rmtree('./debug')
modelpath = Path('./debug/models')
datapath = Path('./debug/test_data')
test_model = FcDeepinShallowoutFinal(
    'fc_deepin_shallowout_final', Path('./debug'), 32, 2)
printc('Model dictionary:\n', test_model.models)
printc('Model name property:\n', test_model.name)
printc('Data keys before build:\n', test_model.data.keys())
test_model.build(32)
test_model.prep_data()
printc('Data keys after build:\n', test_model.data.keys())
printc('Training:')
test_model.train()
test_model.save_core_model()
test_model.gen_test_data()
printc('Data keys after test data generation:\n', test_model.data.keys())
printc('Content in models directory:')
print([str(x.name) for x in modelpath.iterdir() if x.is_file()])
printc('Content in data directory:')
print([str(x.name) for x in datapath.iterdir() if x.is_file()])
printc('Model keys:\n', test_model.models.keys())

printc('Conversions', c='blue')
printc('To float', c='blue')
test_model.to_tf_float()
printc('Model keys:\n', test_model.models.keys())
printc('Models directory before conversion:')
print([str(x.name) for x in modelpath.iterdir() if x.is_file()])
printc('Models directory after conversion:')
test_model.convert_and_save_model('model_float')
print([str(x.name) for x in modelpath.iterdir() if x.is_file()])

printc('To quant', c='blue')
test_model.to_tf_quant()
printc('Model keys:\n', test_model.models.keys())
printc('Models directory before conversion:')
print([str(x.name) for x in modelpath.iterdir() if x.is_file()])
test_model.convert_and_save_model('model_quant')
printc('Models directory after conversion:')
print([str(x.name) for x in modelpath.iterdir() if x.is_file()])

printc('To stripped', c='blue')
printc('Currently broken')
'''
test_model.to_tf_stripped()
print('Model keys:\n', test_model.models.keys())
printc('Models directory before conversion:')
os.listdir('models')
test_model.convert_and_save_model('model_stripped')
printc('Models directory after conversion:')
os.listdir('models')
'''

printc('To xcore', c='blue')
test_model.to_tf_xcore()
printc('Model keys:\n', test_model.models.keys())
printc('Models directory before conversion:')
print([str(x.name) for x in modelpath.iterdir() if x.is_file()])
test_model.convert_and_save_model('model_xcore')
printc('Models directory after conversion:')
print([str(x.name) for x in modelpath.iterdir() if x.is_file()])

printc('Final status', c='blue')
printc('Data keys:\n', test_model.data.keys())
printc('Model keys:\n', test_model.models.keys())