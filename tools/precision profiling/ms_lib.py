import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras.applications.imagenet_utils import decode_predictions
from tflm_interpreter import TFLMInterpreter
from xmos_tools import xformer as xf
import tensorflow_datasets as tfds
import pickle

import numpy as np
from pprint import pprint
import cv2
from keras.preprocessing import image
from openpyxl import Workbook
import matplotlib.pyplot as plt
import seaborn

from tflite.Model import Model

import logging
tf.get_logger().setLevel(logging.ERROR)


class layerArtifacts:
  def __init__(self, tfModel, layerID):
    self.tfModel = tfModel
    self.layerID = layerID

  opList = []

  repDataset = None

  Input = None
  refOutput = None
  outputs = []

  tfliteModel = None
  tfliteRefOutput = None
  tfliteOutputs = []

  xcoreRefOutput = None
  xcoreOutputs = []

  tf_tflite_error = None
  tf_tflite_abserror = None
  tf_tflite_maxabs = None
  tf_tflite_mse = None
  tf_tflite_hist = None

  tf_xcore_error = None
  tf_xcore_abserror = None
  tf_xcore_maxabs = None
  tf_xcore_mse = None
  tf_xcore_hist = None

  tflite_xcore_error = None
  tflite_xcore_abserror = None
  tflite_xcore_maxabs = None
  tflite_xcore_mse = None
  tflite_xcore_hist = None

  def repDatasetGenerator(self):
    for a in self.repDataset:
      yield a

  def createTfliteModel(self):
    os.makedirs('./tflite_models', exist_ok = True)
    print('Generating tflite model ' + str(self.layerID))
    converter = tf.lite.TFLiteConverter.from_keras_model(self.tfModel)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = self.repDatasetGenerator
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS,
      tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    self.tfliteModel = converter.convert()

    with open('./tflite_models/{}.tflite'.format(self.layerID), 'wb') as f:
        f.write(self.tfliteModel)

  def saveTflite(self):
    os.makedirs('./tflite_models', exist_ok = True)
    if os.path.exists('./tflite_models/{}.tflite'.format(self.layerID)):
      os.system('rm ./tflite_models/{}.tflite'.format(self.layerID))
    with open('./tflite_models/' + str(self.layerID) + '.tflite', 'wb') as f:
      f.write(self.tfliteModel)

  def createXcoreModel(self):
    print('Generating xcore model ' + str(self.layerID))
    os.makedirs('./xcore_models', exist_ok = True)
    if os.path.exists('./xcore_models/{}.tflite'.format(self.layerID)):
      os.system('rm ./xcore_models/{}.tflite'.format(self.layerID))
    xf.convert('./tflite_models/{}.tflite'.format(self.layerID), './xcore_models/{}.tflite'.format(self.layerID), params=None)

  def eveluateTf(self, clampOut):
    print('Evaluating TF layer: '+str(self.layerID))
    self.outputs = []

    for datapoint in self.repDataset:
      x = self.tfModel.predict(datapoint[0])
      if clampOut:
        self.outputs.append(clamp(self.outQuantize(x.astype(np.float32))))
      else:
        self.outputs.append(self.outQuantize(x.astype(np.float32)))

  def modelToOpList(self):
    # Update the path to your model
    with open('./tflite_models/{}.tflite'.format(self.layerID), "rb") as model_file:
        buffer = model_file.read()

    # Get Model
    model = Model.GetRootAs(buffer)
    self.opList = []
    for y in range(0, model.Subgraphs(0).OperatorsLength()):
        opcode = model.OperatorCodes(model.Subgraphs(0).Operators(y).OpcodeIndex())
        if opcode.BuiltinCode() == 32:
            self.opList.append(str(opcode.CustomCode()).strip("b'"))
        else:
            self.opList.append(opcode.BuiltinCode())

    f = open('./schema.fbs', "r")
    lines = f.readlines()[108:238]
    for line in lines:
      if '/' in line:
        lines.remove(line)
    for line in lines:
      if '/' in line:
        lines.remove(line)
    for j in range(len(self.opList)):
        for line in lines:
            split = line.split(' = ')
            if str(self.opList[j]) == split[1].strip(',').strip('\n').strip(','):
                self.opList[j] = split[0].strip()
                break

  def evaluateTflite(self):
    print('Evaluating tflite layer: '+str(self.layerID))
    self.tfliteOutputs = []
    interpreter = tf.lite.Interpreter(model_path='./tflite_models/{}.tflite'.format(self.layerID))
    interpreter.allocate_tensors()
    out = interpreter.get_output_details()[0]  # Model has single output.
    inp = interpreter.get_input_details()[0]
    interpreter.set_tensor(inp['index'], self.inQuantize(self.Input))
    interpreter.invoke()
    self.tfliteRefOutput = self.outDequantize(interpreter.get_tensor(out['index']))

    for datapoint in self.repDataset:
      interpreter.set_tensor(inp['index'], self.inQuantize(datapoint[0]))
      interpreter.invoke()
      self.tfliteOutputs.append(interpreter.get_tensor(out['index']).astype(np.float32))

  def evaluateXcore(self):
    print('Evaluating xcore layer: '+str(self.layerID))
    self.xcoreOutputs = []
    interpreter = TFLMInterpreter(model_path='./xcore_models/{}.tflite'.format(self.layerID))
    interpreter.set_input_tensor(0, self.inQuantize(self.Input))
    interpreter.invoke()
    self.xcoreRefOutput = self.outDequantize(interpreter.get_output_tensor(0))

    for datapoint in self.repDataset:
      interpreter.set_input_tensor(0, self.inQuantize(datapoint[0]))
      interpreter.invoke()
      self.xcoreOutputs.append(interpreter.get_output_tensor(0).astype(np.float32))

  def inQuantize(self, data):
    interpreter = tf.lite.Interpreter(model_path='./tflite_models/{}.tflite'.format(self.layerID))
    scale, zero_point = interpreter.get_input_details()[0].get('quantization')
    return np.round(clamp((data/scale)+zero_point)).astype(np.int8)

  def outQuantize(self, data):
    interpreter = tf.lite.Interpreter(model_path='./tflite_models/{}.tflite'.format(self.layerID))
    scale, zero_point = interpreter.get_output_details()[0].get('quantization')
    return clamp((data/scale)+zero_point)

  def outDequantize(self, data):
    interpreter = tf.lite.Interpreter(model_path='./tflite_models/{}.tflite'.format(self.layerID))
    scale, zero_point = interpreter.get_output_details()[0].get('quantization')
    return ((data.astype(np.float32)-zero_point)*scale)

  def calcErrors(self, data1, data2, findError=False):

    print('Calculating {} {} errors for layer: '.format(data1, data2)+str(self.layerID))

    if data1 == 'tf':
      if data2 == 'tflite':
        self.tf_tflite_error, self.tf_tflite_abserror, self.tf_tflite_maxabs, self.tf_tflite_mse, self.tf_tflite_hist = calcErrorSet(self.outputs, self.tfliteOutputs, findError)

      elif data2 == 'xcore':
        self.tf_xcore_error, self.tf_xcore_abserror, self.tf_xcore_maxabs, self.tf_xcore_mse, self.tf_xcore_hist = calcErrorSet(self.outputs, self.xcoreOutputs, findError)
    elif data1 == 'tflite' and data2 == 'xcore':
      self.tflite_xcore_error, self.tflite_xcore_abserror, self.tflite_xcore_maxabs, self.tflite_xcore_mse, self.tflite_xcore_hist = calcErrorSet(self.tfliteOutputs, self.xcoreOutputs, findError)

  def errorHists(self):
    print('Producing hist for layer {}'.format(self.layerID))

    hists(self.tf_tflite_hist, self.layerID, 'tf_tflite', self.opList)
    hists(self.tf_xcore_hist, self.layerID, 'tf_xcore', self.opList)
    hists(self.tflite_xcore_hist, self.layerID, 'tflite_xcore', self.opList)

def clamp(datapoint):
  return np.clip(datapoint, -128, 127)

def hists(histogram, layer, hist_type, opList):
  os.makedirs('./hists', exist_ok = True)
  fig, ax = plt.subplots()
  dim = len(histogram[0])
  xlimit = max(histogram[1][-1], abs(histogram[1][0]))
  dimw = (xlimit*2) / (dim*1.5)
  ax.bar(histogram[1][:-1], histogram[0], width = dimw, align = 'center')
  ax.set_yscale('log')
  ax.set_xlabel("Error")
  ax.set_ylabel("Frequency")
  ax.set_xlim(-xlimit, xlimit)
  ax.set_title("{} layer {} Average Error Histogram (ops: {})".format(hist_type, layer, str(opList)))
  fig.savefig('hists/{}_{}.png'.format(hist_type, layer))
  fig.clf()
  plt.close('all')

def calcErrorSet(out1, out2, findError=False):
  samples = len(out1)

  error = 0
  for x1, x2 in zip(out1, out2):
    error += x1 - x2
  error = error/samples
  error = error.mean()

  abserror = 0
  for x1, x2 in zip(out1, out2):
    abserror += abs(x1 - x2)
  abserror = abserror/samples
  abserror = abserror.mean()

  maxabs = []
  for x1, x2 in zip(out1, out2):
    maxabs.append(np.max(abs(x1 - x2)))
  maxabs = np.max(maxabs)

  mse = 0
  for x1, x2 in zip(out1, out2):
    mse += np.square(x1 - x2)
  mse = mse/samples
  mse = mse.mean()

  hist = []
  histList = []
  i = 0
  for x1, x2 in zip(out1, out2):
    histList.append(list((x1 - x2).flatten()))
  histList = list(np.concatenate(histList).flat)
  # print(histList)
  hist = np.histogram(histList, 40)

  return error, abserror, maxabs, mse, hist

def mainGraphs(Layers, error_type):
  os.makedirs('./graphs', exist_ok = True)
  tf_tflite_err = []
  tf_xcore_err = []
  tflite_xcore_err = []
  for layer in Layers:
    if error_type == 'average':
      tf_tflite_err.append(layer.tf_tflite_error)
      tf_xcore_err.append(layer.tf_xcore_error)
      tflite_xcore_err.append(layer.tflite_xcore_error)
    elif error_type == 'absolute':
      tf_tflite_err.append(layer.tf_tflite_abserror)
      tf_xcore_err.append(layer.tf_xcore_abserror)
      tflite_xcore_err.append(layer.tflite_xcore_abserror)
    elif error_type == 'maximum absolute':
      tf_tflite_err.append(layer.tf_tflite_maxabs)
      tf_xcore_err.append(layer.tf_xcore_maxabs)
      tflite_xcore_err.append(layer.tflite_xcore_maxabs)
    elif error_type == 'mean squared':
      tf_tflite_err.append(layer.tf_tflite_mse)
      tf_xcore_err.append(layer.tf_xcore_mse)
      tflite_xcore_err.append(layer.tflite_xcore_mse)

  width = 0.35  # the width of the bars

  fig, ax = plt.subplots()
  xax = range(0,35)
  rects1 = ax.bar(xax, tf_tflite_err, width, label='tf - tflite {}'.format(error_type))
  rects2 = ax.bar([i+width for i in xax], tf_xcore_err, width, label='tf - xcore {}'.format(error_type))

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Error')
  ax.set_xlabel('Layer')
  ax.set_title('tf - tflite/xcore {} error'.format(error_type))
  ax.legend()

  fig.tight_layout()
  ylim = ax.get_ylim()
  fig.savefig('graphs/tf_quant_{}.png'.format(error_type))
  plt.close('all')

  width = 0.7
  fig, ax = plt.subplots()
  rects1 = ax.bar(xax, tflite_xcore_err, width, label='tflite - xcore {}'.format(error_type))

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Error')
  ax.set_xlabel('Layer')
  ax.set_title('tflite - xcore {} error'.format(error_type))
  ax.legend()
  ax.set_ylim(ylim)
  fig.tight_layout()
  fig.savefig('graphs/tflite_xcore_{}.png'.format(error_type))
  plt.close('all')

# Generates a Mobilenet V1 TensorFlow model with the supplied parameters
# ret: mobilenetv1 tf model
def generateBaseModel(alpha=0.25, inputDim=224):
  model = tf.keras.applications.mobilenet.MobileNet(
  alpha=alpha, input_shape=(inputDim, inputDim, 3), weights='imagenet', classes=1000
  )
  return model

# Loads the image pointed to by 'path', in the form required as input by a tf model
# ret: image as numpy array
def prepare_image(file, dims):
  img = image.load_img(file, target_size=(dims, dims))
  img_array = image.img_to_array(img)
  img_array_expanded_dims = np.expand_dims(img_array, axis=0)
  return img_array_expanded_dims/256.0 - 0.5

# Splits the supplied model apart into layers (group bn and relu with conv)
# ret: list of tf models
def tfLayerArtifacts(model):

  layers_as_models = []
  ret_layers_as_models = []
  for i in range(1, len(model.layers)):
    # Generate Models for each individual layer
    if ("_bn" in model.layers[i].name) or ('_relu' in model.layers[i].name) or ('dropout' in model.layers[i].name):
      layers_as_models[-1].add(model.layers[i])
    else:
      layer_model = tf.keras.Sequential()
      layer_model.add(model.layers[i])
      layers_as_models.append(layer_model)
  for layer_mod in layers_as_models:
    layer_mod.build(input_shape=layer_mod.layers[0].input_shape)
    layer_mod.compile()
    ret_layers_as_models.append(layer_mod)

  print('Number of layers: ' + str(len(layers_as_models)))

  Layers = []
  i = 0
  for layer in layers_as_models:
    artifact = layerArtifacts(layer, i)
    Layers.append(artifact)
    i += 1

  return Layers

# Generates the outputs from each layer of the base model when image is the initial input
# ret: list of outputs
def referenceOutputs(model, image):
  reference_outputs = []

  part_model = tf.keras.Sequential()
  part_model.inputs = model.inputs

  for i in range(1, len(model.layers)):
    # Generate reference outputs for each layer
    part_model.add(model.layers[i])
    part_model.compile()
    if ("conv" in model.layers[i].name) and ('_bn' in model.layers[i+1].name):
      pass
    elif ("_bn" in model.layers[i].name):
      pass
    elif("dropout" in model.layers[i].name):
      pass
    else:
      prediction = part_model.predict(image)
      reference_outputs.append(prediction)

  return reference_outputs

def setInputsOutputs(Layers, ref_outputs, im):
  for x, out in zip(Layers,ref_outputs):
    x.refOutput = out
  Layers[0].Input = im
  for i in range(1, len(Layers)):
    Layers[i].Input = Layers[i-1].refOutput

def setRepDatasets(Layers, samples):
  ds = tfds.load(
    'imagenet_v2',
    split = ('test'),
    with_info = False,
    as_supervised = False
  )

  dataset = []
  for data in ds:
    a = data['image'].numpy()
    a = cv2.resize(a, (224, 224)) #q
    a = (a.astype(np.float32)/255.0) -0.5
    a = [a.reshape(1, *Layers[0].tfModel.input_shape[1:])] #deq
    dataset.append(a)
  Layers[0].repDataset = dataset[:samples]

  for i in range(0, len(Layers)-1):
    print('generating rep dataset {}'.format(i+1))
    rDataset = []
    for datapoint in Layers[i].repDataset:
      res = Layers[i].tfModel.predict(datapoint)
      res = [res.reshape(1, *Layers[i+1].tfModel.input_shape[1:])]
      rDataset.append(res)
    Layers[i+1].repDataset = rDataset

def saveLayers(Layers, stage):
  os.makedirs('./stage0', exist_ok = True)
  os.makedirs('./stage1', exist_ok = True)
  os.makedirs('./stage2', exist_ok = True)
  print('\nSaving Layers...\n')
  for Layer in Layers:
    with open('./stage{}/{}'.format(stage, Layer.layerID),"wb") as f:
      pickle.dump(Layer, f)
    f.close()

def loadLayers(stage):
  print('\nLoading Layers...\n')
  Layers = []

  files = [int(x) for x in os.listdir('./stage{}/'.format(stage))]
  files.sort()
  assert len(files) > 0, "No Layers found to be loaded"
  for file in files:
    with open('./stage{}/'.format(stage) + str(file),'rb') as f:
      layer = pickle.load(f)

      Layers.append(layer)

  return Layers

def errorSpreadsheet(Layers):
  workbook = Workbook()
  sheet = workbook.active
  sheet["A1"] = "Layer"
  sheet["B1"] = "tf - tflite average error"
  sheet["C1"] = "tf - tflite average abs error"
  sheet["D1"] = "tf - tflite max abs error"
  sheet["E1"] = "tf - tflite mse"
  sheet["F1"] = "tf - xcore average error"
  sheet["G1"] = "tf - xcore average abs error"
  sheet["H1"] = "tf - xcore max abs error"
  sheet["I1"] = "tf - xcore mse"
  sheet["J1"] = "tflite - xcore average error"
  sheet["K1"] = "tflite - xcore average abs error"
  sheet["L1"] = "tflite - xcore max abs error"
  sheet["M1"] = "tflite - xcore mse"


  for i in range(0, len(Layers)):
    sheet["A{}".format(i+2)] = Layers[i].layerID
    sheet["B{}".format(i+2)] = Layers[i].tf_tflite_error
    sheet["C{}".format(i+2)] = Layers[i].tf_tflite_abserror
    sheet["D{}".format(i+2)] = Layers[i].tf_tflite_maxabs
    sheet["E{}".format(i+2)] = Layers[i].tf_tflite_mse
    sheet["F{}".format(i+2)] = Layers[i].tf_xcore_error
    sheet["G{}".format(i+2)] = Layers[i].tf_xcore_abserror
    sheet["H{}".format(i+2)] = Layers[i].tf_xcore_maxabs
    sheet["I{}".format(i+2)] = Layers[i].tf_xcore_mse
    sheet["J{}".format(i+2)] = Layers[i].tflite_xcore_error
    sheet["K{}".format(i+2)] = Layers[i].tflite_xcore_abserror
    sheet["L{}".format(i+2)] = Layers[i].tflite_xcore_maxabs
    sheet["M{}".format(i+2)] = Layers[i].tflite_xcore_mse

  workbook.save("error_results.xlsx")