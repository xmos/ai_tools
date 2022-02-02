import ms_lib as ms
from ms_lib import layerArtifacts
import sys

# Interface
if len(sys.argv) == 1:
  print('Option 1 must:\n- generate\n- evaluate\n- compare')
  quit()

elif (sys.argv[1] == 'generate') or (sys.argv[1] == 'evaluate') or (sys.argv[1] == 'compare'):
    pass 
else:
    print('Option 1 must:\n- generate\n- evaluate\n- compare')
    quit()

if sys.argv[1] == 'generate':
  # Generate Base mode and image input
  #########################################
  model = ms.generateBaseModel(0.25, 224)
  im = ms.prepare_image('ostrich.png', 224)
  im2 = ms.prepare_image('goldfish.png', 224)

  # Initialise Layers and Set inputs/outputs
  ############################################
  Layers = ms.tfLayerArtifacts(model)
  ref_outputs = ms.referenceOutputs(model, im)
  ms.setInputsOutputs(Layers, ref_outputs, im)

  # Representative Datasets
  #############################################
  ms.setRepDatasets(Layers, 500)

  # Generate tflite models
  #############################################
  for Layer in Layers:
    Layer.createTfliteModel()
    Layer.createXcoreModel()
    Layer.modelToOpList()
    print(Layer.opList)

  # Save Layer objects as pickles
  #############################################
  ms.saveLayers(Layers, 0)

elif sys.argv[1] == 'evaluate':
  # Load Layer objects
  ###########################################
  Layers = ms.loadLayers(0)

  # Evaluate tflite and xcore layers
  ###########################################
  for layer in Layers:
    layer.eveluateTf(clampOut=True)
    layer.evaluateTflite()
    layer.evaluateXcore()

  ms.saveLayers(Layers, 1)

elif sys.argv[1] == 'compare':
  Layers = ms.loadLayers(1)
  for layer in Layers:
    layer.calcErrors('tf', 'tflite')
    layer.calcErrors('tf', 'xcore')
    layer.calcErrors('tflite', 'xcore')
    layer.errorHists()
  ms.saveLayers(Layers, 2)
  ms.errorSpreadsheet(Layers)
  ms.mainGraphs(Layers, 'average')
  ms.mainGraphs(Layers, 'absolute')
  ms.mainGraphs(Layers, 'maximum absolute')
  ms.mainGraphs(Layers, 'mean squared')
