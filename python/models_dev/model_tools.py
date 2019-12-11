import random
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import scipy.ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

__version__ = '1.1.0'
__author__ = 'Luis Mata'
'''
Tools for model development
'''
# Data
def get_mnist_tf(debug=False, np=False):
    '''
    Get the MNIST dataset in the TensorFlow Dataset format or its equivalent numpy arary.
    \t- debug: for printed output
    \t- np: for numpy format
    '''
    mnist, info = tfds.load('mnist', shuffle_files=True, with_info=True)
    train_data, test_data = mnist['train'], mnist['test']
    if(debug):
        print(info)
        print(train_data)
        print(test_data)
    if np: # for numy arrays format
        return tfds.as_numpy(train_data),tfds.as_numpy(test_data), info
    return train_data, test_data, info

def unfold_gen(size, generator):
    '''
    To unfold a numpy generator, need to be fed with size.
    \t- size: expected size of the unfolded object
    \t- generator: generator object to unfold
    '''
    arr = np.empty(size)
    for i, el in enumerate(generator): arr[i] = el
    return arr

def flatten(ds):
    '''
    Flatten function for a numpy array. It must have 3 dimensions, and the output will have 2.
    '''
    return ds.reshape(ds.shape[0], ds.shape[1]*ds.shape[2])

def get_mnist(rows=28, cols=28, nb_classes=10, categorical=False, padding=2, val_split=True, flatten = False, debug=True, y_float = False):
    '''
    Get the keras MNIST dataset in the specified format.
    \t- rows: height of the image
    \t- cols: columns of the image
    \t- nb_clases: number of clases in the output
    \t- categorical: if categorical labels or not
    \t- padding: if padding of the images or not
    \t- val_split: if divide into validation as well or not
    \t- flatten: if we want the output datasets to have only 2 dims or not
    \t- debug: if we want printed shapes and extra information or not
    \t- y_float: if we want the labels to be float numbers
    '''
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    from sklearn.model_selection import train_test_split
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], rows, cols, 1).astype('float32')/255
    x_test = x_test.reshape(x_test.shape[0], rows, cols, 1).astype('float32')/255
    
    if y_float:
        y_train = y_train.astype('float32')
        y_test = y_test.astype('float32')
    
    if padding:
        x_train = np.pad(x_train, ((0,0),(padding,padding),(padding,padding),(0,0)), 'constant')
        x_test = np.pad(x_test, ((0,0),(padding,padding),(padding,padding),(0,0)), 'constant')

    if categorical:
        y_train = to_categorical(y_train, nb_classes)
        y_test = to_categorical(y_test, nb_classes)
        y_train = y_train.reshape(y_train.shape[0],1,1,10)
        y_test = y_test.reshape(y_test.shape[0],1,1,10)
    
    if val_split:
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
    
    if flatten:
        x_train = flatten(x_train)
        x_test = flatten(x_test)
        x_val = flatten(x_val)
    if debug:
        print('x_train shape: ', x_train.shape)
        print('y_train shape: ', y_train.shape)
        print('x_train type: ', type(x_train))
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        if val_split:
            print(x_val.shape[0], 'validation samples')
            
        if not categorical:
            train_labels_count = np.unique(y_train, return_counts=True)
            print({'count':train_labels_count[1]})
    if val_split:
        return x_train, x_test, x_val, y_train, y_test, y_val
    return x_train, x_test, y_train, y_test

def ecc(nsizex=29, nsizey=29, ch = 1):
    '''
    Crop the dataset images using resize from skimage, consider instead use keras layer Cropping2D.
    '''
    x_train, x_test, x_val, y_train, y_test, y_val = get_mnist(categorical=True, padding=0)
    from skimage.transform import resize
    o_train = resize(x_train, (x_train.shape[0], nsizex, nsizey, ch))
    o_test = resize(x_test, (x_test.shape[0], nsizex, nsizey, ch))
    o_val = resize(x_val, (x_val.shape[0], nsizex, nsizey, ch))
    return o_train, o_test, o_val, y_train, y_test, y_val

# Viz
def sanity_check(ds, labels):
    '''
    Show a random image to perform a sanity check of the data.
    \t- ds: dataset (numpy array)
    \t- labels: dataset labes (numpy array)
    '''
    idx = random.randint(0, len(ds))
    img = ds[idx].squeeze()
    plt.style.use('dark_background')
    plt.figure(figsize=(1,1))
    plt.title('Index: ' + str(labels[idx]) + ' - sanity check')
    plt.imshow(img)
    
def random_pick(ds, labels, categorical=False, dim=32, ch=1, zoom=1):
    '''
    Show and return a random image from a given dataset.
    \t- ds: dataset (numpy array)
    \t- labels: dataset labels (numpy array)
    \t- categorical: if labels are in categorical format
    \t- dim: dimension of the side of the image
    \t- ch: number of channels
    \t- zoom: zoom for the plotting
    '''
    idx = random.randint(0, len(ds))
    exp = labels[idx]
    if categorical:
        exp = np.argmax(exp)
    plt.style.use('dark_background')
    plt.figure(figsize=(zoom,zoom))
    plt.title('Index: ' + str(exp) + ' - random pick')
    plt.imshow(ds[idx].reshape(dim,dim,ch).squeeze())
    return ds[idx]

def random_stack(ds,labels,depth,categorical=False, dim=32, ch=1):
    '''
    Return a random stack of a given dataset.
    \t- ds: dataset (numpy array)
    \t- labels: dataset labels (numpy array)
    \t- depth: number of data instances in the stack
    \t- categorical: if labels are in categorical format
    \t- dim: dimension of the side of the image
    \t- ch: number of channels
    '''
    stack = []
    for i in range(depth):
        stack.append(random_pick(ds,labels,categorical,dim,ch))
    print(np.row_stack(stack).shape)
    return np.row_stack(stack)

def plot(img, title='',zoom=3,dim=32,ch=1):
    '''
    Plot easily an image using matplotlib.
    \t- img: image to plot
    \t- title: title of the plot
    \t- zoom: zoom for the plot
    \t- dim: dimension of the side of the image
    \t- ch: number of channels of the image
    '''
    plt.style.use('dark_background')
    plt.figure(figsize=(zoom,zoom))
    plt.title(title)
    plt.imshow(img.reshape(dim,dim,ch).squeeze())

def multi_plot(imgs, rows, cols, title='', zoom=2):
    '''
    Plot several images easily using matplotlib
    \t- imgs: stack of images to be plotted
    \t- rows: number of rows of the output grid
    \t- cols: number of cols of the output grid
    \t- title: title of the grid
    \t- zoom: zoom of the images in the grid
    '''
    assert rows*cols >= len(imgs)
    plt.style.use('dark_background')
    fig=plt.figure(figsize=(8*zoom, 8*zoom))
    for i in range(1, rows*cols +1):
        img = imgs[i-1]
        plt.title(title)
        fig.add_subplot(rows, cols, i)
        plt.imshow(img.squeeze())
    plt.show()
    
def plot_history(h, title='metrics', zoom=1):
    # list all data in history
    history=h
    plt.style.use('dark_background')
    fig=plt.figure(figsize=(16*zoom, 8*zoom))
    plt.title(title)
    # summarize history for accuracy
    fig.add_subplot(1,2,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    # summarize history for loss
    fig.add_subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
# Augmentation
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape)==2
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)

# Apply elastic distortions to the input
# images: set of images; labels: associated labels
def expand_dataset(images, labels, distortions, sigma=4.0, alpha=60.0, sizex=32, sizey=32):
    '''
    Function to expand a dataset with more images.
    \t- images: original dataset (numpy array)
    \t- labels: original dataset labels (numpy array)
    \t- distortions: number of distortions per image
    \t- sigma: sigma value
    \t- alpha: alpha value
    \t- sizex: size x of the image
    \t- sizey: size y of the image
    '''
    new_images_batch = np.array(
        [elastic_transform(np.reshape(image, (sizex,sizey)), alpha, sigma) for image in images for _ in range(distortions)])
    new_labels_batch = np.array(
        [label for label in labels for _ in range(distortions)])
    # We don't forget to return the original images and labels (hence concatenate)
    x_data, y_data = np.concatenate([np.reshape(images, (-1, sizex, sizey)), new_images_batch]), \
           np.concatenate([labels, new_labels_batch])
    return x_data.reshape(x_data.shape[0], sizex, sizey, 1), y_data

# Model definition
def get_model(t, l1=False):
    from tensorflow.keras import layers
    model = keras.Sequential(name=t)
    if t == 'MLP1':
        model.add(layers.Flatten(input_shape=(32,32,1), name='input'))
        model.add(layers.Dense(420, activation='tanh', name='dense_1'))
        model.add(layers.Dense(300, activation='tanh', name='dense_2'))
    elif t == 'MLP2':
        model.add(layers.Flatten(input_shape=(32,32,1), name='input'))
        model.add(layers.Dense(416, activation='relu', name='dense_1'))
        model.add(layers.Dense(288, activation='relu', name='dense_2'))
    elif t == 'lenet5':
        model.add(keras.Input(shape=(32,32,1), name='input'))
        model.add(layers.Conv2D(6, (5,5), strides=1, activation='tanh', name='conv_1'))
        model.add(layers.AvgPool2D((2,2), strides=2, name='avg_pool_1'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(16, (5,5), strides=1, activation='tanh', name='conv_2'))
        model.add(layers.AvgPool2D((2,2), strides=2, name='avg_pool_2'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(120, (5,5), strides=1, activation='tanh', name='conv_3'))
        model.add(layers.Dense(84, activation='tanh', name='fc_1'))
    elif t == 'lenet5_tuned':
        model.add(keras.Input(shape=(32,32,1), name='input'))
        model.add(layers.Conv2D(8, (5,5), strides=1, activation='relu', name='conv_1'))
        model.add(layers.AvgPool2D((2,2), strides=2, name='avg_pool_1'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(16, (5,5), strides=1, activation='relu', name='conv_2'))
        model.add(layers.AvgPool2D((2,2), strides=2, name='avg_pool_2'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, (5,5), strides=1, activation='relu', name='conv_3'))
        model.add(layers.Dense(96, activation='relu', name='fc_1'))
    elif t == 'simrad':
        model.add(keras.Input(shape=(29,29,1), name='input'))
        model.add(layers.Conv2D(5, (5,5), strides=2, activation='relu', name='conv_1')) # output is 13x13
        model.add(layers.Conv2D(50, (5,5), strides=2, activation='relu', name='conv_2')) # output is 5x5
        model.add(layers.Flatten(name='flaten'))
        model.add(layers.Dense(100, activation='relu', name='fc_1'))
    elif t =='simrad_tuned_a':
        model.add(keras.Input(shape=(29,29,1), name='input'))
        model.add(layers.Conv2D(8, (5,5), strides=2, activation='relu', name='conv_1'))
        model.add(layers.Conv2D(64, (5,5), strides=2, activation='relu', name='conv_2'))
        model.add(layers.Flatten(name='flaten'))
        model.add(layers.Dense(98, activation='relu', name='fc_1'))
    elif t =='simrad_tuned_b':
        model.add(keras.Input(shape=(29,29,1), name='input'))
        model.add(layers.Conv2D(8, (5,5), strides=2, activation='relu', name='conv_1'))
        model.add(layers.Conv2D(64, (5,5), strides=2, activation='relu', name='conv_2'))
        model.add(layers.Flatten(name='flaten'))
        model.add(layers.Dense(128, activation='relu', name='fc_1'))
    elif t =='rodrigob':
        #TODO
        return
    else:
        print('Select a valid option:\n\'MLP1\'\n\'MLP2\'\n\'lenet5\'\n\'lenet5_tuned\'\n')
        return
    if not l1:
        model.add(layers.Dense(10, activation='softmax', name='output'))
    else:
        model.add(layers.Dense(10, activation='softmax', kernel_regularizer=keras.regularizers.l1(1e-5), name='output'))
    return model

def build_model(t):
    model = get_model(t)
    if t != 'MLP1' and t != 'MLP2':
        opt = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-2 / 10) # 10 epochs with categorical data
        model.compile(loss=keras.losses.CategoricalCrossentropy(), 
                          optimizer=opt, metrics=['accuracy'])
    else:
        model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), 
                  optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
                  metrics=['accuracy'])#keras.metrics.SparseCategoricalAccuracy()
    return model    