import tensorflow as tf
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

def create_init(init='random_uniform'):
    initializer = {
        'constant': tf.constant_initializer(), # value
        'identity': tf.keras.initializers.Identity(), # 2D matrix only, gain
        'zeros': tf.zeros_initializer(),
        'ones': tf.ones_initializer(),
        'orthogonal': tf.keras.initializers.Orthogonal(), # 2D matrix only, gain + seed
        'random_normal': tf.random_normal_initializer(), # mean, stddev, seed
        'random_uniform': tf.random_uniform_initializer(), # minval, maxval, seed
        'truncated_normal': tf.keras.initializers.TruncatedNormal(), #mean, stdev, seed
        'glorot_normal': tf.keras.initializers.GlorotNormal(), # seed
        'glorot_uniform': tf.keras.initializers.GlorotUniform(), # seed
        'lecun_normal': tf.keras.initializers.lecun_normal(), # seed
        'lecun_uniform': tf.keras.initializers.lecun_uniform(), # seed
        'he_normal': tf.keras.initializers.he_normal(), # seed
        'he_uniform': tf.keras.initializers.he_uniform(),
        'variance_scaling': tf.keras.initializers.VarianceScaling() # scale, mode, distribution, seed
    }[init]
    return initializer

def get_initializers():
    return {
        'constant': tf.constant_initializer(), # value
        'identity': tf.keras.initializers.Identity(), # 2D matrix only, gain
        'zeros': tf.zeros_initializer(),
        'ones': tf.ones_initializer(),
        'orthogonal': tf.keras.initializers.Orthogonal(), # 2D matrix only, gain + seed
        'random_normal': tf.random_normal_initializer(), # mean, stddev, seed
        'random_uniform': tf.random_uniform_initializer(), # minval, maxval, seed
        'truncated_normal': tf.keras.initializers.TruncatedNormal(), # mean, stdev, seed
        'glorot_normal': tf.keras.initializers.GlorotNormal(), # seed
        'glorot_uniform': tf.keras.initializers.GlorotUniform(), # seed
        'lecun_normal': tf.keras.initializers.lecun_normal(), # seed
        'lecun_uniform': tf.keras.initializers.lecun_uniform(), # seed
        'he_normal': tf.keras.initializers.he_normal(), # seed
        'he_uniform': tf.keras.initializers.he_uniform(),
        'variance_scaling': tf.keras.initializers.VarianceScaling() # scale, mode, distribution, seed
    }

def create_initializer(init='random_uniform', conf={}):
    return create_init(init).from_config(conf)

def plot_histogram(init, title, mu=None, sigma=None, _dtype='float32', shape=[1000], path = Path('.')):
    _bins = 20
    init_values = init(shape=(shape), dtype=_dtype).numpy()
    if sigma == None:
        sigma = np.std(init_values)
    if mu == None:
        mu = np.mean(init_values)
    y = np.random.normal(mu, sigma, shape)
    # plot logic
    plt.style.use('dark_background')
    ax = sns.jointplot(y, init_values, space=0, kind='reg',
                       marginal_kws=dict(bins=_bins, rug=False))
    plt.xlabel('Frequency Histogram of ' + title + r' $\mu$=' + str(mu) +
              ', $\sigma$=' + str(sigma))
    ax.savefig(path/str('histogram_'+title+'.png'), dpi=150, transparent=True)
    

# Creating all initializers
const = create_init('constant') # <- CONST
zeros = create_init('zeros') # <- CONST
ones = create_init('ones') # <- CONST
rand_norm = create_init('random_normal') # <- mean & stddev
rand_unif = create_init('random_uniform') # <- minval maxval
rand_trunc = create_init('truncated_normal') # <- mean & stddev
lecun_normal = create_init('lecun_normal') # <- scale, distribution, modei
lecun_uniform = create_init('lecun_uniform') # <- scale, distribution, mode
he_normal = create_init('he_normal') # <- scale, distribution, mode
he_uniform = create_init('he_uniform') # <- scale, distribution, mode
variance_scaling = create_init('variance_scaling') # scale, mode, distribution, seed
# -------------------------------------------
glorot_normal = create_init('glorot_normal') # <- ?
glorot_uniform = create_init('glorot_uniform') # <- ?
identity = create_init('identity')
orth = create_init('orthogonal')

# Creating a dict with all initalizers
inits = get_initializers()
for key, value in inits.items():
    print(key, value.get_config())

mu = 10
sigma = 0.5
seed = 42
output_path = Path(__file__).parent.joinpath('histograms').resolve()

# Initializers generation
# Random normal
random_normal_initializer = create_initializer('random_normal', conf={
    'mean':mu,'stddev':sigma, 'seed': seed})
plot_histogram(random_normal_initializer, 'random_normal', mu, sigma, shape = [100], path=output_path)

# Random uniform
random_uniform_initializer = create_initializer('random_uniform', conf={
    'minval':8, 'maxval': 12})
plot_histogram(random_uniform_initializer, 'random_uniform', shape = [100], path=output_path)

# Random truncated
random_truncated_initializer = create_initializer('truncated_normal', conf={
    'mean':mu, 'stddev': sigma, 'seed': seed})
plot_histogram(random_truncated_initializer, 'truncated_normal', mu, sigma, shape=[100], path=output_path)

# Lecun normal
lecun_normal_initializer = create_initializer('lecun_normal', conf={
    'scale': 2.0, 'distribution': 'normal', 'seed': seed})
plot_histogram(lecun_normal_initializer, 'lecun_normal', shape=[100], path=output_path)

# Lecun uniform
lecun_uniform_initializer = create_initializer('lecun_uniform', conf={
    'scale': 2.0, 'distribution': 'uniform', 'seed': seed})
plot_histogram(lecun_uniform_initializer, 'lecun_uniform', shape=[100], path=output_path)

# He normal
he_normal_initializer = create_initializer('he_normal', conf={
    'scale': 2.0, 'distribution': 'normal', 'seed': seed})
plot_histogram(he_normal_initializer, 'he_normal', shape=[100], path=output_path)

# He uniform
he_uniform_initializer = create_initializer('he_uniform', conf={
    'scale': 2.0, 'distribution': 'uniform', 'seed': seed})
plot_histogram(he_uniform_initializer, 'he_uniform', shape=[200], path=output_path)

# Glorot_normal
glorot_normal_initializer = create_initializer('glorot_normal', conf={
    'seed': seed})
plot_histogram(glorot_normal_initializer, 'glorot_normal', shape=[200], path=output_path)

# Glorot_uniform
glorot_uniform_initializer = create_initializer('glorot_uniform', conf={
    'seed': seed})
plot_histogram(glorot_uniform_initializer, 'glorot_uniform', shape=[100], path=output_path)

# Const
par = np.random.pareto(3000, 200)
constant_initializer = create_initializer('constant', conf={
    'value': par})
plot_histogram(constant_initializer, 'constant_pareto', shape=[200], path=output_path)

