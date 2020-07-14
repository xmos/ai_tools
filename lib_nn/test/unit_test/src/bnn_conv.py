import numpy as np

# https://github.com/wiseodd/hipsternet


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def conv_forward(X, W, stride=1, padding=0):
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception("Invalid output dimension!")

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    out = W_col @ X_col
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)

    return out


def print_nd(d, indent=0):
    if len(d.shape) == 1:
        print(" " * indent, end="")
        print("{", end="")
        for v in d:
            print(str(v) + ", ", end="")
    else:
        print(" " * indent, end="")
        print("{")
        for i in range(len(d)):
            print_nd(d[i], indent + 1)
        print(" " * indent, end="")

    if indent == 0:
        print("}", end="")
    else:
        print("},")


def make_up_post_activation_vals(Y, output_scale=255):

    effective_post_activation_multiplier = output_scale / (
        np.amax(Y, axis=(0, 1)) - np.amin(Y, axis=(0, 1))
    )

    post_max = np.amax(Y, axis=(0, 1)) * effective_post_activation_multiplier
    post_min = np.amin(Y, axis=(0, 1)) * effective_post_activation_multiplier

    # center the activation
    effective_post_activation_bias = post_min + (post_max - post_min) / 2 - 0.5

    return effective_post_activation_multiplier, effective_post_activation_bias


batch_size = 1  # Don't change this, it's not used
input_channels = 2
output_channels = 8

x_height = 8
x_width = 8
k_height = 3
k_width = 3

print("#define K_HEIGHT " + str(k_height))
print("#define K_WIDTH " + str(k_width))
print("#define CHANS_OUT " + str(output_channels))
print("#define CHANS_IN " + str(input_channels))
print("#define X_HEIGHT " + str(x_height))
print("#define X_WIDTH " + str(x_width))

print("#define Y_HEIGHT (((X_HEIGHT - H_OFFSET) / V_STRIDE) - K_HEIGHT + 1)")
print("#define Y_WIDTH (((X_WIDTH - V_OFFSET) / H_STRIDE) - K_WIDTH + 1)")

k_shape = (output_channels, input_channels, k_height, k_width)
x_shape = (batch_size, input_channels, x_height, x_width)

X = np.random.randint(0, 2, x_shape) * 2 - 1
K = np.random.randint(0, 2, k_shape) * 2 - 1
# X = np.zeros(x_shape, dtype=int) * 2 - 1
# K = np.zeros(k_shape, dtype=int) * 2 - 1

Y = conv_forward(X, K, stride=1, padding=0)

Y = (k_height * k_width * input_channels - Y) // 2  # convert to a pop count

# Transpose to get into the same order as the C impl
Y = np.transpose(Y[0], (1, 2, 0))
K = np.transpose(K, (0, 2, 3, 1))
X = np.transpose(X[0], (1, 2, 0))

# https://github.com/larq/compute-engine/blob/master/larq_compute_engine/core/bconv2d_output_transform.h

int8_output = False
if int8_output:

    # line 28+ OutputTransformBase
    backtransform_add = 0
    Y = backtransform_add - 2 * Y
    Y = np.minimum(Y, np.iinfo(np.int32).max)
    Y = np.maximum(Y, np.iinfo(np.int32).min)

    # scale to use the whole range(maybe a bit more to make the clamping do something)
    (
        effective_post_activation_multiplier,
        effective_post_activation_bias,
    ) = make_up_post_activation_vals(Y, output_scale=255 + 6)

    # line 83+ OutputTransform<AccumScalar, std::int8_t>
    Y = np.asarray(Y, dtype=np.float)
    Y *= effective_post_activation_multiplier
    Y += effective_post_activation_bias

    Y = np.minimum(Y, np.iinfo(np.int8).max)
    Y = np.maximum(Y, np.iinfo(np.int8).min)
    Y = np.asarray(Y, dtype=np.int8)

    print(
        "float WORD_ALIGNED effective_post_activation_multiplier[CHANS_OUT] = ", end=""
    )
    print_nd(effective_post_activation_multiplier)
    print(";")
    print("float WORD_ALIGNED effective_post_activation_bias[CHANS_OUT] = ", end="")
    print_nd(effective_post_activation_bias)
    print(";")

else:

    # this gives a nice spread of results
    thresholds = np.arange(output_channels, dtype=int) + (
        ((k_height * k_width * input_channels) - output_channels) / 2
    )

    # line 64+ struct OutputTransform<AccumScalar, std::int32_t>
    Y = Y > thresholds
    Y = 1 - Y * 2
    Y = np.asarray(Y, dtype=np.int8)  # this is for convinence, it's not bit packed

    print("int16_t WORD_ALIGNED thresholds[CHANS_OUT] = ", end="")
    print_nd(thresholds)
    print(";")

print("bnn_bool_t WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN] = ", end="")
print_nd(X)
print(";")

print("bnn_bool_t WORD_ALIGNED K[CHANS_OUT][K_HEIGHT][K_WIDTH][CHANS_IN] = ", end="")
print_nd(K)
print(";")

print("bnn_bool_t WORD_ALIGNED Y_expected[Y_HEIGHT][Y_WIDTH][CHANS_OUT] = ", end="")
print_nd(Y)
print(";")
