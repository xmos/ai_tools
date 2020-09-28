import numpy as np
import os
import common as c


def round_up_to_multiples_of(x, n):
    return (x + n - 1) // n


def con2d_params_to_kernel_params(
    conv2d_params, output_channels_per_loop, input_channels_per_loop
):

    kernel_params = [
        [
            conv2d_params["x_height"],
            conv2d_params["x_width"],
            round_up_to_multiples_of(
                conv2d_params["output_channels"], output_channels_per_loop
            ),
            conv2d_params["kernel_height"],
            conv2d_params["kernel_width"],
            round_up_to_multiples_of(
                conv2d_params["input_channels"], input_channels_per_loop
            ),
        ]
    ]

    return np.asarray(kernel_params, dtype=int)


def con2d_params_to_bin_kernel_params(conv2d_params):
    return con2d_params_to_kernel_params(conv2d_params, 32, 256)


def con2d_params_to_int8_kernel_params(conv2d_params):
    return con2d_params_to_kernel_params(conv2d_params, 16, 256)


def get_maccs(conv2d_params):
    return (
        conv2d_params["kernel_height"]
        * conv2d_params["kernel_width"]
        * conv2d_params["input_channels"]
        * conv2d_params["x_height"]
        * conv2d_params["x_width"]
        * conv2d_params["output_channels"]
    )


def get_time(conv2d_params, con2d_params_to_kernel_params_fn, coefs):

    # convert conv2d params to kernel params
    params = con2d_params_to_kernel_params_fn(conv2d_params)
    surrogate_params = c.params_to_iterations(params)
    estimated_time = np.dot(surrogate_params, coefs)

    return estimated_time


def get_instructions(timer_ticks, xcore_system_clock_mhz=700, timer_tick_ns=10):

    # 700 is default for XCORE-AI-EXPLORER.xn
    xcore_clock_ns = 5 * 1e9 / (xcore_system_clock_mhz * 1e6)

    return int(np.ceil(timer_ticks * timer_tick_ns / xcore_clock_ns))


def get_stats(data, conv2d_params, con2d_params_to_kernel_params_fn):

    coefs = data["coefs"]
    time = get_time(conv2d_params, con2d_params_to_kernel_params_fn, coefs)
    macc_count = get_maccs(conv2d_params)
    insts = get_instructions(time)

    print("Instructions", insts)
    print(
        "Efficiency",
        np.round(macc_count / insts, 2),
        "maccs per inst",
        np.round(100 * (macc_count / insts) / 256, 1),
        "%",
    )


def profile_conv2d(conv2d_params):
    print(conv2d_params)

    print("int8 output")
    with np.load("int8.npz") as data:
        get_stats(data, conv2d_params, con2d_params_to_int8_kernel_params)

    print("Binary output")
    with np.load("bin.npz") as data:
        get_stats(data, conv2d_params, con2d_params_to_bin_kernel_params)

    print()


if __name__ == "__main__":

    # Example network
    conv2d_params = {
        "kernel_height": 1,
        "kernel_width": 1,
        "input_channels": 256,
        "output_channels": 256,
        "x_height": 128,
        "x_width": 128,
    }
    profile_conv2d(conv2d_params)

    # asymptotic highest efficiency
    conv2d_params = {
        "kernel_height": 1,
        "kernel_width": 1,
        "input_channels": 2 ** 55,
        "output_channels": 32,
        "x_height": 1,
        "x_width": 1,
    }

    profile_conv2d(conv2d_params)
