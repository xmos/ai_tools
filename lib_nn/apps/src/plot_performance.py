import get_time as t
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    with np.load("int8.npz") as data:

        plt.clf()
        for k in range(1, 8, 2):

            x = []
            y = []
            for i in range(256, 2 ** 12 + 1, 256):

                conv2d_params = {
                    "kernel_height": k,
                    "kernel_width": k,
                    "input_channels": i,
                    "output_channels": 2048,
                    "x_height": 16,
                    "x_width": 16,
                }
                macc_count, insts = t.get_stats(
                    data, conv2d_params, t.con2d_params_to_int8_kernel_params
                )

                inst_time = 1 / 700e6
                xor_popcount_per_second = macc_count / (insts * inst_time)
                y.append(xor_popcount_per_second / 1e9)
                x.append(i)

            plt.plot(x, y, label=str(k) + "x" + str(k))
        plt.legend(title="Kernel nxn")
        plt.xlabel("Input Channels")
        plt.ylabel("xor_popcount/Second (Giga)")
        plt.xticks(x)
        plt.show()
