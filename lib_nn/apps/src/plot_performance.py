import get_time as t
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # print(2 ** np.arange(8,13))
    # exit(1)
    with np.load("int8.npz") as data:

        plt.clf()
        fig1, ax1 = plt.subplots()
        for k in range(1, 6, 2):

            x = []
            y = []
            for i in range(256, 2 ** 13 + 1, 256):

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

            plt.loglog(x, y, label=str(k) + "x" + str(k))
        plt.legend(title="Kernel nxn")
        plt.xlabel("Input Channels")
        plt.ylabel("xor/Second (Giga)")

        ax1.set_yticklabels([42, 56, 75, 100, 133])
        plt.xticks(2 ** np.arange(8, 14))
        plt.yticks([42, 56, 75, 100, 133])

        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        plt.grid()
        plt.figtext(
            0.9,
            0.01,
            "Efficiency is independent of output channel count\n1x1 to be optimised",
            ha="right",
            fontsize=9,
        )
        plt.show()

