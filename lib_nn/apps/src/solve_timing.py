import numpy as np
import os
import common as c


def _solve(base_name, output_dir, plot_errors=False):

    csv_filename = base_name + ".csv"

    raw_data = np.genfromtxt(csv_filename, delimiter=",")
    raw_data = np.asarray(raw_data, dtype=int)

    x = raw_data[:, :-1]
    y = raw_data[:, -1, np.newaxis]

    # add the constant time stuff in, i.e. the stuff that doesnt depened on the loops
    surrogate_x = c.params_to_iterations(x)

    coefs, residuals, rank, s = np.linalg.lstsq(surrogate_x, y, rcond=None)
    print("coefs", np.round(coefs[:, 0], 4))
    output_file = os.path.join(output_dir, base_name + ".npz")
    np.savez(output_file, coefs=coefs[:, 0])

    rms_error = np.sqrt(residuals[0]) / len(surrogate_x)
    if rms_error > 0.5:
        print("Error: RMS error is too high")
        exit(1)
    else:
        print("RMS error:", rms_error)

    z = np.dot(surrogate_x, coefs)

    abs_error = abs(z - y)
    abs_error = np.reshape(abs_error, (len(abs_error,)))
    worse_abs_case = np.argmax(abs_error)
    print(
        "worse abs error:",
        abs_error[worse_abs_case],
        raw_data[worse_abs_case],
        z[worse_abs_case][0],
    )

    rel_error = abs(z - y) / y
    rel_error = np.reshape(rel_error, (len(rel_error,)))
    worse_rel_case = np.argmax(rel_error)
    print(
        "worse relative error:",
        rel_error[worse_rel_case],
        raw_data[worse_rel_case],
        z[worse_rel_case][0],
    )

    if plot_errors:
        import matplotlib.pyplot as plt

        plt.clf()
        plt.subplot(211)
        plt.hist(abs_error, bins=200, log=True)
        plt.subplot(212)
        plt.hist(worse_rel_case, bins=200, log=True)
        plt.savefig(base_name + ".pdf")
    return coefs


if __name__ == "__main__":
    _solve("int8", ".")
    _solve("bin", ".")
