import numpy as np
import os

def params_to_iterations(x):
    x = np.c_[np.ones(len(x), dtype=int), x]
    l = x.shape[-1]
    for i in range(l):
        x[:, l - i - 1] = np.prod(x[:, : l - i], axis=1)
    # 1, a,   ab,     abc,     abcd
    #    1,    a,     ab,      abc,
    # 1, a-1, a(b-1), ab(c-1), abc(d-1)
    # This is to account for the fact that the loops exectue n-1 times (the other iteration is owned by the outer loop)
    x[:, 1:] -= x[:, :1]
    return x
