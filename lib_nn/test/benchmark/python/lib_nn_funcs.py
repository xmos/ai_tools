
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

FUNC_LUT = dict()

def func_handler(func):
    fname = str(func).split()[1]
    FUNC_LUT[fname] = func
    return func


@func_handler
def vpu_memcpy(measure, args):

    things = [
        ("small", range(0,  48, 1)),
        ("med",   range(8, 512, 8)),
        ("large", range(32, 10000, 48)),
    ]

    cost = lambda size: 8 + (3*(size//32))

    for name, buff_sizes in things:
        print(f"\t\tvpu_memcpy ({name})")
        op_count = measure(buff_sizes)
        # cycles = [measure(x) for x in buff_sizes]

        plt.figure()
        plt.plot(buff_sizes, op_count,marker='o')
        plt.plot(buff_sizes, [cost(x) for x in buff_sizes])
        plt.title('vpu_memcpy(*, *, size)')
        plt.xlabel('size')
        plt.ylabel('Thread Cycles')
        plt.grid()

        if args.show_plot:
            plt.show()
        else:
            plt.savefig(os.path.join(args.out_dir, f"vpu_memcpy_{name}.png"))


@func_handler
def requantize_16_to_8(measure, args):

    things = [
        ("small", range(0,  48, 1)),
        ("med",   range(8, 512, 8)),
        ("large", range(32, 10000, 48)),
    ]

    cost = lambda size: 10 + (4*(size//16)) + 1*(size >= 16) + 3*(size%16 > 0)

    for name, buff_sizes in things:
        print(f"\t\trequantize_16_to_8 ({name})")
        op_count = measure(buff_sizes)
        # cycles = [measure(x) for x in buff_sizes]

        plt.figure()
        plt.plot(buff_sizes, op_count,marker='o')
        plt.plot(buff_sizes, [cost(x) for x in buff_sizes])
        plt.title('requantize_16_to_8(*, *, size)')
        plt.xlabel('size')
        plt.ylabel('Thread Cycles')
        plt.grid()

        if args.show_plot:
            plt.show()
        else:
            plt.savefig(os.path.join(args.out_dir, f"requantize_16_to_8_{name}.png"))


@func_handler
def lookup8(measure, args):

    things = [
        ("small", range(0,  48, 1)),
        ("med",   range(8, 512, 8)),
        ("large", range(32, 10000, 48)),
    ]

    cost = lambda size: 9 + (61 * size // 16) + 1*(size>0) 

    for name, buff_sizes in things:
        print(f"\t\tlookup8 ({name})")
        op_count = measure(buff_sizes)
        # cycles = [measure(x) for x in buff_sizes]

        plt.figure()
        plt.plot(buff_sizes, op_count,marker='o')
        plt.plot(buff_sizes, [cost(x) for x in buff_sizes])
        plt.title('lookup8(*, *, size)')
        plt.xlabel('size')
        plt.ylabel('Thread Cycles')
        plt.grid()

        if args.show_plot:
            plt.show()
        else:
            plt.savefig(os.path.join(args.out_dir, f"lookup8_{name}.png"))


@func_handler
def nn_conv2d_hstrip_deep(measure, args):

    cols = ['K_h', 'K_w', 'C_in', 'out_cols', 'cycles']

    k_h = [1, 3, 5]
    k_w = [1, 3, 5]
    c_in = [16, 32, 48, 80]
    out_cols = [1, 2, 3]

    params = []
    for prm in itertools.product(k_h, k_w, c_in, out_cols):
        params.extend(prm)

    data = zip(itertools.product(k_h, k_w, c_in, out_cols), measure(params))

    data = [x + (y,) for x,y in data]

    data = pd.DataFrame(columns = cols, data=data, dtype=np.int32)
    data = data.astype({k:np.int32 for k in cols})

    data.to_csv(os.path.join(args.out_dir, "nn_conv2d_hstrip_deep.csv"), index=False)

    ax = data[(data.K_h==1) & (data.K_w==1) & (data.C_in==16)].set_index(['out_cols']).cycles.plot(
        title='K_h = 1; K_w = 1; C_in = 16', marker='o', grid=True, legend=False)
    ax.set_ylabel('cycles')
    ax.locator_params(integer=True)

    if args.show_plot:
        plt.show()
    else:
        plt.savefig(os.path.join(args.out_dir, "nn_conv2d_hstrip_deep1.png"))
    




@func_handler
def conv2d_deep(measure, args):

    cols = ['X_height', 'X_width', 'X_chans', 'Y_height', 'Y_width', 'Y_chans', 
            'K_h', 'K_w', 'start_row', 'start_col', 'vstride', 'hstride', 'cycles']

    case_list = [
        #  X               Y               Win          start        stride
        [  1,  1,  4,      1, 1,  4,       1,  1,       0,  0,       1,  1      ],
        [  1,  1, 16,      1, 1, 16,       1,  1,       0,  0,       1,  1      ],
        [  1,  1, 32,      1, 1, 32,       1,  1,       0,  0,       1,  1      ],
        [  1,  1, 40,      1, 1, 40,       1,  1,       0,  0,       1,  1      ],
        [  1,  3, 40,      1, 3, 40,       1,  1,       0,  0,       1,  1      ],
        [  1,  5, 40,      1, 5, 40,       1,  1,       0,  0,       1,  1      ],
        [  3,  1, 40,      3, 1, 40,       1,  1,       0,  0,       1,  1      ],
        [  5,  1, 40,      5, 1, 40,       1,  1,       0,  0,       1,  1      ],
        [  3,  3, 40,      3, 3, 40,       1,  1,       0,  0,       1,  1      ],
        [  5,  5, 40,      5, 5, 40,       1,  1,       0,  0,       1,  1      ],
        [  3,  3, 40,      3, 1, 40,       1,  3,       0,  0,       1,  1      ],
        [  3,  3, 40,      1, 3, 40,       3,  1,       0,  0,       1,  1      ],
        [  3,  3, 40,      1, 1, 40,       3,  3,       0,  0,       1,  1      ],
        
    ]

    results = []

    for prms in case_list:
        print(f"{tuple(prms)}..")

        results.append(prms + measure(prms))

    data = pd.DataFrame(columns = cols, data=results, dtype=np.int32)
    data.to_csv(os.path.join(args.out_dir, "conv2d_deep.csv"), index=False)




@func_handler
def nn_conv2d_hstrip_deep(measure, args):

    cols = ['K_h', 'K_w', 'C_in', 'out_cols', 'cycles']

    k_h = [1, 3, 5]
    k_w = [1, 3, 5]
    c_in = [16, 32, 48, 80]
    out_cols = [1, 2, 3]

    params = []
    for prm in itertools.product(k_h, k_w, c_in, out_cols):
        params.extend(prm)

    data = zip(itertools.product(k_h, k_w, c_in, out_cols), measure(params))

    data = [x + (y,) for x,y in data]

    data = pd.DataFrame(columns = cols, data=data, dtype=np.int32)
    data = data.astype({k:np.int32 for k in cols})

    data.to_csv(os.path.join(args.out_dir, "nn_conv2d_hstrip_deep/nn_conv2d_hstrip_deep.csv"), index=False)

    ax = data[(data.K_h==1) & (data.K_w==1) & (data.C_in==16)].set_index(['out_cols']).cycles.plot(
        title='K_h = 1; K_w = 1; C_in = 16', marker='o', grid=True, legend=False)
    ax.set_ylabel('cycles')
    ax.locator_params(integer=True)

    if args.show_plot:
        plt.show()
    else:
        plt.savefig(os.path.join(args.out_dir, "nn_conv2d_hstrip_deep1.png"))



@func_handler
def avgpool2d(measure, args):

    params = []

    for out_rows in range(2, 6, 2):
        for out_cols in range(2, 6, 2):
            for channels in range(4, 12, 4):
                for pool_rows in range(1,4):
                    for pool_cols in range(1,4):
                        params.append((out_rows, out_cols, channels, pool_rows, pool_cols, pool_rows, pool_cols))
    

    flattened_params = [y for x in params for y in x]

    products = [a*b*c*d*e*f for a,b,c,d,e,f,g in params]

    op_count = measure(flattened_params, ['avgpool2d', 'avgpool2d_2x2'])

    plt.figure()
    plt.scatter(products, op_count, marker='o')
    plt.title('avgpool2d')
    plt.xlabel('out_rows * out_cols * channels * pool_rows * pool_cols')
    plt.ylabel('Thread Cycles')
    plt.grid()

    if args.show_plot:
        plt.show()
    else:
        plt.savefig(os.path.join(args.out_dir, f"avgpool2d_{name}.png"))