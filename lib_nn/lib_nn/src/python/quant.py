from decimal import localcontext
import numpy as np
import math

# clamp8(((accu*2**A)*(pam*2**M) + pab*2**B)*2**-B)


def compute_result_q(accu, pam, pab, B, A, M):

    accu_q = np.rint(accu*2**A)
    if np.any(accu_q > np.iinfo(np.int16).max) or np.any(accu_q < np.iinfo(np.int16).min):
        raise Exception('accu scaling error')
    accu_q = np.int64(accu_q)

    pam_q = np.rint(pam*2**M)
    if np.any(pam_q > np.iinfo(np.int16).max) or np.any(pam_q < np.iinfo(np.int16).min):
        raise Exception('pam scaling error')
    pam_q = np.int64(pam_q)

    product_q = accu_q*pam_q

    with localcontext() as ctx:
        ctx.prec = 16
        pab_q = np.int64(pab * 2**B)

    sum_q = product_q + pab_q

    sum_q = np.clip(sum_q, np.iinfo(np.int32).min, np.iinfo(np.int32).max)

    # scale down to output
    scaled_q = np.rint(sum_q * 2**-B)
    scaled_q = np.clip(scaled_q, np.iinfo(np.int8).min, np.iinfo(np.int8).max)

    return np.int8(scaled_q)


def compute_result(accu, pam, pab):
    return np.int8(np.clip(np.rint(accu * pam + pab), np.iinfo(np.int8).min, np.iinfo(np.int8).max))


def pick(k_height, k_width, chans_in, chans_out):

    receptive_volume = k_height*k_width*chans_in

    accu_max = receptive_volume//2
    accu_min = -receptive_volume//2

    # print('accu_max', accu_max, 'accu_min', accu_min)

    output_max = np.iinfo(np.int8).max
    output_min = np.iinfo(np.int8).min

    accu_range = accu_max - accu_min
    output_range = output_max - output_min

    m = output_range / accu_range
    post_activation_multiplier = np.random.uniform(-1.5*m, 1.5*m, chans_out)
    post_activation_bias = np.random.uniform(
        output_max//2, output_min//2, chans_out) + output_min - accu_min * m

    vpu_offset = receptive_volume//2
    channel_overlaps = np.random.randint(0, 16, chans_out)

    post_activation_bias += post_activation_multiplier*vpu_offset
    post_activation_bias += post_activation_multiplier*channel_overlaps

    # clamp8(((accu*2**A)*(pam*2**M) + pab*2**B)*2**-B)
    pab_max_exp = max([math.frexp(b)[1] for b in post_activation_bias])
    B_max = 31 - pab_max_exp
    B_min = -pab_max_exp

    accu_max_exp = max([math.frexp(a)[1] for a in [accu_max, accu_min]])
    A_max = 15 - accu_max_exp
    A_min = -accu_max_exp

    pam_max_exp = max([math.frexp(m)[1] for m in post_activation_multiplier])
    M_max = 15 - pam_max_exp
    M_min = -pam_max_exp

    # M + A = B

    max_bits = 0
    best_results = []

    for B in range(B_max, B_min, -1):

        pab_scaled = np.rint(post_activation_bias * 2**B)

        # check for rounding taking the bias out of range
        if np.any(pab_scaled > np.iinfo(np.int32).max):
            continue
        if np.any(pab_scaled < np.iinfo(np.int32).min):
            continue

        for A in range(A_max, A_min, -1):
            M = B - A
            if M <= M_max and M >= M_min:

                pam_bits = pam_max_exp + M
                if A > 0:
                    accu_bits = accu_max_exp
                else:
                    accu_bits = accu_max_exp + A

                product_bits = pam_bits + accu_bits

                if product_bits > max_bits:
                    max_bits = product_bits
                    best_results = []

                if product_bits == max_bits:
                    best_results.append((B, A, M, accu_bits, pam_bits))

    for (B, A, M, accu_bits, pam_bits) in best_results:
        print('B', B, 'A', A, 'M', M, 'accu_bits',
              accu_bits, 'pam_bits', pam_bits)

    (B, A, M, accu_bits, pam_bits) = best_results[0]
    print('B', B, 'A', A, 'M', M, 'accu_bits',
          accu_bits, 'pam_bits', pam_bits)

# TODO
# (necessary) pick the best quantisation constants from the avalaible ones
# (necessary) add explaination
# (inprovement) account for the bias and product accumulation not increasing the number of bits
# (inprovement) measure the performance
