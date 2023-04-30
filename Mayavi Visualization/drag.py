def factor_nn_nt(nn, nt):
    return -1000.77 + -61.84 *nn + 1196.08 *nt + 35.01 *nn*nn + -5.63 *nn*nt + -366.69 * nt*nt + -3.7 *nn*nn*nn + 0.81 * nn*nn *nt + 0.25 * nn * nt*nt + 37.54 *nt*nt*nt


def factor(r_max, ln, lt):
    return (2.619 - 0.2066 * r_max + 3.3801 * (r_max**2)) * (2.619 - 4.7921 * lt + \
            3.35582 * (lt**2)) * (2.619 - 5.8352 * ln + 4.1666 * (ln**2)) / (2.169**2)

def factor_correction(r_max, ln, lt):
    return 2.7206 * ln - 1.5744 * r_max + 2.9460 * lt + 6.2718 * \
        (r_max**2) - 1.866 * ln * ln - 2.3126 * lt * lt - 3.4 * ln * lt + 2.1886 * ln * lt * lt - 1.2521


def total_drag(r_max, nn, nt, ln, lt):
    return (factor_nn_nt(nn, nt) * (factor(r_max, ln, lt) +
                                   factor_correction(r_max, ln, lt)))