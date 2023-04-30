import numpy as np
import numba


@numba.njit
def radius_of_nose(r_max: float, ln: float, nn: float, x_coordinate: float):
    x = x_coordinate
    r_x: float = 0
    # As numpy arg_0 can only take positive values we do the following
    # operation
    temp_abcsissa: float = 1 - np.power(((ln - x) / ln), nn)
    if (temp_abcsissa >= 0):
        r_x = r_max * (np.power(temp_abcsissa, 1 / nn))
    else:
        r_x = (-1) * r_max * (np.power(np.abs(temp_abcsissa), 1 / nn))

    return r_x


@numba.njit
def radius_of_tail(
        r_max: float,
        lt: float,
        nt: float,
        rel_x_coordinate: float):
    x = rel_x_coordinate
    r_x = r_max * (1 - np.power((x / lt), nt))
    return r_x

# Using First principles


@numba.njit
def derivative_of_nose_radius(r_max, nn, x: float, ln: float, divisions=500):
    dx = ln / divisions
    der_r: float = 0
    der_r = (radius_of_nose(r_max, ln, nn, x + dx) -
             radius_of_nose(r_max, ln, nn, x)) / dx
    if (np.isnan(der_r)):
        return 0
    return der_r


@numba.njit
def derivative_of_tail_radius(r_max, nt, x: float, lt: float, divisions=500):
    dx = lt / divisions
    der_r = (radius_of_tail(r_max, lt, nt, x + dx) -
             radius_of_tail(r_max, lt, nt, x)) / dx
    if (np.isnan(der_r)):
        return 0
    return der_r

# Wetted Areas of different sections


@numba.njit
def wetted_area_of_nose(r_max: float, ln: float, nn: float, divisions=1000):
    nose_area: float = 0
    for x in np.arange(0, ln, ln / divisions):
        y = radius_of_nose(r_max, ln, nn, x)
        der_y = derivative_of_nose_radius(r_max, nn, x, ln)
        nose_area = nose_area + 2 * np.pi * y * \
            np.sqrt(1 + np.power(np.abs(der_y), 2)) * (ln / divisions)
        x = x + ln / divisions

    return nose_area


@numba.njit
def wetted_area_of_tail(r_max: float, lt: float, nt: float, divisions=1000):
    tail_area: float = 0
    for x in np.arange(0, lt, lt / divisions):
        y = radius_of_tail(r_max, lt, nt, x)
        der_y = derivative_of_tail_radius(r_max, nt, x, lt)
        tail_area = tail_area + 2 * np.pi * y * \
            np.sqrt(1 + np.power(np.abs(der_y), 2)) * (lt / divisions)
        x = x + lt / divisions

    return tail_area


@numba.njit
def wetted_area_of_midsection(r_max: float, lm: float):
    mid_area = 2 * np.pi * r_max * lm
    return mid_area


@numba.njit
def get_total_surface_area(
        r_max: float,
        nn: float,
        nt: float,
        ln: float,
        lt: float):
    return wetted_area_of_nose(r_max,
                               ln,
                               nn) + wetted_area_of_midsection(r_max,
                                                               3 - ln - lt) + wetted_area_of_tail(r_max,
                                                                                                  lt,
                                                                                                  nt)

# Volumes of different sections


@numba.njit
def volume_of_nose(r_max: float, ln: float, nn: float, divisions=800):
    nose_volume: float = 0
    for x in np.arange(0, ln, ln / divisions):
        y = radius_of_nose(r_max, ln, nn, x)
        nose_volume = nose_volume + np.pi * y**2 * (ln / divisions)

    return nose_volume


@numba.njit
def volume_of_tail(r_max: float, lt: float, nt: float, divisions=800):
    tail_volume: float = 0
    for x in np.arange(0, lt, lt / divisions):
        y = radius_of_tail(r_max, lt, nt, x)
        tail_volume = tail_volume + np.pi * y**2 * (lt / divisions)

    return tail_volume


@numba.njit
def volume_of_midsection(r_max: float, lm: float):
    mid_vol = np.pi * r_max**2 * lm
    return mid_vol


@numba.njit
def get_total_volume(r_max: float, nn: float, nt: float, ln: float, lt: float):
    return volume_of_nose(r_max,
                          ln,
                          nn) + volume_of_midsection(r_max,
                                                     lm=3 - ln - lt) + volume_of_tail(r_max,
                                                                                      lt,
                                                                                      nt)


def get_block_coefficient(r_max: float, auv_vol):
    block_coefficient = auv_vol / (3 * 4 * r_max**2)
    return block_coefficient


def calc_wake_fraction(
        r_max: float,
        block_coefficient: float,
        auv_vol,
        vel=1.0):
    # Calculating the Froude Number for a velocity range of 0.5 - 2 m/s
    Fr = vel / np.sqrt(9.81 * 3.0)

    # Calculating Dw, the wake fraction parameter, Pg 182 of book
    r_prop = 0.25 * r_max
    Dw = np.power(auv_vol, -1 / 6) * (2 * r_max) * np.power(r_prop, -0.5)

    wake_fraction = -0.0458 + 0.3745 * block_coefficient**2 + \
        0.159 * Dw - 0.8635 * Fr + 1.4773 * Fr**2
    return wake_fraction

# Warmup the numba fucntions here so that they get compiled


def warmup_numba_funcs(should_print: bool = 0):
    r_max_test = 0.25
    nn_test = 1
    nt_test = 1
    ln_test = 0.7
    lt_test = 0.7
    temp_vol = get_total_volume(r_max_test, nn_test, nt_test, ln_test, lt_test)
    temp_area = get_total_surface_area(
        r_max_test, nn_test, nt_test, ln_test, lt_test)
    temp_cb = get_block_coefficient(r_max_test, temp_vol)
    temp_wk = calc_wake_fraction(r_max_test, temp_cb, temp_vol)

    if (should_print):
        print(
            "Volume is ",
            temp_vol,
            "\nBlock Coefficient is ",
            temp_cb,
            "\nWake Fraction is ",
            temp_wk,
            "\nArea is ",
            temp_area)

    return

warmup_numba_funcs(0)