# coding: utf-8
import numpy as np

from math import pi
from typing import Callable


def f(x: float):
    return np.sin(x) + 10


def area_under_curve(func: Callable, start: float, end: float, splits: int) -> float:
    """
    :param func: функция, интеграл от которой берём
    :param start: нижний предел интегрирования
    :param end: верхний предел интегрирования
    :param splits: на сколько прямоугольников разбиваем площадь под кривой
    :return: приближённое значение интеграла
    """

    vfunc = np.vectorize(func)

    step: float = (end - start) / splits

    rectangles_bottom_centers: np.ndarray = np.linspace(start, end, splits,
                                                        endpoint=False) + step / 2

    return np.sum(vfunc(rectangles_bottom_centers) * step)


if __name__ == "__main__":

    ACCEPTABLE_PRECISION = 1e-5

    TRUE_VALUE = 2 + 10 * pi

    tens = [10 ** i for i in range(15)]
    approx_area = 0

    for splits in tens:
        print("\nSplits:", splits)
        approx_area = area_under_curve(func=f, start=0, end=pi, splits=splits)
        print("Expected:       %.7f" % TRUE_VALUE)
        print("Approximation:  %.7f" % approx_area)

        if abs(TRUE_VALUE - approx_area) < ACCEPTABLE_PRECISION:
            print("\n\nReasonable precision of [%.5f] achieved at [%d] splits" % (ACCEPTABLE_PRECISION, splits))
            break
