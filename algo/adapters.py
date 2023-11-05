#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np


def svd(a: np.ndarray, source='numpy') -> Tuple:
    if source == 'numpy':
        return np.linalg.svd(a)


if __name__ == '__main__':
    pass