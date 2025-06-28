"""
Data types used throughout the source code for type annotations.
"""

import numpy as np

type NumpyInt32Array1D = np.ndarray[tuple[int], np.dtype[np.int32]]
type NumpyInt32Array2D = np.ndarray[tuple[int, int], np.dtype[np.int32]]
type NumpyFloat32Array1D = np.ndarray[tuple[int], np.dtype[np.float32]]
type NumpyFloat32Array2D = np.ndarray[tuple[int, int], np.dtype[np.float32]]
type NumpyFloat64Array1D = np.ndarray[tuple[int], np.dtype[np.float64]]
