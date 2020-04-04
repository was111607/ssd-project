from numba import cuda
import keras
cuda.select_device(0)
keras.backend.clear_session()
cuda.close()
