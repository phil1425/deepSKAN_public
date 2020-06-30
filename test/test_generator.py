import numpy as np
from deepGTA import onlineDataGenerator
def test_generator_training():
    gen = onlineDataGenerator(8, 8)
    X, Y = gen.__getitem__(0)

    assert X.shape == (8, 256, 64, 1)
    assert Y.shape == (8, 103)
    assert np.amax(X) == 1

    assert np.amin(Y) == 0
    assert np.max(Y) == 1

    assert all(np.isfinite(X.flatten()))

def test_generator_debug():
    gen = onlineDataGenerator(8, 8, debug_mode=True)
    X, Y, K, S, C, I = gen.__getitem__(0)

    assert X.shape == (8, 256, 64, 1)
    assert Y.shape == (8, 103)
    assert K.shape == (8, 5, 5)
    assert S.shape == (8, 5, 64)
    assert C.shape == (8, 256, 5)
    assert I.shape == (8,)

    assert np.amax(X) == 1

    assert np.amax(C) <= 1+1e-8
    assert np.amax(C[:,:,:-1]) <= 0.1+1e-8
    assert np.amin(C) > -1e-6

    assert all(np.isfinite(X.flatten()))
    assert all(np.isfinite(K.flatten()))
    assert all(np.isfinite(S.flatten()))
    assert all(np.isfinite(C.flatten()))

def test_generator_class():
    gen = onlineDataGenerator(1, 1, class_id=64, debug_mode=True)
    X, Y, K, S, C, I = gen.__getitem__(0)

    assert np.argmax(Y[0]) == 64

