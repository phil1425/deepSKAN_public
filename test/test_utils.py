import numpy as np
from deepGTA import generate_binary_matrix, generate_matrix
from deepGTA import generate_viable_ids, data_dy, generate_kinetics

def test_binary_matrix():
    K_test = generate_binary_matrix(0, 5)
    np.testing.assert_array_equal(K_test, np.zeros([5,5]))

    K_test = generate_binary_matrix(1023, 5)
    K_true = [
        [-4 , 0, 0, 0, 0],
        [1 , -3, 0, 0, 0],
        [1 , 1, -2, 0, 0],
        [1 , 1, 1, -1, 0],
        [1 , 1, 1, 1, -0],
    ]
    np.testing.assert_array_equal(K_test, K_true)

def test_matrix():
    #generates binary and standard matrix with same kinetic id
    K_test = generate_matrix(1023, 5, [1e3, 1e6])
    K_b = generate_binary_matrix(1023, 5)

    # checks if the sum of all pathways is zero
    for i in range(5):
        assert np.sum(K_test[:,i]) < 1e-6

    # checks if all entries are in the specified range
    for i in range(5):
        K_test[i,i] = 0

    assert np.amin(K_test) == 0
    assert np.amax(K_test) < 1e-3

    #checks if the binary and standard matrix have the same model
    for i in range(5):
        K_test[i,i] = 0
        K_b[i,i] = 0
        for j in range(5):
            assert (K_test[i,j] != 0) == (K_b[i,j] != 0) 
    
def test_data_dy():
    K = generate_binary_matrix(1023, 5)
    
    #checks if the sum of all changes is zero (makes sure no population gets
    # lost)
    assert np.sum(data_dy(0, [0, 0, 0, 0, 0], K, None)) == 0
    assert np.sum(data_dy(0, [0, 0, 0, 0, 0], K, 1000)) == 0
    assert np.sum(data_dy(0, [1, 1, 1, 1, 1], K, 1000)) == 0

    #at -inf, the change should be zero
    assert data_dy(-1e12, [0, 0, 0, 0, 1], K, 1000)[0] < 1e-8

    #at 0, the change in the first component should be greater than zero
    assert data_dy(0, [0, 0, 0, 0, 1], K, 1000)[0] > 1e-8

def test_generate_kinetics():
    data_t = np.logspace(2, 6, 256)
    K = generate_matrix(1023, 5, [1e3, 1e3])
    data_c = generate_kinetics(data_t, K, 1000)

    assert all(np.isfinite(data_c.flatten()))
    
    # sum at every step is 1 (no population gets lost)
    for i in range(256):
        assert np.abs(np.sum(data_c[i])-1) < 1e-6
    
    # no species gets more populated than 0.1
    for i in range(4):
        assert np.amax(data_c[:,i]) <= 0.1
    
    #no negative population
    assert np.amin(data_c) > -1e-6


def test_ids():
    ids = generate_viable_ids.unique_ids
    assert ids[0] == 0
    assert ids[-1] == 947
    assert ids[32] == 675
    assert ids[-32] == 787