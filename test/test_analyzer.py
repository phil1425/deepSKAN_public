import numpy as np
from deepGTA import GTAnalyzer

def test_testing():
    assert 1 == 1

def test_init():
    ana = GTAnalyzer(np.zeros([256, 64]), np.logspace(2,6,256))

def test_c_irf():
    ana = GTAnalyzer(np.zeros([256, 64]), np.logspace(2,6,256))
    data_t = np.logspace(2,6,256)

    c_test = ana.c_irf(data_t, 1e-3, 0, 0)
    assert all(np.isfinite(c_test))

    c_true = np.exp(-1e-3*data_t)
    assert np.sum(np.abs(c_test-c_true)) < 1e-10

'''
def test_generate_fit():
    ana = GTAnalyzer(np.zeros([256, 64]), np.logspace(2,6,256))
    ana.num_decays = 3
    ana.num_spectra = 3
    ana.offset = False
    ana.iterations = 0
    data_t = np.logspace(2,6,256)

    decays = [1e-3, 1e-4, 1e-5]
    fit = ana.generate_fit(np.array([1e-3, 1e-4, 1e-5, 0, 0]))
    c_true = np.array([ana.c_irf(data_t, d, 0, 0) for d in decays])
    
    assert np.sum(np.abs(ana.c_fit-c_true.T)) < 1e-10
'''