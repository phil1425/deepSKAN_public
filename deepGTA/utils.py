import numpy as np
import random
import h5py
import keras
from tqdm import tqdm
import scipy.interpolate as ip
from scipy.integrate import RK45
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import networkx as nx

def generate_matrix(kinetic_id, num_s, t_0):
    '''
    generates a transfer matrix (K) for a given kinetic id
    with the rate constants (k) for all active pathways
    chosen from a loguniform distribution

    kinetic id (int): number of the kinetic model if created with this funcion.
        All other mentions of kinetic id come from this function.
        The id gets converted into binary and every digit represents one 
        possible pathway which can either be on (1) of off (0)

    num_s (int): number of columns (and also rows) in the transfer matrix.
        Usually, this corresponds to the number of species in the system, 
        including the ground state

    t_0 np.array([t_min, t_max], dtype=float):
        minimum and maximum possible values for the lifetimes of each pathway
        in the matrix. The values get chosen from this interval randomly.

    returns np.array(shape=(num_s,num_s), dtype=float)
    transfer matrix K
    '''
    n_permutable = [0, 1, 3, 6, 10][num_s-1] #number of possible pathways
    bool_list = [bool(kinetic_id & (1<<n)) for n in range(n_permutable)]
    matrix = np.zeros([num_s, num_s])
    for i in range(num_s):
        for j in range(i):
            if bool_list.pop():
                matrix[i,j] = 1/loguniform(t_0[0], t_0[1])

    for i in range(num_s):
        matrix[i,i] = -np.sum(matrix[:,i])
    return matrix

def generate_binary_matrix(kinetic_id, num_s):
    '''
    generates a transfer matrix (K) for a given kinetic id
    where every non-diagonal element is either 0 or 1
    The diagonal elements are calculated so the sum of each column is 0
    i.e the matrix can be simulated and no concentration gets lost
    '''
    n_permutable = [0, 1, 3, 6, 10][num_s-1]
    bool_list = [bool(kinetic_id & (1<<n)) for n in range(n_permutable)]
    matrix = np.zeros([num_s, num_s])
    for i in range(num_s):
        for j in range(i):
            if bool_list.pop():
                matrix[i,j] = 1

    for i in range(num_s):
        matrix[i,i] = -np.sum(matrix[:,i])
    return matrix

def data_dy(t, y_in, kinetic_matrix, irf):
    '''
    converts a kinetic matrix into a set of differential equations
    '''
    num_s = kinetic_matrix.shape[0]
    y_out = np.zeros(num_s)
    for i in range(num_s):
        y_tmp = 0
        for j in range(num_s):
            y_tmp += y_in[j]*kinetic_matrix[i][j] 
        y_out[i] = y_tmp

    if irf != None:
        y_out[0] += 0.1*gauss(t, 0, irf)
        y_out[-1] += -0.1*gauss(t, 0, irf)
    return y_out

def generate_kinetics(data_t, K, sigma_irf, interpolate=True):
    '''
    generates the concentration traces for a given Kinetic matrix and gaussian 
    irf by numerically solving the differential equation with RK45
    '''
    num_s = K.shape[0]
    y_points = data_t.shape[0]
    c_0 = np.array([0, 0, 0, 0, 1]) #concentrations for t = -inf
    dy = lambda t, y: data_dy(t, y, K, sigma_irf) #matrix to diff. equation
    step_c = []
    step_t = []

    solver = RK45(dy, -(10*sigma_irf), c_0, data_t[-1], atol=1e-10, rtol=1e-10)

    while solver.status == 'running':
        solver.step()
        step_c.append(solver.y)
        step_t.append(solver.t)

    plot_c = np.zeros([y_points, 5])
    step_c = np.array(step_c)
    step_t = np.array(step_t)

    if not interpolate:
        return step_c, step_t
    else:
        # interpolate to desired logarithmic timescale
        interpolated2 = [interp1d(step_t, step_c[:,i], 
                         'cubic', assume_sorted=True) for i in range(num_s)]
        for j, f in enumerate(interpolated2):
            plot_c[:,j] = f(data_t)

        return plot_c

def gauss(x, mu, sig):
    '''
    gaussian function
    '''
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)

def uniform(a,b):
    '''
    random variable of the interval [a, b]
    '''
    return np.random.uniform(a, b)

def plot_graph(kinetic_id, num_s, color=None):
    '''
    Draws the Graph to any given kinetic id into a matplotlib figure
    '''
    K = generate_binary_matrix(kinetic_id, num_s)
    for i in range(num_s):
        K[i][i] = 0
    G = nx.from_numpy_matrix(K.transpose(), create_using=nx.DiGraph())
    nx.draw_circular(G, labels={0:'A',1:'B',2:'C',3:'D',4:'E'}, 
                     node_color=color, font_color='white')


def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))

def reciprocaluniform(low, high, size=None):
    return 1/np.random.uniform(1/high, 1/low, size)

def normalize(data):
    data_max = np.amax(data)
    return data/data_max

def top_10_acc(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=10)