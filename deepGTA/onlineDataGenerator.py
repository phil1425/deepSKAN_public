import numpy as np
import secrets

from scipy.interpolate import interp2d, interp1d
from scipy.integrate import RK45

import keras
import keras.backend as K


from .utils import generate_binary_matrix, generate_matrix, data_dy
from .utils import gauss, uniform, loguniform, normalize, generate_kinetics

from .generate_viable_ids import unique_ids

class onlineDataGenerator(keras.utils.Sequence):
    '''
    Generates data online for training or benchmarking.
    '''
    def __init__(self, batch_size, epoch_size, class_id=None, debug_mode=False):
        '''
        Initialization

        batch_size (int): number of items per batch

        epoch_size (int): number of items per epoch

        class_id (int): id of the kinetic model used to produce data. if None,
            random models will be chosen for each example

        debug_mode (bool): determines which data the generator will return.
            if True, all data needed for reproducing the 2d-spectrum will be returned
            (image, kinetic id, kinetic matrix, spectra, transients, response function)
            if False, only the data needed for training the network will be returned to increase
            performance. (image, kinetic id)
        '''
        self.batch_size = batch_size
        self.epoch_size = epoch_size 

        self.add_noise_irf = True # Adds noise+irf if true
        self.debug_mode = debug_mode

        self.data_t = np.logspace(2, 6, 256)

        self.range_irf = [50, 10000] # Min, Max values for sigma of the IRF

        ## Parameters for the spectra
        self.x_points = 64 # Pixels along lambda axis
        self.s_m = 1 # mu of width normal distribution
        self.max_gauss = 8 # number of gauss functions per spectrum

        ## Parameters for the dynamics
        self.y_points = 256 # points along tau axis
        self.t_0 = np.array([1, 1000])*1000 # Minimal and maximal values for the lifetimes

        self.c_0 = np.array([0, 0, 0, 0, 1]) # Concentrations of the species as t=-inf

        # number of species and rate constants
        # only change these if corresponding unique_ids were generated
        self.num_k = 4 # number of different k distributions
        self.num_s = self.num_k + 1 # number of species

        self.unique_ids = unique_ids
        self.num_classes = len(self.unique_ids)

        self.class_id = class_id

        # generate arrays for storing data

        self.X = np.zeros([self.batch_size, 256, 64, 1]) # Images
        self.Y = np.zeros([self.batch_size, 103]) # Kinetic id vectors

        if self.debug_mode:
            self.data_K = np.zeros([self.batch_size, 5, 5]) # transfer Matrices
            self.S = np.zeros([self.batch_size, 5, 64]) # Spectra
            self.C = np.zeros([self.batch_size, 256, 5]) # Transients (concentration traces)
            self.data_irf = np.zeros(self.batch_size) # sigma of the IRF


    def __len__(self):
        '''
        Denotes the number of batches per epoch
        '''
        return int(self.epoch_size / self.batch_size)

    def __getitem__(self, index):
        '''
        Generate one batch of data
        '''
        # get a new (cryptographically) random seed for the pseudo-rng to prevent
        # rng weirdness when using mulitple threads
        np.random.seed(secrets.randbits(32))

        for l in range(self.batch_size):
            # generates random spectra
            self.data_x = np.arange(0, self.x_points)
            self.data_s = np.zeros([self.num_s, self.x_points])

            noise = uniform(0, 0.05)
            for i in range(self.num_s):
                # each spectrum has at least 1 gauss component that is narrow
                num_gauss = np.random.randint(1, self.max_gauss+1)
                amplitude = uniform(0.5, 1)
                center = uniform(0, self.x_points)
                sigma = loguniform(self.s_m, self.x_points/4)
                g = gauss(self.data_x, center, sigma)
                g /= np.sum(g)
                g *= amplitude
                self.data_s[i] += g

                #the rest can be broad
                for _ in range(num_gauss-1):
                    amplitude = uniform(0.1, 1)
                    center = uniform(0, self.x_points)
                    sigma = loguniform(self.s_m, self.x_points)
                    g = gauss(self.data_x, center, sigma)
                    g /= np.sum(g)
                    g *= amplitude
                    self.data_s[i] += g
                self.data_s[i] += np.random.normal(0, noise/64, self.x_points)
            
            # picks class id for the next example
            if self.class_id == None:
                class_id = np.random.randint(0, self.num_classes)
            else:
                class_id = self.class_id

            # gets the kinetic id
            k_id = self.unique_ids[class_id]

            # generates a kinetic id with random rate constants for each pathway
            # and pre-checks if the matrix is worth computing
            found_solvable_kinetic = False
            while not found_solvable_kinetic:
                self.K = generate_matrix(k_id, self.num_s, self.t_0)
                K_b = generate_binary_matrix(k_id, self.num_s)
                for i in range(self.num_s):
                    K_b[i,i] = 0

                K_ = self.K.copy()

                def preselect_k():
                    for i in range(self.num_s):
                        K_[i,i] = 0
                        if np.sum(K_[:,i]) > np.sum(K_[i])*10 and i > 0:
                            return False
                        for j in range(self.num_s):
                            if K_[i,j] > 0:
                                if K_[i,j] < np.amax(K_[:,j])*0.1:
                                    return False
                    return True

                if not preselect_k():
                    continue

                # chooses sigma of IRF
                if self.add_noise_irf: 
                    sigma_irf = uniform(*self.range_irf)
                else:
                    sigma_irf = 1

                #generates the transients for given K
                self.step_c, self.step_t = generate_kinetics(self.data_t, self.K, sigma_irf, interpolate=False)

                found_solvable_kinetic = True
                for i in range(self.num_s):
                    if np.sum(K_b[i]) > 0:
                        if np.amax(self.step_c[:,i]) < 0.01:
                            found_solvable_kinetic = False

            self.plot_c = np.zeros([self.y_points, self.c_0.size])

            interpolated2 = [interp1d(self.step_t, self.step_c[:,i], 'cubic', assume_sorted=True) for i in range(self.c_0.size)]
            for j, f in enumerate(interpolated2):
                self.plot_c[:,j] = f(self.data_t)
            
            # multipy concentration traces by spectra to yield ta signal
            self.data_pure = np.zeros([self.y_points, self.x_points])
            for i in range(5):
                self.data_pure += self.data_s[i]*self.plot_c[:,i].reshape([256, 1])
            self.data_pure -= self.data_s[i]*np.full([256, 1], 1)
            self.data_noise_n = normalize(self.data_pure)

            self.data_y = np.zeros(103)
            self.data_y[class_id] = 1

            self.X[l] = self.data_noise_n.reshape([256, 64, 1])
            self.Y[l] = self.data_y
            if self.debug_mode:
                self.data_K[l] = self.K
                self.S[l] = self.data_s
                self.C[l] = self.plot_c
                self.data_irf[l] = sigma_irf

        if self.debug_mode:
            return self.X, self.Y, self.data_K, self.S, self.C, self.data_irf
        else: 
            return self.X, self.Y