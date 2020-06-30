import numpy as np

from scipy.optimize import minimize, least_squares
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy.integrate import RK45
from scipy.signal import convolve

from itertools import permutations

from .utils import gauss, data_dy, generate_binary_matrix, generate_kinetics
from .utils import uniform
from .generate_viable_ids import unique_ids

class GTAnalyzer():
    '''
    Analyzer for classical Global and Target Analysis
    has to be initialized with a 2d-dataset and the corresponding time scale
    '''
    def __init__(self, x, t):
        self.x_true = x.reshape([256,64])
        self.t = t

    def c_irf(self, t, k, u, D):
        '''
        returns the analytic solution for a single independent decay with a 
        gaussian irf
        '''
        return 0.5*np.exp(-k*t)*np.exp(
            k*(u+((k*D**2)/2)))*(1+erf((t-(u+k*D**2))/(np.sqrt(2)*D))
            )

    def generate_fit(self, p):
        '''
        Generates the transients of n independent decays 
        from the given parameters (*decays, irf, time_zero).
        Calculates the corresponding DADS and returns the fitted 
        2-dimensional data.
        '''
        decays = p[:self.num_decays]
        irf = p[self.num_decays]
        tz = p[self.num_decays+1]

        if self.iterations%10 == 0:
            print((decays*1000).astype(int)/1000, int(irf), int(tz)/1e6, self.iterations)

        self.c_fit = np.zeros([256, self.num_spectra])
        self.dads = np.zeros([self.num_spectra, 64])
        self.x_fit = np.zeros([256, 64])

        for i in range(self.num_decays):
            self.c_fit[:,i] = self.c_irf(self.t, 1/(decays[i]*1000), 0*tz, irf)
        if self.offset:
            self.c_fit[:,-1] = self.c_irf(self.t, 0, 0*tz, irf)

        self.dads = np.dot(np.linalg.pinv(np.nan_to_num(self.c_fit)), self.x_true)

        for i in range(self.num_spectra):
            self.x_fit += self.dads[i]*self.c_fit[:,i].reshape([256, 1])

        self.iterations += 1
        return self.x_fit

    def get_ga(self, decays_start, irf_start=0, tz_start=0, offset=False, 
               max_fev=200):
        '''
        Performs a Global Analysis with the given starting parameters
        returns the fit object of the LMfit optimizer
        '''
        self.num_decays = len(decays_start)
        self.num_spectra = self.num_decays+int(offset)
        self.offset = offset
        
        print('------> Starting Global analysis with %i components <---------'%(self.num_decays))

        d_start = np.array(decays_start)

        p_start = np.array([*d_start.flatten(), 
                                 irf_start, 
                                 tz_start])

        self.x_fit = np.zeros([256, 64])

        self.iterations = 0

        self.err = lambda p: (self.generate_fit(p).flatten()-self.x_true.flatten())
        self.diag = np.array([*[1e-9 for x in decays_start], 1e-9, 1e-9])
        fit = least_squares(self.err, p_start, method='lm', verbose=2, 
        gtol=1e-15, ftol=1e-15, xtol=1e-15, x_scale=self.diag, max_nfev=max_fev)
        print('------> Fit finished after %i function calls <---------'%(self.iterations))
        return fit

    def get_best_ga(self, decay_min, decay_max, num_decays, irf_start=0, 
                    tz_start=0, offset=False, num_tries=10):
        '''
        tries global analysis with different starting conditions, takes the ones
        with the least residual
        '''
        data_decays = np.zeros([num_tries, num_decays])
        data_residual = np.zeros([num_tries])
        for i in range(num_tries):
            decays = np.random.uniform(decay_min, decay_max, num_decays)
            fit = self.get_ga(decays, irf_start, tz_start, offset)
            data_decays[i] = decays
            data_residual[i] = fit.cost
        i_best = np.argmin(data_residual)
        return self.get_ga(data_decays[i_best], irf_start, tz_start, offset)

    
    def get_ta(self, class_id, decays, irf):
        '''
        Performs target analysis
        Kinetic matrix gets calculated by producing binary matrix from kinetic id
        and filling in the time constants from global analysis 
        '''
        K_b = generate_binary_matrix(unique_ids[class_id], 5)
        K = np.zeros([5,5])
        j = 0
        for i in range(5):
            num_p = -K_b[i,i]
            if num_p > 0:
                for k in range(5):
                    if K_b[k,i] > 0:
                        K[k,i] = 1/(K_b[k,i]*decays[j]*1000)/num_p
                K[i, i] = -1/(decays[j]*1000)
                j += 1
        data_c = generate_kinetics(self.t, K, irf)
        data_c[:,-1] = data_c[:,-1]*0

        data_s = np.dot(np.linalg.pinv(data_c), self.x_true)
        return data_c, data_s, K

    def get_best_ta(self, class_id, decays, irf):
        '''
        performs target analysis with all permutations of the decays from GA
        and takes the one where the amplitudes of the spectra are lowest.
        '''
        data_p = []
        data_err = []
        for dec in permutations(decays):
            data_c, data_s, _ = self.get_ta(class_id, dec, irf)

            x_ta = np.zeros([256, 64])
            for i in range(5):
                x_ta += data_s[i]*data_c[:,i].reshape([256, 1])
            x_ta -= data_s[i]*np.full([256, 1], 1)

            err = np.sum([x**2 for x in np.amax(data_s, axis=0) if x!=0])
            data_p.append(dec)
            data_err.append(err)

        i_best = np.argmin(data_err)
        print(data_p[i_best])
        return self.get_ta(class_id, data_p[i_best], irf)