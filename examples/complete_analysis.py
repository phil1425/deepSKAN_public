# deepSKAN
# This is another example implementation wich perfroms a complete analyis of a
# given Transient absorption dataset. 
# The kinetic model is predicted using an artificial neural network and from 
# that classical Global- and Target Analysis is performed. Ideally, no human
# input is needed for this process.
#
# In this example, the dataset is generated on the spot by the same algorithm 
# that was used to generate training data for the network.

import numpy as np
import matplotlib.pyplot as plt

from deepGTA import GTAnalyzer, DLAnalyzer, onlineDataGenerator

# This is special to my machine, not needed else
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


save_csv = False
true_class_id = 38

# initialize generator with model class id 38
gen = onlineDataGenerator(1, 1, class_id=true_class_id, debug_mode=True)

# Generate: Image, Class, Matrix, Spectra, Concentrations, IRF
X, Y, K, S, C, I = gen.__getitem__(0)

# Initialize Deep Learning and Global/Target Analyzers
dl_model = DLAnalyzer(model_path='RES-33L-15M-103C-5S-6-48E-53A')
gta_model = GTAnalyzer(X, np.logspace(2, 6, 256))

# Predict teh kinetic model
P = dl_model.predict(X)
dl_model.draw_prediction(P, path='test.pdf', show=False)
class_id = np.argmax(P)
num_k, offset = dl_model.get_num_decays(P)

# Perform global and target analysis with predicted model
fit = gta_model.get_best_ga(1, 1000, num_k, 10, 1, offset, 30)
fit_decays = fit.x[:num_k]
fit_irf = fit.x[num_k]
plot_c, sads, K_ta = gta_model.get_best_ta(class_id, fit_decays, fit_irf)

# Some output
print(' ')
print('predicted class id: ', np.argmax(P))
print('Decays (global analysis):', fit_decays, fit_irf)
print('Decays (target analysis):',  
      [-1/K_ta[i,i]/1000 for i in range(5) if K_ta[i,i] != 0], fit_irf)
print('Decays (true):', 
      [-1/K[0,i,i]/1000 for i in range(5) if K[0,i,i] != 0], I[0])

# calculate residual
x_ta = np.zeros([256, 64])
for i in range(5):
    x_ta += sads[i]*plot_c[:,i].reshape([256, 1])
x_ta -= sads[i]*np.full([256, 1], 1)

# Save data as csv
if save_csv: 
    print('saving data...')
    np.savetxt('evaluation/ga_benchmark/transients_true.csv', C[0])
    np.savetxt('evaluation/ga_benchmark/spectra_true.csv', S[0].T)
    np.savetxt('evaluation/ga_benchmark/matrix_true.csv', K[0])

    np.savetxt('evaluation/ga_benchmark/transients_gta.csv', plot_c)
    np.savetxt('evaluation/ga_benchmark/spectra_gta.csv', sads.T)
    np.savetxt('evaluation/ga_benchmark/matrix_gta.csv', K_ta)

    data = [fit_irf, I[0], P[class_id]]
    np.savetxt('evaluation/ga_benchmark/info.csv', data)

# Normalize fit SADS
for s in sads:
    if np.sum(s) != 0:
        s /= np.amax(np.abs(s))

# generate true SADS from SAS by removing ground state spectrum abd normalize
for s in S[0]:
    s -= S[0,-1]
    if np.sum(s) != 0:
        s /= np.amax(np.abs(s))

# Plot:
plt.figure(figsize=(15, 7))

# The Time-Resolved Spectrum
plt.subplots_adjust(hspace=0.3)
plt.subplot(332)
plt.title('data')
plt.contourf(X[0,:,:,0].T, levels=64)

# Residuals of global analysis
plt.subplot(331)
plt.title('GA residuals')
plt.contourf(gta_model.err(fit.x).reshape([256,64]).T, levels=128)
plt.colorbar()

# Residuals of target analysis
plt.subplot(334)
plt.title('TA residuals')
plt.contourf((x_ta-X[0,:,:,0]).T, levels=64)
plt.colorbar()

# Concentration traces of Target analysis
plt.subplot(335)
plt.title('TA transients')
plt.plot(plot_c[:,:-1])

# Resulting SADS of target analysis
plt.subplot(336)
plt.title('TA SADS')
plt.plot(sads.T)

# True concentration traces (used to generate the data)
plt.subplot(338)
plt.title('true transients')
plt.plot(C[0,:,:-1])

# True SADS (used to generate the data)
plt.subplot(339)
plt.title('true SADS')
plt.plot(S[0].T)

# Difference between the fitted SADS and the true SADS
plt.subplot(333)
plt.title('SADS residuals')
plt.plot(sads.T-S[0].T)
plt.show()


