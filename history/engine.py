import numpy as np
from nmrmath import nspinspec
from scipy.sparse import lil_matrix
from nmrplot import nmrplot as nmrplt

freq_list = np.array([105, 140, 180, 205])
nspins = len(freq_list)

# Store spins in an nspin x nspin array
scalar_couplings = lil_matrix((nspins, nspins))

# For nuclei 0, 1, and 2:
# J01 = 15, J02 = 11, J12 = 3
scalar_couplings[0, 1] = -12
scalar_couplings[0, 2] = 6
scalar_couplings[0, 3] = 8
scalar_couplings[1, 2] = 3
scalar_couplings[1, 3] = 3
scalar_couplings[2, 3] = 0
scalar_couplings = scalar_couplings + scalar_couplings.T
print(freq_list)
print(scalar_couplings)
spectrum = nspinspec(freq_list, scalar_couplings)
spectrum.sort()
print('New spectrum: ', spectrum)
l_limit = min(freq_list) - 50
r_limit = max(freq_list) + 50

nmrplt(spectrum)
