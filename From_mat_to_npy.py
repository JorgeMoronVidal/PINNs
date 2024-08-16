from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np
n_sources = 32
for i in range(1,n_sources + 1):
    print(i)
    mat_fname = pjoin('DATA_FD/soldata_point_s_'+str(i)+'.mat')
    mat_contents =sio.loadmat(mat_fname)
    U_aux = mat_contents['sol']
    np.save('Input/U_'+str(i-1)+'.npy', U_aux)
