import numpy as np
from PINN_Inverse import PINN

layers_u = [3, 40, 40, 40, 1]
layers_Diff = [2,16,16,16,1]
lb = -2.5
rb = 2.5
bb = -2.5
tb = 2.5
tf = 2.5
nx = 51
ny = 51

t = np.arange(0, tf + 0.01, 0.01)
x = np.arange(lb, rb + 0.1, 0.1)
y = np.arange(bb, tb + 0.1, 0.1)

X, Y, T = np.meshgrid(x, y, t)

X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None], T.flatten()[:, None]))

xx1 = np.hstack((np.array([X[:, :, 0:1].flatten(), ]).T, np.array([Y[:, :, 0:1].flatten(), ]).T,
                 np.array([T[:, :, 0:1].flatten(), ]).T))
xx2 = np.hstack((np.array([X[:, 0:1, :].flatten(), ]).T, np.array([Y[:, 0:1, :].flatten(), ]).T,
                np.array([T[:, 0:1, :].flatten(), ]).T))
xx3 = np.hstack((np.array([X[0:1, :, :].flatten(), ]).T, np.array([Y[0:1, :, :].flatten(), ]).T,
                np.array([T[0:1, :, :].flatten(), ]).T))
xx4 = np.hstack((np.array([X[:, -1:, :].flatten(), ]).T, np.array([Y[:, -1:, :].flatten(), ]).T,
                np.array([T[:, -1:, :].flatten(), ]).T))
xx5 = np.hstack((np.array([X[-1:, :, :].flatten(), ]).T, np.array([Y[-1:, :, :].flatten(), ]).T,
                np.array([T[-1:, :, :].flatten(), ]).T))

X_Collocation = np.vstack([xx1, xx2, xx3, xx4, xx5])

u_sources = []
for source in range(16):
    U = np.load("Input/U_" + str(source) + ".npy")
    u1 = np.array([U[:, :, 0:1].flatten(), ]).T
    u2 = np.array([U[:, 0:1, :].flatten(), ]).T
    u3 = np.array([U[0:1, :, :].flatten(), ]).T
    u4 = np.array([U[:, -1, :].flatten(), ]).T
    u5 = np.array([U[-1, :, :].flatten(), ]).T
    u_Collocation = np.vstack([u1, u2, u3, u4, u5])
    u_sources.append(u_Collocation)

model = PINN(layers_u, layers_Diff, lb, rb, tb, bb, tf, 16)
model.train(5000, 20000, X_Collocation, u_sources, X_star, 10, 2500, 2500)
U_pred, F_pred, Diff_coeff = model.predict(X_star)
np.save("Output/U_0_pred.npy",U_pred)
np.save("Output/F_0_pred.npy",F_pred)
np.save("Output/Diff_coeff_pred.npy",Diff_coeff)