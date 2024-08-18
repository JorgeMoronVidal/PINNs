
# PINNs

*Ver 1.0 by Jorge Morón-Vidal (UC3M), Pedro González (UC3M), Juan Acebrón (UC3M) and Miguel Moscoso (UC3M)*

## Aim and overview
This code is intended to solve inverse dffussion problems 

$$\left\lbrace \begin{array}{l l} 
 \nabla D(\mathbf{x} ) \nabla u_i(\mathbf{x} ,t) = 0, & \mathbf{x} \in \Omega, t \in (0, t_f],\\
2  D(\mathbf{x} )  \dfrac{\partial u_i}{\partial \mathbf{n}} + u_i(\mathbf{x} ,t)  = 0, & \mathbf{x} \in \partial \Omega, t \in (0, t_f],\\
u_i(\mathbf{x},0) = g_i(\mathbf{x}),& x \in \overline{\Omega},\\
\end{array} \right.$$


using Physics informed neural networks in a square domain $\Omega \in \mathbb{R}^2$. In order to do so, it uses the data on $\partial \Omega$ from $N_s$ different sources which correspond to the solution of a problem with the same diffusion coefficient and different initial state $g_i(\mathbf{x} )$.

In this approach to PINNs, the solution for each source $\lbrace u_i (\mathbf{x} ,t) \rbrace_{i = 1, \ldots, N_s}$ is estimated using a $N_s$ neural networks $\lbrace u^{NN}_i (\mathbf{x} ,t) \rbrace_{i = 1, \ldots, N_s}$ such that  $$u_i (\mathbf{x} ,t)   \approx u^{NN}_i (\mathbf{x} ,t), \qquad i = 1, \ldots, N_s.$$ The layer profile of these neural networks is chosen by the variable *layers_u* in the main file. There are no limitations on the number of neurons and layers in the hidden layers but the input layer of this NNs has to be of dimension 3 and the output layer, of dimension 1.

On the other hand, the diffussion coefficient is estimated also via a NN 
$$D(\mathbf{x}) \approx D^{NN}(\mathbf{x}).$$ In this case, the layer profile in the main file is stored in  *layers_Diff*. Again ,we are free to chose any hidden layer profile as long as the input layer of this NN is dimension 2 and the output layer, dimension 1.

The set of $\lbrace u^{NN}_1 (\mathbf{x} ,t), \ldots ,  u^{NN}_{N_S} (\mathbf{x} ,t), D^{NN}(\mathbf{x})\rbrace$ is trained optimizing the loss function which has four components
$$L = L_{f} + L_{\partial \Omega} +  L_{C} + L_{O}$$

 - $L_f$ stands for the loss function contribution coming from a set of $N_f$ points belonging to $\Omega \times (0,t_f]$ and generated using Latin hypercube sampling
 $$L_f = \sum_{i = 1}^{N_S} \dfrac{1}{N_f}\sum_{j = 1}^{N_f} \left[\dfrac{\partial u^{NN}_i(\mathbf{x}_j ,t_j)}{\partial t}-D^{NN}(\mathbf{x}_j)\nabla^2 u^{NN}_i (\mathbf{x}_j ,t_j)\right.  $$$$\left.- (\nabla D^{NN}(\mathbf{x}_j))\cdot(\nabla u^{NN}_i (\mathbf{x}_j ,t_j))\right]^2.$$
 The partial derivatives are computed using the automatic differentiation functions.
 - $L_{\partial \Omega}$ takes care of the Robin boundary conditions from the BVP. Choosing a set of $N_{\partial \Omega}$ points following a Latin hypercube distribution in $\Omega \times (0,t_f]$ $$L_{\partial \Omega} =  \sum_{i = 1}^{N_S}  \dfrac{1}{N_{\partial \Omega}}\sum_{j = 1}^{N_{\partial \Omega}}\left( D^{NN}(\mathbf{x}_j)\dfrac{\partial u^{NN}_i(\mathbf{x}_j ,t_j)}{\partial n} + u^{NN}_i(\mathbf{x}_j ,t_j)\right)^2 $$
 - The third term $L_C$ is the collocation contribution. For each source, the solution of the problem is available at the initial state and on some evenly spaced points on the boundary  that emulate real live detectors. The solution on these points is collocated on the points where it is available$$L_C = \sum_{i = 1}^{N_S}  \dfrac{1}{N_{C}}\sum_{j = 1}^{N_C} \left(u_i(\mathbf{x}_j ,t_j)-u^{NN}_i(\mathbf{x}_j ,t_j)\right)^2$$
 - The last term $L_O$ is there to prevent that $D^{NN}(\mathbf{x}_j)$ takes values that have no physical meaning i.e. values that are lower than the diffusion coefficient of the medium where the  pertubations are embeded $$L_O = N_S \dfrac{1}{N_{O}}\sum_{j = 1}^{N_O} min\left(D^{NN}(\mathbf{x}_j )-1,0\right)^2$$
The factor $N_S$ has been added so the order of magnitude of $L_O$ is similar to the other contributions to the loss function.

In order to optimize the training stage, we opted to train the NNs using a succession of trainsets a each of which we change the points  where $L_f$, $L_{\partial \Omega}$ and $L_O$ are computed. Furthermore, two different optimizers (ADAM and LBFGS) are used at each trainset so the convergence of the training takes advantage of both optimizers. The learning rate of these optimizers is set to decay with each trainset whit the aim of accelerating the algorithm whithout loosing accuracy. 

## Dependencies 

The dependencies of this code are listed in pytorch_cpu.yml . In order to create the associated environment using conda one can run 

  

    conda env create -f pytorch_cpu.yml

Then, before running the main file one has to activate such environment with 

    conda activate pytorch_cpu

## Main file structure

The main file begins by importing both the PINN class and the numpy library. 

    import numpy as np  
    from PINN_Inverse import PINN

Afterwards, the layer profile of the NNs is set. For example, if we want the  $U_i^{NN}$s to have 3 hidden layers with 40 neurons each and $D^{NN}(\mathbf{x})$ to also have 3 hidden layers with 16 neurons each

	   layers_u = [3, 40, 40, 40, 1]  
	   layers_Diff = [2,16,16,16,1]

**lb**,**rb**,**tb** and **bb** are respectively the left, rigth, top and bottom boundary positions so the domain is $[lb,bb]\times[rb,tb]$ and  **tf** is the final time.  If the domain is set to be $[-2.5,2.5]^2$ and the time where the solution is computed is tf = 2.5 then

    lb = -2.5  
	rb = 2.5  	
	bb = -2.5  
	tb = 2.5  
	tf = 2.5  

Once these paremeters are set, we define the numer of the sources **n\_sources** and the  detectors in the x  **n\_x** and y **n\_y** directions.  The  3 column numpy vector   **X\_detectors** is built in a way such that the first column is the x coordinate of the detector, the second column the y coordinate ant the third, the time of the meausrement. The solution of the problem at each $[x,y,t]$ in **X\_detectors** is stored in **u\_detectors** which is a tensor which contains **n\_sources** column vectors with the same number of rows  as **X\_detectors**. The values for **u\_detectors** are provided by a set of files stored in 

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
  
	X_detectors = np.vstack([xx1, xx2, xx3, xx4, xx5])  
	n_sources = 32  
	u_detectors = []  
	for source in range(n_sources):  
	    U = np.load("Input/U_" + str(source) + ".npy")  
	    u1 = np.array([U[:, :, 0:1].flatten(), ]).T  
	    u2 = np.array([U[:, 0:1, :].flatten(), ]).T  
	    u3 = np.array([U[0:1, :, :].flatten(), ]).T  
	    u4 = np.array([U[:, -1, :].flatten(), ]).T  
	    u5 = np.array([U[-1, :, :].flatten(), ]).T  
	    u_Collocation = np.vstack([u1, u2, u3, u4, u5])  
	    u_detectors.append(u_Collocation)

Once all this variables have been defined, an instance of the PINN class -Which stores and manages all the NNs involved in the inverse problem- is created

    model = PINN(layers_u, layers_Diff, lb, rb, tb, bb, tf, n_sources)

Then the training of the NNs is done in the line

    model.train(N_boundary, N_f, X_detectors, u_detectors, X_star, N_trainsets, epochs_ADAM, epochs_LBFGS)

 - **N\_boundary** is the number of points where Robin boundary conditions are checked in $L_{\partial \Omega}$. Equivalent to $N_{\partial \Omega}$ from the previous section.
 - **N_f** is the number of points inside the domain where $L_f$ is cheked. 
 - **X_star** are the points where $D^{NN}(\mathbf{x})$ is  plotted. 
 - **N_trainsets** is the number of different trainsets that are used in the training process. 
 - **epochs_ADAM** indicates the epochs in training that are done with the ADAM optimizer.
 - **epochs_LBFGS** is the number of epochs in training that are done with the LBFGS optimizer.
 
 Once the model is trained, $D^{NN}(\mathbf{x})$ and $U^{NN}_0(\mathbf{x},t)$ are evaluated, plotted and saved 

    U_pred, F_pred, Diff_coeff = model.predict(X_star)  
	np.save("Output/U_0_pred.npy",U_pred)  
	np.save("Output/F_0_pred.npy",F_pred)  
	np.save("Output/Diff_coeff_pred.npy",Diff_coeff)

 **INPUTS**
 
 The inputs required by this code are stored in the **Input folder**. There, the values of the solution for each source labeled by the letter i is stored in a file called **U_i.npy**. For testing porpuses, these files can be created via the finate differences MATLAB program **crea_datos_old.m** which can be found in the folder **DATA_FD**. Then, calling the script **From_mat_to_npy.py** the files produced by the MATLAB code are translated to npy format and stored in the Input folder.

  **OUTPUTS**

Two folders store the outputs of this code: The **Plots** folder for the plots that are produced during the training and the **Output** folder for the numerical files.  To supervise the training process, each 250 epochs the evaluation of $D^{NN}(\mathbf{x})$ is plotted in a regular grid to check the evolution of the algorithm.



 
 
