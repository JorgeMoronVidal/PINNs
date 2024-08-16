
# PINNs

*Ver 1.0 by Jorge Morón-Vidal (UC3M), Pedro González (UC3M), Juan Acebrón (UC3M) y Miguel Moscoso (UC3M)*

## Aim and overview
This code is intended to solve inverse difussion problems 

$$\left\lbrace \begin{array}{c c} 
 \nabla D(\mathbf{x} ) \nabla u(\mathbf{x} ,t) = 0, & \mathbf{x} \in \Omega, t \in (0, t_f]\\
2  D(\mathbf{x} )  \dfrac{\partial u}{\partial \mathbf{n}} + u(\mathbf{x} ,t)  = 0, & \mathbf{x} \in \partial \Omega, t \in (0, t_f]\\
\end{array} \right.$$


using Physics informed neural networks in a square domain $\Omega \in \mathbb{R}^2$. In order to do so, it uses the data on $\partial \Omega$ from $N_s$ different sources which correspond to the solution of a problem with the same diffusion coefficient and different initial state $u(\mathbf{x} ,0)$.

In this approach to PINNs, the solution for each source $\lbrace u_i (\mathbf{x} ,t) \rbrace_{i = 1, \ldots, N_s}$ is estimated using a $N_s$ neural networks $\lbrace u^{NN}_i (\mathbf{x} ,t) \rbrace_{i = 1, \ldots, N_s}$ such that  $$u_i (\mathbf{x} ,t)   \approx u^{NN}_i (\mathbf{x} ,t), \qquad i = 1, \ldots, N_s.$$ The layer profile of these neural networks is chosen by the variable *layers_u* in the main file. There are no limitations on the number of neurons and layers in the hidden layers but the input layer of this NNs has to be of dimension 3 and the output layer, of dimension 1.

On the other hand, the diffussion coefficient is estimated also via a NN 
$$D(\mathbf{x}) \approx D^{NN}(\mathbf{x}).$$ In this case, the layer profile in the main file is stored in  *layers_Diff*. Again ,we are free to chose any hidden layer profile as long as the input layer of this NN is dimension 2 and the output layer, dimension 1.

The set of $\lbrace u^{NN}_1 (\mathbf{x} ,t), \ldots ,  u^{NN}_{N_S} (\mathbf{x} ,t), D^{NN}(\mathbf{x})\rbrace$ is trained optimizing the loss function which has four components
$$L = L_{f} + L_{\partial \Omega} +  L_{C} + L_{O}$$

 - $L_f$ stands for the loss function contribution coming from a set of $N_f$ points belongin to $\Omega \times (0,t_f]$ and generated using Latin hypercube sampling
 $$L_f = \sum_{i = 1}^{N_S} \dfrac{1}{N_f}\sum_{j = 1}^{N_f} \left[\dfrac{\partial u^{NN}_i(\mathbf{x}_j ,t_j)}{\partial t}-D^{NN}(\mathbf{x}_j)\nabla^2 u^{NN}_i (\mathbf{x}_j ,t_j)\right.  $$$$\left.- (\nabla D^{NN}(\mathbf{x}_j))\cdot(\nabla u^{NN}_i (\mathbf{x}_j ,t_j))\right]^2.$$
 The partial derivatives are computed using the automatic differentiation functions.
 - $L_{\partial \Omega}$ takes care of the Robin boundary conditions from the BVP. Choosing a set of $N_{\partial \Omega}$ points following a Latin hypercube distribution in $\Omega \times (0,t_f]$ $$L_{\partial \Omega} =  \sum_{i = 1}^{N_S}  \dfrac{1}{N_{\partial \Omega}}\sum_{j = 1}^{N_{\partial \Omega}}\left( D^{NN}(\mathbf{x}_j)\dfrac{\partial u^{NN}_i(\mathbf{x}_j ,t_j)}{\partial n} + u^{NN}_i(\mathbf{x}_j ,t_j)\right)^2 $$
 - The third term $L_C$ is the collocation contribution. For each source, the solution of the problem is available at the initial state and on some evenly spaced points on the boundary  that emulate real live detectors. The solution on these points is collocated on the points where it is available$$L_C = \sum_{i = 1}^{N_S}  \dfrac{1}{N_{C}}\sum_{j = 1}^{N_C} \left(u_i(\mathbf{x}_j ,t_j)-u^{NN}_i(\mathbf{x}_j ,t_j)\right)^2$$
 - The last term $L_O$ is there to prevent that $D^{NN}(\mathbf{x}_j)$ takes values that have no physical meaning i.e. values that are lower than the diffusion coefficient of the medium where the  pertubations are embeded $$L_O = N_S \dfrac{1}{N_{O}}\sum_{j = 1}^{N_O} min\left(D^{NN}(\mathbf{x}_j )-1,0\right)^2$$
The factor $N_S$ has been added so the order of magnitude of $L_O$ is similar to the other contributions to the loss function.

In order to optimize the training stage, we opted to train the NNs using a succession of trainsets a each of which we change the points  where $L_f$, $L_{\partial \Omega}$ and $L_O$ are computed. Furthermore, two different optimizers (ADAM and LBFGS) are used at each trainset so the convergence of the training takes advantage of both optimizers. The learning rate of these optimizers is set to decay with each trainset whit the aim of accelerating the algorithm whithout loosing accuracy. 

