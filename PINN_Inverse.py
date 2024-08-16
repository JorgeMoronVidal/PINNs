import torch
from DNN_pytorch import DNN
from pyDOE import lhs
from itertools import chain
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
class PINN:
    def __init__(self, layers_u, layers_D, lb, rb, tb, bb, ft, n_sources):

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # boundary conditions
        self.left_boundary = lb
        self.right_boundary = rb
        self.top_boundary = tb
        self.bottom_boundary = bb
        self.final_time = ft
        # data

        self.layers_u = layers_u

        # deep neural networks
        self.dnns = []
        for source_index in range(n_sources):
            self.dnns.append(DNN(layers_u).to(self.device))
        self.dnn_D = DNN(layers_D).to(self.device)
        #optimizers: using the same settings
        self.epoch = 0
        parameters_aux = chain([])
        for dnn in self.dnns:
            parameters_aux = chain(parameters_aux, dnn.parameters())
        parameters_aux = chain(parameters_aux, self.dnn_D.parameters())
        self.optimizer = torch.optim.Adam(parameters_aux, lr=0.002, betas=(0.9, 0.999), eps=1e-08,
                                              weight_decay=0, amsgrad=False)
        self.source_index = 0

    def diff_coeff(self, x, y):
        return self.dnn_D(torch.cat([x, y], dim=1))

    def net_u(self, x, y, t):
        u = self.dnns[self.source_index](torch.cat([x, y, t], dim=1))
        return u

    def net_robin_bc(self, x, y, t):
        u = self.dnns[self.source_index](torch.cat([x, y, t], dim=1))
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_y = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        ones = torch.ones(x.size())
        bool_top = torch.lt((y-ones*self.top_boundary).abs(),1e-05).int()
        bool_bottom = torch.lt((y - ones * self.bottom_boundary).abs(), 1e-05).int()
        bool_left = torch.lt((x - ones * self.left_boundary).abs(), 1e-05).int()
        bool_right = torch.lt((x - ones * self.right_boundary).abs(), 1e-05).int()
        dudn = (+bool_top*u_y - bool_bottom * u_y - bool_left*u_x + bool_right*u_x)/np.sqrt(2.)
        return 2*self.diff_coeff(x,y)*dudn + u

    def net_collocation_bc(self, x, y, t):
        u = self.dnns[self.source_index](torch.cat([x, y, t], dim=1))
        return self.u_collocation[self.source_index] - u

    def net_f(self, x, y, t):
        """ The pytorch autograd version of calculating residual """
        u = self.net_u(x, y, t)
        diff_coeff = self.diff_coeff(x,y)
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_y = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        diff_coeff_x = torch.autograd.grad(
            diff_coeff, x,
            grad_outputs=torch.ones_like(diff_coeff),
            retain_graph=True,
            create_graph=True
        )[0]
        diff_coeff_y = torch.autograd.grad(
            diff_coeff, y,
            grad_outputs=torch.ones_like(diff_coeff),
            retain_graph=True,
            create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]
        u_yy = torch.autograd.grad(
            u_y, y,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True
        )[0]
        f = u_t - diff_coeff * (u_yy + u_xx) - diff_coeff_x * u_x - diff_coeff_y * u_y
        return f

    def loss_func(self):
        self.optimizer.zero_grad()
        loss_colloc = 0.
        loss_robin = 0.
        loss_inner = 0.
        diff_coeff = self.diff_coeff(self.x_f, self.y_f)
        outliers = torch.lt(diff_coeff, 1.).int()
        loss_outliers = torch.mean((outliers * (diff_coeff - 1.)) ** 2) * len(self.u_collocation)
        for self.source_index in range(len(self.u_collocation)):
            colloc_bc = self.net_collocation_bc(self.x_collocation, self.y_collocation, self.t_collocation)
            robin_bc = self.net_robin_bc(self.x_Robin, self.y_Robin, self.t_Robin)
            f_pred = self.net_f(self.x_f, self.y_f, self.t_f)
            # loss_boundary = torch.mean(torch.cat((robin_bc, colloc_bc)) ** 2)
            loss_colloc += torch.mean(colloc_bc ** 2)
            loss_robin += torch.mean(robin_bc ** 2)
            loss_inner += torch.mean(f_pred ** 2)

        loss = (loss_colloc + loss_robin + loss_inner + loss_outliers)
        if self.epoch % 250 == 0:
            print(
                '%d, %.5e, %.5e. %.5e, %.5e, %.5e' % (
                    self.epoch, loss.item(), loss_outliers.item(), loss_colloc.item(), loss_robin.item(),
                    loss_inner.item())
            )
            diff_coeff = self.diff_coeff(self.x_eval, self.y_eval)
            fig_0 = plt.figure(figsize=(9, 9))
            ax_0 = fig_0.add_subplot(111)
            diff_coeff = diff_coeff.reshape(51, 51)
            diff_coeff = diff_coeff.detach().cpu().numpy()
            h_0 = ax_0.imshow(diff_coeff, interpolation='nearest', cmap='rainbow',
                              origin='lower', aspect='auto')
            divider_0 = make_axes_locatable(ax_0)
            cax_0 = divider_0.append_axes("right", size="5%", pad=0.10)
            cbar_0 = fig_0.colorbar(h_0, cax=cax_0)
            cbar_0.ax.tick_params(labelsize=15)
            plt.savefig("Plots/Diff_cent_" + str(self.plot_index))
            plt.close(fig_0)
            self.plot_index += 1
        self.epoch += 1
        loss.backward()
        return loss

    def update_training_sets(self, N_Robin, N_f, X_collocation, u_collocation_array):
        X_f_train = lhs(3, N_f)
        X_f_train[:, 0] = self.left_boundary + (self.right_boundary - self.left_boundary) * X_f_train[:, 0]
        X_f_train[:, 1] = self.bottom_boundary + (self.top_boundary - self.bottom_boundary) * X_f_train[:, 1]
        X_f_train[:, 2] = X_f_train[:, 2]*self.final_time
        X_Robin = lhs(3, N_Robin)
        X_Robin[:, 0] = self.left_boundary + (self.right_boundary - self.left_boundary) * X_Robin[:, 0]
        X_Robin[:, 1] = self.bottom_boundary + (self.top_boundary - self.bottom_boundary) * X_Robin[:, 1]
        X_Robin[:, 2] = X_Robin[:, 2] * self.final_time
        step = int(N_Robin/4)
        X_Robin [0:step, 0] = self.left_boundary
        X_Robin[step:2*step, 0] = self.right_boundary
        X_Robin[2*step:3*step, 1] = self.top_boundary
        X_Robin[3*step:-1, 1] = self.bottom_boundary
        X_u_collocation = X_collocation
        self.x_Robin = torch.tensor(X_Robin[:, 0:1], requires_grad=True).float().to(self.device)
        self.y_Robin = torch.tensor(X_Robin[:, 1:2], requires_grad=True).float().to(self.device)
        self.t_Robin = torch.tensor(X_Robin[:, 2:3], requires_grad=True).float().to(self.device)
        self.x_collocation = torch.tensor(X_u_collocation[:, 0:1], requires_grad=True).float().to(self.device)
        self.y_collocation = torch.tensor(X_u_collocation[:, 1:2], requires_grad=True).float().to(self.device)
        self.t_collocation = torch.tensor(X_u_collocation[:, 2:3], requires_grad=True).float().to(self.device)
        self.x_f = torch.tensor(X_f_train[:, 0:1], requires_grad=True).float().to(self.device)
        self.y_f = torch.tensor(X_f_train[:, 1:2], requires_grad=True).float().to(self.device)
        self.t_f = torch.tensor(X_f_train[:, 2:3], requires_grad=True).float().to(self.device)
        self.u_collocation = []
        for u_collocation in u_collocation_array :
            #u_collocation_aux = u_collocation[idx_collocation, :]
            u_collocation_aux = u_collocation
            self.u_collocation.append(torch.tensor(u_collocation_aux).float().to(self.device))

    def train(self, N_Robin, N_f, X_collocation, u_collocation, X_Eval, trainsets, epochs_ADAM, epochs_LBFGS):
        self.x_eval = torch.tensor(np.array([X_Eval[:,0:1].reshape(51,51,251)[:,:,0].flatten(),]).T,requires_grad = True).float().to(self.device)
        self.y_eval = torch.tensor(np.array([X_Eval[:,1:2].reshape(51,51,251)[:,:,0].flatten(),]).T, requires_grad = True).float().to(self.device)
        for dnn in self.dnns:
            dnn.train()
        self.dnn_D.train()
        self.plot_index = 0
        for trainset in range(trainsets):
            parameters_aux = chain([])
            for dnn in self.dnns:
                parameters_aux = chain(parameters_aux, dnn.parameters())
            parameters_aux = chain(parameters_aux, self.dnn_D.parameters())
            self.update_training_sets(N_Robin, N_f, X_collocation, u_collocation)

            self.optimizer = torch.optim.Adam(parameters_aux, lr=0.02/(1+trainset), betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=0, amsgrad=False)
            self.epoch = 0
            self.max_epochs = epochs_ADAM
            for i in range(self.max_epochs):
                self.loss_func() # backprop
                self.optimizer.step()  # zeroes the gradient buffers of all parameters

            for dnn in self.dnns:
                parameters_aux = chain(parameters_aux, dnn.parameters())
            parameters_aux = chain(parameters_aux, self.dnn_D.parameters())
            self.epoch = 0
            self.optimizer = torch.optim.LBFGS(
                    parameters_aux,
                    lr=0.75+1.0/(1+trainset),
                    max_iter=epochs_LBFGS,
                    max_eval=epochs_LBFGS,
                    history_size=100,
                    tolerance_grad=1e-7,
                    tolerance_change=1.0 * np.finfo(float).eps,
                    line_search_fn="strong_wolfe"  # can be "strong_wolfe"
                )
            self.optimizer.step(self.loss_func)
            self.dnn_D.eval()

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        y = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 2:3], requires_grad=True).float().to(self.device)

        self.dnn_D.eval()
        self.source_index = 0
        u = self.net_u(x, y, t)
        f = self.net_f(x, y, t)
        diff_coeff = self.diff_coeff(x, y)
        diff_coeff = diff_coeff.detach().cpu().numpy()
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return u, f, diff_coeff