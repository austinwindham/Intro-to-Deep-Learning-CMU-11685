import numpy as np
from mytorch.nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.

        self.r = self.r_act.forward(self.Wrx @ x + self.brx + self.Wrh @ h_prev_t + self.brh)
        self.z = self.z_act.forward(self.Wzx @ x + self.bzx + self.Wzh @ h_prev_t + self.bzh)
        self.n = self.h_act.forward(self.Wnx @ x + self.bnx + self.r * (self.Wnh @ h_prev_t + self.bnh))

        h_t = (1- self.z) * self.n + self.z* h_prev_t

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,)  # h_t is the final output of you GRU cell.

        return h_t
        

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        x = self.x.reshape(-1, 1)           # (d, 1)
        h_prev = self.hidden.reshape(-1, 1) # (h, 1)
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        dz = delta * (h_prev.flatten() -self.n)
        dn = delta * (1 - self.z)
        dh_prev_from_ht = delta * self.z

        dn_pre = dn * (1 - self.n ** 2)  
        dn_pre = dn_pre.reshape(-1, 1)  

        a = self.Wnh @ h_prev + self.bnh.reshape(-1, 1)
        dr = dn_pre * a
        r_col = self.r.reshape(-1, 1)
        dr_pre = dr * r_col * (1 - r_col) 
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly

        dz_pre = dz * self.z * (1 - self.z)  # shape (h,)
        dz_pre = dz_pre.reshape(-1, 1)

        # Step 3 continued) Gradients for weights and biases
        self.dWnx += dn_pre @ x.T
        self.dWnh += (dn_pre * r_col) @ h_prev.T
        self.dWrx += dr_pre @ x.T
        self.dWrh += dr_pre @ h_prev.T
        self.dWzx += dz_pre @ x.T
        self.dWzh += dz_pre @ h_prev.T

        self.dbnx += dn_pre.flatten()
        self.dbnh += (dn_pre * r_col).flatten()
        self.dbrx += dr_pre.flatten()
        self.dbrh += dr_pre.flatten()
        self.dbzx += dz_pre.flatten()
        self.dbzh += dz_pre.flatten()

        # Step 4) Compute dx and dh_prev_t

        dx = (self.Wnx.T @ dn_pre +self.Wrx.T @ dr_pre + self.Wzx.T @ dz_pre).flatten()

        dh_prev_t = ( self.Wnh.T @ (dn_pre * r_col) + self.Wrh.T @ dr_pre + self.Wzh.T @ dz_pre + dh_prev_from_ht.reshape(-1, 1)).flatten()



        assert dx.shape == (self.d,)
        assert dh_prev_t.shape == (self.h,)

        return dx, dh_prev_t
       
