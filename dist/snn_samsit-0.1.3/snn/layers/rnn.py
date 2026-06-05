import numpy as np
from .base import Layer
from ..activations import get as get_activation
from ..initializers import get as get_initializer


class SimpleRNN(Layer):
    """
    Vanilla (Elman) Recurrent Neural Network cell.

    Parameters
    ----------
    units : int
        Number of hidden units.
    activation : str or Activation
        Activation applied to the hidden state (default: 'tanh').
    return_sequences : bool
        If True, return output at every time step; else only the last.
    return_state : bool
        If True, return (output, last_hidden_state) tuple.
    kernel_initializer : str or callable
    recurrent_initializer : str or callable
    bias_initializer : str or callable
    """

    def __init__(self, units, activation="tanh", return_sequences=False,
                 return_state=False,
                 kernel_initializer="glorot_uniform",
                 recurrent_initializer="orthogonal",
                 bias_initializer="zeros",
                 name=None):
        super().__init__(name=name)
        self.units = units
        self.activation = get_activation(activation)
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.recurrent_initializer = get_initializer(recurrent_initializer)
        self.bias_initializer = get_initializer(bias_initializer)

        self.Wx = None
        self.Wh = None
        self.b = None

    def build(self, input_shape):
        _, _, input_dim = input_shape
        self.Wx = self.kernel_initializer((input_dim, self.units))
        self.Wh = self._orthogonal((self.units, self.units))
        self.b = self.bias_initializer((1, self.units))
        self._built = True

    @staticmethod
    def _orthogonal(shape):
        flat = np.random.randn(shape[0], shape[1])
        U, _, Vt = np.linalg.svd(flat, full_matrices=False)
        return (U if U.shape == shape else Vt).astype(np.float64)

    def forward(self, x, training=False):
        if not self._built:
            self.build(x.shape)
        n, T, _ = x.shape
        h = np.zeros((n, self.units))
        self._inputs = x
        self._hiddens = [h]
        self._pre_acts = []

        for t in range(T):
            z = x[:, t, :] @ self.Wx + h @ self.Wh + self.b
            self._pre_acts.append(z)
            h = self.activation.forward(z)
            self._hiddens.append(h)

        self._all_h = np.stack(self._hiddens[1:], axis=1)

        if self.return_sequences:
            out = self._all_h
        else:
            out = h

        if self.return_state:
            return out, h
        return out

    def backward(self, grad):
        n, T, _ = self._inputs.shape

        if self.return_sequences:
            dh_seq = grad
        else:
            dh_seq = np.zeros((n, T, self.units))
            dh_seq[:, -1, :] = grad

        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db = np.zeros_like(self.b)
        dx = np.zeros_like(self._inputs)
        dh_next = np.zeros((n, self.units))

        for t in reversed(range(T)):
            dh = dh_seq[:, t, :] + dh_next

            act_clone = type(self.activation)()
            act_clone._out = self.activation._out if hasattr(self.activation, '_out') else None

            z = self._pre_acts[t]
            if hasattr(self.activation, '_out'):
                h_t = self._hiddens[t + 1]
                dz = dh * (1.0 - h_t ** 2)
            else:
                dz = dh

            dWx += self._inputs[:, t, :].T @ dz / n
            dWh += self._hiddens[t].T @ dz / n
            db += np.sum(dz, axis=0, keepdims=True) / n
            dx[:, t, :] = dz @ self.Wx.T
            dh_next = dz @ self.Wh.T

        self._dWx = dWx
        self._dWh = dWh
        self._db = db
        return dx

    @property
    def params(self):
        return {"Wx": self.Wx, "Wh": self.Wh, "b": self.b}

    @property
    def grads(self):
        return {"Wx": self._dWx, "Wh": self._dWh, "b": self._db}


class LSTM(Layer):
    """
    Long Short-Term Memory (LSTM) layer.

    Implements the standard LSTM equations:
        f_t = sigmoid(x_t @ Wf + h_{t-1} @ Uf + bf)
        i_t = sigmoid(x_t @ Wi + h_{t-1} @ Ui + bi)
        g_t = tanh(x_t @ Wg + h_{t-1} @ Ug + bg)
        o_t = sigmoid(x_t @ Wo + h_{t-1} @ Uo + bo)
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)

    Parameters
    ----------
    units : int
    return_sequences : bool
    return_state : bool
    """

    def __init__(self, units, return_sequences=False, return_state=False,
                 kernel_initializer="glorot_uniform",
                 recurrent_initializer="orthogonal",
                 bias_initializer="zeros",
                 name=None):
        super().__init__(name=name)
        self.units = units
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)

        self.W = None
        self.U = None
        self.b = None

    @staticmethod
    def _orthogonal(shape):
        flat = np.random.randn(shape[0], shape[1])
        U, _, Vt = np.linalg.svd(flat, full_matrices=False)
        return (U if U.shape == shape else Vt).astype(np.float64)

    def build(self, input_shape):
        _, _, d = input_shape
        u = self.units
        self.W = self.kernel_initializer((d, 4 * u))
        self.U = self._orthogonal((u, 4 * u))
        self.b = self.bias_initializer((1, 4 * u))
        self._built = True

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x, training=False):
        if not self._built:
            self.build(x.shape)
        n, T, _ = x.shape
        u = self.units
        h = np.zeros((n, u))
        c = np.zeros((n, u))

        self._x = x
        self._h_list = [h]
        self._c_list = [c]
        self._gates = []

        for t in range(T):
            z = x[:, t, :] @ self.W + h @ self.U + self.b
            f, i, g, o = (self._sigmoid(z[:, :u]),
                          self._sigmoid(z[:, u:2*u]),
                          np.tanh(z[:, 2*u:3*u]),
                          self._sigmoid(z[:, 3*u:]))
            c_new = f * c + i * g
            h_new = o * np.tanh(c_new)
            self._gates.append((f, i, g, o, c, c_new))
            c = c_new
            h = h_new
            self._h_list.append(h)
            self._c_list.append(c)

        self._all_h = np.stack(self._h_list[1:], axis=1)
        out = self._all_h if self.return_sequences else h
        if self.return_state:
            return out, h, c
        return out

    def backward(self, grad):
        n, T, _ = self._x.shape
        u = self.units

        if self.return_sequences:
            dh_seq = grad
        else:
            dh_seq = np.zeros((n, T, u))
            dh_seq[:, -1, :] = grad

        dW = np.zeros_like(self.W)
        dU = np.zeros_like(self.U)
        db = np.zeros_like(self.b)
        dx = np.zeros_like(self._x)
        dh_next = np.zeros((n, u))
        dc_next = np.zeros((n, u))

        for t in reversed(range(T)):
            f, i, g, o, c_prev, c_t = self._gates[t]
            h_prev = self._h_list[t]

            dh = dh_seq[:, t, :] + dh_next
            tanh_ct = np.tanh(c_t)
            dc = dh * o * (1.0 - tanh_ct ** 2) + dc_next

            df = dc * c_prev
            di = dc * g
            dg = dc * i
            do = dh * tanh_ct

            df_raw = df * f * (1.0 - f)
            di_raw = di * i * (1.0 - i)
            dg_raw = dg * (1.0 - g ** 2)
            do_raw = do * o * (1.0 - o)

            dz = np.concatenate([df_raw, di_raw, dg_raw, do_raw], axis=-1)

            dW += self._x[:, t, :].T @ dz / n
            dU += h_prev.T @ dz / n
            db += np.sum(dz, axis=0, keepdims=True) / n
            dx[:, t, :] = dz @ self.W.T
            dh_next = dz @ self.U.T
            dc_next = dc * f

        self._dW = dW
        self._dU = dU
        self._db = db
        return dx

    @property
    def params(self):
        return {"W": self.W, "U": self.U, "b": self.b}

    @property
    def grads(self):
        return {"W": self._dW, "U": self._dU, "b": self._db}


class GRU(Layer):
    """
    Gated Recurrent Unit (GRU) layer.

    Equations:
        z_t = sigmoid(x_t @ Wz + h_{t-1} @ Uz + bz)   # update gate
        r_t = sigmoid(x_t @ Wr + h_{t-1} @ Ur + br)   # reset gate
        n_t = tanh(x_t @ Wn + (r_t * h_{t-1}) @ Un + bn)  # new gate
        h_t = (1 - z_t) * n_t + z_t * h_{t-1}

    Parameters
    ----------
    units : int
    return_sequences : bool
    return_state : bool
    """

    def __init__(self, units, return_sequences=False, return_state=False,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 name=None):
        super().__init__(name=name)
        self.units = units
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)
        self.W = None
        self.U = None
        self.b = None

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def build(self, input_shape):
        _, _, d = input_shape
        u = self.units
        self.W = self.kernel_initializer((d, 3 * u))
        self.U = self.kernel_initializer((u, 3 * u))
        self.b = self.bias_initializer((1, 3 * u))
        self._built = True

    def forward(self, x, training=False):
        if not self._built:
            self.build(x.shape)
        n, T, _ = x.shape
        u = self.units
        h = np.zeros((n, u))
        self._x = x
        self._h_list = [h]
        self._gates = []

        for t in range(T):
            xW = x[:, t, :] @ self.W
            z = self._sigmoid(xW[:, :u] + h @ self.U[:, :u] + self.b[:, :u])
            r = self._sigmoid(xW[:, u:2*u] + h @ self.U[:, u:2*u] + self.b[:, u:2*u])
            n_t = np.tanh(xW[:, 2*u:] + (r * h) @ self.U[:, 2*u:] + self.b[:, 2*u:])
            h_new = (1 - z) * n_t + z * h
            self._gates.append((z, r, n_t, h))
            h = h_new
            self._h_list.append(h)

        self._all_h = np.stack(self._h_list[1:], axis=1)
        out = self._all_h if self.return_sequences else h
        if self.return_state:
            return out, h
        return out

    def backward(self, grad):
        n, T, _ = self._x.shape
        u = self.units

        if self.return_sequences:
            dh_seq = grad
        else:
            dh_seq = np.zeros((n, T, u))
            dh_seq[:, -1, :] = grad

        dW = np.zeros_like(self.W)
        dU = np.zeros_like(self.U)
        db = np.zeros_like(self.b)
        dx = np.zeros_like(self._x)
        dh_next = np.zeros((n, u))

        for t in reversed(range(T)):
            z, r, n_t, h_prev = self._gates[t]
            dh = dh_seq[:, t, :] + dh_next

            dn_t = dh * (1 - z) * (1 - n_t ** 2)
            dz = dh * (h_prev - n_t) * z * (1 - z)
            dr = (dn_t @ self.U[:, 2*u:].T) * h_prev * r * (1 - r)

            dz_raw = dz
            dr_raw = dr

            dxW = np.concatenate([dz_raw, dr_raw, dn_t], axis=-1)
            dx[:, t, :] = dxW @ self.W.T

            dW += self._x[:, t, :].T @ dxW / n
            dh_prev_z = dh * z
            dh_prev_r = dn_t @ self.U[:, 2*u:].T * r
            dh_prev_from_dz = dz_raw @ self.U[:, :u].T
            dh_prev_from_dr = dr_raw @ self.U[:, u:2*u].T
            dh_next = dh_prev_z + dh_prev_r + dh_prev_from_dz + dh_prev_from_dr

            dU[:, :u] += h_prev.T @ dz_raw / n
            dU[:, u:2*u] += h_prev.T @ dr_raw / n
            dU[:, 2*u:] += (r * h_prev).T @ dn_t / n
            db += np.sum(dxW, axis=0, keepdims=True) / n

        self._dW = dW
        self._dU = dU
        self._db = db
        return dx

    @property
    def params(self):
        return {"W": self.W, "U": self.U, "b": self.b}

    @property
    def grads(self):
        return {"W": self._dW, "U": self._dU, "b": self._db}
