import numpy as np

class LSTM:
    def __init__(self, Wx_f, Wh_f, b_f, Wx_i, Wh_i, b_i, Wx_o, Wh_o, b_o, Wx, Wh, b):
        self.params = [Wx_f, Wh_f, b_f, Wx_i, Wh_i, b_i, Wx_o, Wh_o, b_o, Wx, Wh, b]
        self.grads = [np.zeros_like(Wx_f), np.zeros_like(Wh_f), np.zeros_like(b_f), \
                      np.zeros_like(Wx_i), np.zeros_like(Wh_i), np.zeros_like(b_i), \
                      np.zeros_like(Wx_o), np.zeros_like(Wh_o), np.zeros_like(b_o), \
                      np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        Wx_f, Wh_f, b_f, \
        Wx_i, Wh_i, b_i, \
        Wx_o, Wx_o, b_o, \
        Wx,   Wh,   b    = self.params

        f = sigmoid(np.dot(x, Wx_f) + np.dot(h_prev, Wh_f) + b_f)
        i = sigmoid(np.dot(x, Wx_i) + np.dot(h_prev, Wh_i) + b_i)
        o = sigmoid(np.dot(x, Wx_o) + np.dot(h_prev, Wh_o) + b_o)
        g = np.tanh(np.dot(x, Wx)   + np.dot(h_prev, Wh)   + b)

        c_next = c_prev * f + g * i
        h_next = tanh(c_next) * o

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx_f, Wh_f, b_f, \
        Wx_i, Wh_i, b_i, \
        Wx_o, Wx_o, b_o, \
        Wx,   Wh,   b    = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)
        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)

        df = ds * c_prev
        di = ds * g
        do = dh_next * tanh_c_next
        dg = ds * i

        dWh_f = np.dot(h_prev.T, df)
        dWx_f = np.dot(x.T, df)
        db_f  = df.sum(axis=0) 
        dWh_i = np.dot(h_prev.T, di)
        dWx_i = np.dot(x.T, di)
        db_i  = di.sum(axis=0) 
        dWh_o = np.dot(h_prev.T, do)
        dWx_o = np.dot(x.T, do)
        db_o  = do.sum(axis=0) 
        dWh   = np.dot(h_prev.T, dg)
        dWx   = np.dot(x.T, dg)
        db    = dg.sum(axis=0) 

        self.grads[ 0][...] = dWx_f
        self.grads[ 1][...] = dWh_f
        self.grads[ 2][...] = db_f
        self.grads[ 3][...] = dWx_i
        self.grads[ 4][...] = dWh_i
        self.grads[ 5][...] = db_i
        self.grads[ 6][...] = dWx_o
        self.grads[ 7][...] = dWh_o
        self.grads[ 8][...] = db_o
        self.grads[ 9][...] = dWx
        self.grads[10][...] = dWh
        self.grads[11][...] = db

        dx_f = np.dot(df, Wx_f.T)
        dx_i = np.dot(di, Wx_i.T)
        dx_o = np.dot(do, Wx_o.T)
        dx   = np.dot(dg, Wx.T)

        dh_prev_f = np.dot(df, Wh_f.T)
        dh_prev_i = np.dot(di, Wh_i.T)
        dh_prev_o = np.dot(do, Wh_o.T)
        dh_prev   = np.dot(dg, Wh.T)

        dc_prev = ds * i
        return dx_f, dx_i, dx_o, dx, \
               dh_prev_f, dh_prev_i, dh_prev_o, dh_prev, \
               dc_prev

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

