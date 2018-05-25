import numpy as np

m = 100
eta = 0.1
seq_length = 25
sig = 0.01
n = 200

def ReadAndProcess(datafile):
    """ Read the .txt file and return two dicts `char_to_ind` and `ind_to_char`. """
    with open(datafile, 'r') as f:
        l = f.read()
    char = list(set(l))
    char.sort()
    K = len(char)
    ind = [i+1 for i in range(len(char))]
    char_to_ind = dict(zip(char, ind))
    ind_to_char = dict(zip(ind ,char))
    return char_to_ind, ind_to_char, K

def Softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def SynthSeq(h0, x0, n, RNN_dict):
    """ Synthesize text with length n. """
    h_t = h0
    ii = []
    for i in range(n):
        a_t = np.dot(RNN_dict['W'], h_t) + np.dot(RNN_dict['U'], x0) + RNN_dict['b']
        h_t = np.tanh(a_t)
        o_t = np.dot(RNN_dict['V'], h_t) + RNN_dict['c']
        p_t = Softmax(o_t)
        cp = np.cumsum(p_t)
        a = np.random.rand()
        ixs = np.where(cp - a > 0)
        ii.append(ixs[0][0] + 1)
    return ii

char_to_ind, ind_to_char, K = ReadAndProcess('goblet_book.txt')

RNN_b = np.zeros((m, 1))
RNN_c = np.zeros((K, 1))
RNN_U = np.random.normal(0, 0.01, (m, K)) * sig
RNN_W = np.random.normal(0, 0.01, (m, m)) * sig
RNN_V = np.random.normal(0, 0.01, (K, m)) * sig
RNN_dict = {'b':RNN_b, 'c':RNN_c, 'U':RNN_U, 'W':RNN_W, 'V':RNN_V}

h0 = np.zeros((m, 1))
x0 = np.zeros((K, 1))
x0[1] = 1

ii = SynthSeq(h0, x0, n, RNN_dict)
ss = [ind_to_char[i] for i in ii]
print(''.join(ss))
