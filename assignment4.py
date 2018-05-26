import numpy as np
import matplotlib.pylab as plt

np.set_printoptions(threshold=np.inf)
m = 100
eta = 0.1
seq_length = 25
sig = 0.01
epsilon = 1e-4

def ReadAndProcess(datafile):
    """ Read the .txt file and return two dicts `char_to_ind` and `ind_to_char`. """
    with open(datafile, 'r') as f:
        l = f.read()
    char = list(set(l))
    K = len(char)
    ind = [i for i in range(len(char))]
    char_to_ind = dict(zip(char, ind))
    ind_to_char = dict(zip(ind ,char))
    return char_to_ind, ind_to_char, K, l

def toOneHot(x, K):
    x_onehot = []
    for i in range(len(x)):
        x0 = np.zeros((K, 1))
        x0[x[i]] = 1
        x_onehot.append(x0)
    return x_onehot

def Softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def EvaluateLayer(h_tm1, x, RNN_dict):
    """ Evaluate every layer/every t. """
    a_t = np.dot(RNN_dict['W'], h_tm1) + np.dot(RNN_dict['U'], x) + RNN_dict['b']
    h_t = np.tanh(a_t)
    o_t = np.dot(RNN_dict['V'], h_t) + RNN_dict['c']
    p_t = Softmax(o_t)
    return a_t, h_t, p_t

def SynthSeq(h0, x0, n, RNN_dict):
    """ Synthesize text with length n. """
    h_t = h0
    seq = [np.where(x0 > 0)[0][0]]
    for i in range(n):
        h_tm1 = h_t
        _, h_t, p_t = EvaluateLayer(h_tm1, x0, RNN_dict)
        # To choose a char random according to possibilities for all characters
        cp = np.cumsum(p_t)
        a = np.random.rand()
        ixs = np.where(cp - a > 0)
        x0 = np.zeros(x0.shape)
        x0[ixs[0][0]] = 1
        seq.append(ixs[0][0])
    return seq

def EvaluateText(h0, text, n, RNN_dict):
    """ Compute the forward pass. """
    h_t = h0
    a = []
    # Note that h has one more element
    h = [h0]
    p = []
    for i in range(n):
        h_tm1 = h_t
        a_t, h_t, p_t = EvaluateLayer(h_tm1, text[i], RNN_dict)
        a.append(a_t)
        h.append(h_t)
        p.append(p_t)
    return a, h, p

def ComputeLoss(seq, y_onehot):
    # Using cross-entropy loss
    l = [-np.log(np.dot(y_onehot[i].T, seq[i])) for i in range(len(seq))]
    l = np.sum(l)
    return l

def ComputeGradients(x_onehot, y_onehot, p, RNN_dict, a, h):
    grad_RNN_dict = {}
    p = np.array(p)
    y_onehot = np.array(y_onehot)
    too_t = p - y_onehot
    grad_c = np.sum(too_t, axis=0)
    grad_RNN_dict['c'] = grad_c

    grad_V = [np.dot(too_t[i], h[i+1].T) for i in range(len(too_t))]
    grad_V = np.sum(grad_V, axis=0)
    grad_RNN_dict['V'] = grad_V

    grad_W = np.zeros(RNN_dict['W'].shape)
    grad_U = np.zeros(RNN_dict['U'].shape)
    grad_b = np.zeros(RNN_dict['b'].shape)
    toa_t = np.zeros(a[0].shape)
    for i in range(len(p))[::-1]:
        toa_t1 = toa_t.reshape(1, -1)
        toh_t = np.dot(too_t[i].T, RNN_dict['V']) + np.dot(toa_t1, RNN_dict['W'])
        toa_t = toh_t * (1 - np.square(np.tanh(a[i]).T))

        grad_b = grad_b + toa_t.T
        grad_W = grad_W + np.dot(toa_t.T, h[i].T)
        grad_U = grad_U + np.dot(toa_t.T, x_onehot[i].T)

    grad_RNN_dict['W'] = grad_W
    grad_RNN_dict['U'] = grad_U
    grad_RNN_dict['b'] = grad_b
    return grad_RNN_dict

def ClipGrad(grad_RNN_dict):
    """ In case the gradients explode. """
    for key in RNN_dict:
        for i in range(grad_RNN_dict[key].shape[0]):
            for j in range(grad_RNN_dict[key].shape[1]):
                grad_RNN_dict[key][i][j] = max(min(grad_RNN_dict[key][i][j], 5), -5)
    return grad_RNN_dict

def ComputeGradTest(x_onehot, y_onehot, p, RNN_dict, a, h):
    """ Compute the gradients numerically to check if the function above is right. """
    h0 = np.zeros((m, 1))
    _, _, seq = EvaluateText(h0, x_onehot, len(y_onehot), RNN_dict)
    loss = ComputeLoss(seq, y_onehot)
    grad_RNN_dict_test = {}
    for key in RNN_dict:
        value = RNN_dict[key]
        grad = np.zeros(value.shape)
        for i in range(value.shape[0]):
            for j in range(value.shape[1]):
                value[i][j] += epsilon
                _, _, seq = EvaluateText(h0, x_onehot, len(y_onehot), RNN_dict)
                loss1 = ComputeLoss(seq, y_onehot)
                grad[i][j] = (loss1 - loss) / epsilon
                value[i][j] -= epsilon
        grad_RNN_dict_test[key] = grad

    return grad_RNN_dict_test


char_to_ind, ind_to_char, K, text = ReadAndProcess('goblet_book.txt')

RNN_b = np.zeros((m, 1))
RNN_c = np.zeros((K, 1))
RNN_U = np.random.normal(0, 1, (m, K)) * sig
RNN_W = np.random.normal(0, 1, (m, m)) * sig
RNN_V = np.random.normal(0, 1, (K, m)) * sig
RNN_dict = {'b':RNN_b, 'c':RNN_c, 'U':RNN_U, 'W':RNN_W, 'V':RNN_V}
m_U = np.zeros((m, K))
m_W = np.zeros((m, m))
m_V = np.zeros((K, m))
m_dict = {'b':RNN_b, 'c':RNN_c, 'U':m_U, 'W':m_W, 'V':m_V}

h0 = np.zeros((m, 1))
x0 = np.zeros((K, 1))
x0[char_to_ind['\n']] = 1

smooth_loss = 0
l = []
iter_per_epo = len(text) // seq_length
for epo in range(4):
    for i in range(iter_per_epo):
        _x = [char_to_ind[c] for c in text[i*seq_length : (i+1)*seq_length]]
        _y = [char_to_ind[c] for c in text[i*seq_length+1 : (i+1)*seq_length+1]]
        x_onehot = toOneHot(_x, K)
        y_onehot = toOneHot(_y, K)

        a, h, seq = EvaluateText(h0, x_onehot, seq_length, RNN_dict)
        if (i+epo*iter_per_epo) % 1000 == 0:
            if (i+epo*iter_per_epo) % 10000 == 0:
                syn_ind = SynthSeq(h0, x_onehot[0], 1000, RNN_dict)
                ss = [ind_to_char[i] for i in syn_ind]
                print('iter = %d, pre_y :' % (i+1+epo*iter_per_epo))
                print(''.join(ss))
            else:
                syn_ind = SynthSeq(h0, x_onehot[0], 200, RNN_dict)
                ss = [ind_to_char[i] for i in syn_ind]
                print('iter = %d, pre_y :' % (i+1+epo*iter_per_epo))
                print(''.join(ss))

        h0 = h[-1] if i > 0 else np.zeros((m, 1))
        loss = ComputeLoss(seq, y_onehot)
        smooth_loss = loss if i == 0 and epo == 0 else 0.999 * smooth_loss + 0.001 * loss
        if (i+epo*iter_per_epo) % 200 == 0:
            print('iter = %d, smooth_loss = %f' % (i+1+epo*iter_per_epo, smooth_loss))
            l.append(smooth_loss)

        grad_RNN_dict = ComputeGradients(x_onehot, y_onehot, seq, RNN_dict, a, h)
        # grad_RNN_dict = ClipGrad(grad_RNN_dict)
        # if i > 0:
        #     grad_RNN_dict_test = ComputeGradTest(x_onehot, y_onehot, seq, RNN_dict, a, h)
        #     for key in grad_RNN_dict:
        #         print(key, ':')
        #         print((grad_RNN_dict_test[key] - grad_RNN_dict[key]) / np.abs(grad_RNN_dict_test[key]))
        #     print(np.max(grad_RNN_dict_test['V'] / grad_RNN_dict['V']))
        #     break

        # Update the network by AdaGrad
        for key in RNN_dict:
            m_dict[key] = m_dict[key] + grad_RNN_dict[key] ** 2
            RNN_dict[key] -= eta * grad_RNN_dict[key] / np.sqrt(m_dict[key] + epsilon)
    print('Epoch %d done!' % (epo+1))

plt.figure(1)
x = np.linspace(0, len(l), len(l))
plt.xlabel("Per 200 iters")
plt.ylabel("Smooth_loss")
plt.plot(x, l)
plt.legend()
plt.show()
