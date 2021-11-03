import pickle
import numpy as np
from math import log, exp

def Alpha(seq, h0, h2h, h2v):
    N = len(seq)
    H = h0.shape[0]
    assert (N > 0 and H > 0)
    alpha = [h0[i] * h2v[i][seq[0]] for i in range(H)]
    print('----alpha----')
    print(alpha)
    alpha_ = [0 for i in range(H)]
    for i in range(1, N):
        for h in range(H):
            alpha_[h] = 0
            for h_ in range(H):
                alpha_[h] += alpha[h_] * h2h[h_][h]
            alpha_[h] *= h2v[h][seq[i]]
        alpha, alpha_ = alpha_, alpha
        print(alpha)
    return alpha

def Beta(seq, h2h, h2v):
    N = len(seq)
    H = h2h.shape[0]
    assert (N > 0 and H > 0)
    beta = [1.0 for i in range(H)]
    print('----beta----')
    print(beta)
    beta_ = [0 for i in range(H)]
    for i in range(1, N):
        for h in range(H):
            beta_[h] = 0
            for h_ in range(H):
                beta_[h] += beta[h_] * h2h[h][h_] * h2v[h_][seq[N - i]]
        beta, beta_ = beta_, beta 
        print(beta)
    return beta

def Viterbi(seq, h0, h2h, h2v):
    N = len(seq)
    H = h2h.shape[0]
    assert(N > 0 and H > 0)
    logP = [[0 for _ in range(H)] for __ in range(N)]
    preH = [[0 for _ in range(H)] for __ in range(N)]
    for h in range(H):
        logP[0][h] = log(h0[h]) + log(h2v[h][seq[0]])
    for i in range(1, N):
        for h in range(H):
            p = -99999.9
            opth = 0
            for h_ in range(H):
                p_ = logP[i-1][h_] + log(h2h[h_][h])
                if p_ > p:
                    p = p_
                    opth = h_ 
            logP[i][h] = p + log(h2v[h][seq[i]])
            preH[i][h] = opth 
    h_seq = [0 for _ in range(N)]
    for h in range(H):
        if logP[N-1][h] > logP[N-1][h_seq[-1]]:
            h_seq[-1] = h 
    for i in range(1, N):
        h_seq[N - i - 1] = preH[N - i][h_seq[N - i]]
    print('----viterbi----')
    for i in range(N):
        print(logP[i], preH[i])
    return h_seq

def sentence2seq(sentence):
    if isinstance(sentence, str):
        sentence = list(sentence.strip())
    return [ord(word) for word in sentence]

def calc_sentence(sentence, h0, h2h, h2v):
    sentence = list(sentence.strip())
    seq = sentence2seq(sentence)
    alpha = Alpha(seq, h0, h2h, h2v)
    beta = Beta(seq, h2h, h2v)
    h = Viterbi(seq, h0, h2h, h2v)
    print('Hidden State: {}'.format(h))
    print(''.join([sentence[i] + '/' if (h[i] == 1 and i != len(seq) - 1) else sentence[i] for i in range(len(seq))]))
    print('Forward: {}'.format(sum(alpha)))
    print('Backward: {}'.format(sum([h0[i] * beta[i] * h2v[i][seq[0]] for i in range(h0.shape[0])])))

if __name__ == '__main__':
    path = './hmm_parameters.pkl'
    with open(path, 'rb') as f:
        hmm_param = pickle.load(f)
    
    start_prob = hmm_param['start_prob']
    trans_mat = hmm_param['trans_mat']
    emission_mat = hmm_param['emission_mat']

    while True:
        sentence = input()
        calc_sentence(sentence, start_prob, trans_mat, emission_mat)
