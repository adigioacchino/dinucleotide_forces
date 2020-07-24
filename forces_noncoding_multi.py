#!/usr/bin/env python
# coding: utf-8

import numpy as np

def _occurrences(string, sub):
    """Return the number of times sub is find inside string."""
    count = start = 0
    while True:
        start = string.find(sub, start) + 1
        if start > 0:
            count+=1
        else:
            return count

def _compute_frequences(seq, alphabet):
    """Return the frequency of each letter in alphabet in seq."""
    freq = np.zeros(len(alphabet))
    L = len(seq)
    for i in range(0, len(alphabet)):
        freq[i] = _occurrences(seq, alphabet[i]) / L
    return freq

def _DimerForce_fast(seq, motifs, tolerance_n=0.01, eps = None, max_iter=100, freqs = [0,0,0,0]):
    """Return the forces on the motifs 'motifs' computed for sequence 'seq' with the fequency bias given.
    Notice that seq is used only to compute the number of observed motifs (and the frequences if they are
    not provided by user). Also notice that if motifs are
    dinucleotides, only up to 12 of them are independent. Finally, tolerance, eps and max_iter are 
    parameters for the Newton-Raphson algorithm used in _calc_force_codon_fast."""
    alphabet = ['A', 'C', 'G', 'T']
    L = len(seq)
    if np.sum(freqs)==0:
        freqs = _compute_frequences(seq, alphabet)
    else:
        pass
    n_obs = [_occurrences(seq, m) for m in motifs]

    dnf = _calc_force_fast(freqs, n_obs, L, motifs, tolerance_n, eps, max_iter)
    return dnf

def _eval_log_Z(freqs, forces, motifs, L):
    """Compute the transfer matrix through _generate_motsM, _generate_freqsM and the forces given,
    then takes the correct power to obtian Z. If a single motif is given, use a faster method that
    exploit the analitical computation of log Z (if the motif is of the form XY this is correct only
    for long sequences, but it is a very good approximation and L>10 should be always fine)."""
    if len(motifs) == 1: 
        mono_2num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        m = motifs[0]
        a = mono_2num[m[0]]
        b = mono_2num[m[1]]
        x = forces[0]
        if a != b: # this trick only works for L large (L about 10 should be enough)
            M = L // 2 
            g = (np.exp(x) - 1) * freqs[a] * freqs[b]
            g1 = 1 + 2 * g
            g2 = np.sqrt(1 + 4 * g)
            lognum = M * np.log(g1 + g2) + np.log(g2 + 1) 
            logden = (M + 1) * np.log(2) + np.log(g2)
            lz = lognum - logden
        else: # this is exact for each L
            g = (np.exp(x) - 1) * freqs[a] * freqs[b]
            lz = (L - 1) * np.log(g + 1)
        return lz
    else:
        n_nt = 4
        v = np.ones(n_nt)
        motsM = _generate_motsM(motifs)
        M = np.ones((n_nt, n_nt))
        for j in range(len(forces)):
            M *= np.exp(forces[j] * motsM[j])
        last_mat = M * _generate_freqsM(freqs, last=True)
        TM = M * _generate_freqsM(freqs, last=False)
        v = np.dot(last_mat, v)
        log_factors = 0
        for i in range(0, L-2):
            f = np.linalg.norm(v)
            log_factors += np.log(f)
            t_v = v / f
            v = np.dot(TM, t_v)
    return np.log(np.sum(v)) + log_factors

def _generate_motsM(motifs):
    """Return the matrices used for the computation of the
    partition function for each motif in motifs. In particular, M_{ij} is the number of times 
    the given motif appears in n(i) + n(j), with n(i) the i-th nucleotide."""
    alphabet = ['A', 'C', 'G', 'T']
    n_row = 4
    n_col = 4    
    # a matrix for each motif, in a list
    Ms = np.array([np.zeros((n_row, n_col)) for i in range(len(motifs))])
    syn1 = alphabet.copy()
    syn2 = alphabet.copy()
    for i in range(len(motifs)):
        for x in range(n_row):
            for y in range(n_col):
                word = syn1[x] + syn2[y]
                t_n = _occurrences(word, motifs[i])
                Ms[i][x][y] = t_n
    return Ms

def _generate_freqsM(freqs, last):
    """Return a matrix with the informations about the nt frequences, so that
    when it is multiplied elementwise with that obtained through 
    _generate_motsM (after implementing the exponentiation with the force, see _eval_log_Z),
    the transfer matrices used to compute Z are finally obtained."""
    # generate all matrices but last 
    alphabet = ['A', 'C', 'G', 'T']
    n_nt = len(alphabet)
    if not last:
        res_mat = np.zeros((n_nt, n_nt))
        for i in range(n_nt):
            p_nt = freqs[i]
            res_mat[i] = [p_nt for k in range(n_nt)]
    # generate last matrix
    if last:
        res_mat = np.zeros((n_nt, n_nt))
        for i in range(n_nt):
            p_nt1 = freqs[i]
            for j in range(n_nt):
                p_nt2 = freqs[j]
                res_mat[i][j] = p_nt1 * p_nt2
    return res_mat

def _calc_force_fast(freqs, n_obs, L, motifs, tolerance_n, eps, max_iter):
    """Compute the optimal estimate of the forces to explain the observed number of motifs,
    by starting from all forces equal to zero and implementing a Newton-Raphson method."""
    if eps is None:
        eps = tolerance_n / 10.
    n_motifs = len(motifs)
    n_obs = np.array(n_obs) # this must be a numpy array
    forces = np.zeros(n_motifs)
    deltas = np.diag(np.full(n_motifs, eps))
    for l in range(0, max_iter):
        lZ = _eval_log_Z(freqs, forces, motifs, L)
        # compute all lZp and lZm and ns - they are matrices with identical columns or row to ease following steps
        lZp = np.zeros((n_motifs, n_motifs))
        lZm = np.zeros((n_motifs, n_motifs))
        for j in range(n_motifs):
            t_p = _eval_log_Z(freqs, forces + deltas[j], motifs, L)
            t_m = _eval_log_Z(freqs, forces - deltas[j], motifs, L)
            lZp[j] = np.full(n_motifs, t_p)
            lZm[j] = np.full(n_motifs, t_m)
        ns = (lZp[:, 0] - lZm[:, 0]) / (2 * eps) # vector of n values
        lZp = np.transpose(lZp) # to compute the jacobian in one line, see below
        # compute all lZpm and jacobian
        lZpm = np.zeros((n_motifs, n_motifs))
        for j in range(n_motifs):
            for k in range(n_motifs):
                if j == k:
                    lZpm[j][k] = lZ
                else:
                    lZpm[j][k] = _eval_log_Z(freqs, forces - deltas[j] + deltas[k], motifs, L)
        lZ_mat = np.full((n_motifs, n_motifs), lZ)
        dn = (lZp - lZpm - lZ_mat + lZm) / eps**2 # jacobian of ns (hessian of logZ)
        # compute next step of Newton-Raphson
        df = np.dot(np.linalg.inv(dn), n_obs - ns)
        if max(abs(n_obs - ns)) <= tolerance_n:
            break
        forces += df
    return forces

def compute_force_noncoding(seq, motifs, sliding_window_length = None, freqs = [0,0,0,0]):
    """Compute forces on motifs 'motifs' sliding windows of sequence 'seq', each sliding
    window being of size 'sliding_window_length' and moving of 1 nt each time.
    If the frequences are not specified, compute them on the whole sequence."""
    if np.sum(freqs) == 0:
        alphabet = ['A', 'C', 'G', 'T']
        freqs = _compute_frequences(seq, alphabet)
    else:
        pass
    if sliding_window_length == None:
        sliding_window_length = len(seq)
    sliding_window_shift = 1
    l = len(seq) - sliding_window_length + 1
    forces = np.zeros((l, len(motifs)))
    for i in range(l):
        w_start = i * sliding_window_shift
        w_end = i * sliding_window_shift + sliding_window_length
        t_seq = seq[w_start:w_end]
        forces[i] = _DimerForce_fast(t_seq, motifs, freqs=freqs)
    return forces

def compute_log_probability(seq, forces, motifs, freqs = [0,0,0,0]):
    """Compute the log-probability of a sequence, given the nucleotide frequences and motifs forces.
    If the frequences are not specified, compute them on the whole sequence."""
    alphabet = ['A', 'C', 'G', 'T']
    if np.sum(freqs) == 0:
        freqs = _compute_frequences(seq, alphabet)
    else:
        pass
    mono_2num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    lp = 0
    # probability of single nucleotides
    for n in seq:
        lp += np.log(freqs[mono_2num[n]])
    # count motif and add force
    n_motifs = [_occurrences(seq, m) for m in motifs]
    lp += np.dot(n_motifs, forces)
    # compute and include partition function
    lZ = _eval_log_Z(freqs, forces, motifs, len(seq))
    lp -= lZ 
    return lp