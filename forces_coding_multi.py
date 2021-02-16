#!/usr/bin/env python
# coding: utf-8

import numpy as np
from Bio.Seq import translate

AA2codons = {'A': ['GCA', 'GCC', 'GCG', 'GCT'], 'C': ['TGT', 'TGC'], 'E': ['GAG', 'GAA'], 'D': ['GAC', 'GAT'], 'G': ['GGT', 'GGG', 'GGA', 'GGC'], 'F': ['TTT', 'TTC'], 'I': ['ATC', 'ATA', 'ATT'], 'H': ['CAT', 'CAC'], 'K': ['AAG', 'AAA'], 'M': ['ATG'], 'L': ['CTT', 'CTG', 'CTA', 'CTC', 'TTA', 'TTG'], 'N': ['AAC', 'AAT'], 'Q': ['CAA', 'CAG'], 'P': ['CCT', 'CCG', 'CCA', 'CCC'], 'S': ['AGC', 'AGT', 'TCT', 'TCG', 'TCC', 'TCA'], 'R': ['AGG', 'AGA', 'CGA', 'CGC', 'CGG', 'CGT'], 'T': ['ACA', 'ACG', 'ACT', 'ACC'], 'W': ['TGG'], 'V': ['GTA', 'GTC', 'GTG', 'GTT'], 'Y': ['TAT', 'TAC'], '*' : ['TAA', 'TAG', 'TGA']}


def _occurrences(string, sub):
    """Return the number of times sub is find inside string."""
    count = start = 0
    while True:
        start = string.find(sub, start) + 1
        if start > 0:
            count+=1
        else:
            return count


def _DimerForceCodons_fast(seq, n_obs, motifs, codon_bias=None, tolerance=0.1, eps=None, max_iter=100, add_pseudocount=True):
    """Return the forces on the motifs 'motifs' computed for sequence 'seq' with the codon bias given.
    Notice that seq is used only to compute the amino acid sequence. Also notice that if motifs are
    dinucleotides, only up to 12 of them are independent. Finally, tolerance, eps and max_iter are 
    parameters for the Newton-Raphson algorithm used in _calc_force_codon_fast."""
    seqAA = translate(seq)
    if codon_bias is None:
        cb = _compute_codon_bias(seq)
    else:
        cb = codon_bias
    mot_Ms = _generate_motif_Ms(seqAA, motifs)
    cod_Ms = _generate_codonbias_Ms(seqAA, cb)
    forces = _calc_force_codons_fast(n_obs, mot_Ms, cod_Ms, tolerance, eps, max_iter, add_pseudocount)
    return forces


def _generate_motsM(aa1, aa2, motifs, last=False):
    """Given two amino acids, aa1 and aa2, return the matrices used for the computation of the
    partition function for each motif in motifs. In particular, M_{ij} is the number of times 
    the given motif appears in the first codon + the first base of the second (or the whole second
    for the last matrix), while i and j move across the synonyms."""
    n_row = len(AA2codons[aa1])
    n_col = len(AA2codons[aa2])    
    # creo una matrice di per ogni motivo, le stampo come lista
    Ms = np.array([np.zeros((n_row, n_col)) for i in range(len(motifs))])
    syn1 = AA2codons[aa1]
    syn2 = AA2codons[aa2]
    for i in range(len(motifs)):
        for x in range(n_row):
            for y in range(n_col):
                if last:
                    word = syn1[x] + syn2[y]
                else:
                    word = syn1[x] + syn2[y][0] # only the first nt of the second is kept
                t_n = _occurrences(word, motifs[i])
                Ms[i][x][y] = t_n
    return Ms

def _generate_motif_Ms(aa_seq, motifs):
    """Use _generate_motsM to generate a list of matrices for each motif in motifs,
    so that they are obtained for each pair of consecutives amino acids in aa_seq."""
    seq_Ms = [[] for m in motifs]
    for i in range(len(aa_seq)-2):
        aa1 = aa_seq[i]
        aa2 = aa_seq[i+1]
        t_ms = _generate_motsM(aa1, aa2, motifs, last=False)
        for j in range(len(t_ms)):
            seq_Ms[j].append(t_ms[j])
    # last matrix
    aa1 = aa_seq[-2]
    aa2 = aa_seq[-1]
    t_ms = _generate_motsM(aa1, aa2, motifs, last=True)
    for j in range(len(t_ms)):
        seq_Ms[j].append(t_ms[j])
    return seq_Ms

def _generate_codonbias_Ms(aa_seq, codon_bias):
    """Return a list of matrices with the informations about the codon biases, so that
    when these matrices are multiplied elementwise with those obtained through 
    _generate_motif_Ms (after implementing the exponentiation with the force, see _eval_log_Z),
    the transfer matrices used to compute Z are finally obtained."""
    # generate all matrices but last 
    codon_ms = []
    for i in range(len(aa_seq)-2):
        aa1 = aa_seq[i]
        aa2 = aa_seq[i+1]
        syn1 = AA2codons[aa1]
        n_col = len(AA2codons[aa2])
        t_m = []
        for s in syn1:
            p_c = codon_bias[s]
            t_m.append([p_c for k in range(n_col)])
        codon_ms.append(np.array(t_m))
    # generate last matrix
    aa1 = aa_seq[-2]
    aa2 = aa_seq[-1]
    syn1 = AA2codons[aa1]
    syn2 = AA2codons[aa2]
    t_m = []
    for x in range(len(syn1)):
        t_m.append([])
        s1 = syn1[x]
        p_c1 = codon_bias[s1]
        for y in range(len(syn2)):
            s2 = syn2[y]
            p_c2 = codon_bias[s2]
            t_m[x].append(p_c1 * p_c2)
    codon_ms.append(np.array(t_m))
    return codon_ms

def _uniform_codon_bias():
    """Return the uniform codon bias."""
    syns = list(AA2codons.values())
    cb = {}
    for ss in syns:
        t_b = 1/len(ss)
        for s in ss:
            cb[s] = t_b
    return cb

def _compute_codon_bias(seq):
    """Return the codon bias of sequence seq."""
    if len(seq) % 3 != 0:
        print('Warning: sequence lenght not multiple of 3, negletcing last 1 or 2 nucleotides.')
    seq_in_codons = [seq[i*3:(i+1)*3] for i in range(0, len(seq)//3)]
    syns = list(AA2codons.values())
    cb = {}
    counts = []
    # count codons
    for ss in syns:
        counts.append([])
        for s in ss:
            counts[-1].append(seq_in_codons.count(s))
    # normalize
    for i in range(len(counts)):
        t_tot = np.sum(counts[i])
        for j in range(len(counts[i])):
            cb[syns[i][j]] = counts[i][j] / t_tot
    return cb

def _eval_log_Z(Ms, codon_ms, forces):
    """Given the matrices with info on which codon contains which motif and the codon
    bias matrices, add the information about the forces to obtain the actual transfer matrices,
    then perform the computation through transfer matrix method to obtian Z."""
    vec = np.ones(codon_ms[-1].shape[1])
    log_factors = 0
    for i in np.arange(len(codon_ms))[::-1]:
        M = np.ones(codon_ms[i].shape)
        for j in range(len(forces)):
            M *= np.exp(forces[j] * Ms[j][i])
        M *= codon_ms[i]
        # trick to deal with very large or very small numbers
        t_factor = vec[0]
        #print('vec:', vec)
        #t_factor = np.linalg.norm(vec)
        vec_mod = vec / t_factor
        #print('vec_mod:', vec_mod)
        vec = np.dot(M, vec_mod)
        log_factors += np.log(t_factor)
    return np.log(np.sum(vec)) + log_factors

def _calc_force_codons_fast(n_obs, Ms, freqs_Ms, tolerance_n, eps, max_iter, add_pseudocount):
    """Compute the optimal estimate of the forces to explain the observed number of motifs,
    by starting from all forces equal to zero and implementing a Newton-Raphson method."""
    if eps is None:
        eps = tolerance_n / 10.
    n_motifs = len(Ms)
    n_obs = np.array(n_obs) # this must be a numpy array
    if add_pseudocount:
        for i, no in enumerate(n_obs):
            if no == 0:
                n_obs[i] = 1
    forces = np.zeros(n_motifs)
    deltas = np.diag(np.full(n_motifs,eps))
    for l in range(0, max_iter):
#        print('forces', forces)
        lZ = _eval_log_Z(Ms, freqs_Ms, forces)
        # compute all lZp and lZm and ns - they are matrices with identical columns or row to ease following steps
        lZp = np.zeros((n_motifs, n_motifs))
        lZm = np.zeros((n_motifs, n_motifs))
        for j in range(n_motifs):
            t_p = _eval_log_Z(Ms, freqs_Ms, forces + deltas[j])
            t_m = _eval_log_Z(Ms, freqs_Ms, forces - deltas[j])
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
                    lZpm[j][k] = _eval_log_Z(Ms, freqs_Ms, forces - deltas[j] + deltas[k])
        lZ_mat = np.full((n_motifs, n_motifs), lZ)
        dn = (lZp - lZpm - lZ_mat + lZm) / eps**2 # jacobian of ns (hessian of logZ)
        #print('eigvals:', np.linalg.eigvals(dn))
        # compute next step of Newton-Raphson
        df = np.dot(np.linalg.inv(dn), n_obs - ns)
#        print('forces, ns, n_obs, dn, df, iteration:', forces, ns, n_obs, dn, df,l)
#        print('forces, ns, n_obs, df, iteration:', forces, ns, n_obs, df, l)
#        print('lZp, lZm, lZ, lZpm:', lZp, lZm, lZ,lZpm)
        if max(abs(n_obs - ns)) <= tolerance_n:
            break
        forces += df
    return forces

def _count_and_drop_gaps(seq):
    """Return the sequence without gaps."""
    count = _occurrences(seq, '-')
    return count, seq.replace('-','')
   

def compute_forces_coding(seq, motifs, sliding_window_length = None, codon_bias=None,
                          adaptive_sliding_windows=False, add_pseudocount=True):
    """ Compute forces on motifs 'motifs' sliding windows of sequence 'seq', each sliding
    window being of size 'sliding_window_length' and moving of 3 nt each time.
    Takes also into account gaps: the sliding window is computed on the sequence with gaps,
    then the gap are removed and the force is computed on the remaining part. This should be 
    fine, provided that the number of gaps is not too high (in that case the 
    sliding window lenght should be increased). Use codon_bias = 'human' to use human codon bias
    from file. When adaptive_sliding_windows=True, the sliding window size shrinks at the 
    sequence ends. If add_pseudocount, when the number of motif is 0 in seq (or sliding window), 
    the force is computed for num motif = 1."""
    sliding_window_shift = 3 # shift of 1 amino acid each time
    warning = True
    if codon_bias == 'human': # load human codon bias from file
        c_bias = np.zeros(64)
        with open('human_codon_bias.dat') as f:
            i = 0
            for line in f: 
                c_bias[i] = line.split(' ')[1]
                i += 1
    elif codon_bias == 'virus':
        c_bias = _count_codons(seq, normalize=True)
    elif codon_bias == 'uniform':
        c_bias = _count_codons(seq, normalize=True, uniform=True)
    else:
        c_bias = codon_bias
    if sliding_window_length == None:
        print('Not using sliding windows.')
        n_obs = [_occurrences(seq, m) for m in motifs]
        return _DimerForceCodons_fast(seq, n_obs, motifs, codon_bias=c_bias)
    while (sliding_window_length//2) % 3 != 0:
        print('Half sliding_window_length must be mutiple of 3. Correcting...')
        sliding_window_length += 1
        print('new sliding_window_length:', sliding_window_length)
    if adaptive_sliding_windows:
        l = len(seq) // sliding_window_shift
        half_len = sliding_window_length // 2
    else:   
        l =  (len(seq) - sliding_window_length)//sliding_window_shift + 1
    forces = np.zeros((l, len(motifs)))
    for i in range(l):
        if adaptive_sliding_windows:
            pos = sliding_window_shift * i
            w_start = max(0, pos - half_len)
            w_end = min(len(seq), pos + half_len)
        else:
            w_start = i * sliding_window_shift
            w_end = i * sliding_window_shift + sliding_window_length
        t_seq = seq[w_start:w_end]
        num_gap, t_seq = _count_and_drop_gaps(t_seq)
        if num_gap >= sliding_window_length / 3 and warning:
            print('Warning: many gaps are present, expect lower quality result:', num_gap ,'gaps in', 
                    sliding_window_length,'nt long window (further warning suppressed).')
            warning = False
        n_obs = [_occurrences(t_seq, m) for m in motifs]
        forces[i] = _DimerForceCodons_fast(t_seq, n_obs, motifs, codon_bias=c_bias, add_pseudocount=add_pseudocount)
    return forces


def compute_log_probability(seq, forces, motifs, codon_bias):
    """Compute the logarithm of the probability of observing seq, according to the model."""
    log_prob = 0
    # conta numero di motivi n, fai x * n, aggiungi a log prob
    n_obs = [_occurrences(seq, m) for m in motifs]
    log_prob += np.dot(forces, n_obs)
    # cicla sui codoni, calcola le prob, fai log(p) e somma a log prob
    seq_in_codons = [seq[i*3:(i+1)*3] for i in range(0, len(seq)//3)]
    for i in range(len(seq_in_codons)):
        p_i = codon_bias[seq_in_codons[i]]
        log_prob += np.log(p_i)
    # calcola log Z
    aa_seq = translate(seq)
    Ms = _generate_motif_Ms(aa_seq, motifs)
    codon_Ms = _generate_codonbias_Ms(aa_seq, codon_bias)
    lZ = _eval_log_Z(Ms, codon_Ms, forces)    
    log_prob -= lZ
    return log_prob