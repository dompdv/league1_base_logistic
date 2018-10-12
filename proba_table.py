import numpy as np
import csv
from itertools import product

def proba_table2(NCAT, matrices=None):
    # construit la matrice de probabilités
    rp = matrices.copy()
    # renormalisaton à cause du petit 0.0001
    for Aa, Ad, Ba, Bd in product(range(NCAT), repeat=4):
        rp[Aa, Ad, Ba, Bd, :, :] += 0.00001
        t = np.sum(rp[Aa, Ad, Ba, Bd, :, :])
        rp[Aa, Ad, Ba, Bd, :, :] /= t
    return rp

