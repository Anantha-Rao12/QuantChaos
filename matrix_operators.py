import numpy as np

N = 4 

def get_xp_operators(N:int,s:float,hbar=1) -> np.ndarray:
    """Using the representation of position and momentum operators in the number-operator basis of the Harmonic oscillator and truncating it to an integer value N, we obtain the x,p operator in matrix forms"""
    diag_elems = np.sqrt(np.arange(1,N))
    x = (s/np.sqrt(2))*(np.diag(diag_elems,-1) + np.diag(diag_elems,+1))
    p = 1j/(s*np.sqrt(2))*(np.diag(diag_elems,-1) - np.diag(diag_elems,+1))
    return x,p


N = 20
x,p = get_xp_operators(N,1)

# Quadratic/ Harmonic Oscillator
H = (p @ p)/2  + (x @ x )/2

egval,_ = np.linalg.eig(H)
print('Eigenvalues of the Hamiltonian for the Quantum Harmonic Oscillator :\n',egval)

# Quartic Oscillator
# H0 = (p @ p)/2  + (x @ x @ x @ x)/2
# 
# egval,egvec = np.linalg.eig(H0)
# 
# print(np.sort(egval))

