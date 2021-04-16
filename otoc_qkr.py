import numpy as np
from numpy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt


def cis_theta(array: np.ndarray) -> np.ndarray:
    return np.cos(array) + 1j * np.sin(array)


Ux = lambda x, K: cis_theta(-K * np.cos(x))
Up = lambda p, M: cis_theta((-(p ** 2)) / 2 * M)
Ux_dagger = lambda x, K: cis_theta(K * np.cos(x))
Up_dagger = lambda p, M: cis_theta((p ** 2) / 2 * M)


def coeff_b(nrows, ncols) -> np.ndarray:
    """Compute the coefficient matrix of b which is a Toeplitz matrix"""
    coeff_array = np.zeros((nrows, ncols), dtype="complex_")
    for idx, _ in np.ndenumerate(coeff_array):
        coeff_array[idx] = 1j * (idx[0] - idx[1])
    return coeff_array


def evolve(
    operator: np.ndarray, basis, x: np.ndarray, p: np.ndarray, params: dict
) -> np.ndarray:
    """Compute the time-evolution of Heisenberg operators using split-step-FT
    method. During each iteration, the new operator is computed using the
    previous position and momentum operators. The iteration is performed by
    splitting the Floquet operator in appropriate fourier basis"""
    N, M, K, T = params.values()
    try:
        if basis == "position":
            F = Ux(x, K) * ifft(Up(p, M))
            F_dagger = ifft(Up_dagger(p, M) * fft(Ux_dagger(x, K)))
            return F_dagger * x * F

        elif basis == "momentum":
            F = fft(Ux(x, K) * ifft(Up(p, M)))
            F_dagger = Up_dagger(p, M) * fft(Ux_dagger(x, K))
            return F_dagger * p * F

        else:
            raise (NameError)

    except ValueError as v:
        print("Error!")
        print("Shape of x and p operator do not match with operator tobe evolved")
        print(v)

    except NameError:
        print("Error!")
        print(f"Basis type is either 'position' or 'momentum'.You supplied '{basis}'")

    except Exception as e:
        print("Error!")
        print("Something unexpected happened...")
        print(e)


def init_compute(params):
    """Initialise arrays that will hold momentum and position matrices
    in their respective basis (only diagonal entries) under unitary time-evolution
    and computes the momentum and position operators after each iteration and stores
    them in tensors. Additionally, computes the microcanonical OTOC (c)"""

    # unzip dictionary and assign parameters
    N, M, K, T = params.values()

    p0 = np.fft.fftfreq(N, 1.0 / N)
    x0 = np.arange(0, 2 * np.pi, 2 * np.pi / N)

    p_time_evolution = np.zeros((N, T), dtype="complex_")
    x_time_evolution = np.zeros((N, T), dtype="complex_")
    p_time_evolution[:, 0] = p0
    x_time_evolution[:, 0] = x0

    # Initialise b,c tensor to hold b,c matrices iteravtively
    b_tensor = np.zeros((N, N, T), dtype="complex_")
    c_tensor = np.zeros((N, N, T), dtype="complex_")
    c_trace = np.zeros(T, dtype="complex_")
    for i in range(T - 1):  # Evolve
        b = coeff_b(N, N) * p_time_evolution[:, i]
        c = b @ b.T.conj()
        print(c.trace())
        c_trace[i] = c.trace()
        p_time_evolution[:, i + 1] = evolve(
            p_time_evolution[:, i],
            "momentum",
            x_time_evolution[:, i],
            p_time_evolution[:, i],
            params,
        )
        x_time_evolution[:, i + 1] = evolve(
            x_time_evolution[:, i],
            "position",
            x_time_evolution[:, i],
            p_time_evolution[:, i],
            params,
        )
        b_tensor[:, :, i], c_tensor[:, :, i] = b, c
    return b_tensor, c_tensor


def main(N, M, K, T):
    params = {"N": N, "M": M, "K": K, "T": T}
    b, c = init_compute(params)
