import math
from numpy import zeros, empty, cos, sin, any, copy
import numpy as np
import copy as cop
from numpy.linalg import solve, LinAlgError
from numpy.random import rand, randn, uniform, default_rng
from numba import float32, float64, jit, NumbaPerformanceWarning
import warnings
from numba import cuda


warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


def drss_matrices(
        states, inputs, outputs, strictly_proper=False, mag_range=(0.5, 0.97), phase_range=(0, math.pi / 2),
        dtype="float64", rng=None):
    """Generate a random state space.

    This does the actual random state space generation expected from rss and
    drss.  cdtype is 'c' for continuous systems and 'd' for discrete systems.

    """

    #print (mag_range, phase_range)
    if rng is None:
        rng = default_rng(None)
    # Probability of repeating a previous root.
    pRepeat = 0.05
    # Probability of choosing a real root.  Note that when choosing a complex
    # root, the conjugate gets chosen as well.  So the expected proportion of
    # real roots is pReal / (pReal + 2 * (1 - pReal)).
    pReal = 0.6
    # Probability that an element in B or C will not be masked out.
    pBCmask = 0.8
    # Probability that an element in D will not be masked out.
    pDmask = 0.3
    # Probability that D = 0.
    pDzero = 0.5

    # Check for valid input arguments.
    if states < 1 or states % 1:
        raise ValueError("states must be a positive integer.  states = %g." %
                         states)
    if inputs < 1 or inputs % 1:
        raise ValueError("inputs must be a positive integer.  inputs = %g." %
                         inputs)
    if outputs < 1 or outputs % 1:
        raise ValueError("outputs must be a positive integer.  outputs = %g." %
                         outputs)

    # Make some poles for A.  Preallocate a complex array.
    poles = zeros(states) + zeros(states) * 0.j
    i = 0

    while i < states:
        if rng.random() < pRepeat and i != 0 and i != states - 1:
            # Small chance of copying poles, if we're not at the first or last
            # element.
            if poles[i - 1].imag == 0:
                # Copy previous real pole.
                poles[i] = poles[i - 1]
                i += 1
            else:
                # Copy previous complex conjugate pair of poles.
                poles[i:i + 2] = poles[i - 2:i]
                i += 2
        elif rng.random() < pReal or i == states - 1:
            # No-oscillation pole.
            #
            poles[i] = rng.uniform(mag_range[0], mag_range[1], 1)
            i += 1
        else:
            mag = rng.uniform(mag_range[0], mag_range[1], 1)
            phase = rng.uniform(phase_range[0], phase_range[1], 1)

            poles[i] = complex(mag * cos(phase), mag * sin(phase))
            poles[i + 1] = complex(poles[i].real, -poles[i].imag)
            i += 2

    # Now put the poles in A as real blocks on the diagonal.
    A = zeros((states, states))
    i = 0
    while i < states:
        if poles[i].imag == 0:
            A[i, i] = poles[i].real
            i += 1
        else:
            A[i, i] = A[i + 1, i + 1] = poles[i].real
            A[i, i + 1] = poles[i].imag
            A[i + 1, i] = -poles[i].imag
            i += 2
    # Finally, apply a transformation so that A is not block-diagonal.
    while True:
        T = rng.normal(size=(states, states))
        try:
            A = solve(T, A) @ T  # A = T \ A @ T
            break
        except LinAlgError:
            # In the unlikely event that T is rank-deficient, iterate again.
            pass

    # Make the remaining matrices.
    B = rng.normal(size=(states, inputs))
    C = rng.normal(size=(outputs, states))
    D = rng.normal(size=(outputs, inputs))

    # Make masks to zero out some of the elements.
    while True:
        Bmask = rng.random(size=(states, inputs)) < pBCmask
        if any(Bmask):  # Retry if we get all zeros.
            break
    while True:
        Cmask = rng.random(size=(outputs, states)) < pBCmask
        if any(Cmask):  # Retry if we get all zeros.
            break
    if rng.random() < pDzero:
        Dmask = zeros((outputs, inputs))
    else:
        Dmask = rng.random(size=(outputs, inputs)) < pDmask

    # Apply masks.
    B = B * Bmask
    C = C * Cmask
    D = D * Dmask if not strictly_proper else zeros(D.shape) # strictly proper -> D = 0 -> 1 input delay

    return A.astype(dtype), B.astype(dtype), C.astype(dtype), D.astype(dtype)


signatures = [
    float32[:, :](float32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:]),
    float64[:, :](float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:, :], float64[:])
]


@jit(signatures, nopython=True, cache=True)
def _dlsim(A, B, C, D, u, x0):
    seq_len = u.shape[0]
    nx, nu = B.shape
    ny, _ = C.shape
    y = empty(shape=(seq_len, ny), dtype=u.dtype)
    x_step = copy(x0)  # x_step = zeros((nx,), dtype=u.dtype)
    for idx in range(seq_len):
        u_step = u[idx]
        y[idx] = C.dot(x_step) + D.dot(u_step)
        x_step = A.dot(x_step) + B.dot(u_step)
    return y


def dlsim(A, B, C, D, u, x0=None):
    if x0 is None:
        nx = A.shape[0]
        x0 = zeros((nx,), dtype=u.dtype)
    return _dlsim(A, B, C, D, u, x0)

import torch
# cuda_device = "cuda:3"
# no_cuda = False
# use_cuda = not no_cuda and torch.cuda.is_available()
# device_name  = cuda_device if use_cuda else "cpu"
# device = torch.device(device_name)
# device_type = 'cuda' if 'cuda' in device_name else 'cpu'


def nn_fun(x,w1,b1,w2,b2):#, device_cuda):
    # print(device_cuda, device_name)
    # assert device_cuda == device_name, "change the cuda for non linear function in WH"
    # print(x.dtype)
    # print(x)
    x_gpu = torch.tensor(x)
    w1_gpu = torch.tensor(w1.transpose())
    b1_gpu = torch.tensor(b1)
    w2_gpu = torch.tensor(w2.transpose())
    b2_gpu = torch.tensor(b2)
    # Perform matrix multiplication on GPU
    out = torch.matmul(x_gpu, w1_gpu) + b1_gpu
    out = torch.tanh(out)
    out = torch.matmul(out, w2_gpu) + b2_gpu
    out = out.cpu().double().numpy()

    return out


from typing import Callable
from numba import float64, int64, jit, njit
import numpy as np

# @jit(nopython=False)
def RK4_step(f: Callable[[float], np.ndarray], x0: float, y0: float64[:], 
             h: float, FJss_seq: np.ndarray, seq_len: int, par_shifted: float64[:], 
             par_fixed: float64[:], par_changed: float64[:]):
    """
    Perform a single step of the 4th-order Runge-Kutta method.
    """
    k1 = f(x0, y0, FJss_seq, seq_len, par_shifted, par_fixed, par_changed)
    k2 = f(x0 + 0.5*h, y0 + 0.5*h * k1, FJss_seq, seq_len, par_shifted, par_fixed, par_changed)
    k3 = f(x0 + 0.5*h, y0 + 0.5*h * k2, FJss_seq, seq_len, par_shifted, par_fixed, par_changed)
    k4 = f(x0 + h, y0 + h * k3, FJss_seq, seq_len, par_shifted, par_fixed, par_changed)

    y_next = y0 + h * (k1 + 2*k2 + 2*k3 + k4) / 6.0

    return y_next

# @jit(nopython=False)
def RK4_system(f: Callable[[float], float64[:]], x0: float, y0: float64[:], 
               xN: float, N: int, FJss_seq: float64[:], par_shifted: float64[:], 
               par_fixed: float64[:], par_changed: float64[:]):
    """
    Solve an initial value problem using the 4th-order Runge-Kutta method.
    """
    # Define a finer step size
    M = 1 * N  # 10 times more steps for finer resolution
    h = (xN - x0) / M
    x_fine = np.linspace(x0, xN, M + 1)
    y_fine = np.zeros((M + 1, y0.shape[0]), dtype=y0.dtype)
    y_fine[0] = y0

    for i in range(M):
        y_fine[i + 1] = RK4_step(f, x_fine[i], y_fine[i], h, FJss_seq, N, par_shifted, par_fixed, par_changed)

    # Resample to get N+1 points using higher-order polynomial interpolation (Lagrange)

    return x_fine, y_fine

# @jit(nopython=True)
def generate_random_binary_function_rep_2(t: float, c: float64[:]):
    '''
    From a binary signal return its respective function.
    '''
    seq_len = len(c)
    t0, tN = 0, seq_len * 20
    h = (tN - t0) / seq_len
    idx = int((t - t0) / h)
    if idx >= seq_len:
        return c[-1]
    return c[idx]   
    # In order to match the dimensions, the last element is taken as the previous one

# @jit(nopython=True)
def f2(t: float, y: float64[:], FJss_seq: float64[:], seq_len: int, 
      par_shifted:float64[:], par_fixed: float64[:], par_changed: float64[:]):
    [R, rho, cp, U, rhoJ, cJ, Fss, TRss, Cass] = par_fixed
    [kss, VR, D, AJ] = par_changed
    [E, lam, k0, Ca0, T0, TCin, UAJ] = par_shifted

    Qss = (Ca0 - Cass) * Fss * (-lam) - cp * rho * Fss * (TRss - T0)
    # TJss = TRss - Qss / (U * AJ)
    VJ = 1 / 3 * AJ
    T1 = y[1]
    exp_factor = y[0] * (k0 * np.exp(-E / (T1 * R)))

    f1 = Fss / VR * (Ca0 - y[0]) - exp_factor
    f2 = Fss / VR * (T0 - T1) - lam * exp_factor / (rho * cp) - UAJ * (T1 - y[2]) / (VR * rho * cp)
    f3 = generate_random_binary_function_rep_2(t, FJss_seq) / VJ * (TCin - y[2]) + UAJ * (T1 - y[2]) / (VJ * rhoJ * cJ)

    return np.array([f1, f2, f3])

