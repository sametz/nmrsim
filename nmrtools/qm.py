"""qm contains functions for the quantum-mechanical (second-order)
calculation of NMR spectra.

Because numpy.matrix is marked as deprecated, in Winter/Spring 2019 the qm
code was refactored to a) accommodate this deprecation and b) speed up the
calculations. The fastest calculations rely on:

1. the pydata/sparse library. SciPy's sparse depends on numpy.matrix,
and they currently recommend that pydata/sparse be used for now.

2. Caching partial solutions for spin operators and transition matrices as
.npz files.

If the pydata/sparse package is no longer available, and/or if distributing
the library with .npz files via PyPI is problematic, then a backup is
required. The qm module for now provides two sets of functions for
calculating second-order spectra: one using pydata/sparse and caching,
and the other using neither.
"""
import os

import numpy as np
import sparse
from nmrtools.math import normalize_spectrum

CACHE = True  # saving of partial solutions is allowed
SPARSE = True  # the sparse library is available


def so_dense(nspins):
    """
    Calculate spin operators required for constructing the spin hamiltonian.

    Parameters
    ----------
    nspins: int
        the number of spins in the spin system

    Returns
    -------
    (Lz, Lproduct): a tuple of:
        Lz: 3d array of shape (n, 2^n, 2^n) representing [Lz1, Lz2, ...Lzn]
        Lproduct: 4d array of shape (n, n, 2^n, 2^n), representing an n x n
        array (cartesian product) for all combinations of
        Lxa*Lxb + Lya*Lyb + Lza*Lzb, where 1 <= a, b <= n.
    """
    sigma_x = np.array([[0, 1 / 2], [1 / 2, 0]])
    sigma_y = np.array([[0, -1j / 2], [1j / 2, 0]])
    sigma_z = np.array([[1 / 2, 0], [0, -1 / 2]])
    unit = np.array([[1, 0], [0, 1]])

    L = np.empty((3, nspins, 2 ** nspins, 2 ** nspins),
                 dtype=np.complex128)  # consider other dtype?
    for n in range(nspins):
        Lx_current = 1
        Ly_current = 1
        Lz_current = 1

        for k in range(nspins):
            if k == n:
                Lx_current = np.kron(Lx_current, sigma_x)
                Ly_current = np.kron(Ly_current, sigma_y)
                Lz_current = np.kron(Lz_current, sigma_z)
            else:
                Lx_current = np.kron(Lx_current, unit)
                Ly_current = np.kron(Ly_current, unit)
                Lz_current = np.kron(Lz_current, unit)

        L[0][n] = Lx_current
        L[1][n] = Ly_current
        L[2][n] = Lz_current

    # ref:
    # https://stackoverflow.com/questions/47752324/matrix-multiplication-on-4d-numpy-arrays
    L_T = L.transpose(1, 0, 2, 3)
    Lproduct = np.tensordot(L_T, L, axes=((1, 3), (0, 2))).swapaxes(1, 2)

    return L[2], Lproduct


def so_sparse(nspins):
    """Either load a presaved set of spin operators as numpy arrays, or
    calculate them and save them if a presaved set wasn't found.

    Parameters
    ----------
    nspins: int
        the number of spins in the spin system

    Returns
    -------
    (Lz, Lproduct): a tuple of:
        Lz: 3d sparse.COO array of shape (n, 2^n, 2^n) representing
        [Lz1, Lz2, ...Lzn]
        Lproduct: 4d sparse.COO array of shape (n, n, 2^n, 2^n), representing
        an n x n array (cartesian product) for all combinations of
        Lxa*Lxb + Lya*Lyb + Lza*Lzb, where 1 <= a, b <= n.

    Side Effect
    -----------
    Saves the results as .npz files to the bin directory if they were not
    found there.
    """
    filename_Lz = f'Lz{nspins}.npz'
    filename_Lproduct = f'Lproduct{nspins}.npz'
    bin_dir = os.path.join(os.path.dirname(__file__), 'bin')
    path_Lz = os.path.join(bin_dir, filename_Lz)
    path_Lproduct = os.path.join(bin_dir, filename_Lproduct)

    try:
        Lz = sparse.load_npz(path_Lz)
        Lproduct = sparse.load_npz(path_Lproduct)
        return Lz, Lproduct
    except FileNotFoundError:
        print('no SO file ', filename_Lz, ' found in: ', bin_dir)
        print(f'creating {filename_Lz} and {filename_Lproduct}')
    Lz, Lproduct = so_dense(nspins)
    Lz_sparse = sparse.COO(Lz)
    Lproduct_sparse = sparse.COO(Lproduct)
    sparse.save_npz(path_Lz, Lz_sparse)
    sparse.save_npz(path_Lproduct, Lproduct_sparse)

    return Lz_sparse, Lproduct_sparse


def hamiltonian_dense(v, J):
    nspins = len(v)
    Lz, Lproduct = so_dense(nspins)  # noqa
    H = np.tensordot(v, Lz, axes=1)
    scalars = 0.5 * J
    H += np.tensordot(scalars, Lproduct, axes=2)
    return H


def hamiltonian_sparse(v, J):
    """

        Parameters
        ----------
        v: array-like
            list of frequencies in Hz
        J: 2D array-like
            matrix of coupling constants

        Returns
        -------
        H: sparse.COO
            a sparse spin Hamiltonian
        """
    nspins = len(v)
    Lz, Lproduct = so_sparse(nspins)  # noqa
    # On large spin systems, converting v and J to sparse improved speed, so:
    H = sparse.tensordot(sparse.COO(v), Lz, axes=1)
    scalars = 0.5 * sparse.COO(J)
    H += sparse.tensordot(scalars, Lproduct, axes=2)
    return H


def transition_matrix_dense(nspins):
    """
    Creates a matrix of allowed transitions.

    The integers 0-`n`, in their binary form, code for a spin state
    (alpha/beta). The (i,j) cells in the matrix indicate whether a transition
    from spin state i to spin state j is allowed or forbidden.
    See the ``is_allowed`` function for more information.

    Parameters
    ---------
    nspins : number of spins in the system.

    Returns
    -------
    numpy.ndarray
        a transition matrix that can be used to compute the intensity of
    allowed transitions.

    """
    # function was optimized by only calculating upper triangle and then adding
    # the lower.
    n = 2 ** nspins
    T = np.zeros((n, n))  # sparse matrix created
    for i in range(n - 1):
        for j in range(i + 1, n):
            if bin(i ^ j).count('1') == 1:
                T[i, j] = 1
    T += T.T
    return T


def nspinspec_dense(freqs, couplings, normalize=True):
    """
    Calculates second-order spectral data (freqency and intensity of signals)
    for *n* spin-half nuclei.

    Parameters
    ---------
    freqs : [float...]
        a list of *n* nuclei frequencies in Hz
    couplings : array-like
        an *n, n* array of couplings in Hz. The order
        of nuclei in the list corresponds to the column and row order in the
        matrix, e.g. couplings[0][1] and [1]0] are the J coupling between
        the nuclei of freqs[0] and freqs[1].
    normalize: bool
        True if the intensities should be normalized so that total intensity
        equals the total number of nuclei.

    Returns
    -------
    spectrum : [[float, float]...] numpy 2D array
         of [frequency, intensity] pairs.
    """
    nspins = len(freqs)
    H = hamiltonian_dense(freqs, couplings)
    E, V = np.linalg.eigh(H)
    V = V.real
    T = transition_matrix_dense(nspins)
    I = np.square(V.T.dot(T.dot(V)))
    spectrum = new_compile_spectrum(I, E)
    if normalize:
        spectrum = normalize_spectrum(spectrum, nspins)
    return spectrum


def cache_tm(nspins):
    """

    Parameters
    ----------
    nspins

    Returns
    -------

    """
    """spin11 test indicates this leads to faster overall simsignals().

    11 spin x 6: 29.6 vs. 35.1 s
    8 spin x 60: 2.2 vs 3.0 s"""
    filename = f'T{nspins}.npz'
    bin_dir = os.path.join(os.path.dirname(__file__), 'bin')
    path = os.path.join(bin_dir, filename)
    try:
        T = sparse.load_npz(path)
        return T
    except FileNotFoundError:
        print(f'creating {filename}')
        T = transition_matrix_dense(nspins)
        T_sparse = sparse.COO(T)
        sparse.save_npz(path, T_sparse)
        return T_sparse


def intensity_and_energy(H, nspins):
    """Calculate intensity matrix and energies (eigenvalues) from Hamiltonian.

    Parameters
    ----------
    H:  numpy.ndarray
        Spin Hamiltonian
    nspins: int
        number of spins in spin system

    Returns
    -------
    (I, E): (numpy.ndarray, numpy.ndarray) tuple of:
        I: (relative) intensity 2D array
        V: 1-D array of relative energies.
    """
    E, V = np.linalg.eigh(H)
    V = V.real
    T = cache_tm(nspins)
    I = np.square(V.T.dot(T.dot(V)))
    return I, E


def new_compile_spectrum(I, E):
    """

    Parameters
    ----------
    I: numpy.ndarray (2D)
        matrix of relative intensities
    E: numpy.ndarray (1D)
        array of energies

    Returns
    -------
    numpy.ndarray (2D)
        [[frequency, intensity]...]
    """
    I_upper = np.triu(I)
    E_matrix = np.abs(E[:, np.newaxis] - E)
    E_upper = np.triu(E_matrix)
    combo = np.stack([E_upper, I_upper])
    iv = combo.reshape(2, I.shape[0] ** 2).T

    return iv[iv[:, 1] >= 0.01]


def vectorized_simsignals(H, nspins):
    """
    Calculates frequencies and intensities of signals from a spin Hamiltonian
    and number of spins.

    TODO: with a spin Hamiltonian, nspins could be inferred?

    Parameters
    ----------
    H: numpy.ndarray (2D)
        The spin Hamiltonian
    nspins : int
        The number of spins in the system

    Returns
    -------
    [[float, float]...] numpy 2D array of frequency, intensity pairs.
    """
    I, E = intensity_and_energy(H, nspins)
    return new_compile_spectrum(I, E)


def nspinspec_sparse(freqs, couplings, normalize=True):
    """
    Calculates second-order spectral data (freqency and intensity of signals)
    for *n* spin-half nuclei.

    Parameters
    ---------
    freqs : [float...]
        a list of *n* nuclei frequencies in Hz
    couplings : array-like
        an *n, n* array of couplings in Hz. The order
        of nuclei in the list corresponds to the column and row order in the
        matrix, e.g. couplings[0][1] and [1]0] are the J coupling between
        the nuclei of freqs[0] and freqs[1].
    normalize: bool
        True if the intensities should be normalized so that total intensity
        equals the total number of nuclei.

    Returns
    -------
    spectrum : [[float, float]...] numpy 2D array
        of [frequency, intensity] pairs.
    """
    nspins = len(freqs)
    H = hamiltonian_sparse(freqs, couplings)
    spectrum = vectorized_simsignals(H.todense(), nspins)
    if normalize:
        spectrum = normalize_spectrum(spectrum, nspins)
    return spectrum


def spectrum(*args, cache=CACHE, sparse=SPARSE, **kwargs):
    # for key, val in kwargs.items():
    if not (cache and sparse):
        return nspinspec_dense(*args, **kwargs)
    return nspinspec_sparse(*args, **kwargs)
