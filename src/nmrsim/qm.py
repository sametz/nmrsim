"""qm contains functions for the quantum-mechanical (second-order)
calculation of NMR spectra.

The qm module provides the following attributes:

* CACHE : bool (default True)
    Whether saving to disk of partial solutions is allowed.
* SPARSE : bool (default True)
    Whether the sparse library can be used.

The qm module provides the following functions:

* qm_spinsystem: The high-level function for computing a second-order
  simulation from frequency and J-coupling data.
* hamiltonian_dense: Calculate a spin Hamiltonian using dense arrays
  (slower).
* hamiltonian_sparse: Calculate a spin Hamiltonian using cached sparse arrays
  (faster).
* solve_hamiltonian: Calculate a peaklist from a spin Hamiltonian.
* secondorder_dense: Calculate a peaklist for a second-order spin system,
  using dense arrays (slower).
* secondorder_sparse: Calculate a peaklist for a second-order spin system,
  using cached sparse arrays (faster).

Notes
-----
Because numpy.matrix is marked as deprecated, starting with Version 0.2.0 the
qm code was refactored to a) accommodate this deprecation and b) speed up the
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
import sys

import scipy.sparse

if sys.version_info >= (3, 7):
    from importlib import resources
else:
    import importlib_resources as resources

import numpy as np  # noqa: E402
import sparse  # noqa: E402
import itertools  # need itertools to generate_spin_states

import nmrsim.bin  # noqa: E402
from nmrsim.math import normalize_peaklist  # noqa: E402

CACHE = True  # saving of partial solutions is allowed
SPARSE = True  # the sparse library is available


def _bin_path():
    """Return a Path to the nmrsim/bin directory."""
    init_path_context = resources.path(nmrsim.bin, "__init__.py")
    with init_path_context as p:
        init_path = p
    bin_path = init_path.parent
    return bin_path


def _so_dense(spins):
    """
    Calculate spin operators required for constructing the spin hamiltonian,
    using dense (numpy) arrays.

    Parameters
    ----------
    spins : array-like of float
        A list or array containing the spin quantum number (I) for each nucleus.
        E.g., np.array([0.5, 0.5, 1.0]) for two spin-1/2 and one spin-1 nucleus.

    Returns
    -------
    (Lz, Lproduct) : a tuple of:
        Lz : 3d array of shape (n, dim_total, dim_total) representing [Lz1, Lz2, ...Lzn]
        Lproduct : 4d array of shape (n, n, dim_total, dim_total), representing an n x n
            array (cartesian product) for all combinations of
            Lxa*Lxb + Lya*Lyb + Lza*Lzb, where 1 <= a, b <= n.
    """
    nspins = len(spins)  # Number of nuclei in the system

    # Define spin-1/2 Pauli matrices
    sigma_x_half = np.array([[0, 1 / 2], [1 / 2, 0]])
    sigma_y_half = np.array([[0, -1j / 2], [1j / 2, 0]])
    sigma_z_half = np.array([[1 / 2, 0], [0, -1 / 2]])
    unit_half = np.array([[1, 0], [0, 1]])

    # Define spin-1 matrices
    sigma_x_1 = np.array([[0, np.sqrt(2) / 2, 0], [np.sqrt(2) / 2, 0, np.sqrt(2) / 2], [0, np.sqrt(2) / 2, 0]])
    sigma_y_1 = np.array([[0, (-1j * np.sqrt(2)) / 2, 0], [(1j * np.sqrt(2)) / 2, 0, (-1j * np.sqrt(2)) / 2], [0, (1j * np.sqrt(2)) / 2, 0]])
    sigma_z_1 = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    unit_1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Calculate the total dimension of the Hilbert space
    # (Product of (2I + 1) for each spin)
    dim_total = 1
    for s_val in spins:
        dim_total *= int(2 * s_val + 1)

    # Preallocate space for L (Lx, Ly, Lz for each nucleus)
    # The dimensions are (3, number_of_nuclei, total_hilbert_space_dim, total_hilbert_space_dim)
    L = np.empty((3, nspins, dim_total, dim_total), dtype=np.complex128)

    # Construct individual spin operators for each nucleus
    for n in range(nspins):  # Iterate through each nucleus (n) for which we're building operators
        Lx_current = 1
        Ly_current = 1
        Lz_current = 1

        for k in range(nspins):  # Iterate through each position in the Kronecker product
            # If this is the nucleus 'n' for which we're building the operator
            if k == n:
                if spins[n] == 0.5:
                    Lx_current = np.kron(Lx_current, sigma_x_half)
                    Ly_current = np.kron(Ly_current, sigma_y_half)
                    Lz_current = np.kron(Lz_current, sigma_z_half)
                elif spins[n] == 1.0:
                    Lx_current = np.kron(Lx_current, sigma_x_1)
                    Ly_current = np.kron(Ly_current, sigma_y_1)
                    Lz_current = np.kron(Lz_current, sigma_z_1)
                else:
                    raise ValueError(f"Unsupported spin quantum number: {spins[n]}. Only 0.5 and 1.0 are supported.")
            # If this is not the nucleus 'n', use the identity matrix for its sub-space
            else:
                if spins[k] == 0.5:
                    Lx_current = np.kron(Lx_current, unit_half)
                    Ly_current = np.kron(Ly_current, unit_half)
                    Lz_current = np.kron(Lz_current, unit_half)
                elif spins[k] == 1.0:
                    Lx_current = np.kron(Lx_current, unit_1)
                    Ly_current = np.kron(Ly_current, unit_1)
                    Lz_current = np.kron(Lz_current, unit_1)
                else:
                    raise ValueError(f"Unsupported spin quantum number: {spins[k]}. Only 0.5 and 1.0 are supported.")

        # Store the constructed operators for nucleus 'n'
        L[0, n] = Lx_current
        L[1, n] = Ly_current
        L[2, n] = Lz_current

    # ref:
    # https://stackoverflow.com/questions/47752324/matrix-multiplication-on-4d-numpy-arrays
    # Construct the Lproduct (Lx_i*Lx_j + Ly_i*Ly_j + Lz_i*Lz_j) for coupling terms
    L_T = L.transpose(1, 0, 2, 3)

    Lproduct = np.tensordot(L_T, L, axes=((1, 3), (0, 2))).swapaxes(1, 2)

    return L[2], Lproduct  # L[2] contains all Lz operators


def _so_sparse(spins):
    """
    Either load a presaved set of spin operators as numpy arrays, or
    calculate them and save them if a presaved set wasn't found.

    Parameters
    ----------
    spins : array-like of float
        The array containing the spin quantum number (I) for each nucleus.

    Returns
    -------
    (Lz, Lproduct) : a tuple of:
        Lz : 3d sparse.COO array of shape (n, dim_total, dim_total) representing
             [Lz1, Lz2, ...Lzn]
        Lproduct : 4d sparse.COO array of shape (n, n, dim_total, dim_total), representing
             an n x n array (cartesian product) for all combinations of
             Lxa*Lxb + Lya*Lyb + Lza*Lzb, where 1 <= a, b <= n.

    Side Effect
    -----------
    Saves the results as .npz files to the bin directory if they were not
    found there.
    """
    # TODO: once nmrsim demonstrates installing via the PyPI *test* server,
    # need to determine how the saved solutions will be handled. For example,
    # part of the final build may be generating these files then testing.
    # Also, need to consider different users with different system capabilities
    # (e.g. at extreme, Raspberry Pi). Some way to let user select, or select
    # for user?
    # Determine a unique identifier for the spins array for caching
    # A simple way for now is to convert to string or a hash

    spins_str = "_".join(map(str, spins.round(1)))  # e.g., "0.5_0.5_1.0"
    filename_Lz = f"Lz_spins_{spins_str}.npz"  # Update filename
    filename_Lproduct = f"Lproduct_spins_{spins_str}.npz"  # Update filename

    bin_path = _bin_path()
    path_Lz = bin_path.joinpath(filename_Lz)
    path_Lproduct = bin_path.joinpath(filename_Lproduct)

    try:
        Lz = sparse.load_npz(path_Lz)
        Lproduct = sparse.load_npz(path_Lproduct)
        return Lz, Lproduct
    except FileNotFoundError:
        print("no SO file ", path_Lz, " found.")
        print(f"creating {filename_Lz} and {filename_Lproduct}")
    # Pass 'spins' to _so_dense
    Lz, Lproduct = _so_dense(spins) # <--- Pass 'spins' here
    Lz_sparse = sparse.COO(Lz)
    Lproduct_sparse = sparse.COO(Lproduct)
    sparse.save_npz(path_Lz, Lz_sparse)
    sparse.save_npz(path_Lproduct, Lproduct_sparse)

    return Lz_sparse, Lproduct_sparse


def hamiltonian_dense(v, J, spins): 
    """
    Calculate the spin Hamiltonian as a dense array.

    Parameters
    ----------
    v : array-like
        list of frequencies in Hz (in the absence of splitting) for each
        nucleus.
    J : 2D array-like
        matrix of coupling constants. J[m, n] is the coupling constant between
        v[m] and v[n].
    spins : array-like of float
        The array containing the spin quantum number (I) for each nucleus.

    Returns
    -------
    H : numpy.ndarray
        a sparse spin Hamiltonian.
    """
    Lz, Lproduct = _so_dense(spins)  # <--- Pass 'spins' here
    H = np.tensordot(v, Lz, axes=1)
    if not isinstance(J, np.ndarray):
        J = np.array(J)
    scalars = 0.5 * J
    H += np.tensordot(scalars, Lproduct, axes=2)
    return H


def hamiltonian_sparse(v, J, spins):
    """
    Calculate the spin Hamiltonian as a sparse array.

    Parameters
    ----------
    v : array-like
        list of frequencies in Hz (in the absence of splitting) for each
        nucleus.
    J : 2D array-like
        matrix of coupling constants. J[m, n] is the coupling constant between
        v[m] and v[n].
    spins : array-like of float
        The array containing the spin quantum number (I) for each nucleus.

    Returns
    -------
    H : sparse.COO
        a sparse spin Hamiltonian.
    """
    Lz, Lproduct = _so_sparse(spins)  # <--- Pass 'spins' here
    # TODO: remove the following lines once tests pass
    # print("From hamiltonian_sparse:")
    # print("Lz is type: ", type(Lz))
    # print("Lproduct is type: ", type(Lproduct))
    assert isinstance(Lz, (sparse.COO, np.ndarray, scipy.sparse.spmatrix))
    # On large spin systems, converting v and J to sparse improved speed of
    # sparse.tensordot calls with them.
    # First make sure v and J are a numpy array (required by sparse.COO)
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    if not isinstance(J, np.ndarray):
        J = np.array(J)
    H = sparse.tensordot(sparse.COO(v), Lz, axes=1)
    scalars = 0.5 * sparse.COO(J)
    H += sparse.tensordot(scalars, Lproduct, axes=2)
    return H


def generate_spin_states(spins):
    """
    Generates all possible combinations of mI states for a given set of spins.

    Args:
        spins (np.array): Array of spin values for each nucleus (e.g., [0.5, 1]).

    Returns:
        list of tuples: Each tuple represents a spin state, e.g., (0.5, 0.5, -1.0).
    """
    state_values = []
    for s_val in spins:
        # mI values for a spin I are -I, -I+1, ..., I-1, I
        state_values.append(np.arange(-s_val, s_val + 0.1, 1.0)) # Add 0.1 for float precision with arange

    # Use itertools.product to get all combinations
    all_states = list(itertools.product(*state_values))
    return all_states


def _transition_matrix_dense(spins):
    """
    Creates a matrix of allowed transitions, as a dense array.

    The (i,j) cells in the matrix indicate whether a transition
    from spin state i to spin state j is allowed or forbidden,
    based on the NMR selection rule: only one nucleus's mI value
    changes by +/- 1.

    Parameters
    ---------
    spins : array-like of float
        A list or array containing the spin quantum number (I) for each nucleus.
        E.g., np.array([0.5, 0.5, 1.0]) for two spin-1/2 and one spin-1 nucleus.

    Returns
    -------
    numpy.ndarray
        A transition matrix that can be used to compute the intensity of
        allowed transitions.
    """

    all_states = generate_spin_states(spins)
    num_states = len(all_states)
    T = np.zeros((num_states, num_states), dtype=int)

    # Optimized loop: only calculate upper triangle and then add the lower.
    # This also avoids checking i == j, as i < j naturally.
    for i in range(num_states):
        for j in range(i + 1, num_states): # Start j from i + 1
            state1 = all_states[i]
            state2 = all_states[j]

            diff_count = 0
            allowed_change_magnitude = True

            for k in range(len(spins)): # Iterate through each nucleus
                diff = state1[k] - state2[k]

                if diff != 0:
                    diff_count += 1
                    # Check if the change is exactly +/- 1.0 for this nucleus
                    if not np.isclose(abs(diff), 1.0):  # Use np.isclose for float comparison
                        allowed_change_magnitude = False
                        break  # Not an allowed +/-1 change for this nucleus

            # If exactly one nucleus changed, and that change was +/-1
            if diff_count == 1 and allowed_change_magnitude:
                T[i, j] = 1  # Set upper triangle
    T += T.T  # Add the lower triangle by transposing and adding

    return T


def secondorder_dense(freqs, couplings, s=None, normalize=True, **kwargs):
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
    s : array-like of float, optional (default = None)
        A list or array containing the spin quantum number (I) for each nucleus.
        If None, all nuclei are assumed to be spin-1/2 (I=0.5).
    normalize: bool
        True if the intensities should be normalized so that total intensity
        equals the total number of nuclei.

    Returns
    -------
    peaklist : [[float, float]...]
        numpy 2D array of [frequency, intensity] pairs.

    Other Parameters
    ----------------
    cutoff : float
        The intensity cutoff for reporting signals (default is 0.001).
    """
    nspins_count = len(freqs) 

    # Determine the actual spins array, defaulting to spin-1/2
    if s is None:
        spins_actual = np.full(nspins_count, 0.5)
    else:
        spins_actual = np.array(s)  

    # Pass 'spins_actual' to hamiltonian_dense and _transition_matrix_dense
    H = hamiltonian_dense(freqs, couplings, spins=spins_actual)  # Pass spins
    E, V = np.linalg.eigh(H)
    V = V.real
    T = _transition_matrix_dense(spins_actual)  # Pass spins
    I = np.square(V.T.dot(T.dot(V)))
    peaklist = _compile_peaklist(I, E, **kwargs)
    if normalize:
        peaklist = normalize_peaklist(peaklist, nspins_count)  # Use nspins_count for total intensity
    return peaklist


def _tm_cache(spins):
    """
    Loads a saved sparse transition matrix if it exists, or creates and saves
    one if it is not.

    Parameters
    ----------
    spins : array-like of float
        The array containing the spin quantum number (I) for each nucleus.

    Returns
    -------
    T_sparse : sparse.COO
        The sparse transition matrix.

    Side Effects
    ------------
    Saves a sparse array to the bin folder if the required array was not
    found there.
    """
    # Determine a unique identifier for the spins array for caching
    spins_str = "_".join(map(str, spins.round(1)))  # e.g., "0.5_0.5_1.0"
    filename = f"T_spins_{spins_str}.npz"  # Update filename

    bin_path = _bin_path()
    path = bin_path.joinpath(filename)
    try:
        T_sparse = sparse.load_npz(path)
        return T_sparse
    except FileNotFoundError:
        print(f"creating {filename}")
    # Pass 'spins' to _transition_matrix_dense
    T_sparse = _transition_matrix_dense(spins)  # <--- Pass 'spins' here
    T_sparse = sparse.COO(T_sparse)
    print("_tm_cache will save on path: ", path)
    sparse.save_npz(path, T_sparse)
    return T_sparse


def _intensity_and_energy(H, spins):
    """
    Calculate intensity matrix and energies (eigenvalues) from Hamiltonian.

    Parameters
    ----------
    H :  numpy.ndarray
        Spin Hamiltonian
    spins : array-like of float
        The array containing the spin quantum number (I) for each nucleus.

    Returns
    -------
    (I, E) : (numpy.ndarray, numpy.ndarray) tuple of:
        I : (relative) intensity 2D array
        V : 1-D array of relative energies.
    """
    E, V = np.linalg.eigh(H)
    V = V.real
    T = _tm_cache(spins)
    I = np.square(V.T.dot(T.dot(V)))
    return I, E


def _compile_peaklist(I, E, cutoff=0.001):
    """
    Generate a peaklist from intensity and energy matrices.

    Parameters
    ----------
    I : numpy.ndarray (2D)
        matrix of relative intensities
    E : numpy.ndarray (1D)
        array of energies
    cutoff : float, optional
        The intensity cutoff for reporting signals.

    Returns
    -------
    numpy.ndarray (2D)
        A [[frequency, intensity]...] peaklist.
    """
    I_upper = np.triu(I)
    E_matrix = np.abs(E[:, np.newaxis] - E)
    E_upper = np.triu(E_matrix)
    combo = np.stack([E_upper, I_upper])
    iv = combo.reshape(2, I.shape[0] ** 2).T
    return iv[iv[:, 1] >= cutoff]


def solve_hamiltonian(H, spins, **kwargs):
    """
    Calculates frequencies and intensities of signals from a spin Hamiltonian
    and number of spins.

    Parameters
    ----------
    H : numpy.ndarray (2D)
        The spin Hamiltonian
    spins : array-like of float
        The array containing the spin quantum number (I) for each nucleus.

    Returns
    -------
    [[float, float]...] numpy 2D array of frequency, intensity pairs.

    Other Parameters
    ----------------
    cutoff : float
        The intensity cutoff for reporting signals (default is 0.001).
    """
    I, E = _intensity_and_energy(H, spins)
    return _compile_peaklist(I, E, **kwargs)


def secondorder_sparse(freqs, couplings, s=None, normalize=True, **kwargs):
    """
    Calculates second-order spectral data (frequency and intensity of signals)
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
    s : array-like of float, optional (default = None)
        A list or array containing the spin quantum number (I) for each nucleus.
        If None, all nuclei are assumed to be spin-1/2 (I=0.5).
    normalize: bool
        True if the intensities should be normalized so that total intensity
        equals the total number of nuclei.

    Returns
    -------
    peaklist : [[float, float]...] numpy 2D array
        of [frequency, intensity] pairs.

    Other Parameters
    ----------------
    cutoff : float
        The intensity cutoff for reporting signals (default is 0.001).
    """
    nspins_count = len(freqs)  # Keep for normalize_peaklist

    # Determine the actual spins array, defaulting to spin-1/2
    if s is None:
        spins_actual = np.full(nspins_count, 0.5)
    else:
        spins_actual = np.array(s)  

    H = hamiltonian_sparse(freqs, couplings, spins=spins_actual)  
    peaklist = solve_hamiltonian(H.todense(), spins=spins_actual, **kwargs) 
    if normalize:
        peaklist = normalize_peaklist(peaklist, nspins_count)
    return peaklist


def qm_spinsystem(freqs, couplings, s=None, cache=CACHE, sparse=SPARSE, normalize=True, **kwargs):
    """
    Calculates second-order spectral data (frequency and intensity of signals)
    for *n* spin-half nuclei.

    Currently, n is capped at 11 spins.

    Parameters
    ----------
    freqs : [float...]
        a list of *n* nuclei frequencies in Hz.
    couplings : array-like
        An *n, n* array of couplings in Hz. The order of nuclei in the list
        corresponds to the column and row order in the matrix, e.g.
        couplings[0][1] and [1]0] are the J coupling between the nuclei of
        freqs[0] and freqs[1].
    s : array-like of float, optional (default = None)
        A list or array containing the spin quantum number (I) for each nucleus.
        If None, all nuclei are assumed to be spin-1/2 (I=0.5).
    normalize: bool (optional keyword argument; default = True)
        True if the intensities should be normalized so that total intensity
        equals the total number of nuclei.

    Returns
    -------
    peaklist : [[float, float]...] numpy 2D array
        of [frequency, intensity] pairs.

    Other Parameters
    ----------------
    cache: bool (default = nmrsim.qm.CACHE)
        Whether caching of partial solutions (for acceleration) is allowed.
        Currently CACHE = True, but this provides a hook to modify nmrsim for
        platforms such as Raspberry Pi where storage space is a concern.
    sparse: bool (default = nmrsim.qm.SPARSE)
        Whether the pydata sparse library for sparse matrices is available.
        Currently SPARSE = True, but this provides a hook to modify nmrsim
        should the sparse library become unavailable (see notes).
    cutoff : float
        The intensity cutoff for reporting signals (default is 0.001).

    Notes
    -----
    With numpy.matrix marked for deprecation, the scipy sparse array
    functionality is on shaky ground, and the current recommendation is to use
    the pydata sparse library. In case a problem arises in the numpy/scipy/
    sparse ecosystem, SPARSE provides a hook to use a non-sparse-dependent
    alternative.
    """
    if not (cache and sparse):
        # Pass 's' explicitly, then remaining kwargs
        return secondorder_dense(freqs, couplings, s=s, normalize=normalize, **kwargs)
    # Pass 's' explicitly, then remaining kwargs
    return secondorder_sparse(freqs, couplings, s=s, normalize=normalize, **kwargs)
