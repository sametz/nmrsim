"""A collection of functions for processing simulated NMR spectra.

Terms used:
signal: a pair (e.g. tuple) of frequency, intensity values
peaklist: a list (or 1D-array-like) of signals.

Provides the following functions:

* add_peaks: combines a list of signals into one signal of average frequency
  and summed intensity.

* reduce_peaks: processes a peaklist so that signals within a frequency
  tolerance are added together.

* normalize_peaklist: scales a peaklist so that intensities add to a specific
  value.
* lorentz: given a frequency, a signal and a linewidth, calculates an
  intensity. Used to calculate Lorentzian lineshapes for signals.

* get_intensity: given a lineshape and a frequency, find the intensity at the
  datapoint closest to that frequency.
"""
import numpy as np


def add_peaks(plist):
    """
    Reduces a list of (frequency, intensity) tuples to an
    (average frequency, total intensity) tuple.

    Parameters
    ----------
    plist: [(float, float)...]
        a list of (frequency, intensity) tuples

    Returns
    -------
    (float, float)
        a tuple of (average frequency, total intensity)
    """
    v_total = 0
    i_total = 0
    for v, i in plist:
        v_total += v
        i_total += i
    return v_total / len(plist), i_total


def reduce_peaks(plist_, tolerance=0):
    """
    Takes a list of (frequency, intensity) peaks and combines those whose frequencies
    are within a specified tolerance limit.

    This function is used to simplify peak lists by merging very close or
    numerically identical peaks that should appear as a single signal in a spectrum.

    Parameters
    ----------
    plist_ : list of (float, float) tuples or np.ndarray
        The input peak list. Can be:
        - A standard Python list of (frequency, intensity) tuples.
        - A 2D NumPy array where each row is a [frequency, intensity] pair.
    tolerance : float
        The maximum absolute difference in frequency (x-value) for two peaks
        to be considered "close enough" to be combined. Frequencies are in Hz.

    Returns
    -------
    list of (float, float) tuples
        A new list of (frequency, intensity) tuples where closely spaced
        peaks have been combined. Frequencies are sorted in ascending order.
    """

    # Convert NumPy array input to a list of tuples to ensure consistent processing
    # by the subsequent sorting and reduction logic, which expects a list of tuples.
    if isinstance(plist_, np.ndarray):
        # Handle empty NumPy array explicitly to prevent errors in list comprehension.
        if plist_.size == 0:
            return []
        
        # Convert each NumPy array row (e.g., [frequency, intensity]) into a tuple.
        # This is crucial for `sorted()` to work correctly on the elements.
        plist = [tuple(row) for row in plist_]
    else:
        # If the input is not a NumPy array, assume it's already a list of tuples
        # or a compatible sequence, and proceed directly. This maintains
        # backward compatibility with original usage.
        plist = plist_

    # Sorts the peak list by frequency (the first element of each tuple).
    # This is essential for the reduction algorithm to correctly group
    # adjacent peaks within the specified tolerance.
    plist_sorted = sorted(plist)

    res = []
    work = []  
    
    for peak in plist_sorted:
        if not work:
            work.append(peak)
            continue
        
        # Check if the current peak's frequency is within tolerance of the last
        # peak added to the `work` group.
        if peak[0] - work[-1][0] <= tolerance:
            work.append(peak)
        else:
            res.append(add_peaks(work))
            work = [peak]
    # After the loop finishes, there might be peaks left in `work` that
    # haven't been added to `res`. Combine and add them.            
    if work:  
        res.append(add_peaks(work))

    return res


def _normalize(intensities, n=1):
    """
    Scale a list of intensities so that they sum to the total number of
    nuclei.

    Parameters
    ---------
    intensities : [float...]
        A list of intensities.
    n : int (optional)
        Number of nuclei (default = 1).
    """
    factor = n / sum(intensities)
    intensities[:] = [factor * i for i in intensities]


def normalize_peaklist(peaklist, n=1):
    """
    Normalize the intensities in a peaklist so that total intensity equals
    value n (nominally the number of nuclei giving rise to the signal).

    Parameters
    ---------
    peaklist : [(float, float)...]
        a list of (frequency, intensity) tuples.
    n : int or float (optional)
        total intensity to normalize to (default = 1).
    """
    freq, int_ = [x for x, y in peaklist], [y for x, y in peaklist]
    _normalize(int_, n)
    return list(zip(freq, int_))


def lorentz(v, v0, I, w):
    """
    A lorentz function that takes linewidth at half intensity (w) as a
    parameter.

    When `v` = `v0`, and `w` = 0.5 (Hz), the function returns intensity I.

    Arguments
    ---------
    v : float
        The frequency (x coordinate) in Hz at which to evaluate intensity (y
        coordinate).
    v0 : float
        The center of the distribution.
    I : float
        the relative intensity of the signal
    w : float
        the peak width at half maximum intensity

    Returns
    -------
    float
        the intensity (y coordinate) for the Lorentzian distribution
        evaluated at frequency `v`.
    """
    # Adding a height scaling factor so that peak intensities are lowered as
    # they are more broad. If I is the intensity with a default w of 0.5 Hz:
    scaling_factor = 0.5 / w  # i.e. a 1 Hz wide peak will be half as high
    return scaling_factor * I * ((0.5 * w) ** 2 / ((0.5 * w) ** 2 + (v - v0) ** 2))


def add_lorentzians(linspace, peaklist, w):
    """
    Given a numpy linspace, a peaklist of (frequency, intensity)
    tuples, and a linewidth, returns an array of y coordinates for the
    total line shape.

    Arguments
    ---------
    linspace : array-like
        Normally a numpy.linspace of x coordinates corresponding to frequency
        in Hz.
    peaklist : [(float, float)...]
        A list of (frequency, intensity) tuples.
    w : float
        Peak width at half maximum intensity.

    Returns
    -------
    [float...]
        an array of y coordinates corresponding to intensity.
    """
    # TODO: consider naming, and confusion with .math.add_peaks
    # TODO: function looks clunky. Refactor?
    result = lorentz(linspace, peaklist[0][0], peaklist[0][1], w)
    for v, i in peaklist[1:]:
        result += lorentz(linspace, v, i, w)
    return result


def get_intensity(lineshape, x):
    """
    A crude method to find the intensity of data point closest to
    frequency x. Better: interpolate between two data points if match isn't
    exact (TODO?)

    Parameters
    ----------
    lineshape : tuple of (x, y) arrays for frequency, intensity data
    x : frequency lookup

    Returns
    -------
    float : the intensity at that frequency
    """
    nearest_x_index = np.abs(lineshape[0] - x).argmin()
    return lineshape[1][nearest_x_index]


def get_maxima(lineshape):
    """
    Crude function that returns maxima in the lineshape.

    Parameters
    ----------
    lineshape : tuple of frequency, intensity arrays

    Returns
    -------
    a list of (frequency, intensity) tuples for individual maxima.
    """
    res = []
    for n, val in enumerate(lineshape[1][1:-2]):
        index = n + 1  # start at lineshape[1][1]
        lastvalue = lineshape[1][index - 1]
        nextvalue = lineshape[1][index + 1]

        if lastvalue < val and nextvalue < val:
            print("MAXIMUM FOUND AT: ")
            print((lineshape[0][index], val))
            res.append((lineshape[0][index], val))
    return res


def ppm_to_hz_from_nuclei_info(ppm_positions, nuclei_types, gyromagnetic_ratios_MHz_per_T, spectrometer_1H_MHz):
    """
    Converts an array of ppm chemical shifts to absolute frequencies (Hz) based on
    nuclear types and user-provided gyromagnetic ratios, given a reference 1H spectrometer frequency.

    Parameters
    ----------
    ppm_positions : np.ndarray
        Array of chemical shifts in ppm for each nucleus.
    nuclei_types : list of str
        List of nuclear symbols (e.g., '1H', '2H', '13C') corresponding to ppm_positions.
        This list's order should match `gyromagnetic_ratios_MHz_per_T`.
    gyromagnetic_ratios_MHz_per_T : list of float
        List of gyromagnetic ratios in MHz/Tesla for each nucleus in the order
        they appear in `nuclei_types`. This allows users to input exotic nuclei.
    spectrometer_1H_MHz : float
        The reference frequency of the spectrometer in MHz (e.g., 600.0 MHz for 1H).
        This value, along with the hardcoded 1H gyromagnetic ratio, is used to calculate
        the B0 magnetic field strength.

    Returns
    -------
    np.ndarray
        An array of absolute frequencies in Hz for each nucleus, suitable for `SpinSystem(v, ...)`.

    Raises
    ------
    ValueError
        If input array lengths do not match.
    """

    # Hardcoded 1H gyromagnetic ratio in MHz/Tesla
    # This is a fundamental constant used to calculate B0 from the spectrometer's 1H frequency.
    _GAMMA_1H_MHZ_PER_T = 42.577478461 

    if not (len(ppm_positions) == len(nuclei_types) == len(gyromagnetic_ratios_MHz_per_T)):
        raise ValueError("All input arrays (ppm_positions, nuclei_types, gyromagnetic_ratios_MHz_per_T) "
                         "must have the same length.")

    # Calculate the B0 field strength in Tesla using the hardcoded 1H gamma
    # and the user-provided spectrometer's 1H frequency.
    B0_Tesla = spectrometer_1H_MHz / _GAMMA_1H_MHZ_PER_T

    v_calculated = []
    for i, ppm in enumerate(ppm_positions):
        gamma_nucleus_MHz_per_T = gyromagnetic_ratios_MHz_per_T[i]

        # Calculate the Larmor frequency of *this specific nucleus* at the calculated B0 field, in Hz
        nucleus_larmor_freq_Hz = gamma_nucleus_MHz_per_T * B0_Tesla * 1_000_000 # Convert MHz to Hz

        # Calculate the actual frequency of the signal in Hz based on its ppm shift.
        chemical_shift_in_Hz = ppm * (nucleus_larmor_freq_Hz / 1_000_000) # (Hz/ppm) * ppm

        v_calculated.append(chemical_shift_in_Hz)

    return np.array(v_calculated)