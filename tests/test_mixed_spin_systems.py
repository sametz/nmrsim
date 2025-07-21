import numpy as np
from nmrsim import SpinSystem 
from nmrsim.math import ppm_to_hz_from_nuclei_info, reduce_peaks 


# --- Common Test Parameters (if any, otherwise defined per test) ---
SPECTROMETER_1H_MHZ = 600 # Spectrometer frequency in MHz


# --- Test 1: Direct Hz Frequency Input ---

# Input Parameters for Test 1
V_FREQUENCIES_HZ_TEST1 = np.array([600, 276.30236475])
S_SPINS_TEST1 = np.array([0.5, 1])
J_COUPLINGS_TEST1 = np.array([[0, 10],
                              [10, 0]])

# Expected Output for Test 1 (from your manual Hz input script)
EXPECTED_FINAL_SIGNALS_PPM_TEST1 = np.array([
    [0.45190926, 0.95566885],
    [0.45217855, 0.95747711],
    [0.46858387, 1.04298499],
    [0.46882933, 1.04482266],
    [0.98359469, 1.04433115],
    [1.00051476, 0.99770018],
    [1.01692008, 0.95701501]
])


# --- TEST FUNCTION ---
def test_mixed_spin_system_final_signals_ppm():
    """
    Tests the final signals (ppm scale) generation for a mixed spin system
    with direct Hz frequency input.
    """
    # 1. Initialize SpinSystem with the direct Hz frequencies
    system = SpinSystem(V_FREQUENCIES_HZ_TEST1, J_COUPLINGS_TEST1, S_SPINS_TEST1)

    # 2. Get the unnormalized peaklist (to match your processing steps)
    sim_signals_raw = system.peaklist(normalize=False)

    # Ensure it's a NumPy array for consistent processing
    sim_signals_np = np.array(sim_signals_raw)

    # 3. Convert frequencies to PPM scale using the spectrometer's 1H frequency
    # (as done in your script: `ppm_scale = sim_signals[:,0]/spectrometer_1H_MHz`)
    ppm_scale = sim_signals_np[:, 0] / SPECTROMETER_1H_MHZ
    intensities = sim_signals_np[:, 1]

    # 4. Combine ppm and intensities, then sort by ppm
    signals_combined = np.vstack((ppm_scale, intensities)).T
    actual_final_signals = signals_combined[signals_combined[:, 0].argsort(kind='mergesort')]

    # 5. Compare with the expected values
    np.testing.assert_allclose(actual_final_signals, EXPECTED_FINAL_SIGNALS_PPM_TEST1, atol=1e-8)  # Adjust atol if needed

# ---------------------------------------------------------------------------------------------------------------------

# --- Test 2: PPM Input using ppm_to_hz_from_nuclei_info ---

# Input Parameters for Test 2
PPM_POSITIONS_INPUT_TEST2 = np.array([1.0, 3.0]) # 1H at 1 ppm, D at 3 ppm
NUCLEI_TYPES_INPUT_TEST2 = ['1H', '2H'] # Using '2H' for Deuterium
# Ensure these match the values used in your math.py tests to ensure consistency
GYROMAGNETIC_RATIOS_INPUT_MHZ_PER_T_TEST2 = [42.577478461, 6.53569888]
S_SPINS_TEST2 = np.array([0.5, 1])
J_COUPLINGS_TEST2 = np.array([[0, 10],
                              [10, 0]])

# Expected Output for Test 2 (This should be identical to Test 1's output
# since the input parameters for v are designed to yield the same result)
EXPECTED_FINAL_SIGNALS_PPM_TEST2 = np.array([
    [0.45190926, 0.95566885],
    [0.45217855, 0.95747711],
    [0.46858387, 1.04298499],
    [0.46882933, 1.04482266],
    [0.98359469, 1.04433115],
    [1.00051476, 0.99770018],
    [1.01692008, 0.95701501]
])


def test_mixed_spin_system_full_pipeline_with_ppm_to_hz():
    """
    Tests the full simulation pipeline starting from ppm input,
    using ppm_to_hz_from_nuclei_info, and verifying the final ppm output.
    """
    # 1. Convert ppm_positions to Hz using the new function
    v_frequencies_hz = ppm_to_hz_from_nuclei_info(
        PPM_POSITIONS_INPUT_TEST2, NUCLEI_TYPES_INPUT_TEST2,
        GYROMAGNETIC_RATIOS_INPUT_MHZ_PER_T_TEST2, SPECTROMETER_1H_MHZ
    )

    # 2. Initialize SpinSystem with the calculated Hz frequencies
    system = SpinSystem(v_frequencies_hz, J_COUPLINGS_TEST2, S_SPINS_TEST2)

    # 3. Get the unnormalized peaklist
    sim_signals_raw = system.peaklist(normalize=False)
    sim_signals_np = np.array(sim_signals_raw)

    # 4. Convert frequencies to PPM scale using the spectrometer's 1H frequency
    ppm_scale = sim_signals_np[:, 0] / SPECTROMETER_1H_MHZ
    intensities = sim_signals_np[:, 1]

    # 5. Combine ppm and intensities, then sort by ppm
    signals_combined = np.vstack((ppm_scale, intensities)).T
    actual_final_signals = signals_combined[signals_combined[:, 0].argsort(kind='mergesort')]

    # 6. Compare with the expected values
    np.testing.assert_allclose(actual_final_signals, EXPECTED_FINAL_SIGNALS_PPM_TEST2, atol=1e-8)

# --- New Test: DMSO CD3CD2HSO isotopomer Full Spectrum Simulation (Proton Quintet + Deuterium Doublet) ---

# Input Parameters for DMSO CD3CD2HSO isotopomer
PPM_POSITIONS_DMSO_FULL = np.array([2.7, 2.7, 2.7])  # 1H at 2.7 ppm in proton spectrum. 2H, 2H at 2.7 ppm in deuterium spectrum
NUCLEI_TYPES_DMSO_FULL = ['1H', '2H', '2H']
GYROMAGNETIC_RATIOS_MHZ_PER_T_DMSO_FULL = [42.577478461, 6.53569888, 6.53569888]
S_SPINS_DMSO_FULL = np.array([0.5, 1, 1])  # 1H (I=1/2), two 2H (I=1)
J_COUPLINGS_DMSO_FULL = np.array([[0, 1.9, 1.9],  # J(1H-2H) = 1.9 Hz
                                  [1.9, 0, 0],
                                  [1.9, 0, 0]])

# **EXPECTED OUTPUT**: This is the output that includes both the proton quintet and the deuterium doublet(s) in PPM.
EXPECTED_FINAL_SIGNALS_PPM_DMSO_FULL = np.array([
    [0.41286912, 11.97649277],
    [0.41603578, 12.02351875],
    [2.69367106, 1.00392891],
    [2.69683992, 2.00391223],
    [2.70000585, 2.99998146],
    [2.70317324, 1.99607466],
    [2.70633771, 0.99609122]
])


def test_dmso_CD3CD2HSO_full_spectrum_simulation():
    """
    Tests the full simulation of the DMSO CD3CD2HSO isotopomer, including
    both the 1H quintet and 2H doublet signals in the output.
    """
    # 1. Convert ppm_positions to Hz using ppm_to_hz_from_nuclei_info
    v_frequencies_hz = ppm_to_hz_from_nuclei_info(
        PPM_POSITIONS_DMSO_FULL, NUCLEI_TYPES_DMSO_FULL,
        GYROMAGNETIC_RATIOS_MHZ_PER_T_DMSO_FULL, SPECTROMETER_1H_MHZ
    )

    # 2. Initialize SpinSystem
    system = SpinSystem(v_frequencies_hz, J_COUPLINGS_DMSO_FULL, S_SPINS_DMSO_FULL)

    # 3. Get the unnormalized peaklist and reduce it with the specified tolerance
    sim_signals_raw = system.peaklist(normalize=False)
    sim_reduced_signals = reduce_peaks(sim_signals_raw, tolerance=0.01)

    # Convert reduced signals to NumPy array for further processing
    sim_reduced_signals_np = np.array(sim_reduced_signals)

    # 4. Convert frequencies to PPM scale (relative to 1H spectrometer frequency)
    ppm_scale = sim_reduced_signals_np[:, 0] / SPECTROMETER_1H_MHZ
    intensities = sim_reduced_signals_np[:, 1]

    # 5. Combine ppm and intensities, then sort by ppm
    signals_combined = np.vstack((ppm_scale, intensities)).T
    #actual_final_signals = signals_combined[signals_combined[:, 0].argsort(kind='mergesort')]

    # 6. Compare with the expected values
    # Use a small atol for floating-point comparison.
    # Adjust if necessary based on your exact numerical precision.
    np.testing.assert_allclose(signals_combined, EXPECTED_FINAL_SIGNALS_PPM_DMSO_FULL, atol=1e-8)
