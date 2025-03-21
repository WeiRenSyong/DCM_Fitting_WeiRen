# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.signal import find_peaks

# %% Define Some Useful Functions
######################################
#### Input:                       ####
#### (Column 1: Time)             ####
#### Column 1: Frequency in Hz    ####
#### Column 2: Magnitude in dB    ####
#### Column 3: Phase in degrees   ####
######################################
#### Output:                      ####
#### Column 1 : Frequency in Hz   ####
#### Column 2 : Magnitude in lin  ####
#### Column 3 : Phase in radians  ####
######################################
print(f"Define Organize_Date function...")
def Organize_Data(raw_data):
    """
    Function to organize raw data to:
        - Column 1 : Frequency in Hz
        - Column 2 : Magnitude in linear
        - Column 3 : Phase in radians

    Parameters:
        raw_data (numpy array): N×3 (Nx4) matrix containing:
            - (Column 1: Time)
            - Column 1: Frequency in Hz
            - Column 2: Magnitude in dB
            - Column 3: Phase in degrees
    """
    # Ensure the input is a NumPy array
    raw_data = np.array(raw_data, dtype=str)

    # Check if the raw_data has an extra time column (4 columns instead of 3)
    if raw_data.shape[1] == 4:
        raw_data = raw_data[:, 1:]  # Ignore the first column (time column)
        print('The raw data are Nx4 matrix.')
    else:
        print('The raw data are Nx3 matrix.')

    # Convert each column to numeric values, removing non-numeric rows
    freq, mag, phase = [], [], []
    for column in raw_data:
        try:
            f = float(column[0])
            m = float(column[1])
            p = float(column[2])
            freq.append(f)
            mag.append(m)
            phase.append(p)
        except ValueError:
            continue  # Skip rows with non-numeric entries
    freq = np.array(freq)
    mag = np.array(mag)
    phase = np.array(phase)

    # Convert frequency to Hz if given in GHz
    if np.max(freq) < 1e9:  # Assuming min frequency is 1 GHz
        freq_Hz = freq * 1e9
    else:
        freq_Hz = freq

    # Convert magnitude from dB to linear scale
    if np.min(mag) < 0: # If min mag < 0, assume dB
        mag_linear = 10 ** (mag / 20)   # The reading of the VNA is a ratio of voltage, so 20 is here
    else:
        mag_linear = mag

    # Convert phase to radians if in degrees
    if np.max(np.abs(phase)) > 2 * np.pi:  # If max phase > 2π, assume degrees
        phase_rad = np.deg2rad(phase)
    else:
        phase_rad = phase

    # Stack the processed data
    organized_data = np.column_stack((freq_Hz, mag_linear, phase_rad))

    return organized_data
###########################################
#### Input:                            ####
#### Column 1: Frequency in Hz         ####
#### Column 2: Magnitude in lin        ####
#### Column 3: Phase in rad            ####
###########################################
#### Output:                           ####
#### 1. Frequency (GHz) vs Mag (dB)    ####
#### 2. Frequency (GHz) vs Phase (deg) ####
#### 3. Real(S21) vs Imag(S21)         ####
###########################################
print(f"Define the Plot_Data function...")
def Plot_Data(organized_data):
    """
    Function to plot raw data in three subplots:
    1. Frequency (GHz) vs Magnitude (dB)
    2. Frequency (GHz) vs Phase (degrees)
    3. Real(S21) vs Imag(S21)

    Parameters:
        organized_data: N×3 matrix containing:
            - Column 1: Frequency in Hz
            - Column 2: Magnitude in linear
            - Column 3: Phase in radians
    """
    # Extract values for plotting
    freq_Hz = organized_data[:, 0]
    mag_linear = organized_data[:, 1]
    phase_rad = organized_data[:, 2]

    # Convert frequency to GHz for plotting
    freq_GHz = freq_Hz / 1e9

    # Convert magnitude to dB for plotting
    mag_dB = 20 * np.log10(mag_linear)
    
    # Convert phase to degree for plotting
    phase_deg = np.rad2deg(phase_rad)
    
    # Compute real and imaginary parts of S21
    S21 = mag_linear * np.exp(1j * phase_rad)
    real_S21 = S21.real
    imag_S21 = S21.imag

    # Create figure
    fig = plt.figure(figsize=(10, 5))

    # Plot Frequency vs Magnitude (dB) - Left Top
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(freq_GHz, mag_dB, 'o', markersize=3, label="Magnitude (dB)", color='blue')
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title("Frequency vs Magnitude")
    ax1.grid(True)
    ax1.legend()
    ax1.set_xticks([np.min(freq_GHz), (np.min(freq_GHz)+np.max(freq_GHz))/2, np.max(freq_GHz)])

    # Plot Frequency vs Phase (degrees) - Left Bottom
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.plot(freq_GHz, phase_deg, 'o', markersize=3, label="Phase (deg)", color='orange')
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("Phase (deg)")
    ax2.set_title("Frequency vs Phase")
    ax2.grid(True)
    ax2.legend()
    ax2.set_xticks([np.min(freq_GHz), (np.min(freq_GHz)+np.max(freq_GHz))/2, np.max(freq_GHz)])

    # Plot Real(S21) vs Imag(S21) - Right
    ax3 = fig.add_subplot(1, 2, 2)  # Ensures a single, right-side wide plot
    ax3.plot(real_S21, imag_S21, 'o', markersize=3, label="S21 Complex Plane", color='green')
    ax3.set_xlabel("Real(S21)")
    ax3.set_ylabel("Imag(S21)")
    ax3.set_title("S21 Complex Plane")
    ax3.grid(True)
    ax3.legend()
    ax3.axis("equal")  # Ensures proper scaling of real/imag axes

    # Improve layout spacing
    plt.tight_layout()
    plt.show()

    #######################################

# %% Preprocessing (1) - fit_cable_delay, fit_alpha
#######################################
#### Input:                        ####
#### para 1 : organized_data       ####
#######################################
#### Output:                       ####
#### para 1 : cable_delay(tau)     ####
#### para 2 : phase offset (alpha) ####
#######################################
print(f"Define fit_cable_delay function...")
def fit_cable_delay(organized_data):
    """
    Function to finds the cable delay to remove the slope of the phase response:
    1. cable_delay

    Parameters:
        organized_data (numpy array): N×3 matrix containing:
            - Column 1: Frequency in Hz
            - Column 2: Magnitude in linear
            - Column 3: Phase in radians
    """
    # Extract values for plotting
    freq_Hz = organized_data[:, 0]
    mag_linear = organized_data[:, 1]
    phase_rad = organized_data[:, 2]

    # Unwrap phase to prevent discontinuities
    phase_rad = np.unwrap(phase_rad)  

    # Select the wings first and last few points
    num_points = 2  # Number of points to use from each wing
    freq_bg = np. concatenate((freq_Hz[:num_points], freq_Hz[-num_points:]))
    phase_bg = np.concatenate((phase_rad[:num_points], phase_rad[-num_points:]))

    # Perform linear fit to get A(slope) and B (offset)
    minus_two_pi_f_tau, alpha = np.polyfit(freq_bg, phase_bg, 1)    # Linear fit (degree = 1)

    tau = minus_two_pi_f_tau / (2 * np.pi)
    print(f"cable_delay (tau) is {tau*1e9:.2f} ns")
    print(f"phase offset (alpha) is {np.rad2deg(alpha):.2f} deg")

    return tau, alpha
#######################################
#### Input:                        ####
#### para 1 : organized_data       ####
#### para 2 : cable_delay(tau)     ####
#### para 3 : phase offset (alpha) ####
#######################################
#### Output:                       ####
#### reorganized_data              ####
#######################################
print(f"Define remove_cable_delay function...")
def remove_cable_delay(organized_data, tau, alpha):
    """
    Function to finds the cable delay to remove the slope of the phase response:
    1. z_corrected : Complex-valued data after removing the cable delay.

    Parameters:
    1. organized_data (numpy array): N×3 matrix containing:
        - Column 1: Frequency in Hz
        - Column 2: Magnitude in linear
        - Column 3: Phase in radians
    2. tau_fit, phi0_fit
    """

    print(f"Preprocess the phase correction...")
    # Extract values for plotting
    freq_Hz = organized_data[:, 0]
    mag_linear = organized_data[:, 1]
    phase_rad = organized_data[:, 2]

    # Compute the background phase caused by cable delay
    background_phase = -2 * np.pi * freq_Hz * tau - alpha # in radians

    z = mag_linear * np.exp(1j * phase_rad)
    # Remove the background phase
    z_corrected = z * np.exp(1j * background_phase)  # Multiply by exp(+j*phase) to correct

    # Arrange z_corrected     
    freq_Hz = freq_Hz
    mag_linear = np.abs(z_corrected)
    phase_rad = np.angle(z_corrected)

    # Stack the processed data
    remove_cable_delay_organized_data = np.column_stack((freq_Hz, mag_linear, phase_rad))

    return remove_cable_delay_organized_data

#%% Preprocessing (2) - remove_mg_bg
#######################################
#### Input:                        ####
#### para 1 : organized_data       ####
#######################################
#### Output:                       ####
#### reorganized_data              ####
#######################################
print(f"Define remove_mag_bg function...")
def remove_mag_bg(organized_data):
    
    print(f"Preprocess the background mag removal...")
    
    # Extract values for plotting
    freq_Hz = organized_data[:, 0]
    mag_linear = organized_data[:, 1]
    phase_rad = organized_data[:, 2]

    # Unwrap phase to prevent discontinuities
    # phase_rad = np.unwrap(np.angle(phase_rad))  

    # Select the wings first and last few points
    num_points = 2  # Number of points to use from each wing
    freq_bg = np. concatenate((freq_Hz[:num_points], freq_Hz[-num_points:]))
    mag_bg = np.concatenate((mag_linear[:num_points], mag_linear[-num_points:]))

    # Perform linear fit to get A(slope) and B (offset)
    q = np.polyfit(freq_bg, mag_bg, 1)    # Linear fit (degree = 1)

    # Define the background to substrate (Ax + B)
    background_mag = np.polyval(q, freq_Hz)

    z = mag_linear * np.exp(1j * phase_rad)
    # Remove the background mag
    z_corrected = z / background_mag  # Multiply by exp(+j*phase) to correct

    # Arrange z_corrected     
    freq_Hz = freq_Hz
    mag_linear = np.abs(z_corrected)
    phase_rad = np.angle(z_corrected)

    # Stack the processed data
    remove_mag_bg_data = np.column_stack((freq_Hz, mag_linear, phase_rad))

    return remove_mag_bg_data

# # %% Set a Testing Point - Check the complex circle after removing environment factors
# ################################
# #### Use to be a Test Point ####
# #### 1. Correct the phase   ####
# #### 2. Correct the mag     ####
# ################################ 
# # Define folder path and file name separately
# folder_path = r"C:\Users\user\Documents\GitHub\Cooldown_56_Line_5-NW_Ta2O5_15nm_01\2024_10_17_Final_Data_Modify_Fitting_Codes\2024_10_14_Most_Recent_Datasets_Organizing\Resonator_2_5p837GHz_Nf100"
# file_name = r"NW_Ta2O5_15nm_01_5p837GHz_-62dBm_-1000mK.csv"

# # Combine folder path and file name
# file_path = os.path.join(folder_path, file_name)

# # Load data
# raw_data = pd.read_csv(file_path)

# # Print confirmation
# print(f"In the folder: {folder_path}")
# print(f"Load data from: {file_name}")

# organized_data = Organize_Data(raw_data)
# Plot_Data(organized_data)

# tau_fit, phi0_fit = fit_cable_delay(organized_data)
# remove_cable_delay_data = remove_cable_delay(organized_data, tau_fit, phi0_fit)
# Plot_Data(remove_cable_delay_data)

# remove_mag_bg_data = remove_mag_bg(remove_cable_delay_data)
# Plot_Data(remove_mag_bg_data)

# %% Initial Guessing
#####################################################
#### Input:                                      ####
#### para 1 : reorganized_data                   ####
#####################################################
#### Output: Initial Guessing Fitting Parameters ####
#### 1. resonance frequency (fc)                 ####
#### 2. mismatch phase (phi)                     ####
#### 3. loaded quality factor (Q)                ####
#### 4. coupling quality factor (Qc)             ####
#####################################################
print(f"Define find_fc function...")
def find_fc(organized_data):
    """
    Function to finds the resonance frequency (fc):
    1. Resonance frequency (fc) in Hz.

    Parameters:
    1. organized_data (numpy array): N×3 matrix containing:
        - Column 1: Frequency in Hz
        - Column 2: Magnitude in lin
        - Column 3: Phase in rad
    """
    # Extract frequency and magnitude
    freq_Hz = organized_data[:, 0]  # frequency in Hz
    mag_lin = organized_data[:, 1]  # magnitude in lin

    # Find index of minimum magnitude
    min_idx = np.argmin(mag_lin)

    # Get resonance frequency
    fc = freq_Hz[min_idx]

    return fc

print(f"Define find_phi function...")
def find_phi(organized_data, guess_fc):
    """
    Function to finds the phase mismatch (phi):
    1. phi mismatch (phi) in rad.

    Parameters:
    1. organized_data : N×3 matrix containing:
        - Column 1: Frequency in Hz
        - Column 2: Magnitude in lin
        - Column 3: Phase in rad
    2. guess_fc: Estimated resonance frequency (Hz)
    """
    # Extract frequency, magnitude, and phase
    freq_Hz = organized_data[:, 0]  # Frequency in Hz
    mag_lin = organized_data[:, 1]  # Magnitude in linear scale
    phase_rad = organized_data[:, 2]  # Phase in radians

    # Step (1): Calculate complex S21
    S21 = mag_lin * np.exp(1j * phase_rad)

    # Step (2): Find the index where frequency is closest to guess_fc
    guess_fc = find_fc(organized_data)
    idx_fc = np.argmin(np.abs(freq_Hz - guess_fc))
    S21_fc = S21[idx_fc]  # S21 value at resonance frequency

    # Step (3): Compute the angle of the extension line from (1,0) to S21_fc
    phi = np.pi - np.angle(S21_fc - 1)  # Phase of (S21_fc - (1+0j)) in rad
    return phi

print(f"Define find_Q function...")
def find_Q(organized_data):
    """
    Function to find the quality factor (Q) = fc/FWHM:

    Parameters:
    1. organized_data : N×3 matrix containing:
        - Column 1: Frequency in Hz
        - Column 2: Magnitude in lin
        - Column 3: Phase in rad
    """
    freq_Hz = organized_data[:, 0]  # Frequency in Hz
    mag_lin = organized_data[:, 1]  # Magnitude in linear scale

    # Step (1): Find resonance frequency fc
    fc = find_fc(organized_data)

    # Step (2): Compute half maximum magnitude
    peak_idx, _ = find_peaks(mag_lin)  # Find peaks
    if len(peak_idx) == 0:
        raise ValueError("No peak found in absorption spectrum")
    res_idx = peak_idx[np.argmax(mag_lin[peak_idx])]
    background = (mag_lin[1] + mag_lin[-1]) / 2
    half_max = (background + np.min(mag_lin)) / 2
    left_idx = np.where(mag_lin[:res_idx] >= half_max)[0]
    right_idx = np.where(mag_lin[res_idx:] >= half_max)[0] + res_idx
    if len(left_idx) == 0 or len(right_idx) == 0:
        raise ValueError("FWHM calculation failed")
    
    f_left = freq_Hz[left_idx[-1]]
    f_right = freq_Hz[right_idx[0]]
    FWHM = np.abs(f_right - f_left)  # Full Width at Half Maximum
    
    # Step (3): Calculate Q = fc / FWHM
    Q = fc / FWHM if FWHM > 0 else np.nan

    return Q

print(f"Define find_Qc function...")
def find_Qc(organized_data, guess_fc):
    """
    Function to find the coupling quality factor (Qc) :

    Parameters:
        organized_data:
            - Column 0: Frequency in Hz
            - Column 1: Magnitude in lin
            - Column 2: Phase in rad
        guess_fc : Initial guess for the resonance frequency (fc)
    """
    freq = organized_data[:, 0]  # Frequency in Hz
    mag_lin = organized_data[:, 1]  # Magnitude in linear scale
    phase_rad = organized_data[:, 2]  # Phase in radians

    # Step (1): Compute complex S21
    S21 = mag_lin * np.exp(1j * phase_rad)

    # Step (2): Find the point where fc is closest
    idx_fc = np.argmin(np.abs(freq - guess_fc))
    S21_fc = S21[idx_fc]  # S21 value at resonance

    # Step (3): Fit the S21 data to a circle
    def circle_residuals(params, S21_data):
        """ Compute residuals for circle fitting. """
        x0, y0, r = params
        return np.abs(np.abs(S21_data - (x0 + 1j * y0)) - r)
    # Initial guess: Circle centered around mean and approximate radius
    x0_init, y0_init = np.mean(S21.real), np.mean(S21.imag)
    r_init = np.max(np.abs(S21 - (x0_init + 1j * y0_init)))
    # Perform least squares circle fitting
    result = opt.least_squares(circle_residuals, x0=[x0_init, y0_init, r_init], args=(S21,))
    x0_opt, y0_opt, r_opt = result.x  # Fitted circle parameters

    # Step (4): Compute Qc using Qc = Q / diameter
    Q = find_Q(organized_data)  # Compute Q using previously defined function
    Qc = Q / r_opt if r_opt > 0 else np.nan  # Ensure non-zero division

    return Qc

# %% Set a Testing Point - Check if the initial guessings (fc, phi, Q, Qc) are found succuessfully.
##############################################
#### Use to be a Test Point               ####
#### 1. find resonance frequecny (fc)     ####
#### 2. find phase mismatch (phi)         ####
#### 3. find loaded quality factor (Q)    ####
#### 4. find coupling quality factor (Qc) ####
##############################################
# Define folder path and file name separately
folder_path = r"C:\Users\user\Documents\GitHub\Cooldown_56_Line_4-Tony_Ta_NbSi_03\Resonator_0_5p734GHz"
file_name = r"Tony_Ta_NbSi_03_5p734GHz_-30dBm_-1000mK.csv"

# Combine folder path and file name
file_path = os.path.join(folder_path, file_name)

# Load data
raw_data = pd.read_csv(file_path)

# Print confirmation
print(f"In the folder: {folder_path}")
print(f"Load data from: {file_name}")

organized_data = Organize_Data(raw_data)
tau_fit, phi0_fit = fit_cable_delay(organized_data)
remove_cable_delay_data = remove_cable_delay(organized_data, tau_fit, phi0_fit)
reorganized_data = remove_mag_bg(remove_cable_delay_data)
Plot_Data(reorganized_data)

guess_fc = find_fc(reorganized_data)
guess_phi = find_phi(reorganized_data, guess_fc)
guess_Q = find_Q(organized_data)
guess_Qc = find_Qc(organized_data, guess_fc)
print(f"Initial guess fc: {guess_fc/1e9:.4f} GHz")
print(f"Initial guess phi: {np.rad2deg(guess_phi):.4f} deg")
print(f"Initial guess Q: {guess_Q/1e6:.4f} × 10\u2076")
print(f"Initial guess Qc: {guess_Qc/1e6:.4f} x 10\u2076")
print(f"==============================================")
print(f"The infered internal quality factor (Qi): {(guess_Q*guess_Qc)/(guess_Qc-guess_Q)/1e6:.4f} x 10\u2076")
# %% Start Fitting Using Initial Guessing
