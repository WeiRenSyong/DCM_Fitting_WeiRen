# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.interpolate import interp1d
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
    # Ensure the input is a NumPy array
    raw_data = np.array(raw_data, dtype=str)

    # Check if the raw_data has an extra time column (4 columns instead of 3)
    if raw_data.shape[1] == 3:
        print('The raw data are Nx3 matrix.')
    elif raw_data.shape[1] == 4:
        print('The raw data are Nx4 matrix.')
        raw_data = raw_data[:, 1:]  # Ignore the first column (time column)
    else:
        print('Please check the raw data due to worng format.')
    
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
        mag_lin = 10 ** (mag / 20)   # The reading of the VNA is a ratio of voltage, so 20 is here
    else:
        mag_lin = mag

    # Convert phase to radians if in degrees
    if np.max(np.abs(phase)) > 2 * np.pi:  # If max phase > 2π, assume degrees
        phase_rad = np.deg2rad(phase)
    else:
        phase_rad = phase

    # Stack the processed data
    organized_data = np.column_stack((freq_Hz, mag_lin, phase_rad))

    return organized_data

###########################################
#### Input:                            ####
#### 1. organized data                 ####
###########################################
#### Output:                           ####
#### 1. Frequency (GHz) vs Mag (dB)    ####
#### 2. Frequency (GHz) vs Phase (deg) ####
#### 3. Real(S21) vs Imag(S21)         ####
###########################################
print(f"Define the Plot_Data function...")
def Plot_Data(organized_data):
    # Extract values for plotting
    freq_Hz = organized_data[:, 0]
    mag_lin = organized_data[:, 1]
    phase_rad = organized_data[:, 2]

    # Convert frequency to GHz for plotting
    freq_GHz = freq_Hz / 1e9

    # Convert magnitude to dB for plotting
    mag_dB = 20 * np.log10(mag_lin)
    
    # Convert phase to degree for plotting
    phase_deg = np.rad2deg(phase_rad)
    
    # Compute real and imaginary parts of S21
    S21 = mag_lin * np.exp(1j * phase_rad)
    real_S21 = np.real(S21)
    imag_S21 = np.imag(S21)

    # Create figure
    fig = plt.figure(figsize=(10, 5))

    # Plot Frequency vs Magnitude (dB) - Left Top
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(freq_GHz, mag_dB, color='blue', s=50, marker='o', label="Mag (dB)", alpha=1)
    ax1.set_xlabel("Freq (GHz)")
    ax1.set_ylabel("Mag (dB)")
    ax1.set_title("Freq vs Mag")
    ax1.grid(True)
    ax1.legend()
    ax1.set_xticks([np.min(freq_GHz), (np.min(freq_GHz)+np.max(freq_GHz))/2, np.max(freq_GHz)])

    # Plot Frequency vs Phase (degrees) - Left Bottom
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.scatter(freq_GHz, phase_deg, color='orange', s=50, marker='o', label="Phase (deg)", alpha=1)
    ax2.set_xlabel("Freq (GHz)")
    ax2.set_ylabel("Phase (deg)")
    ax2.set_title("Freq vs Phase")
    ax2.grid(True)
    ax2.legend()
    ax2.set_xticks([np.min(freq_GHz), (np.min(freq_GHz)+np.max(freq_GHz))/2, np.max(freq_GHz)])

    # Plot Real(S21) vs Imag(S21) - Right
    ax3 = fig.add_subplot(1, 2, 2)  # Ensures a single, right-side wide plot
    ax3.scatter(real_S21, imag_S21, color='green', s=50, marker='o', label="S21 Complex Plane", alpha=1)
    ax3.set_xlabel("Real(S21)")
    ax3.set_ylabel("Imag(S21)")
    ax3.set_title("S21 Complex Plane")
    ax3.grid(True)
    ax3.legend()
    ax3.axis("equal")  # Ensures proper scaling of real/imag axes

    # Improve layout spacing
    plt.tight_layout()
    plt.show()

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
    # Extract values for plotting
    freq_Hz = organized_data[:, 0]
    mag_lin = organized_data[:, 1]
    phase_rad = organized_data[:, 2]

    # Unwrap phase to prevent discontinuities
    phase_rad = np.unwrap(phase_rad)  

    # Select the wings first and last few points
    num_points = 2  # Number of points to use from each wing
    freq_bg = np. concatenate((freq_Hz[:num_points], freq_Hz[-num_points:]))
    phase_bg = np.concatenate((phase_rad[:num_points], phase_rad[-num_points:]))

    # Perform linear fit to get A(slope) and B (offset)
    minus_two_pi_tau, alpha = np.polyfit(freq_bg, phase_bg, 1)    # Linear fit (degree = 1)

    tau = minus_two_pi_tau / (-2 * np.pi)
    print(f"cable_delay (tau) is {tau * 1e9:.2f} ns")
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
    print(f"Preprocess the phase correction...")
    # Extract values for plotting
    freq_Hz = organized_data[:, 0]
    mag_lin = organized_data[:, 1]
    phase_rad = organized_data[:, 2]

    # Compute the background phase caused by cable delay
    phase_bg = -2 * np.pi * freq_Hz * tau + alpha # in rad

    z = mag_lin * np.exp(1j * phase_rad)
    # Remove the background phase
    z_corrected = z * np.exp(-1j * phase_bg)  # Multiply by exp(+j*phase) to correct

    # Arrange z_corrected     
    freq_Hz = freq_Hz
    mag_lin = np.abs(z_corrected)
    phase_rad = np.angle(z_corrected)

    # Stack the processed data
    reorganized_data = np.column_stack((freq_Hz, mag_lin, phase_rad))

    return reorganized_data

#%% Preprocessing (2) - remove_mg_bg
##########################################
#### Input:                           ####
#### para 1 : organized_data          ####
##########################################
#### Output:                          ####
#### reorganized_data: bg_mag_removal ####
##########################################
print(f"Define remove_mag_bg function...")
def remove_mag_bg(organized_data):
    print(f"Preprocess the background mag removal...")
    # Extract values for plotting
    freq_Hz = organized_data[:, 0]
    mag_lin = organized_data[:, 1]
    phase_rad = organized_data[:, 2]

    # Select the wings first and last few points
    num_points = 2  # Number of points to use from each wing
    freq_bg = np. concatenate((freq_Hz[:num_points], freq_Hz[-num_points:]))
    mag_bg = np.concatenate((mag_lin[:num_points], mag_lin[-num_points:]))

    # Perform linear fit to get A(slope) and B (offset)
    q = np.polyfit(freq_bg, mag_bg, 1)    # Linear fit (degree = 1)

    # Define the background to substrate (Ax + B)
    mag_bg = np.polyval(q, freq_Hz)

    z = mag_lin * np.exp(1j * phase_rad)
    # Remove the background mag
    z_corrected = z / mag_bg

    # Arrange z_corrected     
    freq_Hz = freq_Hz
    mag_lin = np.abs(z_corrected)
    phase_rad = np.angle(z_corrected)

    # Stack the processed data
    reorganized_data = np.column_stack((freq_Hz, mag_lin, phase_rad))

    return reorganized_data

# # %% Set a Testing Point - Check the complex circle after removing environment factors
# ################################
# #### Use to be a Test Point ####
# #### 1. Correct the phase   ####
# #### 2. Correct the mag     ####
# ################################ 
# # Define folder path and file name separately
# folder_path = r"C:\Users\user\Documents\GitHub\Cooldown_56_Line_5-NW_Ta2O5_15nm_01\2024_10_18_Final_Organized_Data\All_csv_raw_data_and_fitting_results\Resonator_0_5p750GHz"
# file_name = r"NW_Ta2O5_15nm_01_5p750GHz_-30dBm_-1000mK.csv"

# # Combine folder path and file name
# file_path = os.path.join(folder_path, file_name)

# # Load data
# raw_data = pd.read_csv(file_path)

# # Print confirmation
# print(f"In the folder: {folder_path}")
# print(f"Load data from: {file_name}")

# organized_data = Organize_Data(raw_data)
# Plot_Data(organized_data)

# tau, alpha = fit_cable_delay(organized_data)
# remove_cable_delay_data = remove_cable_delay(organized_data, tau, alpha)
# Plot_Data(remove_cable_delay_data)

# remove_mag_bg_data = remove_mag_bg(remove_cable_delay_data)
# Plot_Data(remove_mag_bg_data)

# %% Initial Guessing
#####################################################
#### Input:                                      ####
#### para 1 : organized_data                     ####
#####################################################
#### Output:                                     ####
#### 1. center (zc_fit)                          ####
#### 2. diameter (d_fit)                         ####
#####################################################
print(f"Define find_circle finctino...")
def find_circle(organized_data):
    # Extract frequency, magnitude, and phase from organized data
    freq_Hz = organized_data[:, 0]  # frequency in Hz
    mag_lin = organized_data[:, 1]  # magnitude in lin
    phase_rad = organized_data[:, 2]  # phase in rad

    # Calculate S21
    S21 = mag_lin * np.exp(1j * phase_rad)
    
    S21_real = np.real(S21)
    S21_imag = np.imag(S21)

    # Define the circle equation: (x - xc)^2 + (y - yc)^2 = r^2
    def circle_equation(params, x, y): 
        xc, yc, r = params
        return (x - xc)**2 + (y - yc)**2 - r**2

    # Initial guess for the center (xc, yc) and radius (r)
    xc_guess = np.mean(S21_real)
    yc_guess = np.mean(S21_imag)
    r_guess = np.mean(np.sqrt((S21_real - xc_guess)**2 + (S21_imag - yc_guess)**2))
    # Initial parameter guess: [xc, yc, r]
    initial_guess = [xc_guess, yc_guess, r_guess]
    
    # Perform the least squares fitting
    result = opt.least_squares(circle_equation, initial_guess, args=(S21_real, S21_imag))
    # Get the fitted parameters
    xc_fit, yc_fit, r_fit = result.x
    zc_fit = xc_fit + 1j * yc_fit
    d_fit = r_fit * 2

    return zc_fit, d_fit

#####################################################
#### Input:                                      ####
#### para 1 : organized_data                     ####
#####################################################
#### Output:                                     ####
#### 1. Resonance frequency (fc) in Hz           ####
#####################################################
print(f"Define find_fc function...")
def find_fc(organized_data):
    # Extract frequency, magnitude, and phase from organized data
    freq_Hz = organized_data[:, 0]  # frequency in Hz
    mag_lin = organized_data[:, 1]  # magnitude in linear scale
    phase_rad = organized_data[:, 2]  # phase in radians

    # Create the complex S21 data
    S21 = mag_lin * np.exp(1j * phase_rad)
    S21_real = np.real(S21)
    S21_imag = np.imag(S21)

    # Get the center (zc) and diameter (d) of the fitted circle
    zc, d = find_circle(organized_data)

    # Assuming z_fc is defined as the resonance point
    z_fc = 1 + (zc - 1) * 2

    # Function to calculate the Euclidean distance
    def variance(x_fc, y_fc, x, y):
        return (x - x_fc)**2 + (y - y_fc)**2

    # Find the closest point to z_fc
    distances = []
    for xi, yi in zip(S21_real, S21_imag):
        d = variance(np.real(z_fc), np.imag(z_fc), xi, yi)
        distances.append(d)

    # Get the index of the closest point
    closest_index = np.argmin(distances)

    # Get the frequency corresponding to the closest point
    fc = freq_Hz[closest_index]

    return fc

#####################################################
#### Input:                                      ####
#### para 1 : organized_data                     ####
#####################################################
#### Output:                                     ####
#### 1. phase mismatch (phi) in rad              ####
#####################################################
print(f"Define find_phi function...")
def find_phi(organized_data):
    zc, d = find_circle(organized_data)
    phi = np.angle(1 - zc)  # Phase of ((1+0j) - S21_fc) in rad
    return phi

#####################################################
#### Input:                                      ####
#### para 1 : organized_data                     ####
#####################################################
#### Output:                                     ####
#### 1. quality factor (Q) = fc / FWHM           ####
#####################################################
print(f"Define find_Q function...")
def find_Q(organized_data, plot=False):
    # Extract frequency and magnitude
    freq_Hz = organized_data[:, 0]  # frequency in Hz
    mag_lin = organized_data[:, 1]  # magnitude in linear scale
    phase_rad = organized_data[:, 2]  # phase in radians

    def find_FWHM(freq_Hz, mag_lin):
        # Find the minimum magnitude (resonance dip)
        min_mag_lin = np.min(mag_lin)

        # Estimate background level using the maximum magnitude
        background = (mag_lin[0] + mag_lin[-1]) / 2  

        # Compute Half-Maximum Value
        half_mag_lin = (background + min_mag_lin) / 2

        # Find indices where magnitude crosses the half-maximum value
        below_half = np.where(mag_lin <= half_mag_lin)[0]  # Indices of points below half-max

        idx_lower, idx_upper = below_half[-1], below_half[0]  # Corrected indexing

        freq_lower = freq_Hz[idx_lower]
        freq_upper = freq_Hz[idx_upper]

        # Compute Full Width at Half Maximum (FWHM)
        FWHM = abs(freq_upper - freq_lower)

        return FWHM, freq_lower, freq_upper, idx_lower, idx_upper

    # Get FWHM and corresponding frequencies
    FWHM, freq_lower, freq_upper, idx_lower, idx_upper = find_FWHM(freq_Hz, mag_lin)

    # Get resonance frequency fc
    fc = find_fc(organized_data)

    # Compute Quality Factor Q
    Q = fc / FWHM

    if plot:
        freq_GHz = freq_Hz / 1e9
        mag_dB = 20 * np.log10(mag_lin)  # Convert to dB for clarity

        plt.figure(figsize=(6, 4))
        plt.scatter(freq_GHz, mag_dB, color='blue', s=50, marker='o', label="Organized Data", alpha=1)
        plt.scatter(freq_lower / 1e9, 20 * np.log10(mag_lin[idx_lower]), color='red', s=500, marker='*', zorder=5, label="freq_lower")
        plt.scatter(freq_upper / 1e9, 20 * np.log10(mag_lin[idx_upper]), color='green', s=500, marker='*', zorder=5, label="freq_upper")

        plt.xlabel("Freq (GHz)")
        plt.ylabel("Mag (dB)")
        plt.title("Resonance Dip and FWHM")
        plt.grid(True)
        plt.xticks(np.linspace(np.min(freq_GHz), np.max(freq_GHz), 3))
        plt.legend()
        plt.show()

    return Q

#####################################################
#### Input:                                      ####
#### para 1 : organized_data                     ####
#####################################################
#### Output:                                     ####
#### 1. coupling quality factor (Qc)             ####
#####################################################
print(f"Define find_Qc function...")
def find_Qc(organized_data):
    zc, d = find_circle(organized_data)
    Q_over_absQc = d
    phi = find_phi(organized_data)
    Q_over_Qc = Q_over_absQc * np.cos(phi)
    Q = find_Q(organized_data)  # Compute Q using previously defined function
    Qc = Q / Q_over_Qc

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
file_name = r"Tony_Ta_NbSi_03_5p734GHz_-81dBm_-1000mK.csv"

# Combine folder path and file name
file_path = os.path.join(folder_path, file_name)

# Load data
raw_data = pd.read_csv(file_path)

# Print confirmation
print(f"In the folder: {folder_path}")
print(f"Load data from: {file_name}")

organized_data = Organize_Data(raw_data)
tau, alpha = fit_cable_delay(organized_data)
remove_cable_delay_data = remove_cable_delay(organized_data, tau, alpha)
reorganized_data = remove_mag_bg(remove_cable_delay_data)
Plot_Data(reorganized_data)

guess_fc = find_fc(reorganized_data)
guess_phi = find_phi(reorganized_data)
guess_Q = find_Q(reorganized_data, plot=False)
guess_Qc = find_Qc(reorganized_data)
print(f"Initial guess Q: {guess_Q/1e6:.4f} × 10\u2076")
print(f"The infered internal quality factor (Qi): {(1 / guess_Q - 1 / guess_Qc) **(-1) / 1e6:.4f} x 10\u2076")
print(f"Initial guess |Qc|: {guess_Qc/1e6:.4f} x 10\u2076")
print(f"Initial guess phi: {np.rad2deg(-guess_phi):.4f} deg")
print(f"Initial guess fc: {guess_fc / 1e9:.9f} GHz")
  
# %% Doing the Monte Carlo to Adjust the Fit Parameters
guess_fc = find_fc(reorganized_data)
guess_Qc = find_Qc(reorganized_data)
guess_Q = find_Q(reorganized_data, plot=False)
guess_phi = find_phi(reorganized_data)

freq_Hz = reorganized_data[:, 0]
mag_lin = reorganized_data[:, 1]
phase_rad = reorganized_data[:, 2]
S21_data = mag_lin * np.exp(1j * phase_rad)

# Define Theoretical S21 Model
def S21_model(freq, fc, phi, Q, Qc):
    return 1 - (Q / Qc) * np.exp(1j * phi) / (1 + 2j * Q * (freq / fc - 1))

def Monte_Carlo_fit_complex_circle(freq, S21_data, guess_fc, guess_phi, guess_Q, guess_Qc, num_samples=10):
    
    test_S21_data = S21_model(freq, guess_fc, guess_phi, guess_Q, guess_Qc)
    best_cost = np.sum(np.abs(S21_data - test_S21_data) ** 2)

    best_Qc, best_Q, best_phi, best_fc = guess_Qc, guess_Q, guess_phi, guess_fc

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Labels for axes
    ax1.set_title("Qc Evolution")
    ax2.set_title("Q Evolution")
    ax3.set_title("phi Evolution")
    ax4.set_title("fc Evolution")

    for i in range(num_samples):
        # Propose new parameter values
        param_Qc = np.random.normal(best_Qc, 5e4)
        param_Q = np.random.normal(best_Q, 1e4)
        param_phi = np.random.normal(best_phi, 0.002)
        param_fc = np.random.normal(best_fc, 10)

        # Evaluate cost for new parameter set
        test_S21_data = S21_model(freq, param_fc, param_phi, param_Q, param_Qc)
        cost_S21_data = np.sum(np.abs(S21_data - test_S21_data) ** 2)

        # Accept if cost is lower
        if cost_S21_data < best_cost:
            best_cost = cost_S21_data
            best_Qc, best_Q, best_phi, best_fc = param_Qc, param_Q, param_phi, param_fc

        # Scatter plot updates
        ax1.scatter(i, best_Qc, color='black', s=50, marker='o', alpha=0.7)
        ax2.scatter(i, best_Q, color='red', s=50, marker='o', alpha=0.7)
        ax3.scatter(i, best_phi, color='green', s=50, marker='o', alpha=0.7)
        ax4.scatter(i, best_fc, color='blue', s=50, marker='o', alpha=0.7)

    plt.tight_layout()
    plt.show()

    return best_fc, best_phi, best_Q, best_Qc

best_fc, best_phi, best_Q, best_Qc = Monte_Carlo_fit_complex_circle(freq_Hz, S21_data, guess_fc, guess_phi, guess_Q, guess_Qc, num_samples=500)

print(f"Final Q: {best_Q/1e6:.4f} × 10\u2076")
print(f"The infered internal quality factor (Qi): {(1 / best_Q - 1 / best_Qc) **(-1) / 1e6:.4f} x 10\u2076")
print(f"Final guess |Qc|: {best_Qc/1e6:.4f} x 10\u2076")
print(f"Final guess phi: {np.rad2deg(-best_phi):.4f} deg")
print(f"Final guess fc: {best_fc / 1e9:.9f} GHz")


# %% Draw the Initial Guessing Fitting
# def Plot_Fit_Data(organized_data):
#     freq_Hz = organized_data[:, 0]  
#     mag_lin = organized_data[:, 1] 
#     phase_rad = organized_data[:, 2] 

#     S21 = mag_lin * np.exp(1j * phase_rad)
#     S21_real = np.real(S21)
#     S21_imag = np.imag(S21)

#     freq_GHz = freq_Hz / 1e9
#     mag_dB = 20 * np.log10(mag_lin) 
#     phase_deg = np.rad2deg(phase_rad)

#     # Draw the fitted curve
#     zc_fit, d_fit = find_circle(organized_data)
#     phi = find_phi(organized_data)
#     theta = np.linspace(0, 2 * np.pi, len(freq_Hz))
#     z_fit = zc_fit + d_fit / 2 * np.exp(1j * theta) * np.exp(-1j * phi)
#     x_fit = np.real(z_fit)
#     y_fit = np.imag(z_fit)
    
#     mag_lin_fit = np.abs(z_fit)
#     mag_dB_fit = 20 * np.log10(mag_lin_fit)
#     phase_rad_fit = np.angle(z_fit) 
#     phase_deg_fit = np.rad2deg(phase_rad_fit)
    
#     # Assuming z_fc is defined as the resonance point
#     z_fc_fit = 1 + (zc_fit - 1) * 2

#     fc_Hz = find_fc(organized_data)
#     fc_GHz = fc_Hz / 1e9

#     mag_lin_fc_fit = np.abs(z_fc_fit)
#     mag_dB_fc_fit = 20 * np.log10(mag_lin_fc_fit)

#     phase_rad_fc_fit = np.angle(z_fc_fit)
#     phase_deg_fc_fit = np.rad2deg(phase_rad_fc_fit)

#     # Create figure
#     fig = plt.figure(figsize=(10, 5))

#     # Plot Frequency vs Magnitude (dB) - Left Top
#     ax1 = fig.add_subplot(2, 2, 1)
#     ax1.scatter(freq_GHz, mag_dB, color='blue', s=50, marker='o', label="Mag", alpha=1)
#     ax1.scatter(fc_GHz, mag_dB_fc_fit, color='red', s=500, marker='*', zorder=5, label="Resonance")
#     ax1.plot(freq_GHz, mag_dB_fit, label="Fitted Mag", color="green")
#     ax1.set_xlabel("Freq (GHz)")
#     ax1.set_ylabel("Mag (dB)")
#     ax1.set_title("Freq vs Mag")
#     ax1.grid(True)
#     ax1.legend()
#     ax1.set_xticks(np.linspace(np.min(freq_GHz), np.max(freq_GHz), 3))

#     # Plot Frequency vs Phase (degrees) - Left Bottom
#     ax2 = fig.add_subplot(2, 2, 3)
#     ax2.scatter(freq_GHz, phase_deg, color='orange', s=50, marker='o', label="Phase", alpha=1)
#     ax2.scatter(fc_GHz, phase_deg_fc_fit, color='red', s=500, marker='*', zorder=5, label="Resonance")
#     ax2.plot(freq_GHz, phase_deg_fit, label="Fitted Phase", color="green")
#     ax2.set_xlabel("Freq (GHz)")
#     ax2.set_ylabel("Phase (deg)")
#     ax2.set_title("Freq vs Phase")
#     ax2.grid(True)
#     ax2.legend()
#     ax2.set_xticks(np.linspace(np.min(freq_GHz), np.max(freq_GHz), 3))

#     # Plot Real(S21) vs Imag(S21) - Right
#     ax3 = fig.add_subplot(1, 2, 2)  # Ensures a single, right-side wide plot
#     ax3.scatter(S21_real, S21_imag, color='green', s=50, marker='o', label="Reorganized Data", alpha=1)
#     ax3.scatter(np.real(z_fc_fit), np.imag(z_fc_fit), color='red', s=500, marker='*', zorder=5, label="Resonance")
#     ax3.plot(x_fit, y_fit, label="Fitted Circle", color="green")
#     # Adding dashed lines at x=1 and y=0
#     ax3.axvline(x=1, color='black', linestyle='--', label=None)
#     ax3.axhline(y=0, color='black', linestyle='--', label=None)
#     # Plotting the gray line connecting the star and (1, 0)
#     ax3.plot([np.real(z_fc_fit), 1], [np.imag(z_fc_fit), 0], color='red', linestyle='-', linewidth=2, label=None)
#     ax3.set_xlabel("Real(S21)")
#     ax3.set_ylabel("Imag(S21)")
#     ax3.set_title("S21 Complex Plane")
#     ax3.grid(True)
#     ax3.legend()
#     ax3.axis("equal")  # Ensures proper scaling of real/imag axes

#     # Improve layout spacing
#     plt.tight_layout()
#     plt.show()

# Plot_Fit_Data(reorganized_data)

# %% Function to plot final results
def Plot_Final_Fit_Data(organized_data, fc, phi, Q, Qc):
    freq_Hz = organized_data[:, 0]  
    mag_lin = organized_data[:, 1] 
    phase_rad = organized_data[:, 2] 

    S21 = mag_lin * np.exp(1j * phase_rad)
    S21_real = np.real(S21)
    S21_imag = np.imag(S21)

    freq_GHz = freq_Hz / 1e9
    mag_dB = 20 * np.log10(mag_lin) 
    phase_deg = np.rad2deg(phase_rad)

    # Use the S21 model for the final fit
    S21_final_fit = S21_model(freq_Hz, fc, phi, Q, Qc)
    mag_lin_final_fit = np.abs(S21_final_fit)
    mag_dB_final_fit = 20 * np.log10(mag_lin_final_fit)
    phase_deg_final_fit = np.rad2deg(np.angle(S21_final_fit))
    S21_real_final_fit = np.real(S21_final_fit)
    S21_imag_final_fit = np.imag(S21_final_fit)
    
    fc_GHz = fc / 1e9
    
    # Find the closest frequency point to fc
    closest_index = np.argmin(np.abs(freq_Hz - fc))
    mag_dB_fc_final_fit = mag_dB[closest_index]
    phase_deg_fc_final_fit = phase_deg[closest_index]
    z_fc_final_fit = S21_real[closest_index] + 1j * S21_imag[closest_index]

    # Create figure
    fig = plt.figure(figsize=(10, 5))

    # Plot Frequency vs Magnitude (dB)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(freq_GHz, mag_dB, color='blue', s=50, marker='o', label="Measured Mag", alpha=1)
    ax1.scatter(fc_GHz, mag_dB_fc_final_fit, color='red', s=500, marker='*', zorder=5, label="Resonance")
    ax1.plot(freq_GHz, mag_dB_final_fit, label="Fitted Mag", color="green")
    ax1.set_xlabel("Freq (GHz)")
    ax1.set_ylabel("Mag (dB)")
    ax1.set_title("Freq vs Mag")
    ax1.grid(True)
    ax1.legend()
    
    # Plot Frequency vs Phase (degrees)
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.scatter(freq_GHz, phase_deg, color='orange', s=50, marker='o', label="Measured Phase", alpha=1)
    ax2.scatter(fc_GHz, phase_deg_fc_final_fit, color='red', s=500, marker='*', zorder=5, label="Resonance")
    ax2.plot(freq_GHz, phase_deg_final_fit, label="Fitted Phase", color="green")
    ax2.set_xlabel("Freq (GHz)")
    ax2.set_ylabel("Phase (deg)")
    ax2.set_title("Freq vs Phase")
    ax2.grid(True)
    ax2.legend()

    # Plot Real(S21) vs Imag(S21)
    ax3 = fig.add_subplot(1, 2, 2)
    ax3.scatter(S21_real, S21_imag, color='green', s=50, marker='o', label="Measured Data", alpha=1)
    ax3.scatter(np.real(z_fc_final_fit), np.imag(z_fc_final_fit), color='red', s=500, marker='*', zorder=5, label="Resonance")
    ax3.plot(S21_real_final_fit, S21_imag_final_fit, label="Fitted Circle", color="green")
    ax3.axvline(x=1, color='black', linestyle='--', label=None)
    ax3.axhline(y=0, color='black', linestyle='--', label=None)
    ax3.plot([np.real(z_fc_final_fit), 1], [np.imag(z_fc_final_fit), 0], color='red', linestyle='-', linewidth=2, label=None)
    ax3.set_xlabel("Real(S21)")
    ax3.set_ylabel("Imag(S21)")
    ax3.set_title("S21 Complex Plane")
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout()
    plt.show()

Plot_Final_Fit_Data(reorganized_data, best_fc, best_phi, best_Q, best_Qc)