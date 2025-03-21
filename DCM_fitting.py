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

# %% Set a Testing Point - Check the complex circle after removing environment factors
# ################################
# #### Use to be a Test Point ####
# #### 1. Correct the phase   ####
# #### 2. Correct the mag     ####
# ################################ 
# # Define folder path and file name separately
# folder_path = r"C:\Users\user\Documents\GitHub\Cooldown_56_Line_4-Tony_Ta_NbSi_03\Resonator_3_5p863GHz"
# file_name = r"Tony_Ta_NbSi_03_5p863GHz_-90dBm_-1000mK.csv"

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
print(f"Define find_circle finctino...")
def find_circle(organized_data):
    """
    Function to finds the center (zc), diameter (d) of the circle:
    1. center (zc).
    2. diameter (d).

    Parameters:
    1. organized_data (numpy array): N×3 matrix containing:
        - Column 1: Frequency in Hz
        - Column 2: Magnitude in lin
        - Column 3: Phase in rad
    """
    # Extract frequency, magnitude, and phase from organized data
    freq_Hz = organized_data[:, 0]  # frequency in Hz
    mag_lin = organized_data[:, 1]  # magnitude in lin
    phase_rad = organized_data[:, 2]  # magnitude in lin

    # Calculate S21
    S21 = mag_lin * np.exp(1j * phase_rad)
    
    x = np.real(S21)
    y = np.imag(S21)
    # Define the circle equation: (x - x0)^2 + (y - y0)^2 = r^2
    def circle_equation(params, x, y): 
        xc, yc, r = params
        return (x - xc)**2 + (y - yc)**2 - r**2

    # Initial guess for the center (x0, y0) and radius (r)
    xc_guess = np.mean(x)
    yc_guess = np.mean(y)
    r_guess = np.mean(np.sqrt((x - xc_guess)**2 + (y - yc_guess)**2))
    
    # Initial parameter guess: [x0, y0, r]
    initial_guess = [xc_guess, yc_guess, r_guess]
    
    # Perform the least squares fitting
    result = opt.least_squares(circle_equation, initial_guess, args=(x, y))

    # Get the fitted parameters
    xc_fit, yc_fit, r_fit = result.x
    zc_fit = xc_fit + 1j * yc_fit
    
    return zc_fit, r_fit

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

    # Extract frequency, magnitude, and phase from organized data
    freq_Hz = organized_data[:, 0]  # frequency in Hz
    mag_lin = organized_data[:, 1]  # magnitude in lin
    phase_rad = organized_data[:, 2]  # phase in rad

    # Create the complex S21 data
    S21 = mag_lin * np.exp(1j * phase_rad)
    x = np.real(S21)
    y = np.imag(S21)

    # Get the center (zc) and radius (r) of the fitted circle
    zc, r = find_circle(organized_data)
    
    # Assuming z_fc is defined as the resonance point
    z_fc = 1 + (zc - 1) * 2

    # Function to calculate the Euclidean distance between two points
    def calculate_distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    # Find the closest point to z_fc
    distances = [calculate_distance(xi, yi, z_fc.real, z_fc.imag) for xi, yi in zip(x, y)]

    # Get the index of the closest point
    closest_index = np.argmin(distances)

    # Get the frequency corresponding to the closest point
    fc = organized_data[closest_index, 0]

    return fc

print(f"Define find_phi function...")
def find_phi(organized_data):
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
    zc, r = find_circle(organized_data)

    phi = np.pi - np.angle(zc - 1)  # Phase of (S21_fc - (1+0j)) in rad
    return -phi

print(f"Define find_Q function...")
def find_Q(organized_data, plot=None):
    """
    Function to find the quality factor (Q) = fc / FWHM:

    Parameters:
    1. organized_data : N×3 matrix containing:
        - Column 1: Frequency in Hz
        - Column 2: Magnitude in lin
        - Column 3: Phase in rad
    """
    fc = find_fc(organized_data)
    
    # Extract frequency and magnitude
    freq_Hz = organized_data[:, 0]  # frequency in Hz
    mag_lin = organized_data[:, 1]  # magnitude in lin

    def find_FWHM(freq_Hz, mag_lin):
        # Find the minimum of the magnitude (resonance dip)
        min_mag_lin = np.min(mag_lin)
        background = (mag_lin[0] + mag_lin[-1]) / 2
        
        # Half of the minimum magnitude (assuming resonance dip)
        half_mag_lin = background + (min_mag_lin - background) / 2
        
        # Find the indices where the magnitude crosses the half maximum value
        idx_upper = np.where(mag_lin <= half_mag_lin)[0][0]  # First index at or below half max
        idx_lower = np.where(mag_lin <= half_mag_lin)[0][-1]  # Last index at or below half max
        
        # Get the frequencies at these indices
        freq_upper = freq_Hz[idx_upper]
        freq_lower = freq_Hz[idx_lower]
        
        # FWHM is the difference between the frequencies at the half maximum
        FWHM = freq_upper - freq_lower
        return FWHM, idx_lower, idx_upper

    FWHM, idx_lower, idx_upper = find_FWHM(freq_Hz, mag_lin)
        
    Q = fc / FWHM

    if plot:
        freq_GHz = freq_Hz / 1e9
        plt.figure(figsize=(8, 6))
        plt.plot(freq_GHz, mag_lin, color='black', linestyle='-', linewidth=2, label="mag_lin")
        plt.scatter(freq_Hz[idx_lower] / 1e9, mag_lin[idx_lower], color='red', s=500, marker='*', zorder=5, label="freq_lower")
        plt.scatter(freq_Hz[idx_upper] / 1e9, mag_lin[idx_upper], color='green', s=500, marker='*', zorder=5, label="freq_upper")

        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Magnitude (dB)")
        plt.title("Frequency vs Magnitude")
        plt.grid(True)
        plt.xticks([np.min(freq_GHz), (np.min(freq_GHz) + np.max(freq_GHz)) / 2, np.max(freq_GHz)])
        plt.legend()
        plt.show()

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
    zc_fit, r_fit = find_circle(organized_data)
    Q_over_absQc = 2 * r_fit
    phi = find_phi(organized_data)
    Q_over_Qc = Q_over_absQc * np.cos(phi)

    # Step (4): Compute Qc using Qc = Q / diameter
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
folder_path = r"C:\Users\user\Documents\GitHub\Cooldown_56_Line_4-Tony_Ta_NbSi_03\Resonator_3_5p863GHz"
file_name = r"Tony_Ta_NbSi_03_5p863GHz_-90dBm_-1000mK.csv"

# Combine folder path and file name
file_path = os.path.join(folder_path, file_name)

# Load data
raw_data = pd.read_csv(file_path)

# Print confirmation
print(f"In the folder: {folder_path}")
print(f"Load data from: {file_name}")

organized_data = Organize_Data(raw_data)
Plot_Data(organized_data)
tau_fit, phi0_fit = fit_cable_delay(organized_data)
remove_cable_delay_data = remove_cable_delay(organized_data, tau_fit, phi0_fit)
reorganized_data = remove_mag_bg(remove_cable_delay_data)
Plot_Data(reorganized_data)

guess_fc = find_fc(reorganized_data)
guess_phi = find_phi(reorganized_data)
guess_Q = find_Q(organized_data, plot=None)
guess_Qc = find_Qc(organized_data, guess_fc)
print(f"Initial guess fc: {guess_fc/1e9:.4f} GHz")
print(f"Initial guess phi: {np.rad2deg(guess_phi):.4f} deg")
print(f"Initial guess Q: {guess_Q/1e6:.4f} × 10\u2076")
print(f"Initial guess Qc: {guess_Qc/1e6:.4f} x 10\u2076")
print(f"==============================================")
print(f"The infered internal quality factor (Qi): {(guess_Q*guess_Qc)/(guess_Qc-guess_Q)/1e6:.4f} x 10\u2076")

# %% Draw the Initial Guessing Fitting
# Mark fc on the Complex Plot
freq_Hz = reorganized_data[:, 0]  # Frequency in Hz
mag_lin = reorganized_data[:, 1]  # Magnitude in linear scale
phase_rad = reorganized_data[:, 2]  # Phase in radians

zc_fit_reorganized, r_fit_reorganized = find_circle(reorganized_data)
xc_fit_reorganized = np.real(zc_fit_reorganized)
yc_fit_reorganized = np.imag(zc_fit_reorganized)
# Generate values for theta (0 to 2*pi)
theta = np.linspace(0, 2 * np.pi, 100)
# Parametric equation of the circle
x = xc_fit_reorganized + r_fit_reorganized * np.cos(theta)
y = yc_fit_reorganized + r_fit_reorganized * np.sin(theta)

# Mark specific points with a different color and marker
idx_fc = np.argmin(np.abs(reorganized_data[:,0] - guess_fc))
S21 = mag_lin * np.exp(1j * phase_rad)
S21_fc = S21[idx_fc]  # S21 value at resonance frequency
plt.scatter(np.real(S21), np.imag(S21), color='blue', s=50, marker='o', label="Reorganized Data", alpha=1)
plt.scatter(np.real(S21_fc), np.imag(S21_fc), color='orange', s=500, marker='*', zorder=5, label="Resonance")
plt.plot(x, y, label="Fitted Circle", color="green")

# Adding labels and title
plt.xlabel("Re(S21)")
plt.ylabel("Im(S21)")
plt.title("Reorganized Data on Complex Plane with Marked Resonance")

# Adding dashed lines at x=1 and y=0
plt.axvline(x=1, color='black', linestyle='--', label="x = 1 (Dashed Line)")
plt.axhline(y=0, color='black', linestyle='--', label="y = 0 (Dashed Line)")

# Plotting the gray line connecting the star and (1, 0)
plt.plot([np.real(S21_fc), 1], [np.imag(S21_fc), 0], color='red', linestyle='-', linewidth=2, label="Connection Line")

# Ensuring the x and y axes have the same scale
plt.axis('equal')

# Show the plot
plt.show()

# %% Start Fitting Using Initial Guessing
