import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# Read the CSV data
df = pd.read_csv('transformer-iron.csv')

# Calculate voltage increments and find smallest non-zero step size
A_increments = np.abs(np.diff(df['A/V']))
B_increments = np.abs(np.diff(df['B/V']))

# Remove zero increments as they don't represent measurement steps
A_increments = A_increments[A_increments > 0]
B_increments = B_increments[B_increments > 0]

# Use the smallest non-zero increment as the resolution
A_resolution = np.min(A_increments)
B_resolution = np.min(B_increments)

print(f"Empirical A voltage resolution: {A_resolution:.6f} V")
print(f"Empirical B voltage resolution: {B_resolution:.6f} V")

# Constants with uncertainties
n_p = 400  
n_p_err = 1
L_p = 0.0437  
L_p_err = 5E-4
R_p = 2.35
R_p_err = 0.05
R_i = 2160
R_i_err = 20
C = 0.4385E-6
C_err = 0.2E-9
n_s = 500
n_s_err = 1
A_s = 7.7E-6
A_s_err = 0.1E-6
mu_0 = 1.25663706E-6  # Permeability of free space
L_s = 0.05  # 5 cm in meters
L_s_err = 0.001  # 0.1 cm in meters

# Apply Savitzky-Golay filter for smoothing
window_length = 201
polyorder = 3

# Smooth A and B voltage data
A_smooth = savgol_filter(df['A/V'], window_length, polyorder)
B_smooth = savgol_filter(df['B/V'], window_length, polyorder)

# Calculate H and B - CORRECTED channel assignment
H = (n_p/(L_p * R_p)) * A_smooth  # Now using A channel
B = (R_i * C/(n_s * A_s)) * B_smooth  # Now using B channel

# Modified error calculation functions
def calculate_H_error(V_A):  # Now takes A channel voltage
    rel_n_p = n_p_err/n_p
    rel_L_p = L_p_err/L_p
    rel_R_p = R_p_err/R_p
    rel_V = A_resolution/np.abs(V_A)  # Using A channel resolution
    
    rel_H = np.sqrt(rel_n_p**2 + rel_L_p**2 + rel_R_p**2 + rel_V**2)
    return np.abs(H) * rel_H

def calculate_B_error(V_B):  # Now takes B channel voltage
    rel_R_i = R_i_err/R_i
    rel_C = C_err/C
    rel_n_s = n_s_err/n_s
    rel_A_s = A_s_err/A_s
    rel_V = B_resolution/np.abs(V_B)  # Using B channel resolution
    
    rel_B = np.sqrt(rel_R_i**2 + rel_C**2 + rel_n_s**2 + rel_A_s**2 + rel_V**2)
    return np.abs(B) * rel_B

# Calculate error bars
H_errors = calculate_H_error(A_smooth)  # Now using A channel
B_errors = calculate_B_error(B_smooth)  # Now using B channel

# Create the plot
plt.figure(figsize=(12, 8))

# Plot with error bars
plt.errorbar(H, B, 
            xerr=H_errors, yerr=B_errors,
            fmt='r-', linewidth=1.5, 
            ecolor='lightgray',
            elinewidth=0.5,
            capsize=0,
            label='B-H curve')

plt.title('B-H Hysteresis Loop with Empirical Errors', fontsize=14)
plt.xlabel('H (A/m)', fontsize=12)
plt.ylabel('B (T)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

# Set axis limits with padding
x_padding = (H.max() - H.min()) * 0.05
y_padding = (B.max() - B.min()) * 0.05
plt.xlim(H.min() - x_padding, H.max() + x_padding)
plt.ylim(B.min() - y_padding, B.max() + y_padding)

plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('BH_hysteresis_loop_empirical_errors.png', dpi=300, bbox_inches='tight')
plt.show()

# Print uncertainty analysis
print("\nUncertainty Analysis:")
print(f"Average relative uncertainty in H: {np.mean(H_errors/np.abs(H))*100:.1f}%")
print(f"Average relative uncertainty in B: {np.mean(B_errors/np.abs(B))*100:.1f}%")

# Print contribution of each error source for a typical point
def print_error_contributions(V_A_typical, V_B_typical):
    # For H (using A channel)
    rel_n_p = (n_p_err/n_p)**2
    rel_L_p = (L_p_err/L_p)**2
    rel_R_p = (R_p_err/R_p)**2
    rel_V_A = (A_resolution/np.abs(V_A_typical))**2
    
    H_total_squared = rel_n_p + rel_L_p + rel_R_p + rel_V_A
    
    print("\nH error contributions:")
    print(f"Primary turns uncertainty: {(rel_n_p/H_total_squared)*100:.1f}%")
    print(f"Length uncertainty: {(rel_L_p/H_total_squared)*100:.1f}%")
    print(f"Resistance uncertainty: {(rel_R_p/H_total_squared)*100:.1f}%")
    print(f"Voltage measurement: {(rel_V_A/H_total_squared)*100:.1f}%")
    print(f"Sum of contributions: {((rel_n_p + rel_L_p + rel_R_p + rel_V_A)/H_total_squared)*100:.1f}%")
    
    # For B (using B channel)
    rel_R_i = (R_i_err/R_i)**2
    rel_C = (C_err/C)**2
    rel_n_s = (n_s_err/n_s)**2
    rel_A_s = (A_s_err/A_s)**2
    rel_V_B = (B_resolution/np.abs(V_B_typical))**2
    
    B_total_squared = rel_R_i + rel_C + rel_n_s + rel_A_s + rel_V_B
    
    print("\nB error contributions:")
    print(f"Input resistance uncertainty: {(rel_R_i/B_total_squared)*100:.1f}%")
    print(f"Capacitance uncertainty: {(rel_C/B_total_squared)*100:.1f}%")
    print(f"Secondary turns uncertainty: {(rel_n_s/B_total_squared)*100:.1f}%")
    print(f"Area uncertainty: {(rel_A_s/B_total_squared)*100:.1f}%")
    print(f"Voltage measurement: {(rel_V_B/B_total_squared)*100:.1f}%")
    print(f"Sum of contributions: {((rel_R_i + rel_C + rel_n_s + rel_A_s + rel_V_B)/B_total_squared)*100:.1f}%")

# Calculate contributions for a typical point (using median absolute values)
typical_VA = np.median(np.abs(A_smooth[A_smooth != 0]))
typical_VB = np.median(np.abs(B_smooth[B_smooth != 0]))
print_error_contributions(typical_VA, typical_VB)






# Calculate μᵣ for all points first
dB_dH = np.gradient(B, H)
mu_r = dB_dH / mu_0

# Apply smoothing to all points
mu_r_smooth = savgol_filter(mu_r, 100, polyorder)

# Now filter for analysis (but keep original arrays for plotting)
mask = (H >= -20000) & (H <= 20000)
mu_r_filtered = mu_r[mask]
mu_r_smooth_filtered = mu_r_smooth[mask]
H_filtered = H[mask]

# Filter out invalid values from the filtered range
valid_mask = np.isfinite(mu_r_filtered)
valid_mu_r = mu_r_filtered[valid_mask]
valid_H = H_filtered[valid_mask]

# Print relative permeability statistics using filtered values
print("\nRelative Permeability (μᵣ) Analysis:")
print(f"Maximum μᵣ: {np.max(valid_mu_r):.0f}")
print(f"Minimum μᵣ: {np.min(valid_mu_r):.0f}")
print(f"Median μᵣ: {np.median(valid_mu_r):.0f}")

# Create plot using full range data
plt.figure(figsize=(12, 8))
plt.plot(H, mu_r_smooth, 'b-', linewidth=1.5, label='μᵣ vs H')
plt.title('Relative Permeability vs Magnetic Field Strength', fontsize=14)
plt.xlabel('H (A/m)', fontsize=12)
plt.ylabel('Relative Permeability (μᵣ)', fontsize=12)
plt.xlim(-20000,20000)
plt.ylim(0,250)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

# Add axis lines
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('relative_permeability.png', dpi=300, bbox_inches='tight')
plt.show()

def calculate_mu_r_error(dB_dH, H_errors, B_errors, H, B):
    # Calculate relative errors for a given point
    rel_H_error = H_errors/np.abs(H)
    rel_B_error = B_errors/np.abs(B)
    
    # Combine errors (adding gradient uncertainty would make this more complex)
    rel_mu_r_error = np.sqrt(rel_B_error**2 + rel_H_error**2)
    
    return np.abs(dB_dH/mu_0) * rel_mu_r_error

# Calculate errors using all points
mu_r_errors = calculate_mu_r_error(dB_dH, H_errors, B_errors, H, B)

# Filter errors for analysis
mu_r_errors_filtered = mu_r_errors[mask]
valid_mu_r_errors = mu_r_errors_filtered[valid_mask]

print(f"\nRelative Permeability Uncertainty Analysis:")
print(f"Typical relative uncertainty in μᵣ: {np.median(valid_mu_r_errors/np.abs(valid_mu_r))*100:.1f}%")

# Print range with uncertainty using filtered values
typical_mu_r = np.median(valid_mu_r)
typical_mu_r_error = np.median(valid_mu_r_errors)
print(f"\nTypical μᵣ value: {typical_mu_r:.0f} ± {typical_mu_r_error:.0f}")


# Calculate maximum permeability from filtered values
max_mu_r = np.max(valid_mu_r)
max_index = np.argmax(valid_mu_r)
max_mu_r_error = valid_mu_r_errors[max_index]
print(f"Maximum μᵣ: {max_mu_r:.0f} ± {max_mu_r_error:.0f}")

# Calculate minimum permeability from filtered values
min_mu_r = np.min(valid_mu_r)
min_index = np.argmin(valid_mu_r)
min_mu_r_error = valid_mu_r_errors[min_index]
print(f"Minimum μᵣ: {min_mu_r:.2f} ± {min_mu_r_error:.2f}")

#energy dissipation calculation
def calculate_energy_loss(H, B, B_errors):
    # Main integral for energy
    energy = np.trapz(B, H)
    
    # Calculate dH values
    dH = np.abs(np.diff(H))  # Take absolute value of H increments
    
    # Calculate uncertainty by summing up |error_B|*|dH| for each segment
    uncertainty = np.sum(np.abs(B_errors[:-1]) * dH)
    
    # Calculate percentage error
    percent_error = (uncertainty/abs(energy)) * 100
    
    print("\nEnergy Loss Analysis:")
    print(f"Energy dissipated per cycle: {energy:.2e} ± {uncertainty:.2e} J/m³")
    print(f"Relative uncertainty: {percent_error:.1f}%")
    
    return energy, uncertainty

# Call the function after B-H loop calculation
energy, uncertainty = calculate_energy_loss(H, B, B_errors)
