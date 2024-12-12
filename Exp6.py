#!/usr/bin/env python
# coding: utf-8

# In[75]:



from picosdk.discover import find_unit
from picosdk.device import ChannelConfig, TimebaseOptions
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import pandas as pd

# Capture data from Picoscope
with find_unit() as device:
    channel_configs = [
        ChannelConfig('A', True, 'AC', 1.0),
        ChannelConfig('B', True, 'AC', 1)
    ]
    timebase_options = TimebaseOptions(min_collection_time=20e-3)
    times, voltages, overflow_warnings = device.capture_block(timebase_options, channel_configs)

# Constants
A_s = 7.7 * 10**-6  # Area for mild steel

# Extract and downsample data
t = times[::1000]
v_x = 10*voltages["A"][::1000]
v_y = 10*voltages["B"][::1000]

# Compute H and B
H=(v_x*400)/(0.0437*2.3)
B=(v_y*2160*438.5*1e-9)/(500*A_s)

# Smooth H and B
H_smooth = savgol_filter(H, window_length=501, polyorder=3)
B_smooth = savgol_filter(B, window_length=501, polyorder=3)

# Compute dB/dH with safe handling of division by zero
# Use np.diff to calculate differences and add epsilon to avoid zero differences
dH = np.diff(H_smooth)
dB = np.diff(B_smooth)

# Handle zero differences to avoid division by zero
dH[dH == 0] = 100

# Compute dB/dH
dB_dH = dB / dH

# Pad dB/dH to match the length of original H and B arrays
dB_dH = np.pad(dB_dH, (1, 0), 'edge')

# Replace NaN or Inf values in dB/dH
dB_dH = np.nan_to_num(dB_dH, nan=0.0, posinf=0.0, neginf=0.0)

# Compute relative permeability (mu_r)
mu_r = dB_dH / (4 * np.pi * 10**-7)

# Plot B-H curve
plt.plot(H_smooth, B_smooth)
plt.xlabel('H (A/m)')
plt.ylabel('B (T)')
plt.title('B-H curve for Cu-Ni Alloy 28$^\circ$C')
plt.grid()
plt.savefig(r'C:\Users\Classes\Desktop\fabricated-data\alloy28.jpeg', bbox_inches='tight', dpi=600)
plt.show()

df = pd.DataFrame({"time/seconds":t,
                "A/V":v_x,
                "B/V":v_y})
df.to_csv(r'C:\Users\Classes\Desktop\fabricated-data\alloy-28.csv', index=False)
# Plot mu_r against H
plt.plot(H_smooth, mu_r)
plt.title('The Dependence of $\mu_{r}$ on H for Cu/Ni alloy 28$^\circ$C')
plt.xlabel('H (A/m)')
plt.ylabel('$\mu_{r}$')
plt.xlim(-15000,15000)
plt.ylim(0,200)
plt.grid()
plt.savefig(r'C:\Users\Classes\Desktop\fabricated-data\mu_r_CuNi_28.png', bbox_inches='tight', dpi=600)
plt.show()

# Calculate area under the B-H curve
area = np.trapz(B_smooth, H_smooth)
print(f"Area under the curve: {area:.6f} J/m³")


