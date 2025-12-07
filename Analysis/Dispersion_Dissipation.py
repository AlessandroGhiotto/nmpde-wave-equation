import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

#u(t)_x0_y0.csv

csv_file = "grafico.csv"
df = pd.read_csv(csv_file, quotechar='"')
df.rename(columns=lambda x: x.strip('"'), inplace=True)

timestep = df["Time"].values.astype(float)
u_num = df["avg(u)"].values.astype(float)

N = len(timestep)
dt = np.mean(np.diff(timestep)) 

delta_t_real = 0.0035  #timestep
t = timestep * delta_t_real   
dt_real = dt * delta_t_real

#Computing FFT
U = fft(u_num)
freqs = fftfreq(N, dt_real)

#Only positive frequencies
mask = freqs > 0
freqs_pos = freqs[mask]
U_pos = U[mask]

#Dominant frequency
idx_max = np.argmax(np.abs(U_pos))
f_num = freqs_pos[idx_max]
omega_num = 2 * np.pi * f_num


#Plot of the signals at P point over the time
plt.figure(figsize=(10,4))
plt.plot(t, u_num, label="Numerical u(t)")
plt.xlabel("Time [s]")
plt.ylabel("u")
plt.title("Signal at point (0.5 , 0.5)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("=========== NUMERICAL DOMINANT FREQUENCY ===========")
print(f"f_num = {f_num:.6f} Hz")
print(f"ω_num = {omega_num:.6f} rad/s")



# FFT plot
plt.figure(figsize=(10,4))
plt.plot(freqs_pos, np.abs(U_pos), label="FFT Numerical")
plt.axvline(f_num, color='r', linestyle='--', label=f"Dominant f_num = {f_num:.4f} Hz")
plt.axvline(f_exact, color='g', linestyle='--', label=f"Theoretical f_exact = {f_exact:.4f} Hz")
plt.xlabel("Frequency [Hz]")
plt.ylabel("|U(f)|")
plt.title("FFT of the signal with numerical vs theoretical frequency")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


#Dissipation
peaks, _ = find_peaks(u_num)
A_peaks = u_num[peaks]

print("\n=========== Numerical Dissipation ===========")
if len(A_peaks) > 2:
    alpha_values = -np.log(A_peaks[1:] / A_peaks[:-1]) / (t[peaks][1:] - t[peaks][:-1])
    alpha_mean = np.mean(alpha_values)
    print(f"Estimated numerical damping: α ≈ {alpha_mean:.6e} 1/s")
else:
    print("Not enough points to estimate dissipation")


# Theoretical frequency for u(x,y,t) = sin(pi*x) sin(pi*y) cos(sqrt(2)*pi t)
omega_exact = np.sqrt(2) * np.pi
f_exact = omega_exact / (2*np.pi)

delta_omega = omega_num - omega_exact

print("\n=========== Theoretical comparison ===========")
print(f"Theoretical frequency: f_exact = {f_exact:.6f} Hz")
print(f"Therotical pulsation: ω_exact = {omega_exact:.6f} rad/s")
print(f"Error of numerical dispersion: Δω = {delta_omega:.6f} rad/s")
