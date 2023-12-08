# Calculations in support of Sec 5.2 of Bloot++ 2024
# Phenomenology and periodicity of radio emission from the stellar
# system AU Microscopii
# Author: Harish Vedantham
# Dec 2023
#
import numpy as np
import matplotlib.pyplot as plt
#
# Energy loss rate of gyro-magnetic emitting charges in Au Mic
# Approximate calculation based on homogeneous region of size R*^3
# Use expression for nu_peak from Guedel 2002 to get value of density of 
# high energy charges
# Then use a simple collisional energy loss rate to find the power needed to 
# maintain the charge distribution
# delta is a parameter (slope of e- energy distyribution)
#
C_ev2ergs = 1.60218e-12		# eV to ergs
nupeak = 3e9			# Peak frequency (opt. thick to thin transition)
delta = np.arange(2.0,7.0,0.1)	# Power-law slope of e- energy distribution
B = 1000.0			# Magnetic field strength (Gauss)
L = 0.75*7e10			# Size of the emitting region (cm)
eps0 = 10e3 * C_ev2ergs		# Low energy cut off of power-law spectrum
n10 = 0.1			# Thermal plasma density (10^10 cm^-3)
#
# Equation 18 Guedel's review for peak frequency
NL = (nupeak/(10**(3.41+0.27*delta) * B**(0.68+0.03*delta)))**(1./(0.32-0.03*delta))
N = NL/L # Volume density of high energy charges (cm^-3)
#
# Numerical integration of energy loss over e- energy spectrum
#
log_eps = np.linspace(np.log(eps0),np.log(10e6*C_ev2ergs),1000)
d_log_eps = log_eps[1]-log_eps[0]
eps = np.exp(log_eps)
Edot = np.zeros(eps.shape) # Energy loss rate for diff e- energies (given delta)
energy_rate = np.zeros(delta.shape) # Energy loss rate for each delta
#
for i in range(len(delta)):
   d = delta[i]
   # Use eqn 2 of Bai & Ramaty for energy loss calculation
   # Edot = 4.9e-9 * n * E^-1/2 (keV/s) for E<=160 keV
   # Edot = 3.8e-10 * n (keV/s) for E>160 keV
   I = np.where(eps<=160e3*C_ev2ergs)[0]
   Edot[I] = 49.0*n10*(eps[I]/1e3/C_ev2ergs)**-0.5 * 1e3*C_ev2ergs
   I = np.where(eps>160e3*C_ev2ergs)[0]
   Edot[I] = 3.8*n10 * 1e3*C_ev2ergs
   # The e- distribution is (eqn 4 of Guedel's review)
   # N * (d-1)*eps0^(d-1)*eps^-d
   # L^3 is the volume
   # x by eps for logarithmic interval integration
   # integrand is the distribution function x Edot 
   # to give the tital energy loss rate in the volume due to collisions
   energy_rate[i] = L**3*N[i]*d_log_eps*np.sum((d-1)*eps0**(d-1)*eps**(-d+1)*Edot)

plt.plot(delta, energy_rate)
plt.xlabel(r"$\delta$")
plt.ylabel(r"Energy loss rate [erg/s]")
plt.tight_layout()
plt.savefig("aumic_coll_loss_rate.pdf")
plt.show()
plt.close()
