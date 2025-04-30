import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.title("Benzene Sparging Model")

# --- User Inputs ---
st.sidebar.header("Input Parameters")

C_i = st.sidebar.number_input("Initial Benzene Concentration, $C_i$ (mg/L)", value=1.0)
V_i = st.sidebar.number_input("Bio-oil Batch Size, $V_i$ (gallons)", value=8000.0)
Vtot_i = st.sidebar.number_input("Sparging Tank Volume, $V_{tot,i}$ (gallons)", value=12000.0)
benzene_threshold = st.sidebar.number_input("Benzene Threshold, $C_l$ (mg/L)", value=0.4)

temp_input = st.sidebar.text_input(
    "Temperatures to test (°F, comma-separated)", value="100, 110, 120"
)
try:
    temp_list = [float(t.strip()) for t in temp_input.split(",") if t.strip()]
except ValueError:
    st.error("Invalid temperature input. Please enter numbers separated by commas.")
    temp_list = []

st.sidebar.markdown("### Sparging Ramp (Time hr, Flow scfm)")
default_data = {
    "Time (hr)": [0, 1, 5, 50],
    "Flow (scfm)": [10, 50, 135, 135]
}
sparge_df = st.sidebar.data_editor(pd.DataFrame(default_data), num_rows="dynamic")
spargeramp_i = sparge_df.dropna().to_numpy()

# --- Model Functions ---
def kHcalc(T_1):  # T_0 and T_1 in K
    kH_0 = 118.15  # unitless
    T_0 = 273.15 + 40  # K
    delta_H_sol = -30000  # J/mol
    R = 8.314  # J/mol·K
    kH_1 = kH_0 * np.exp((delta_H_sol / R) * (1/T_0 - 1/T_1))
    return kH_1

def process_temperature(T_i):
    T = (T_i - 32) * 5/9  # °F → °C
    T_actual_K = T + 273.15
    kH = kHcalc(T_actual_K)
    return kH, T

def eqmodel(params, t_eval):
    n0, kLa, kH, Vliq, Vgas, Vdot = params

    def n_t(t):
        return n0 * np.exp(-Vdot * t / (Vgas + kH * Vliq))

    n_values = n_t(t_eval)
    Cl_values = kH * n_values / (Vgas + kH * Vliq)
    Cg_values = (n_values - Cl_values * Vliq) / Vgas
    results = np.column_stack((t_eval, n_values, Cl_values, Cg_values))
    return results

def run_simulation(T_i, spargeramp_i):
    kH, T = process_temperature(T_i)
    C = C_i / 78.11 / 1000
    Vliq = V_i * 3.78541
    Vgas = (Vtot_i - V_i) * 3.78541
    n_00 = C * Vliq

    hr_to_sec = 3600
    scfm_to_Lps_std = 28.3168 / 60
    T_std_K = 293.15
    T_actual_K = T + 273.15

    spargeramp = np.empty_like(spargeramp_i)
    spargeramp[:, 0] = spargeramp_i[:, 0] * hr_to_sec
    spargeramp[:, 1] = spargeramp_i[:, 1] * (T_actual_K / T_std_K) * scfm_to_Lps_std

    n_0 = n_00
    kLa = np.nan
    Vdot = spargeramp[0, 1]
    results = []
    tspace = 60 * 15
    for i in range(len(spargeramp) - 1):
        params1 = [n_0, kLa, kH, Vliq, Vgas, Vdot]
        start = spargeramp[i, 0]
        stop = spargeramp[i + 1, 0]
        num_points = int((stop - start) / tspace)
        t_eval1 = np.linspace(start, stop, num_points)
        t_eval1_zeroed = t_eval1 - t_eval1[0]
        results1 = eqmodel(params1, t_eval1_zeroed)
        results1[:, 0] = t_eval1
        if i == 0:
            results = results1
        else:
            results = np.vstack((results, results1))
        n_0 = results1[-1, 1]
        Vdot = spargeramp[i + 1, 1]
    return results

# --- Run Simulation ---
if st.button("Run Model"):
    #temps = [100, 110, 120]
    results_dict = {}
    tclp_times = {}

    for temp in temp_list:
        results = run_simulation(temp, spargeramp_i)
        time_hr = results[:, 0] / 3600
        Cl_mg_L = results[:, 2] * 78.11 * 1000
        Cg_mg_L = results[:, 3] * 78.11 * 1000
        results_dict[temp] = (time_hr, Cl_mg_L, Cg_mg_L)

        # Find first time Cl drops below user-specified threshold
        below_threshold = np.where(Cl_mg_L < benzene_threshold)[0]
        if below_threshold.size > 0:
            tclp_times[temp] = time_hr[below_threshold[0]]
        else:
            tclp_times[temp] = None

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    time = spargeramp_i[:, 0]
    flow = spargeramp_i[:, 1]
    axs[0].step(time, flow, where='post')
    axs[0].set_xlabel('Time (hr)', fontsize=14)
    axs[0].set_ylabel('Sparging Flow Rate (scfm)', fontsize=14)
    axs[0].set_title('Input: Air Flow Rate vs Time', fontsize=14)
    axs[0].set_ylim(bottom=0)
    axs[0].grid(True)

    for temp, (t, Cl, Cg) in results_dict.items():
        axs[1].plot(t, Cl, label=f'{temp}°F')
        axs[2].plot(t, Cg, label=f'{temp}°F')

    axs[1].set_xlabel('Time (hr)', fontsize=14)
    axs[1].set_ylabel('$C_l$ (mg benzene/L)', fontsize=14)
    axs[1].set_title('Output: Liquid Conc vs Time', fontsize=14)
    axs[1].grid(True)
    axs[1].set_ylim(bottom=0)
    axs[1].axhline(0.5, color='black', linestyle='--', linewidth=1, label='TCLP limit (0.5 mg/L)')
    axs[1].axhline(benzene_threshold, color='red', linestyle='--', linewidth=1, label=f'Threshold ({benzene_threshold} mg/L)')
    axs[1].legend()

    axs[2].set_xlabel('Time (hr)', fontsize=14)
    axs[2].set_ylabel('$C_g$ (mg benzene/L)', fontsize=14)
    axs[2].set_title('Output: Gas Conc vs Time', fontsize=14)
    axs[2].grid(True)
    axs[2].set_ylim(bottom=0)
    axs[2].legend()

    plt.tight_layout()
    st.pyplot(fig)

    # Display results
    st.markdown(f"### Time to reach user-defined threshold ({benzene_threshold} mg/L):")
    for temp in temp_list:
      t = tclp_times[temp]
      if t is not None:
        st.markdown(f"- **{temp}°F**: {t:.2f} hours")
      else:
        st.markdown(f"- **{temp}°F**: Not reached within simulation window")