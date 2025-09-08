import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# ---------------------------------------
# Helper: Plot membership functions
# ---------------------------------------
def plot_mf(var, title):
    fig, ax = plt.subplots()
    for term in var.terms:
        ax.plot(var.universe, var[term].mf, label=term)
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

# ---------------------------------------
# Define FLC-I
# ---------------------------------------
def build_flc_I():
    dTin = ctrl.Antecedent(np.arange(-10, 11, 0.1), 'ΔTin')
    dHin = ctrl.Antecedent(np.arange(-50, 51, 0.5), 'ΔHin')

    Cooling = ctrl.Consequent(np.arange(0, 51, 1), 'Cooling')
    Heating = ctrl.Consequent(np.arange(0, 901, 1), 'Heating')
    Humidification = ctrl.Consequent(np.arange(0, 51, 1), 'Humidification')
    Dehumidification = ctrl.Consequent(np.arange(0, 51, 1), 'Dehumidification')

    # Membership functions (inputs)
    #dTin
    dTin['NB'] = fuzz.trapmf(dTin.universe, [-10, -10, -5, -1])
    dTin['NM'] = fuzz.trimf(dTin.universe, [-5, -2.5, -0.2])
    dTin['Z']  = fuzz.trimf(dTin.universe, [-0.4, 0, 0.4])
    dTin['PM'] = fuzz.trimf(dTin.universe, [0.2, 2.5, 5])
    dTin['PB'] = fuzz.trapmf(dTin.universe, [3, 5, 10, 10])
    #dHin
    dHin['NB'] = fuzz.trapmf(dHin.universe, [-50, -50, -30, -20])
    dHin['NM'] = fuzz.trimf(dHin.universe, [-30, -15, -4])
    dHin['Z']  = fuzz.trimf(dHin.universe, [-5, 0, 5])
    dHin['PM'] = fuzz.trimf(dHin.universe, [4, 15, 30])
    dHin['PB'] = fuzz.trapmf(dHin.universe, [20, 30, 50, 50])

    # Membership functions (outputs)
    Cooling['Z'] = fuzz.trimf(Cooling.universe, [0, 0, 8])
    Cooling['M'] = fuzz.trimf(Cooling.universe, [1, 16, 30])
    Cooling['H'] = fuzz.trimf(Cooling.universe, [15, 30, 50])

    Heating['Z'] = fuzz.trimf(Heating.universe, [0, 0, 18])
    Heating['M'] = fuzz.trimf(Heating.universe, [17, 200, 450])
    Heating['H'] = fuzz.trimf(Heating.universe, [450, 700, 900])

    Humidification['Z'] = fuzz.trimf(Humidification.universe, [0, 0, 5])
    Humidification['M'] = fuzz.trimf(Humidification.universe, [4, 20, 35])
    Humidification['H'] = fuzz.trimf(Humidification.universe, [32.5, 40, 50])

    Dehumidification['Z'] = fuzz.trimf(Dehumidification.universe, [0, 0, 4])
    Dehumidification['M'] = fuzz.trimf(Dehumidification.universe, [3, 20, 35])
    Dehumidification['H'] = fuzz.trimf(Dehumidification.universe, [32.5, 40, 50])

    # Rules
    rules = [
    # dTin = NB
        ctrl.Rule(dTin['NB'] & dHin['NB'], (Cooling['H'], Heating['Z'], Humidification['Z'], Dehumidification['H'])),
        ctrl.Rule(dTin['NB'] & dHin['NM'], (Cooling['H'], Heating['Z'], Humidification['Z'], Dehumidification['M'])),
        ctrl.Rule(dTin['NB'] & dHin['Z'],  (Cooling['H'], Heating['Z'], Humidification['Z'], Dehumidification['Z'])),
        ctrl.Rule(dTin['NB'] & dHin['PM'], (Cooling['H'], Heating['Z'], Humidification['M'], Dehumidification['Z'])),
        ctrl.Rule(dTin['NB'] & dHin['PB'], (Cooling['H'], Heating['Z'], Humidification['H'], Dehumidification['Z'])),

    # dTin = NM
        ctrl.Rule(dTin['NM'] & dHin['NB'], (Cooling['M'], Heating['Z'], Humidification['Z'], Dehumidification['H'])),
        ctrl.Rule(dTin['NM'] & dHin['NM'], (Cooling['M'], Heating['Z'], Humidification['Z'], Dehumidification['M'])),
        ctrl.Rule(dTin['NM'] & dHin['Z'],  (Cooling['M'], Heating['Z'], Humidification['Z'], Dehumidification['Z'])),
        ctrl.Rule(dTin['NM'] & dHin['PM'], (Cooling['M'], Heating['Z'], Humidification['M'], Dehumidification['Z'])),
        ctrl.Rule(dTin['NM'] & dHin['PB'], (Cooling['M'], Heating['Z'], Humidification['H'], Dehumidification['Z'])),

    # dTin = Z
        ctrl.Rule(dTin['Z'] & dHin['NB'], (Cooling['Z'], Heating['Z'], Humidification['Z'], Dehumidification['H'])),
        ctrl.Rule(dTin['Z'] & dHin['NM'], (Cooling['Z'], Heating['Z'], Humidification['Z'], Dehumidification['M'])),
        ctrl.Rule(dTin['Z'] & dHin['Z'],  (Cooling['Z'], Heating['Z'], Humidification['Z'], Dehumidification['Z'])),
        ctrl.Rule(dTin['Z'] & dHin['PM'], (Cooling['Z'], Heating['Z'], Humidification['M'], Dehumidification['Z'])),
        ctrl.Rule(dTin['Z'] & dHin['PB'], (Cooling['Z'], Heating['Z'], Humidification['H'], Dehumidification['Z'])),

    # dTin = PM
        ctrl.Rule(dTin['PM'] & dHin['NB'], (Cooling['Z'], Heating['M'], Humidification['Z'], Dehumidification['H'])),
        ctrl.Rule(dTin['PM'] & dHin['NM'], (Cooling['Z'], Heating['M'], Humidification['Z'], Dehumidification['M'])),
        ctrl.Rule(dTin['PM'] & dHin['Z'],  (Cooling['Z'], Heating['M'], Humidification['Z'], Dehumidification['Z'])),
        ctrl.Rule(dTin['PM'] & dHin['PM'], (Cooling['Z'], Heating['M'], Humidification['M'], Dehumidification['Z'])),
        ctrl.Rule(dTin['PM'] & dHin['PB'], (Cooling['Z'], Heating['M'], Humidification['H'], Dehumidification['Z'])),

        # dTin = PB
        ctrl.Rule(dTin['PB'] & dHin['NB'], (Cooling['Z'], Heating['H'], Humidification['Z'], Dehumidification['H'])),
        ctrl.Rule(dTin['PB'] & dHin['NM'], (Cooling['Z'], Heating['H'], Humidification['Z'], Dehumidification['M'])),
        ctrl.Rule(dTin['PB'] & dHin['Z'],  (Cooling['Z'], Heating['H'], Humidification['Z'], Dehumidification['Z'])),
        ctrl.Rule(dTin['PB'] & dHin['PM'], (Cooling['Z'], Heating['H'], Humidification['M'], Dehumidification['Z'])),
        ctrl.Rule(dTin['PB'] & dHin['PB'], (Cooling['Z'], Heating['H'], Humidification['H'], Dehumidification['Z']))
    ]

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    return sim, [dTin, dHin, Cooling, Heating, Humidification, Dehumidification]

# ---------------------------------------
# Define FLC-II
# ---------------------------------------
def build_flc_II():
    dTin = ctrl.Antecedent(np.arange(-10, 11, 0.1), 'ΔTin')
    dTout = ctrl.Antecedent(np.arange(-20, 21, 0.1), 'ΔTout')
    dHin = ctrl.Antecedent(np.arange(-50, 51, 0.5), 'ΔHin')
    dHout = ctrl.Antecedent(np.arange(-50, 51, 0.5), 'ΔHout')

    Cooling = ctrl.Consequent(np.arange(0, 51, 1), 'Cooling')
    Heating = ctrl.Consequent(np.arange(0, 901, 1), 'Heating')
    Humidification = ctrl.Consequent(np.arange(0, 51, 1), 'Humidification')
    Dehumidification = ctrl.Consequent(np.arange(0, 51, 1), 'Dehumidification')
    Ventilation = ctrl.Consequent(np.arange(0, 51, 1), 'Ventilation')

    # Inputs membership functions
    dTin['NB'] = fuzz.trapmf(dTin.universe, [-10, -10, -7, -5])
    dTin['NM'] = fuzz.trimf(dTin.universe, [-6, -3.5, -1])
    dTin['Z']  = fuzz.trimf(dTin.universe, [-2, 0, 2])
    dTin['PM'] = fuzz.trimf(dTin.universe, [1, 3.5, 6])
    dTin['PB'] = fuzz.trapmf(dTin.universe, [5, 7, 10, 10])

    dTout['NB'] = fuzz.trapmf(dTout.universe, [-20, -20, -15, -10])
    dTout['NM'] = fuzz.trimf(dTout.universe, [-12, -7, -2])
    dTout['Z']  = fuzz.trimf(dTout.universe, [-4, 0, 4])
    dTout['PM'] = fuzz.trimf(dTout.universe, [2.5, 7, 12])
    dTout['PB'] = fuzz.trapmf(dTout.universe, [10, 15, 20, 20])

    dHin['NB'] = fuzz.trapmf(dHin.universe, [-50, -50, -40, -20])
    dHin['NM'] = fuzz.trimf(dHin.universe, [-30, -15, -4])
    dHin['Z']  = fuzz.trimf(dHin.universe, [-5, 0, 5])
    dHin['PM'] = fuzz.trimf(dHin.universe, [4, 15, 30])
    dHin['PB'] = fuzz.trapmf(dHin.universe, [20, 30, 50, 50])

    dHout['NB'] = fuzz.trapmf(dHout.universe, [-50, -50, -40, -20])
    dHout['NM'] = fuzz.trimf(dHout.universe, [-30, -15, -4])
    dHout['Z']  = fuzz.trimf(dHout.universe, [-5, 0, 5])
    dHout['PM'] = fuzz.trimf(dHout.universe, [4, 15, 30])
    dHout['PB'] = fuzz.trapmf(dHout.universe, [20, 30, 50, 50])

    # Cooling (0–50)
    Cooling['Z']  = fuzz.trimf(Cooling.universe, [0, 0, 10])
    Cooling['L']  = fuzz.trimf(Cooling.universe, [5, 15, 20])
    Cooling['M']  = fuzz.trimf(Cooling.universe, [15, 25, 35])
    Cooling['H']  = fuzz.trimf(Cooling.universe, [30, 40, 45])
    Cooling['VH'] = fuzz.trimf(Cooling.universe, [40, 50, 50])

    # Heating (0–900)
    Heating['Z']  = fuzz.trimf(Heating.universe, [0, 0, 50])
    Heating['L']  = fuzz.trimf(Heating.universe, [50, 200, 350])
    Heating['M']  = fuzz.trimf(Heating.universe, [300, 450, 600])
    Heating['H']  = fuzz.trimf(Heating.universe, [550, 700, 800])
    Heating['VH'] = fuzz.trimf(Heating.universe, [750, 900, 900])

    # Humidification (0–50)
    Humidification['Z']  = fuzz.trimf(Humidification.universe, [0, 0, 10])
    Humidification['L']  = fuzz.trimf(Humidification.universe, [5, 15, 20])
    Humidification['M']  = fuzz.trimf(Humidification.universe, [15, 25, 35])
    Humidification['H']  = fuzz.trimf(Humidification.universe, [30, 40, 45])
    Humidification['VH'] = fuzz.trimf(Humidification.universe, [40, 50, 50])

    # Dehumidification (0–50)
    Dehumidification['Z']  = fuzz.trimf(Dehumidification.universe, [0, 0, 10])
    Dehumidification['L']  = fuzz.trimf(Dehumidification.universe, [5, 15, 20])
    Dehumidification['M']  = fuzz.trimf(Dehumidification.universe, [15, 25, 35])
    Dehumidification['H']  = fuzz.trimf(Dehumidification.universe, [30, 40, 45])
    Dehumidification['VH'] = fuzz.trimf(Dehumidification.universe, [40, 50, 50])

    # Membership function
    Ventilation['Z']  = fuzz.trimf(Ventilation.universe, [0, 0, 10])   # off / very low
    Ventilation['L']  = fuzz.trimf(Ventilation.universe, [5, 15, 20])  # low speed
    Ventilation['M']  = fuzz.trimf(Ventilation.universe, [15, 25, 35]) # medium speed
    Ventilation['H']  = fuzz.trimf(Ventilation.universe, [30, 40, 45]) # high speed
    Ventilation['VH'] = fuzz.trimf(Ventilation.universe, [40, 50, 50]) # max speed


    temp_rules = [
    # Indoor NB (too hot, need cooling) → outdoor cooler air helps more
    ctrl.Rule(dTin['NB'] & dTout['NB'], (Cooling['VH'], Heating['Z'], Ventilation['Z'])),
    ctrl.Rule(dTin['NB'] & dTout['NM'], (Cooling['H'],  Heating['Z'], Ventilation['L'])),
    ctrl.Rule(dTin['NB'] & dTout['Z'],  (Cooling['M'],  Heating['Z'], Ventilation['M'])),
    ctrl.Rule(dTin['NB'] & dTout['PM'], (Cooling['L'],  Heating['Z'], Ventilation['H'])),
    ctrl.Rule(dTin['NB'] & dTout['PB'], (Cooling['Z'],  Heating['Z'], Ventilation['VH'])),

    # Indoor NM (slightly hot → cooling, weaker intensity)
    ctrl.Rule(dTin['NM'] & dTout['NB'], (Cooling['H'],  Heating['Z'], Ventilation['Z'])),
    ctrl.Rule(dTin['NM'] & dTout['NM'], (Cooling['M'],  Heating['Z'], Ventilation['Z'])),
    ctrl.Rule(dTin['NM'] & dTout['Z'],  (Cooling['L'],  Heating['Z'], Ventilation['L'])),
    ctrl.Rule(dTin['NM'] & dTout['PM'], (Cooling['L'],  Heating['Z'], Ventilation['M'])),
    ctrl.Rule(dTin['NM'] & dTout['PB'], (Cooling['Z'],  Heating['Z'], Ventilation['VH'])),

    # Indoor Z (near setpoint → minimal corrections; outdoor nudges small)
    ctrl.Rule(dTin['Z'] & dTout['NB'], (Cooling['M'],  Heating['Z'], Ventilation['Z'])),
    ctrl.Rule(dTin['Z'] & dTout['NM'], (Cooling['L'],  Heating['Z'], Ventilation['Z'])),
    ctrl.Rule(dTin['Z'] & dTout['Z'],  (Cooling['Z'],  Heating['Z'], Ventilation['Z'])),
    ctrl.Rule(dTin['Z'] & dTout['PM'], (Cooling['Z'],  Heating['L'], Ventilation['Z'])),
    ctrl.Rule(dTin['Z'] & dTout['PB'], (Cooling['Z'],  Heating['M'], Ventilation['Z'])),

    # Indoor PM (slightly cold → heating, outdoor warmer helps less)
    ctrl.Rule(dTin['PM'] & dTout['NB'], (Cooling['Z'],  Heating['L'], Ventilation['M'])),
    ctrl.Rule(dTin['PM'] & dTout['NM'], (Cooling['Z'],  Heating['M'], Ventilation['L'])),
    ctrl.Rule(dTin['PM'] & dTout['Z'],  (Cooling['Z'],  Heating['H'], Ventilation['Z'])),
    ctrl.Rule(dTin['PM'] & dTout['PM'], (Cooling['Z'],  Heating['H'], Ventilation['Z'])),
    ctrl.Rule(dTin['PM'] & dTout['PB'], (Cooling['Z'],  Heating['VH'], Ventilation['Z'])),

    # Indoor PB (too cold → strong heating; outdoor also cold → VH heating)
    ctrl.Rule(dTin['PB'] & dTout['NB'], (Cooling['Z'],  Heating['M'], Ventilation['VH'])),
    ctrl.Rule(dTin['PB'] & dTout['NM'], (Cooling['Z'],  Heating['H'], Ventilation['H'])),
    ctrl.Rule(dTin['PB'] & dTout['Z'],  (Cooling['Z'],  Heating['H'], Ventilation['M'])),
    ctrl.Rule(dTin['PB'] & dTout['PM'], (Cooling['Z'],  Heating['VH'], Ventilation['L'])),
    ctrl.Rule(dTin['PB'] & dTout['PB'], (Cooling['Z'],  Heating['VH'], Ventilation['Z']))
    ]


    humid_rules = [
    # Indoor NB (too dry → humidification needed; more if outdoor is also dry)
    ctrl.Rule(dHin['NB'] & dHout['NB'], (Humidification['VH'], Dehumidification['Z'], Ventilation['Z'])),
    ctrl.Rule(dHin['NB'] & dHout['NM'], (Humidification['H'],  Dehumidification['Z'], Ventilation['L'])),
    ctrl.Rule(dHin['NB'] & dHout['Z'],  (Humidification['M'],  Dehumidification['Z'], Ventilation['M'])),
    ctrl.Rule(dHin['NB'] & dHout['PM'], (Humidification['L'],  Dehumidification['Z'], Ventilation['H'])),
    ctrl.Rule(dHin['NB'] & dHout['PB'], (Humidification['Z'],  Dehumidification['Z'], Ventilation['VH'])),

    # Indoor NM (slightly dry → humidification, weaker)
    ctrl.Rule(dHin['NM'] & dHout['NB'], (Humidification['H'],  Dehumidification['Z'], Ventilation['Z'])),
    ctrl.Rule(dHin['NM'] & dHout['NM'], (Humidification['M'],  Dehumidification['Z'], Ventilation['Z'])),
    ctrl.Rule(dHin['NM'] & dHout['Z'],  (Humidification['L'],  Dehumidification['Z'], Ventilation['L'])),
    ctrl.Rule(dHin['NM'] & dHout['PM'], (Humidification['Z'],  Dehumidification['Z'], Ventilation['H'])),
    ctrl.Rule(dHin['NM'] & dHout['PB'], (Humidification['Z'],  Dehumidification['L'], Ventilation['VH'])),

    # Indoor Z (balanced → minimal action, outdoor nudges)
    ctrl.Rule(dHin['Z'] & dHout['NB'], (Humidification['M'],  Dehumidification['Z'], Ventilation['Z'])),
    ctrl.Rule(dHin['Z'] & dHout['NM'], (Humidification['L'],  Dehumidification['Z'], Ventilation['Z'])),
    ctrl.Rule(dHin['Z'] & dHout['Z'],  (Humidification['Z'],  Dehumidification['Z'], Ventilation['Z'])),
    ctrl.Rule(dHin['Z'] & dHout['PM'], (Humidification['Z'],  Dehumidification['L'], Ventilation['Z'])),
    ctrl.Rule(dHin['Z'] & dHout['PB'], (Humidification['Z'],  Dehumidification['M'], Ventilation['Z'])),

    # Indoor PM (slightly humid → dehumidification, outdoor dryness helps)
    ctrl.Rule(dHin['PM'] & dHout['NB'], (Humidification['Z'],  Dehumidification['L'], Ventilation['L'])),
    ctrl.Rule(dHin['PM'] & dHout['NM'], (Humidification['Z'],  Dehumidification['M'], Ventilation['M'])),
    ctrl.Rule(dHin['PM'] & dHout['Z'],  (Humidification['Z'],  Dehumidification['H'], Ventilation['H'])),
    ctrl.Rule(dHin['PM'] & dHout['PM'], (Humidification['Z'],  Dehumidification['H'], Ventilation['Z'])),
    ctrl.Rule(dHin['PM'] & dHout['PB'], (Humidification['Z'],  Dehumidification['VH'], Ventilation['Z'])),

    # Indoor PB (too humid → strong dehumidification; VH if outside also humid)
    ctrl.Rule(dHin['PB'] & dHout['NB'], (Humidification['Z'],  Dehumidification['L'], Ventilation['VH'])),
    ctrl.Rule(dHin['PB'] & dHout['NM'], (Humidification['Z'],  Dehumidification['L'], Ventilation['VH'])),
    ctrl.Rule(dHin['PB'] & dHout['Z'],  (Humidification['Z'],  Dehumidification['H'], Ventilation['H'])),
    ctrl.Rule(dHin['PB'] & dHout['PM'], (Humidification['Z'],  Dehumidification['VH'], Ventilation['Z'])),
    ctrl.Rule(dHin['PB'] & dHout['PB'], (Humidification['Z'],  Dehumidification['VH'], Ventilation['Z']))
    ]

    # Simplified rule base: demonstration (paper has full Mamdani tables)
    rules = temp_rules + humid_rules

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    return sim, [dTin, dTout, dHin, dHout, Cooling, Heating, Humidification, Dehumidification, Ventilation]


# ---------------------------------------
# Streamlit UI
# ---------------------------------------
st.title("🌿 Smart Greenhouse Fuzzy Controller")

model_choice = st.sidebar.radio("Choose Fuzzy Controller:", ["FLC-I", "FLC-II"])

if model_choice == "FLC-I":
    st.header("FLC-I: Indoor Only")
    Topt = st.sidebar.slider("Optimal Temperature (°C)", 15, 35, 25)
    Hopt = st.sidebar.slider("Optimal Humidity (%)", 30, 80, 60)
    Tin = st.sidebar.slider("Indoor Temp (°C)", Topt-10, Topt+10, 28)
    Hin = st.sidebar.slider("Indoor Humidity (%)", Hopt-50, Hopt+50, 50)
    # Tin = st.sidebar.slider("Current Indoor Temperature (°C)", 10, 40, 28)
    # Hin = st.sidebar.slider("Current Indoor Humidity (%)", 20, 90, 50)

    dTin_val = Topt - Tin
    dHin_val = Hopt - Hin

    st.write(f"ΔTin = {dTin_val:.2f}, ΔHin = {dHin_val:.2f}")
    sim, vars = build_flc_I()
    sim.input['ΔTin'] = dTin_val
    sim.input['ΔHin'] = dHin_val
    sim.compute()

    st.subheader("Control Actions")
    for out in ['Cooling','Heating','Humidification','Dehumidification']:
        st.write(f"{out}: **{sim.output[out]:.2f}**")

    st.subheader("📊 Membership Functions")
    for v in vars:
        plot_mf(v, v.label)

else:
    st.header("FLC-II: Indoor + Outdoor")
    Topt = st.sidebar.slider("Optimal Temperature (°C)", 15, 35, 25)
    Hopt = st.sidebar.slider("Optimal Humidity (%)", 30, 80, 60)
    # Indoor Temp slider → ensures ΔTin ∈ [-10, 10]
    Tin = st.sidebar.slider("Indoor Temp (°C)", Topt-10, Topt+10, Topt)
    # Outdoor Temp slider → ensures ΔTout ∈ [-20, 20]
    Tout = st.sidebar.slider("Outdoor Temp (°C)", Topt-20, Topt+20, Topt)
    # Indoor Humidity slider → ensures ΔHin ∈ [-50, 50]
    Hin = st.sidebar.slider("Indoor Humidity (%)", max(0, Hopt-50), min(100, Hopt+50), Hopt)
    # Outdoor Humidity slider → ensures ΔHout ∈ [-50, 50]
    Hout = st.sidebar.slider("Outdoor Humidity (%)", max(0, Hopt-50), min(100, Hopt+50), Hopt)

    # Tin = st.sidebar.slider("Indoor Temp (°C)", 10, 40, 28)
    # Tout = st.sidebar.slider("Outdoor Temp (°C)", -5, 45, 30)
    # Hin = st.sidebar.slider("Indoor Humidity (%)", 20, 90, 50)
    # Hout = st.sidebar.slider("Outdoor Humidity (%)", 10, 90, 70)

    dTin_val = Topt - Tin
    dTout_val = Topt - Tout
    dHin_val = Hopt - Hin
    dHout_val = Hopt - Hout

    st.write(f"ΔTin = {dTin_val:.2f}, ΔTout = {dTout_val:.2f}")
    st.write(f"ΔHin = {dHin_val:.2f}, ΔHout = {dHout_val:.2f}")

    sim, vars = build_flc_II()
    sim.input['ΔTin'] = dTin_val
    sim.input['ΔTout'] = dTout_val
    sim.input['ΔHin'] = dHin_val
    sim.input['ΔHout'] = dHout_val
    sim.compute()

    # Collect raw outputs
    outputs = {out: sim.output[out] for out in ['Cooling','Heating','Humidification','Dehumidification','Ventilation']}


    st.subheader("🚀 Control Actions")
    for out, val in outputs.items():
        st.write(f"{out}: **{val:.2f}**")


    st.subheader("📊 Membership Functions")
    for v in vars:
        plot_mf(v, v.label)
