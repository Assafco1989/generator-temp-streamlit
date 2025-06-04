import streamlit as st
import numpy as np
import onnxruntime as ort
import datetime
import pandas as pd
import os

session = ort.InferenceSession("bearing_temperature_model.onnx")
input_name = session.get_inputs()[0].name

st.set_page_config(page_title="U1 Generator TE Bearing Temp Prediction", layout="centered")
st.title("🔧 U1 Generator TE Bearing Temp Prediction")
st.markdown("Designed by **Eng. Mohammed Assaf**")

mw = st.slider("Active Power (MW)", 100, 300, 277)
mvar = st.slider("Reactive Power (MVAR)", -120, 40, -20)
oil_pressure = st.number_input("Oil Pressure (kPa)", min_value=270.0, max_value=320.0, value=295.0)

if st.button("Predict"):
    input_data = np.array([[mw, mvar, oil_pressure]], dtype=np.float32)
    result = session.run(None, {input_name: input_data})
    temp = result[0][0][0]

    vibration = round(1.2 + 0.005 * abs(mvar), 2)
    shaft = round(20 + 0.01 * abs(mvar), 2)

    if temp < 95 and vibration <= 3 and shaft <= 30:
        status = "🟢 Normal"
    elif 95 <= temp <= 98 or 3 < vibration <= 4 or 30 < shaft <= 35:
        status = "🟡 Warning"
    else:
        status = "🔴 Alarm"

    st.metric("Temperature", f"{temp:.2f} °C")
    st.metric("Vibration", f"{vibration} mm/s")
    st.metric("Shaft Displacement", f"{shaft} mm")
    st.markdown(f"### Status: {status}")

    log = {
        "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "MW": mw, "MVAR": mvar, "Oil Pressure": oil_pressure,
        "Temperature": temp, "Vibration": vibration, "Shaft": shaft, "Status": status
    }

    df = pd.DataFrame([log])
    if os.path.exists("prediction_log.csv"):
        df.to_csv("prediction_log.csv", mode='a', header=False, index=False)
    else:
        df.to_csv("prediction_log.csv", index=False)

if st.checkbox("Show Prediction Log"):
    if os.path.exists("prediction_log.csv"):
        st.dataframe(pd.read_csv("prediction_log.csv"))
    else:
        st.info("No log available.")
