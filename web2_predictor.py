import streamlit as st
import numpy as np
import onnxruntime as ort
import datetime
import pandas as pd
import os
from PIL import Image
from io import BytesIO

# -------------------- Set page config FIRST --------------------
st.set_page_config(page_title="U1 Generator TE Bearing Temp Prediction", layout="centered")

# -------------------- Language Toggle --------------------
LANG = st.selectbox("🌐 Language / اللغة", ["English", "Arabic"])

labels = {
    "English": {
        "title": "U1 Generator TE Bearing Temp Prediction",
        "mw": "Active Power (MW)",
        "mvar": "Reactive Power (MVAR)",
        "oil": "Oil Pressure (kPa)",
        "predict": "Predict",
        "log": "Show Prediction Log",
        "download": "Download Log as CSV",
        "clear": "Clear Log",
        "designer": "Designed by Eng. Mohammed Assaf",
        "status": "Status",
        "temp": "Temperature",
        "model_info": "Model Info",
        "trained": "Trained Date: 2025-06-04",
        "algo": "Algorithm: Gradient Boosting",
        "importance": "Input Importance"
    },
    "Arabic": {
        "title": "توقع درجة حرارة كرسي تحميل المولد الأول",
        "mw": "القدرة الفعالة (MW)",
        "mvar": "القدرة غير الفعالة (MVAR)",
        "oil": "ضغط الزيت (kPa)",
        "predict": "تنبؤ",
        "log": "عرض سجل التنبؤات",
        "download": "تحميل السجل",
        "clear": "مسح السجل",
        "designer": "تصميم م. محمد عساف",
        "status": "الحالة",
        "temp": "درجة الحرارة",
        "model_info": "معلومات النموذج",
        "trained": "تاريخ التدريب: 2025-06-04",
        "algo": "الخوارزمية: الانحدار المعزز",
        "importance": "أهمية المدخلات"
    }
}
l = labels[LANG]

# -------------------- Title --------------------
st.title(f"🔧 {l['title']}")

# -------------------- Load ONNX Model --------------------
session = ort.InferenceSession("bearing_temperature_model.onnx")
input_name = session.get_inputs()[0].name

# -------------------- Logo --------------------
try:
    logo = Image.open("OMCO_Logo.png")
    st.image(logo, width=150)
except:
    st.info("Logo not found (OMCO_Logo.png)")

# -------------------- Inputs --------------------
mw = st.slider(l["mw"], 100, 300, 277)
mvar = st.slider(l["mvar"], -120, 40, -20)
oil_pressure = st.number_input(l["oil"], min_value=270.0, max_value=320.0, value=295.0)

if st.button(l["predict"]):
    input_data = np.array([[mw, mvar, oil_pressure]], dtype=np.float32)
    result = session.run(None, {input_name: input_data})
    temp = result[0][0][0]

    # Temperature-based status
    if temp < 95:
        status = "🟢 Normal" if LANG == "English" else "🟢 طبيعي"
    elif 95 <= temp <= 98:
        status = "🟡 Warning" if LANG == "English" else "🟡 تحذير"
    else:
        status = "🔴 Alarm" if LANG == "English" else "🔴 إنذار"

    st.markdown(f"<h2 style='color:darkred;'>{l['temp']}: {temp:.2f} °C</h2>", unsafe_allow_html=True)
    st.markdown(f"### {l['status']}: {status}")

    # Save log
    log = {
        "Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "MW": mw, "MVAR": mvar, "Oil Pressure": oil_pressure,
        "Temperature": temp, "Status": status
    }
    df = pd.DataFrame([log])
    if os.path.exists("prediction_log.csv"):
        df.to_csv("prediction_log.csv", mode='a', header=False, index=False)
    else:
        df.to_csv("prediction_log.csv", index=False)

# -------------------- Log Viewer --------------------
st.markdown("---")
if st.checkbox(l["log"]):
    if os.path.exists("prediction_log.csv"):
        df_log = pd.read_csv("prediction_log.csv")
        st.dataframe(df_log)

        # Download as CSV
        csv = df_log.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 " + l["download"],
                           data=csv,
                           file_name="prediction_log.csv",
                           mime="text/csv")
    else:
        st.info("📭 No predictions logged yet.")

# -------------------- Clear Log --------------------
if st.button("🧹 " + l["clear"]):
    if os.path.exists("prediction_log.csv"):
        os.remove("prediction_log.csv")
        st.success("✅ Log cleared successfully.")

# -------------------- Model Info --------------------
st.markdown("---")
with st.expander(l["model_info"]):
    st.write(f"📅 {l['trained']}")
    st.write(f"🧠 {l['algo']}")
    st.markdown(f"**{l['importance']}:**")
    importance_list = [("MVAR", 0.45), ("MW", 0.35), ("Oil Pressure", 0.20)]
    for name, val in importance_list:
        bar = "█" * int(val * 20)
        st.write(f"{name}: {bar} {int(val * 100)}%")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown(f"<p style='text-align: center;'>{l['designer']}</p>", unsafe_allow_html=True)
