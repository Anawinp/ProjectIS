import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import sklearn

# โหลดโมเดล Neural Network
@st.cache_resource
def load_nn_model():
    return tf.keras.models.load_model("heart_disease_model_updated.h5")

# โหลดโมเดล Machine Learning
@st.cache_resource
def load_ml_model():
    with open("diabetes_model.pkl", "rb") as file:
        return pickle.load(file)

# Sidebar สำหรับเลือกหน้า
st.sidebar.title("เมนู")
page = st.sidebar.radio("เลือกหน้า:", [
    "อธิบายการพัฒนา Machine Learning",
    "อธิบายการพัฒนา Neural Network",
    "Demo Machine Learning",
    "Demo Neural Network"
])

# หน้าอธิบายการพัฒนา Machine Learning
if page == "อธิบายการพัฒนา Machine Learning":
    st.title("การพัฒนา Machine Learning")
    st.write("""
    ### แนวคิดพื้นฐานของ Machine Learning
    Machine Learning เป็นเทคนิคที่ใช้สำหรับสร้างแบบจำลองจากข้อมูลโดยอัตโนมัติ
    - มีประเภทหลัก เช่น Supervised Learning, Unsupervised Learning และ Reinforcement Learning
    - ใช้กระบวนการฝึกสอนโมเดลจากข้อมูลตัวอย่าง

    ### ขั้นตอนการพัฒนา Machine Learning
    1. การเตรียมข้อมูล (Data Preprocessing)
    2. การเลือกและเทรนโมเดล (Model Selection & Training)
    3. การทดสอบและประเมินผล (Evaluation)
    4. การปรับแต่ง (Hyperparameter Tuning)
    5. การใช้งานโมเดล (Deployment)
    """)

# หน้าอธิบายการพัฒนา Neural Network
elif page == "อธิบายการพัฒนา Neural Network":
    st.title("การพัฒนา Neural Network")
    st.write("""
    ### แนวคิดพื้นฐานของ Neural Network
    Neural Network เป็นสาขาย่อยของ Machine Learning ที่ได้รับแรงบันดาลใจจากโครงสร้างของสมองมนุษย์
    - ใช้โครงสร้างของเครือข่ายประสาท (Neuron) ซึ่งมีหลายชั้น (Layers)
    - ใช้กระบวนการ Backpropagation และ Gradient Descent ในการฝึกสอน

    ### ขั้นตอนการพัฒนา Neural Network
    1. การเตรียมข้อมูล (Data Preparation)
    2. การออกแบบโครงสร้างเครือข่าย (Model Architecture)
    3. การกำหนดค่าพารามิเตอร์และการเรียนรู้ (Training & Optimization)
    4. การทดสอบและการประเมินผล (Testing & Validation)
    5. การนำไปใช้งานจริง (Deployment)
    """)


# หน้า Demo Machine Learning
elif page == "Demo Machine Learning":
    st.title("Demo Machine Learning - ทำนายโรคเบาหวาน")
    st.write("กรอกข้อมูลเพื่อทำนายโรคเบาหวาน")

    # ให้ผู้ใช้กรอกค่าฟีเจอร์ต่าง ๆ ตามโมเดลที่เทรน
    age = st.number_input("อายุ", min_value=0, max_value=120, value=30)
    bmi = st.number_input("ค่าดัชนีมวลกาย (BMI)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    glucose = st.number_input("ระดับน้ำตาลในเลือด", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("ความดันโลหิต", min_value=0, max_value=200, value=80)

    # ปุ่มให้พยากรณ์
    if st.button("ทำนาย"):
        model = load_ml_model()
        input_data = np.array([[age, bmi, glucose, blood_pressure]])
        prediction = model.predict(input_data)
        risk_levels = {0: "High", 1: "Low", 2: "Medium"}
        # แสดงผลลัพธ์ตามระดับความเสี่ยง
        predicted_class = int(prediction[0])  # แปลงค่าผลลัพธ์ให้อยู่ในรูปแบบ int
        risk_level = risk_levels.get(predicted_class, "ไม่ทราบ")
        st.write(f"ผลการพยากรณ์: ระดับความเสี่ยง {risk_level}")

# หน้า Demo Neural Network
elif page == "Demo Neural Network":
    st.title("Demo Neural Network - ทำนายโรคหัวใจ")
    st.write("กรอกข้อมูลเพื่อทำนายโรคหัวใจ")

    # ให้ผู้ใช้กรอกค่าฟีเจอร์ต่าง ๆ ตาม dataset ใหม่
    age = st.number_input("อายุ", min_value=0, max_value=120, value=50)
    gender = st.selectbox("เพศ (0 = หญิง, 1 = ชาย)", [0, 1])
    cholesterol = st.number_input("ระดับคอเลสเตอรอล (mg/dL)", min_value=100, max_value=600, value=220)
    bp_systolic = st.number_input("ความดันโลหิตตัวบน (BP_Systolic)", min_value=50, max_value=250, value=120)
    bp_diastolic = st.number_input("ความดันโลหิตตัวล่าง (BP_Diastolic)", min_value=30, max_value=150, value=80)
    bmi = st.number_input("ค่าดัชนีมวลกาย (BMI)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    hba1c = st.number_input("ระดับน้ำตาลสะสมในเลือด (HbA1c)", min_value=3.0, max_value=15.0, value=5.5, step=0.1)
    smoking = st.selectbox("สูบบุหรี่ (0 = ไม่, 1 = ใช่)", [0, 1])
    diabetes = st.selectbox("เบาหวาน (0 = ไม่, 1 = ใช่)", [0, 1])

    # ปุ่มให้พยากรณ์
    if st.button("ทำนาย"):
        model = load_nn_model()
        input_data = np.array([[
            age, gender, cholesterol, bp_systolic, bp_diastolic, bmi, hba1c, smoking, diabetes
        ]])
        prediction = model.predict(input_data)

        # แสดงผลลัพธ์
        st.write(f"🔍 ค่าทำนายจากโมเดล (Raw Prediction): {prediction[0][0]}")
