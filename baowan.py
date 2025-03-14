import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

# โหลดโมเดล Neural Network
@st.cache_resource
def load_nn_model():
    return tf.keras.models.load_model("heart_disease_model.h5")

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

    # ให้ผู้ใช้กรอกค่าฟีเจอร์ต่าง ๆ
    pregnancies = st.number_input("จำนวนครั้งที่ตั้งครรภ์", min_value=0, max_value=20, value=1)
    glucose = st.number_input("ระดับน้ำตาลในเลือด", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("ความดันโลหิต", min_value=0, max_value=200, value=80)
    skin_thickness = st.number_input("ความหนาของผิวหนัง (mm)", min_value=0, max_value=100, value=20)
    insulin = st.number_input("ระดับอินซูลินในเลือด", min_value=0, max_value=900, value=80)
    bmi = st.number_input("ค่าดัชนีมวลกาย (BMI)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    diabetes_pedigree = st.number_input("ประวัติเบาหวานในครอบครัว", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("อายุ", min_value=0, max_value=120, value=30)

    # ปุ่มให้พยากรณ์
    if st.button("ทำนาย"):
        model = load_ml_model()
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
        prediction = model.predict(input_data)

        # แสดงผลลัพธ์
        st.write(f"ผลการพยากรณ์: {'เป็นโรคเบาหวาน' if prediction[0] == 1 else 'ไม่เป็นโรคเบาหวาน'}")

# หน้า Demo Neural Network
elif page == "Demo Neural Network":
    st.title("Demo Neural Network")
    st.write("กรอกข้อมูลเพื่อทำนายโรคหัวใจ")

    # ให้ผู้ใช้กรอกค่าฟีเจอร์ต่าง ๆ
    age = st.number_input("อายุ", min_value=0, max_value=120, value=50)
    sex = st.selectbox("เพศ (0 = หญิง, 1 = ชาย)", [0, 1])
    trestbps = st.number_input("ความดันโลหิตขณะพัก (mmHg)", min_value=50, max_value=250, value=140)
    chol = st.number_input("ระดับคอเลสเตอรอล (mg/dL)", min_value=100, max_value=600, value=220)
    fbs = st.selectbox("น้ำตาลในเลือดสูง (>120 mg/dL) (0 = ไม่, 1 = ใช่)", [0, 1])
    restecg = st.selectbox("ผล ECG (0-2)", [0, 1, 2])
    thalach = st.number_input("อัตราการเต้นของหัวใจสูงสุด", min_value=50, max_value=250, value=150)
    exang = st.selectbox("มีอาการเจ็บหน้าอกขณะออกกำลังกาย (0 = ไม่, 1 = ใช่)", [0, 1])
    oldpeak = st.number_input("ST depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope ของ ST segment (0-2)", [0, 1, 2])

    # ปุ่มให้พยากรณ์
    if st.button("ทำนาย"):
        model = load_nn_model()
        input_data = np.array([[age, sex, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]])
        prediction = model.predict(input_data)

        # แสดงผลลัพธ์
        st.write(f"ผลการพยากรณ์: {'มีความเสี่ยง' if prediction[0][0] > 0.5 else 'ไม่มีความเสี่ยง'} (ค่าคะแนน: {prediction[0][0]:.4f})")