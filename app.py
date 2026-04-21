import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_assets():
    model = joblib.load('xgb_battery_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le_battery = joblib.load('le_battery_type.pkl')
    le_driving = joblib.load('le_driving_style.pkl')
    return model, scaler, le_battery, le_driving

model, scaler, le_battery, le_driving = load_assets()
expected_features = scaler.feature_names_in_

# --- 1. CẤU HÌNH TRANG (PHẢI ĐẶT ĐẦU TIÊN) ---
st.set_page_config(page_title="EV Battery SoH", page_icon="")

# --- 2. CHÈN LOGO VÀ TIÊU ĐỀ ---
# Chia cột để logo nằm bên trái, tiêu đề nằm bên phải
col_logo, col_text = st.columns([1, 3]) 

with col_logo:
    # Hệ thống sẽ tìm file AEE.jpeg trong cùng thư mục trên GitHub của bạn
    st.image("aee.jpeg", width=500) 

with col_text:
    st.title("Predicting current SoH percent")

st.write("Please enter the operating parameters below:")

# --- 3. GIAO DIỆN NHẬP LIỆU ---
with st.container():
    col1, col2 = st.columns(2)
    user_inputs = {}

    for i, col_name in enumerate(expected_features):
        target_col = col1 if i % 2 == 0 else col2
        
        with target_col:
            if col_name == 'Battery_Type':
                user_inputs[col_name] = st.selectbox("Battery type", options=le_battery.classes_)
            elif col_name == 'Driving_Style':
                user_inputs[col_name] = st.selectbox("Driving style", options=le_driving.classes_)
            elif col_name == 'Battery_Capacity_kWh':
                user_inputs[col_name] = st.number_input("Battery capacity (kWh)", value=75.0)
            elif col_name == 'Vehicle_Age_Months':
                user_inputs[col_name] = st.number_input("Vehicle age (Months)", value=12)
            elif col_name == 'Total_Charging_Cycles':
                user_inputs[col_name] = st.number_input("Total charging cycles", value=100)
            elif col_name == 'Avg_Temperature_C':
                user_inputs[col_name] = st.number_input("Average temperature (°C)", -10.0, 60.0, 25.0)
            elif col_name == 'Fast_Charge_Ratio':
                user_inputs[col_name] = st.number_input("Fast charge ratio", 0.0, 1.0, 0.1)
            elif col_name == 'Avg_Discharge_Rate_C':
                user_inputs[col_name] = st.number_input("Average discharge rate (C-rate)", value=1.0)
            else:
                user_inputs[col_name] = st.number_input(f"Enter {col_name}", value=0.0)

# --- 4. XỬ LÝ DỰ ĐOÁN ---
st.markdown("---")
if st.button("📊 SoH percentage prediction", use_container_width=True):
    input_df = pd.DataFrame([user_inputs])
    input_df['Battery_Type'] = le_battery.transform(input_df['Battery_Type'])
    input_df['Driving_Style'] = le_driving.transform(input_df['Driving_Style'])
    input_df = input_df[expected_features]
    
    input_scaled = scaler.transform(input_df)
    soh_result = model.predict(input_scaled)[0]
    
    st.balloons()
    st.markdown(f"<h2 style='text-align: center;'>SoH prediction: <span style='color: #2ecc71;'>{soh_result:.2f}%</span></h2>", unsafe_allow_html=True)
    st.progress(min(max(int(soh_result), 0), 100))
    
    if soh_result < 80:
        st.warning("⚠️ Warning: The battery is showing signs of degradation (below 80%).")
    else:
        st.success("✅ The battery is still in good working order.")
