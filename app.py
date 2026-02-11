import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. CẤU HÌNH GIAO DIỆN
st.set_page_config(page_title="Phân Tích Quyết Định Mua Xe", page_icon="🚗", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 2.2rem; color: #0068C9; font-weight: bold; text-align: center; margin-bottom: 20px;}
    .sub-text {font-size: 1.1rem; color: #555; text-align: center; font-style: italic;}
    .metric-box {background-color: #F0F2F6; padding: 20px; border-radius: 10px; border-left: 5px solid #0068C9;}
    .stButton>button {background-color: #0068C9; color: white; width: 100%; font-weight: bold; height: 50px; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚗 HỆ THỐNG MÔ PHỎNG SỨC MUA Ô TÔ</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Phân tích độ nhạy: Thay đổi các yếu tố kỹ thuật để xem tác động đến quyết định mua (Total)</div>', unsafe_allow_html=True)
st.markdown("---")

# 2. HÀM TẢI TÀI NGUYÊN
@st.cache_resource
def load_resources():
    try:
        # Đọc trực tiếp từ folder trong Repo
        df = pd.read_csv('data/processed_carbuyers.csv')
        model = joblib.load('models/car_purchase_model.joblib')
        
        # Load danh sách cột
        try:
            model_cols = joblib.load('models/model_columns.joblib')
        except:
            model_cols = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
            
        return df, model, model_cols
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        return None, None, None

df_org, model, model_cols = load_resources()

# 3. THANH ĐIỀU KHIỂN (SIDEBAR)
if df_org is not None and model is not None:
    st.sidebar.header("🛠️ THIẾT LẬP THÔNG SỐ XE")

    # --- A. CHỌN HÃNG XE ---
    manus = sorted(df_org['Manufacturer'].unique().tolist())
    if 'Ford' in manus: manus.insert(0, manus.pop(manus.index('Ford')))
    sel_manu = st.sidebar.selectbox("Hãng xe (Manufacturer)", manus)

    # --- B. CHỌN NHIÊN LIỆU (Đã lọc lỗi Automatic) ---
    raw_fuels = [x for x in df_org['Fuel'].unique() if isinstance(x, str) and x.lower() != 'automatic']
    sel_fuel = st.sidebar.selectbox("Nhiên liệu (Fuel Type)", raw_fuels)

    st.sidebar.markdown("---")

    # --- C. THÔNG SỐ KỸ THUẬT ---
    min_price, max_price = int(df_org['Price'].min()), int(df_org['Price'].max())
    sel_price = st.sidebar.slider(f"Giá bán (Price - Nghìn USD)", min_price, max_price, 25)

    min_power, max_power = int(df_org['Power'].min()), int(df_org['Power'].max())
    sel_power = st.sidebar.slider("Công suất (Power - Mã lực/HP)", min_power, max_power, 150)

    min_engine, max_engine = int(df_org['Engine CC'].min()), int(df_org['Engine CC'].max())
    sel_engine = st.sidebar.number_input("Dung tích (Engine CC)", min_engine, max_engine, 2000)

    # Hộp số
    sel_trans = st.sidebar.slider("Hộp số (Transmission - Số cấp)", 0, 10, 5)

    # 4. XỬ LÝ DỰ BÁO
    if st.sidebar.button("🚀 CHẠY MÔ PHỎNG PHÂN TÍCH"):
        input_df = pd.DataFrame(0, index=[0], columns=model_cols)

        if 'Price' in model_cols: input_df['Price'] = sel_price
        if 'Power' in model_cols: input_df['Power'] = sel_power
        if 'Engine CC' in model_cols: input_df['Engine CC'] = sel_engine
        if 'Transmission' in model_cols: input_df['Transmission'] = sel_trans

        if f"Manufacturer_{sel_manu}" in model_cols: input_df[f"Manufacturer_{sel_manu}"] = 1
        if f"Fuel_{sel_fuel}" in model_cols: input_df[f"Fuel_{sel_fuel}"] = 1

        try:
            prediction = model.predict(input_df)[0]

            c1, c2 = st.columns([1.5, 1])
            with c1:
                st.markdown("### 📊 KẾT QUẢ DỰ BÁO")
                st.markdown(f"""
                <div class="metric-box">
                    <h4 style="margin:0; color:#555">SỨC MUA DỰ KIẾN (TOTAL)</h4>
                    <h1 style="margin:0; color:#0068C9">{prediction:,.2f}</h1>
                    <p>Khách hàng tiềm năng</p>
                </div>
                """, unsafe_allow_html=True)
                
                avg = df_org['Total'].mean()
                diff = prediction - avg
                
                st.write("")
                if diff > 0:
                    st.success(f"📈 **Tiềm năng:** Cao hơn trung bình thị trường (+{diff:,.0f}).")
                else:
                    st.warning(f"📉 **Rủi ro:** Thấp hơn trung bình thị trường ({diff:,.0f}).")

            with c2:
                st.markdown("### 🔍 SO SÁNH")
                chart_data = pd.DataFrame({"Loại": ["Trung bình TT", "Cấu hình này"], "Total": [avg, prediction]})
                st.bar_chart(chart_data.set_index("Loại"), color="#0068C9")

        except Exception as e:
            st.error(f"Lỗi xử lý dự báo: {str(e)}")
else:
    st.info("⏳ Đang tải dữ liệu...")