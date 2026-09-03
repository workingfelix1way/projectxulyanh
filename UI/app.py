import streamlit as st
from PIL import Image

st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")
st.title("Hệ thống Phân vùng Khối u Não (MRI)")

st.info("Vui lòng tải ảnh MRI 2D để hệ thống xử lý.")
uploaded_file = st.file_uploader("Tải ảnh lên (.tif, .png, .jpg)", type=["tif", "png", "jpg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh MRI Gốc")
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("Kết quả Phân vùng")
        st.info("Mô hình đang được huấn luyện. Kết quả sẽ hiển thị tại đây.")
else:
    st.write("Vui lòng tải ảnh ở khung bên trên.")