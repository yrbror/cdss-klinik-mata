import os
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Sistem CDSS Klinik Mata", page_icon="🏥", layout="wide")
st.title("🏥 Sistem CDSS & Deteksi Retinopati Diabetik")
st.markdown("Sistem pendukung keputusan klinis hybrid menggunakan AI (MobileNetV2) & Rule-Based Logic.")

# ==========================================
# 2. LOAD MODEL (Di-cache agar tidak loading terus)
# ==========================================
@st.cache_resource
def load_ai_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'model_retina_terbaik.keras')
    # Load model murni tanpa parameter tambahan karena sudah sesuai dengan Keras 3
    return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

model_terbaik = load_ai_model()
last_conv_layer_name = 'out_relu'

def buat_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def gabungkan_heatmap(img_array, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img_array
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    return heatmap, superimposed_img

# ==========================================
# 3. TAHAP INPUT (SIDEBAR)
# ==========================================
st.sidebar.header("Data Klinis Pasien")
nama_pasien = st.sidebar.text_input("Nama Pasien", "Pasien A")
usia_pasien = st.sidebar.number_input("Usia Pasien", min_value=1, max_value=120, value=30)
jenis_kelamin = st.sidebar.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
riwayat_diabetes = st.sidebar.selectbox("Riwayat Diabetes", ["Tidak", "Ya"])

uploaded_file = st.sidebar.file_uploader("Unggah Foto Fundus Mata", type=["jpg", "png", "jpeg"])

# ==========================================
# 4. PEMROSESAN & OUTPUT
# ==========================================
if uploaded_file is not None:
    with st.spinner('Sistem sedang menganalisis gambar & data klinis...'):
        # Preprocessing gambar
        image = Image.open(uploaded_file).convert('RGB')
        img_resized = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_tensor = np.expand_dims(img_array, axis=0) / 255.0

        # Prediksi AI
        prediksi_normal = model_terbaik.predict(img_tensor, verbose=0)[0][0]
        risiko_dasar_ai = 1.0 - prediksi_normal
        
        # Kalkulasi XAI
        heatmap = buat_gradcam_heatmap(img_tensor, model_terbaik, last_conv_layer_name)
        img_asli_cv = np.array(img_resized) # Konversi PIL ke Numpy
        _, gambar_gabungan = gabungkan_heatmap(img_asli_cv, heatmap)

        # Kalkulasi Rule-Based CDSS
        risiko_akhir = risiko_dasar_ai
        catatan_medis = []
        
        if riwayat_diabetes == 'Ya':
            risiko_akhir += 0.15
            catatan_medis.append("Adanya riwayat diabetes secara signifikan meningkatkan risiko Retinopati Diabetik.")
        if usia_pasien >= 50:
            risiko_akhir += 0.10
            catatan_medis.append("Faktor usia lanjut meningkatkan kerentanan degenerasi area makula.")
            
        risiko_akhir = min(risiko_akhir, 1.0)

    # Tampilkan Hasil Teks
    st.markdown("---")
    st.subheader(f"Hasil Screening: Sdr/i {nama_pasien}")
    
    col1, col2 = st.columns(2)
    col1.metric("Risiko Visual AI", f"{risiko_dasar_ai * 100:.1f}%")
    col2.metric("Risiko Total (AI + Klinis)", f"{risiko_akhir * 100:.1f}%")

    if risiko_akhir >= 0.50:
        st.error("🚨 **Diagnosis Sistem: TERINDIKASI ABNORMAL TINGGI**\n\n**Tindakan:** SEGERA RUJUK KE DOKTER SPESIALIS.")
    elif risiko_akhir >= 0.40:
        st.warning("⚠️ **Diagnosis Sistem: TERINDIKASI ABNORMAL (BORDERLINE)**\n\n**Tindakan:** Rujuk untuk observasi manual. Cek visualisasi peta Grad-CAM.")
    else:
        st.success("✅ **Diagnosis Sistem: NORMAL**\n\n**Tindakan:** Tidak ditemukan kelainan darurat.")

    if catatan_medis:
        st.info("**Catatan Evaluasi Klinis:**\n" + "\n".join([f"- {cat}" for cat in catatan_medis]))

    # Tampilkan Hasil Visualisasi
    st.markdown("---")
    st.subheader("Visualisasi Explainable AI (Grad-CAM)")
    img_col1, img_col2, img_col3 = st.columns(3)
    
    with img_col1:
        st.image(image, caption="1. Foto Mata Asli", use_column_width=True)
    with img_col2:
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        st.image(heatmap_colored, caption="2. Peta Fokus AI", use_column_width=True)
    with img_col3:
        st.image(gambar_gabungan, caption="3. Visualisasi Deteksi XAI", use_column_width=True)
