import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from datetime import datetime
from pathlib import Path
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from tensorflow.keras.models import load_model

# ============================================
# ESTILO PREMIUM MODO OSCURO
# ============================================
def load_dark_gold_css():
    st.markdown("""
    <style>
        :root {
            --gold: #FFD700;
            --dark-gold: #C5A100;
            --bg-dark: #121212;
            --bg-darker: #0A0A0A;
            --card-dark: #1E1E1E;
            --text-light: #E0E0E0;
        }
        
        .main {
            background-color: var(--bg-dark) !important;
            color: var(--text-light) !important;
        }
        
        .stApp {
            background: linear-gradient(135deg, var(--bg-darker) 0%, var(--bg-dark) 100%) !important;
        }
        
        .header {
            color: var(--gold) !important;
            text-align: center;
            padding: 1.5rem;
            border-radius: 12px;
            background: rgba(30, 30, 30, 0.8);
            border: 1px solid var(--dark-gold);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            margin-bottom: 2rem;
            background: linear-gradient(135deg, #1E1E1E 0%, #2A2A2A 100%);
        }
        
        .card {
            background: var(--card-dark);
            border-radius: 12px;
            padding: 1.8rem;
            border: 1px solid rgba(255, 215, 0, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            margin-bottom: 2rem;
            color: var(--text-light);
        }
        
        .metric-card {
            background: linear-gradient(135deg, #252525 0%, #1A1A1A 100%);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid rgba(255, 215, 0, 0.15);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
            color: var(--gold);
        }
        
        .progress-container {
            background: #2D2D2D;
            border-radius: 10px;
            height: 24px;
            margin-top: 1rem;
            overflow: hidden;
            border: 1px solid rgba(255, 215, 0, 0.1);
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--gold) 0%, var(--dark-gold) 100%);
            transition: width 0.5s ease;
        }
        
        .sidebar .sidebar-content {
            background: linear-gradient(135deg, #1A1A1A 0%, #121212 100%) !important;
            border-right: 1px solid rgba(255, 215, 0, 0.1);
        }
        
        .stButton>button {
            background: linear-gradient(135deg, var(--gold) 0%, var(--dark-gold) 100%) !important;
            color: #121212 !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.7rem 1.5rem !important;
            font-weight: bold !important;
            box-shadow: 0 2px 10px rgba(255, 215, 0, 0.3) !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 15px rgba(255, 215, 0, 0.4) !important;
        }
        
        .stSelectbox, .stTextInput {
            background: #252525 !important;
            color: var(--text-light) !important;
            border: 1px solid rgba(255, 215, 0, 0.3) !important;
            border-radius: 8px !important;
        }
        
        .stDataFrame {
            background: #252525 !important;
            color: var(--text-light) !important;
            border: 1px solid rgba(255, 215, 0, 0.2) !important;
        }
        
        .stExpander {
            background: #252525 !important;
            border: 1px solid rgba(255, 215, 0, 0.2) !important;
            border-radius: 8px !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: var(--gold) !important;
        }
        
        .device-selector {
            background: #252525 !important;
            color: var(--gold) !important;
            border: 1px solid var(--gold) !important;
        }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# COMPONENTES PREMIUM
# ============================================
def create_premium_header():
    st.markdown("""
    <div class="header">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">🍌 GOLDEN BANANA CLASSIFIER</h1>
        <p style="color: #AAAAAA; font-size: 1.1rem;">Sistema premium de clasificación con tecnología AI</p>
    </div>
    """, unsafe_allow_html=True)

def create_device_selector():
    devices = ["Cámara Principal", "Cámara Secundaria", "Dispositivo Móvil"]
    selected = st.selectbox(
        "SELECT DEVICE", 
        devices, 
        key="device_selector",
        format_func=lambda x: f"📷 {x}"
    )
    return selected

def create_gold_metrics(class_name, confidence):
    with st.container():
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="border-bottom: 1px solid rgba(255, 215, 0, 0.3); padding-bottom: 0.5rem; margin-bottom: 1rem;">ÚLTIMA PREDICCIÓN</h3>
            <div style="font-size: 28px; font-weight: bold; margin: 1rem 0; color: var(--gold);">{class_name}</div>
            <div style="font-size: 24px; margin: 1rem 0; color: var(--gold);">{confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="margin-top: 1.5rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; color: var(--text-light);">
                <span>Precisión:</span>
                <span>{confidence:.1%}</span>
            </div>
            <div class="progress-container">
                <div class="progress-bar" style="width: {confidence*100}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# CLASE BANANACLASSIFIER (Implementación completa)
# ============================================
class BananaClassifier(VideoTransformerBase):
    def __init__(self):
        self.prev_gray = None
        self.user = None
        self.status = "Iniciando..."
        self.last_prediction = ("", 0.0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def preprocess_for_model(self, frame):
        """Prepara la imagen para el modelo"""
        img = cv2.resize(frame, (224, 224))
        img = img.astype("float32") / 255.0
        return np.expand_dims(img, axis=0)

    def detect_movement(self, current_frame):
        """Detección de movimiento simple"""
        if self.prev_gray is None:
            self.prev_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return False
        
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_pixels = np.count_nonzero(thresh)
        self.prev_gray = gray
        return motion_pixels > 5000  # Ajusta este umbral según necesites

    def detect_yellow(self, frame):
        """Detección de color amarillo (banano)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([20, 100, 100])
        upper = np.array([30, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        return np.count_nonzero(mask), mask

    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        display_frame = frame.copy()
        
        # Detecciones
        movement = self.detect_movement(frame)
        yellow_pixels, yellow_mask = self.detect_yellow(frame)
        yellow_detected = yellow_pixels > 3000  # Ajusta este umbral
        
        # Predicción
        class_name, confidence = "", 0.0
        model = load_model("modelo_completo.keras")  # Asegúrate de tener esta línea o manejar el modelo adecuadamente
        
        if model and (movement or yellow_detected):
            processed_img = self.preprocess_for_model(frame)
            predictions = model.predict(processed_img, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])
            
            if confidence > 0.65:  # Ajusta el umbral de confianza
                class_name = ["Banano Maduro", "Banano Verde", "Otro"][class_idx]  # Ajusta según tus clases
                self.last_prediction = (class_name, confidence)
                self.log_event(frame, class_name, confidence)

        # Visualización
        status_text = f"{class_name} ({confidence:.1%})" if class_name else \
                     "Banano detectado (color)" if yellow_detected else \
                     "Movimiento detectado" if movement else "Listo"
        
        color = (0, 255, 0) if class_name else \
                (0, 255, 255) if yellow_detected else \
                (0, 0, 255) if movement else (255, 255, 255)
        
        cv2.putText(display_frame, status_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(display_frame, (10, 10), (frame.shape[1]-10, frame.shape[0]-10), color, 2)
        
        self.status = status_text
        return display_frame

    def log_event(self, frame, class_name, confidence):
        """Registra evento y guarda snapshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar en CSV
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user": self.user or "Anónimo",
            "event": "clasificación",
            "class": class_name,
            "confidence": confidence
        }
        pd.DataFrame([log_entry]).to_csv("registro_eventos.csv", mode='a', 
                                       header=not Path("registro_eventos.csv").exists(), 
                                       index=False)
        
        # Guardar imagen
        Path("snapshots").mkdir(exist_ok=True)
        snapshot_path = f"snapshots/{timestamp}_{class_name}.jpg"
        cv2.imwrite(snapshot_path, frame)

# ============================================
# INTERFAZ PRINCIPAL
# ============================================
def main():
    # Configuración inicial
    st.set_page_config(
        page_title="🍌 Golden Banana Classifier",
        page_icon="🍌",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_dark_gold_css()
    create_premium_header()
    
    # Barra lateral premium
    with st.sidebar:
        st.markdown("## ⚙️ CONFIGURACIÓN PREMIUM")
        
        with st.expander("👤 REGISTRO DE USUARIOS", expanded=True):
            st.markdown("<p style='color: var(--gold);'>Autenticación facial avanzada</p>", unsafe_allow_html=True)
            new_user_img = st.camera_input("Captura rostro para registro", key="reg_cam")
            if new_user_img is not None:
                user_name = st.text_input("Nombre de usuario", key="user_name")
                if st.button("REGISTRAR USUARIO PREMIUM", key="reg_btn"):
                    img = Image.open(new_user_img)
                    Path("registered_faces").mkdir(exist_ok=True)
                    img.save(f"registered_faces/{user_name}.jpg")
                    st.success(f"Usuario {user_name} registrado en sistema premium")
        
        with st.expander("🎛️ AJUSTES AVANZADOS"):
            st.slider("Umbral de confianza", 0.5, 1.0, 0.65, 0.01, 
                     format="%.2f", key="conf_threshold")
            st.checkbox("Mostrar detalles técnicos", value=False, key="show_tech_details")
            st.checkbox("Modo diagnóstico", value=False, key="diagnostic_mode")

    # Contenido principal
    col1, col2 = st.columns([3, 1])
    
    with col1:
        with st.container():
            st.markdown("### 📡 CLASIFICACIÓN EN TIEMPO REAL")
            selected_device = create_device_selector()
            
            ctx = webrtc_streamer(
                key="golden-banana-classifier",
                video_transformer_factory=BananaClassifier,
                rtc_configuration=RTCConfiguration({
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                }),
                async_transform=True,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 1280},
                        "height": {"ideal": 720},
                        "deviceId": selected_device
                    },
                    "audio": False
                }
            )
    
    with col2:
        with st.container():
            st.markdown("### 📊 RESULTADOS PREMIUM")
            
            # Ejemplo de métricas (conectar con tu modelo real)
            class_name = "BANANO MADURO"  # Reemplazar con salida real
            confidence = 0.92  # Reemplazar con confianza real
            create_gold_metrics(class_name, confidence)
            
            st.markdown("### 📜 REGISTRO DE EVENTOS")
            if Path("registro_eventos.csv").exists():
                events = pd.read_csv("registro_eventos.csv")
                st.dataframe(
                    events.tail(5).style.set_properties(**{
                        'background-color': '#252525',
                        'color': 'var(--gold)',
                        'border': '1px solid rgba(255, 215, 0, 0.1)'
                    }),
                    height=300,
                    use_container_width=True
                )
                
                with open("registro_eventos.csv", "rb") as f:
                    st.download_button(
                        "💾 DESCARGAR REGISTRO COMPLETO",
                        f.read(),
                        file_name="golden_banana_log.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

if __name__ == "__main__":
    main()