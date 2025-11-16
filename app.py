import streamlit as st
from PIL import Image
import numpy as np, io, os, json
from utils.preprocessing import extract_hand_landmarks_from_image, preprocess_landmarks
from utils.translator import translate_text
from utils.tts import text_to_speech

st.set_page_config('ISL Translator', layout='centered')
st.title('Indian Sign Language → Text → Multilingual Speech')

MODEL_PATH = 'model/isl_model.h5'
LABEL_PATH = 'model/label_map.json'
model = None
inv_label_map = None

if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_PATH):
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    with open(LABEL_PATH,'r') as f:
        label_map = json.load(f)
    inv_label_map = {int(k):v for k,v in label_map.items()}
else:
    st.warning('Model or label_map not found. Predictions disabled until model is added.')

st.sidebar.header('Options')
use_camera = st.sidebar.checkbox('Use webcam', value=True)
use_upload = st.sidebar.checkbox('Allow image upload', value=True)
target_lang = st.sidebar.text_input('Target language code (e.g. en, hi, bn, fr, de)', 'en')
do_translate = st.sidebar.checkbox('Translate output', value=True)
do_tts = st.sidebar.checkbox('Enable Text-to-Speech', value=True)

img_file_buffer = None
if use_camera:
    img_file_buffer = st.camera_input('Capture an image')
if use_upload:
    uploaded = st.file_uploader('Or upload an image', type=['jpg','jpeg','png'])
    if uploaded and img_file_buffer is None:
        img_file_buffer = uploaded

def predict_label(image):
    if model is None: return None, None
    lm = extract_hand_landmarks_from_image(image)
    if lm is None: return None, None
    plm = preprocess_landmarks(lm)
    if plm is None: return None, None
    X = np.expand_dims(plm, axis=0)
    preds = model.predict(X)
    idx = int(np.argmax(preds))
    conf = float(preds[0][idx])
    label = inv_label_map.get(idx, str(idx))
    return label, conf

if img_file_buffer is not None:
    image = Image.open(io.BytesIO(img_file_buffer.getvalue())).convert('RGB')
    st.image(image, caption='Captured Image', use_column_width=True)
    img = np.array(image)[:, :, ::-1]
    label, conf = predict_label(img)
    if label is None:
        st.warning('No gesture detected or model missing.')
    else:
        st.success(f'Predicted Gesture: {label} (confidence {conf:.2f})')
        text_output = label
        if do_translate:
            try:
                text_output = translate_text(label, dest_lang=target_lang)
                st.write(f'Translated ({target_lang}): {text_output}')
            except Exception as e:
                st.error(f'Translation failed: {e}')
        if do_tts:
            try:
                mp3_path = text_to_speech(text_output, lang=target_lang)
                st.audio(mp3_path)
            except Exception as e:
                st.error(f'TTS failed: {e}')
else:
    st.info('Please capture or upload an image to start.')
