# Indian Sign Language (ISL) Detection → Text → Multilingual Speech (Streamlit)

This project detects Indian Sign Language (ISL) gestures using MediaPipe landmarks, predicts the corresponding text with a trained model, translates it into a chosen language using `googletrans`, and converts the translated text into speech via `gTTS`.

## Features
- Live webcam or image upload input
- Static gesture recognition (A–Z or custom gestures)
- Multilingual translation via Google Translate API
- Speech output using gTTS
- Streamlit-based web interface

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Place your trained Keras model at `model/isl_model.h5`.
- Label map JSON (`model/label_map.json`) defines class indices → gesture names.
- You can deploy easily on Streamlit Cloud.
