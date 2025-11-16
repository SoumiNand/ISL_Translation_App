from gtts import gTTS
import tempfile

def text_to_speech(text, lang='en'):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tts = gTTS(text=text, lang=lang)
    tts.save(tmp.name)
    return tmp.name
