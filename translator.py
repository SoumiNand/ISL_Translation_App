from googletrans import Translator

translator = Translator()

def translate_text(text, dest_lang='en'):
    try:
        res = translator.translate(text, dest=dest_lang)
        return res.text
    except Exception as e:
        raise e
