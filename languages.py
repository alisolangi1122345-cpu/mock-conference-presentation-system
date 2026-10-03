import os
import re
import string
import unicodedata

# ==========================================
# AI-Based Mock Conference Presentation System
# Language settings (English, Urdu, Hindi, Arabic)
#
# To add another language: copy one block in RAW, give it a new code,
# and fill in its Whisper code, filler words, intro/conclusion cues,
# linking words and stopwords.
# ==========================================

# Languages the app currently offers. The others stay in RAW but are switched off.
# To turn more on later, e.g.: ENABLED_LANGUAGES = ["en", "ur", "hi", "ar"]
ENABLED_LANGUAGES = ["en"]

OUTPUT_DIR = "output"
LANGUAGE_FILE = os.path.join(OUTPUT_DIR, "language.txt")
DEFAULT_CODE = "en"

EN_MODEL = "all-MiniLM-L6-v2"
MULTI_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

PUNCT = "".join(set(string.punctuation) | set("۔،؛؟«»…—–।॥“”‘’"))

_DIACRITICS = re.compile("[\u064B-\u065F\u0670\u0640\u06D6-\u06ED]")
# Makes Urdu and Arabic letter variants compare equal (applied to text AND cues)
_AR_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ی": "ي", "ې": "ي", "ے": "ي", "ۓ": "ي",
    "ک": "ك", "ں": "ن",
    "ہ": "ه", "ھ": "ه", "ۃ": "ه", "ة": "ه", "ۂ": "ه",
})


RAW = {
    "en": {
        "name": "English",
        "whisper": "en",
        "prompt": "Um, so, uh, like, you know, basically, I will present my work.",
        "embed_model": EN_MODEL,
        "min_len": 3,
        "hard": "um umm uh uhh er erm ah hmm hm",
        "soft": "like so basically actually well",
        "intro": [
            "hello", "hi everyone", "good morning", "good afternoon", "good evening",
            "welcome", "today i", "my topic", "i will present", "i am going to",
            "i'm going to", "going to talk", "going to present", "my name is",
            "presentation on", "present my",
        ],
        "conclusion": [
            "thank you", "thanks", "in conclusion", "to conclude", "to sum up",
            "in summary", "to summarize", "finally", "any questions", "that's all",
            "that is all",
        ],
        "transitions": [
            "first", "second", "third", "next", "then", "also", "because",
            "for example", "for instance", "however", "moreover", "another",
            "in addition", "therefore", "as a result", "finally",
        ],
        "stopwords": """a an the and or but if so of to in on at by for with from as is are
            was were be been being am i me my we our you your he she it its they them their
            this that these those there here what which who whom will would can could should
            may might do does did done have has had not no yes very just also then than too
            about into over under up down out um uh er ah hmm like want say something going""",
    },
    "ur": {
        "name": "Urdu (اردو)",
        "whisper": "ur",
        "prompt": "ام، اچھا، مطلب، یعنی، میں آج اپنی پریزنٹیشن پیش کروں گا۔",
        "embed_model": MULTI_MODEL,
        "min_len": 2,
        "hard": "ام امم اممم ہمم آہ اہ",
        "soft": "مطلب یعنی بس تو اچھا وغیرہ",
        "intro": [
            "السلام علیکم", "آداب", "خوش آمدید", "صبح بخیر", "میرا نام", "میرا موضوع",
            "آج میں", "میں آج", "پیش کرنے", "پیش کروں", "بات کروں", "پریزنٹیشن",
        ],
        "conclusion": [
            "شکریہ", "آخر میں", "اختتام", "خلاصہ", "نتیجہ", "اللہ حافظ", "خدا حافظ",
            "سوالات", "شکر گزار",
        ],
        "transitions": [
            "پہلا", "پہلے", "دوسرا", "دوسری", "تیسرا", "اس کے بعد", "پھر", "کیونکہ",
            "مثال کے طور پر", "مثلاً", "لیکن", "اس کے علاوہ", "اسی لیے", "اس لیے",
            "آخر میں",
        ],
        "stopwords": """اور کی کا کے میں ہے ہیں ہو سے کو نے پر یہ وہ کہ اس ان تھا تھی تھے
            ہم آپ بھی تو جو کر کیا ایک لیے لئے ساتھ بہت ہوتا ہوتی ہوتے گا گی گے رہا رہی
            رہے والا والی والے سب کچھ کوئی مگر لیکن اگر پھر جب تک ہی نہیں نہ آج میرا میری
            میرے اپنا اپنی اپنے ہوں ہوگا ہوگی مطلب یعنی""",
    },
    "hi": {
        "name": "Hindi (हिन्दी)",
        "whisper": "hi",
        "prompt": "उम, अच्छा, मतलब, यानी, आज मैं अपना प्रेजेंटेशन प्रस्तुत करूंगा।",
        "embed_model": MULTI_MODEL,
        "min_len": 2,
        "hard": "उम उम्म हम्म अं आं",
        "soft": "मतलब यानी बस तो अच्छा वो",
        "intro": [
            "नमस्ते", "नमस्कार", "स्वागत", "सुप्रभात", "मेरा नाम", "मेरा विषय",
            "आज मैं", "प्रस्तुत", "बात करने", "प्रेजेंटेशन",
        ],
        "conclusion": [
            "धन्यवाद", "शुक्रिया", "अंत में", "निष्कर्ष", "सारांश", "आभार", "प्रश्न",
        ],
        "transitions": [
            "पहला", "पहले", "दूसरा", "तीसरा", "इसके बाद", "फिर", "क्योंकि",
            "उदाहरण के लिए", "जैसे", "लेकिन", "इसके अलावा", "इसलिए", "अंत में",
        ],
        "stopwords": """और का की के में है हैं हो से को ने पर यह वह कि इस उस था थी थे हम आप
            भी तो जो कर क्या एक लिए साथ बहुत होता होती होते गा गी गे रहा रही रहे सब कुछ
            कोई मगर लेकिन अगर जब तक ही नहीं न मैं मेरा मेरी मेरे अपना अपनी अपने मतलब यानी""",
    },
    "ar": {
        "name": "Arabic (العربية)",
        "whisper": "ar",
        "prompt": "امم، يعني، طيب، اليوم سأقدم عرضي.",
        "embed_model": MULTI_MODEL,
        "min_len": 2,
        "hard": "امم اممم اه همم ااه",
        "soft": "يعني طيب بس هكذا",
        "intro": [
            "السلام عليكم", "مرحبا", "اهلا", "صباح الخير", "مساء الخير", "اسمي",
            "موضوعي", "اليوم سوف", "اليوم سا", "سأتحدث", "سأقدم", "عرضي", "اود ان",
        ],
        "conclusion": [
            "شكرا", "في الختام", "ختاما", "خلاصة", "في النهاية", "الاسئلة",
            "جزاكم الله", "اخيرا",
        ],
        "transitions": [
            "اولا", "ثانيا", "ثالثا", "ثم", "بعد ذلك", "لان", "على سبيل المثال",
            "مثلا", "لكن", "بالاضافة", "لذلك", "اخيرا",
        ],
        "stopwords": """في من على الى إلى عن مع هذا هذه ذلك تلك هو هي هم نحن انا انت كان كانت
            يكون ان أن إن ما لا لم لن قد كل بعض او أو و ثم لكن حتى اذا إذا هنا هناك الذي
            التي الذين عند بين بعد قبل كما ايضا أيضا جدا فقط يعني""",
    },
}


# ---------- text helpers (work for every script) ----------

def normalize(text):
    text = unicodedata.normalize("NFC", text).lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = _DIACRITICS.sub("", text)
    return text.translate(_AR_MAP)


def tokenize(text):
    tokens = []
    for raw in normalize(text).split():
        t = raw.strip(PUNCT)
        if t and not t.isdigit():
            tokens.append(t)
    return tokens


def clean_word(word):
    t = tokenize(word)
    return t[0] if t else ""


def split_sentences(text):
    parts = re.split(r"(?<=[.!?۔؟।])\s+", text.strip())
    return [p.strip() for p in parts if len(p.split()) >= 2]


def has_cue(text, cues):
    n = normalize(text)
    return any(c in n for c in cues)


# ---------- language lookup ----------

_CACHE = {}


def _prepare(code):
    raw = RAW[code]
    return {
        "code": code,
        "name": raw["name"],
        "whisper": raw["whisper"],
        "prompt": raw["prompt"],
        "embed_model": raw["embed_model"],
        "min_len": raw["min_len"],
        "hard": {normalize(w) for w in raw["hard"].split()},
        "soft": {normalize(w) for w in raw["soft"].split()},
        "intro": [normalize(c) for c in raw["intro"]],
        "conclusion": [normalize(c) for c in raw["conclusion"]],
        "transitions": [normalize(c) for c in raw["transitions"]],
        "stopwords": {normalize(w) for w in raw["stopwords"].split()},
    }


def available_languages():
    return [(code, RAW[code]["name"]) for code in ENABLED_LANGUAGES]


def current_language_code():
    try:
        with open(LANGUAGE_FILE, encoding="utf-8") as f:
            code = f.read().strip()
        if code in ENABLED_LANGUAGES:
            return code
    except OSError:
        pass
    return DEFAULT_CODE


def save_language(code):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(LANGUAGE_FILE, "w", encoding="utf-8") as f:
        f.write(code)


def get_language(code=None):
    code = code or current_language_code()
    if code not in RAW:
        raise ValueError(f"Unknown language '{code}'. Choose from: {', '.join(RAW)}")
    if code not in _CACHE:
        _CACHE[code] = _prepare(code)
    return _CACHE[code]
