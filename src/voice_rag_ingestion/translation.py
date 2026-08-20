"""Translation abstraction layer for multilingual query support.

Provides a BaseTranslator protocol and two implementations:
- SarvamTranslator: Production translator using Sarvam AI API
- DictionaryTranslator: Testing fallback using static bilingual mappings
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from .language import SARVAM_LANG_MAP

logger = logging.getLogger(__name__)


class BaseTranslator(ABC):
    """Abstract base for translation providers."""

    @abstractmethod
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text from source_lang to target_lang.

        Args:
            text: The text to translate.
            source_lang: ISO 639-1 language code (e.g., 'hi', 'gu').
            target_lang: ISO 639-1 language code (e.g., 'en').

        Returns:
            Translated text.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the translator backend is ready."""
        ...


class SarvamTranslator(BaseTranslator):
    """Production translator using Sarvam AI REST API.

    Lazy-initialized: the HTTP client is created on first use.
    Thread-safe: each call is independent.
    """

    TRANSLATE_URL = "https://api.sarvam.ai/translate"

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self._session: Any = None  # lazy httpx.Client

    def _get_session(self) -> Any:
        if self._session is None:
            import httpx
            self._session = httpx.Client(timeout=10.0)
        return self._session

    def is_available(self) -> bool:
        return bool(self._api_key)

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text or not text.strip():
            return text
        if source_lang == target_lang:
            return text
        if not self._api_key:
            logger.warning("Sarvam API key not configured; returning original text")
            return text

        src_code = SARVAM_LANG_MAP.get(source_lang, f"{source_lang}-IN")
        tgt_code = SARVAM_LANG_MAP.get(target_lang, f"{target_lang}-IN")

        payload = {
            "input": text,
            "source_language_code": src_code,
            "target_language_code": tgt_code,
            "model": "mayura:v1",
        }
        headers = {
            "api-subscription-key": self._api_key,
            "Content-Type": "application/json",
        }

        try:
            session = self._get_session()
            resp = session.post(self.TRANSLATE_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            translated = data.get("translated_text", text)
            return translated if translated else text
        except Exception as exc:
            logger.warning("Sarvam translation failed: %s; returning original", exc)
            return text

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None


# ── Dictionary-based fallback for testing ─────────────────────────────────

_DICTIONARY: dict[tuple[str, str], dict[str, str]] = {
    # ── Hindi → English ───────────────────────────────────────────────────
    ("hi", "en"): {
        "कंपनी क्या है?": "What is a company?",
        "निगम क्या है?": "What is a corporation?",
        "निगम की परिभाषा क्या है?": "What is the definition of a corporation?",
        "शेयरधारक क्या है?": "What is a shareholder?",
        "सीमित देयता क्या है?": "What is limited liability?",
        "निदेशक मंडल क्या है?": "What is a board of directors?",
        "कॉर्पोरेशन क्या है?": "What is a corporation?",
        "कॉर्पोरेट गवर्नेंस क्या है?": "What is corporate governance?",
        "कानून में निगम क्या है?": "What is a corporation in law?",
        "व्यवसाय इकाई क्या है?": "What is a business entity?",
        "शेयर क्या होते हैं?": "What are shares?",
        "निगम क्यों बनाया जाता है?": "Why is a corporation formed?",
        "कारोबार कैसे शुरू करें?": "How to start a business?",
        "निगम का गठन कैसे होता है?": "How is a corporation formed?",
        "निगम में लाभ कैसे बंटता है?": "How is profit distributed in a corporation?",
        "कॉर्पोरेशन में सीमित देयता का अर्थ": "Meaning of limited liability in a corporation",
        "कॉर्पोरेशन के लक्षण क्या हैं?": "What are the characteristics of a corporation?",
        "भारत की राजधानी क्या है?": "What is the capital of India?",
        "पृथ्वी से चंद्रमा की दूरी कितनी है?": "What is the distance from Earth to the Moon?",
        "मुंबई का तापमान क्या है?": "What is the temperature in Mumbai?",
        "कंपनी की संरचना क्या है?": "What is corporate structure?",
        "निगम के फायदे क्या हैं?": "What are the benefits of incorporation?",
        "व्यापार कैसे चलता है?": "How does a business work?",
        "कंपनी और निगम में क्या अंतर है?": "What is the difference between a company and a corporation?",
        "शेयर बाजार क्या है?": "What is the stock market?",
        "कंपनी का मालिक कौन होता है?": "Who owns a company?",
        "निगम कैसे काम करता है?": "How does a corporation work?",
        "व्यवसाय का अर्थ क्या है?": "What is the meaning of business?",
        "कंपनी कानून क्या है?": "What is company law?",
        "पूंजी क्या होती है?": "What is capital in a company?",
    },

    # ── Gujarati → English ────────────────────────────────────────────────
    ("gu", "en"): {
        "કંપની શું છે?": "What is a company?",
        "નિગમ એટલે શું?": "What is a corporation?",
        "કાનૂની સંસ્થા એટલે શું?": "What is a legal entity?",
        "શેરધારક શું છે?": "What is a shareholder?",
        "સીમિત જવાબદારી એટલે શું?": "What is limited liability?",
        "ડિરેક્ટર બોર્ડ એટલે શું?": "What is a board of directors?",
        "કૉર્પોરેટ ગવર્નન્સ શું છે?": "What is corporate governance?",
        "કંપની કેવી રીતે બને છે?": "How is a company formed?",
        "કંપનીનું માળખું શું છે?": "What is corporate structure?",
        "વ્યાપાર એટલે શું?": "What is a business entity?",
        "નિગમના ફાયદા શું છે?": "What are the benefits of incorporation?",
        "શેર એટલે શું?": "What are shares?",
        "કંપની કાયદો શું છે?": "What is company law?",
        "મૂડી એટલે શું?": "What is capital in a company?",
        "ખાનગી કંપની શું છે?": "What is a private company?",
        "જાહેર કંપની શું છે?": "What is a public company?",
        "નિગમ કેવી રીતે કામ કરે છે?": "How does a corporation work?",
        "વ્યાપાર કેવી રીતે ચાલે છે?": "How does a business work?",
        "કંપનીના લક્ષણો શું છે?": "What are the characteristics of a corporation?",
        "નિગમ શા માટે બનાવવામાં આવે છે?": "Why do companies incorporate?",
        "કંપની એક કાનૂની સંસ્થા છે": "A company is a legal entity",
    },

    # ── Assamese → English ────────────────────────────────────────────────
    ("as", "en"): {
        "কৰ্পোৰেচন কি?": "What is a corporation?",
        "কোম্পানী এটা কেনেকৈ গঠন কৰা হয়?": "How is a company formed?",
        "নিগমৰ অৰ্থ কি?": "What is the meaning of corporation?",
        "কৰ্পোৰেচনৰ বৈশিষ্ট্য কি?": "What are the characteristics of a corporation?",
        "শ্বেয়াৰহোল্ডাৰ কি?": "What is a shareholder?",
        "সীমিত দায়বদ্ধতা কি?": "What is limited liability?",
        "পৰিচালনা সমিতি কি?": "What is a board of directors?",
        "নিগমিতকৰণ মানে কি?": "What is incorporation?",
        "ব্যক্তিগত কোম্পানী কি?": "What is a private company?",
        "কৰ্পোৰেচন মানে কি?": "What is meant by the term corporation?",
        "ব্যৱসায়িক সত্তা কি?": "What is a business entity?",
        "কৰ্পোৰেট পৰিচালনা কি?": "What is corporate governance?",
        "কোম্পানী আইনত কি?": "What is a corporation in law?",
        "কোম্পানীত ইকুইটি কি?": "What is equity in a company?",
        "কৰ্পোৰেচনত শ্বেয়াৰ কি?": "What are shares in a corporation?",
        "সীমিত দায় কোম্পানী কি?": "What is a limited liability company?",
        "সংস্থা কি?": "What is an entity or corporation?",
        "কোম্পানীৰ গাঁথনি কি?": "What is corporate structure?",
        "কৰ্পোৰেচনৰ মালিকানা কি?": "Who owns a corporation?",
        "ভাৰতৰ ৰাজধানী কি?": "What is the capital of India?",
    },

    # ── Bengali → English ─────────────────────────────────────────────────
    ("bn", "en"): {
        "কোম্পানী কি?": "What is a company?",
        "কোম্পানি কি?": "What is a company?",
        "নিগম কি?": "What is a corporation?",
        "শেয়ারধারী কি?": "What is a shareholder?",
        "সীমিত দায় কি?": "What is limited liability?",
        "পরিচালনা পর্ষদ কি?": "What is a board of directors?",
        "কোম্পানি কিভাবে গঠিত হয়?": "How is a company formed?",
        "ব্যবসা কি?": "What is a business entity?",
        "কোম্পানির কাঠামো কি?": "What is corporate structure?",
        "নিগমের সুবিধা কি?": "What are the benefits of incorporation?",
        "শেয়ার কি?": "What are shares?",
        "কোম্পানি আইন কি?": "What is company law?",
        "মূলধন কি?": "What is capital in a company?",
        "বেসরকারি কোম্পানি কি?": "What is a private company?",
        "সরকারি কোম্পানি কি?": "What is a public company?",
        "নিগম কিভাবে কাজ করে?": "How does a corporation work?",
        "ব্যবসা কিভাবে চলে?": "How does a business work?",
        "কোম্পানি ও নিগমের পার্থক্য কি?": "What is the difference between a company and a corporation?",
        "কর্পোরেট গভর্ন্যান্স কি?": "What is corporate governance?",
        "ইক্যুইটি কি?": "What is equity in a company?",
        "কোম্পানির মালিক কে?": "Who owns a company?",
    },

    # ── Tamil → English ───────────────────────────────────────────────────
    ("ta", "en"): {
        "நிறுவனம் என்றால் என்ன?": "What is a company?",
        "கூட்டு நிறுவனம் என்றால் என்ன?": "What is a corporation?",
        "பங்குதாரர் என்றால் என்ன?": "What is a shareholder?",
        "வரையறுக்கப்பட்ட பொறுப்பு என்ன?": "What is limited liability?",
        "இயக்குநர் குழு என்றால் என்ன?": "What is a board of directors?",
        "நிறுவனம் எவ்வாறு உருவாகிறது?": "How is a company formed?",
        "வணிகம் என்றால் என்ன?": "What is a business entity?",
        "நிறுவன அமைப்பு என்ன?": "What is corporate structure?",
        "பங்கு என்றால் என்ன?": "What are shares?",
        "நிறுவனத்தின் நன்மைகள் என்ன?": "What are the benefits of incorporation?",
        "நிறுவன சட்டம் என்ன?": "What is company law?",
        "மூலதனம் என்றால் என்ன?": "What is capital in a company?",
        "தனியார் நிறுவனம் என்ன?": "What is a private company?",
        "பொது நிறுவனம் என்ன?": "What is a public company?",
        "நிறுவனம் எவ்வாறு செயல்படுகிறது?": "How does a company work?",
        "வணிகம் எவ்வாறு நடக்கிறது?": "How does a business work?",
        "கார்ப்பரேட் நிர்வாகம் என்ன?": "What is corporate governance?",
        "நிறுவன உரிமையாளர் யார்?": "Who owns a company?",
        "நிறுவனத்தின் பண்புகள் என்ன?": "What are the characteristics of a corporation?",
        "ஈக்விட்டி என்றால் என்ன?": "What is equity in a company?",
    },

    # ── Telugu → English ──────────────────────────────────────────────────
    ("te", "en"): {
        "కంపెనీ అంటే ఏమిటి?": "What is a company?",
        "కార్పొరేషన్ అంటే ఏమిటి?": "What is a corporation?",
        "షేర్‌హోల్డర్ అంటే ఏమిటి?": "What is a shareholder?",
        "పరిమిత బాధ్యత అంటే ఏమిటి?": "What is limited liability?",
        "డైరెక్టర్ల బోర్డు అంటే ఏమిటి?": "What is a board of directors?",
        "కంపెనీ ఎలా ఏర్పడుతుంది?": "How is a company formed?",
        "వ్యాపారం అంటే ఏమిటి?": "What is a business entity?",
        "కంపెనీ నిర్మాణం ఏమిటి?": "What is corporate structure?",
        "షేర్ అంటే ఏమిటి?": "What are shares?",
        "కంపెనీ చట్టం ఏమిటి?": "What is company law?",
        "మూలధనం అంటే ఏమిటి?": "What is capital in a company?",
        "ప్రైవేట్ కంపెనీ అంటే ఏమిటి?": "What is a private company?",
        "పబ్లిక్ కంపెనీ అంటే ఏమిటి?": "What is a public company?",
        "కార్పొరేట్ పరిపాలన ఏమిటి?": "What is corporate governance?",
        "కంపెనీ ఎలా పనిచేస్తుంది?": "How does a company work?",
    },

    # ── Marathi → English ─────────────────────────────────────────────────
    ("mr", "en"): {
        "कंपनी म्हणजे काय?": "What is a company?",
        "निगम म्हणजे काय?": "What is a corporation?",
        "भागधारक म्हणजे काय?": "What is a shareholder?",
        "मर्यादित दायित्व म्हणजे काय?": "What is limited liability?",
        "संचालक मंडळ म्हणजे काय?": "What is a board of directors?",
        "कंपनी कशी तयार होते?": "How is a company formed?",
        "व्यवसाय म्हणजे काय?": "What is a business entity?",
        "कंपनीची रचना काय आहे?": "What is corporate structure?",
        "शेअर म्हणजे काय?": "What are shares?",
        "कंपनी कायदा म्हणजे काय?": "What is company law?",
        "भांडवळ म्हणजे काय?": "What is capital in a company?",
        "खाजगी कंपनी म्हणजे काय?": "What is a private company?",
        "सार्वजनिक कंपनी म्हणजे काय?": "What is a public company?",
        "कॉर्पोरेट गव्हर्नन्स म्हणजे काय?": "What is corporate governance?",
        "कंपनी कशी काम करते?": "How does a company work?",
    },

    # ── Malayalam → English ───────────────────────────────────────────────
    ("ml", "en"): {
        "കമ്പനി എന്താണ്?": "What is a company?",
        "നിഗമം എന്താണ്?": "What is a corporation?",
        "ഓഹരിയുടമ എന്താണ്?": "What is a shareholder?",
        "പരിമിത ബാധ്യത എന്താണ്?": "What is limited liability?",
        "ഡയറക്ടർ ബോർഡ് എന്താണ്?": "What is a board of directors?",
        "കമ്പനി എങ്ങനെ രൂപീകരിക്കുന്നു?": "How is a company formed?",
        "വ്യാപാരം എന്താണ്?": "What is a business entity?",
        "ഓഹരി എന്താണ്?": "What are shares?",
        "കമ്പനി നിയമം എന്താണ്?": "What is company law?",
        "മൂലധനം എന്താണ്?": "What is capital in a company?",
    },

    # ── Kannada → English ─────────────────────────────────────────────────
    ("kn", "en"): {
        "ಕಂಪನಿ ಎಂದರೇನು?": "What is a company?",
        "ನಿಗಮ ಎಂದರೇನು?": "What is a corporation?",
        "ಷೇರುದಾರ ಎಂದರೇನು?": "What is a shareholder?",
        "ಸೀಮಿತ ಹೊಣೆಗಾರಿಕೆ ಎಂದರೇನು?": "What is limited liability?",
        "ನಿರ್ದೇಶಕರ ಮಂಡಳಿ ಎಂದರೇನು?": "What is a board of directors?",
        "ಕಂಪನಿ ಹೇಗೆ ರಚನೆಯಾಗುತ್ತದೆ?": "How is a company formed?",
        "ವ್ಯಾಪಾರ ಎಂದರೇನು?": "What is a business entity?",
        "ಷೇರು ಎಂದರೇನು?": "What are shares?",
        "ಕಂಪನಿ ಕಾನೂನು ಎಂದರೇನು?": "What is company law?",
        "ಬಂಡವಾಳ ಎಂದರೇನು?": "What is capital in a company?",
    },

    # ── Punjabi → English ─────────────────────────────────────────────────
    ("pa", "en"): {
        "ਕੰਪਨੀ ਕੀ ਹੈ?": "What is a company?",
        "ਕਾਰਪੋਰੇਸ਼ਨ ਕੀ ਹੈ?": "What is a corporation?",
        "ਸ਼ੇਅਰਧਾਰਕ ਕੀ ਹੈ?": "What is a shareholder?",
        "ਕਾਰੋਬਾਰ ਕੀ ਹੈ?": "What is a business entity?",
        "ਕੰਪਨੀ ਕਿਵੇਂ ਬਣਦੀ ਹੈ?": "How is a company formed?",
    },

    # ── Odia → English ────────────────────────────────────────────────────
    ("or", "en"): {
        "କମ୍ପାନୀ କଣ?": "What is a company?",
        "ନିଗମ କଣ?": "What is a corporation?",
        "ଅଂଶୀଦାର କଣ?": "What is a shareholder?",
        "ବ୍ୟବସାୟ କଣ?": "What is a business entity?",
        "କମ୍ପାନୀ କିପରି ଗଠିତ ହୁଏ?": "How is a company formed?",
    },

    # ── English → Indic (Answer translations) ──────────────────────────────
    ("en", "hi"): {
        "A company is a legal entity": "कंपनी एक कानूनी संस्था है",
        "A corporation is a legal entity formed by a group of individuals":
            "निगम व्यक्तियों के समूह द्वारा गठित एक कानूनी संस्था है",
    },
    ("en", "gu"): {
        "A company is a legal entity": "કંપની એક કાનૂની સંસ્થા છે",
        "A corporation is a legal entity formed by a group of individuals":
            "નિગમ એ વ્યક્તિઓના જૂથ દ્વારા રચાયેલી કાનૂની સંસ્થા છે",
    },
    ("en", "as"): {
        "A company is a legal entity": "কোম্পানী এক আইনী সত্তা",
        "A corporation is a legal entity formed by a group of individuals":
            "কৰ্পোৰেচন হৈছে ব্যক্তিৰ এটা দলৰ দ্বাৰা গঠিত এক আইনী সত্তা",
    },
    ("en", "bn"): {
        "A company is a legal entity": "কোম্পানি একটি আইনি সত্তা",
        "A corporation is a legal entity formed by a group of individuals":
            "নিগম হলো ব্যক্তিদের একটি দল দ্বারা গঠিত একটি আইনি সত্তা",
    },
    ("en", "ta"): {
        "A company is a legal entity": "நிறுவனம் என்பது ஒரு சட்டப்பூர்வ அமைப்பாகும்",
        "A corporation is a legal entity formed by a group of individuals":
            "கார்ப்பரேஷன் என்பது நபர்களின் குழுவால் உருவாக்கப்பட்ட ஒரு சட்ட அமைப்பாகும்",
    },
    ("en", "te"): {
        "A company is a legal entity": "కంపెనీ అనేది ఒక చట్టబద్ధమైన సంస్థ",
        "A corporation is a legal entity formed by a group of individuals":
            "కార్పొరేషన్ అనేది వ్యక్తుల సమూహం ద్వారా ఏర్పడిన చట్టపరమైన సంస్థ",
    },
    ("en", "mr"): {
        "A company is a legal entity": "कंपनी ही एक कायदेशीर संस्था आहे",
        "A corporation is a legal entity formed by a group of individuals":
            "निगम ही व्यक्तींच्या समूहाने स्थापन केलेली कायदेशीर संस्था आहे",
    },
    ("en", "ml"): {
        "A company is a legal entity": "കമ്പനി ഒരു നിയമപരമായ സ്ഥാപനമാണ്",
        "A corporation is a legal entity formed by a group of individuals":
            "കോർപ്പറേഷൻ എന്നത് വ്യക്തികളുടെ ഒരു കൂട്ടം രൂപീകരിച്ച നിയമപരമായ സ്ഥാപനമാണ്",
    },
    ("en", "kn"): {
        "A company is a legal entity": "ಕಂಪನಿಯು ಒಂದು ಕಾನೂನುಬದ್ಧ ಸಂಸ್ಥೆಯಾಗಿದೆ",
        "A corporation is a legal entity formed by a group of individuals":
            "ನಿಗಮವು ವ್ಯಕ್ತಿಗಳ ಗುಂಪಿನಿಂದ ರಚಿಸಲ್ಪಟ್ಟ ಕಾನೂನು ಘಟಕವಾಗಿದೆ",
    },
    ("en", "pa"): {
        "A company is a legal entity": "ਕੰਪਨੀ ਇੱਕ ਕਾਨੂੰਨੀ ਸੰਸਥਾ ਹੈ",
        "A corporation is a legal entity formed by a group of individuals":
            "ਕਾਰਪੋਰੇਸ਼ਨ ਵਿਅਕਤੀਆਂ ਦੇ ਸਮੂਹ ਦੁਆਰਾ ਬਣਾਈ ਗਈ ਇੱਕ ਕਾਨੂੰਨੀ ਸੰਸਥਾ ਹੈ",
    },
    ("en", "or"): {
        "A company is a legal entity": "କମ୍ପାନୀ ଏକ ଆଇନଗତ ସଂସ୍ଥା ଅଟେ",
        "A corporation is a legal entity formed by a group of individuals":
            "ନିଗମ ହେଉଛି ବ୍ୟକ୍ତିବିଶେଷଙ୍କ ଗୋଷ୍ଠୀ ଦ୍ୱାରା ଗଠିତ ଏକ ଆଇନଗତ ସଂସ୍ଥା",
    },
}


class DictionaryTranslator(BaseTranslator):
    """Static dictionary-based translator for testing without API keys.

    Falls back to returning the original text if no mapping exists.
    """

    def __init__(self, dictionary: dict[tuple[str, str], dict[str, str]] | None = None) -> None:
        self._dict = dictionary or _DICTIONARY

    def is_available(self) -> bool:
        return True

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text or not text.strip():
            return text
        if source_lang == target_lang:
            return text

        key = (source_lang, target_lang)
        lang_dict = self._dict.get(key, {})
        result = lang_dict.get(text.strip())
        if result:
            return result

        # Try case-insensitive match
        text_lower = text.strip().lower()
        for k, v in lang_dict.items():
            if k.lower() == text_lower:
                return v

        logger.debug("No dictionary entry for '%s' (%s→%s)", text[:50], source_lang, target_lang)
        return text


def build_translator(provider: str = "sarvam", **kwargs: Any) -> BaseTranslator:
    """Factory to build the appropriate translator."""
    if provider == "sarvam":
        translator = SarvamTranslator(**kwargs)
        if translator.is_available():
            return translator
        logger.info("Sarvam API key not available; falling back to dictionary translator")
        return DictionaryTranslator()
    elif provider == "dictionary":
        return DictionaryTranslator(**kwargs)
    else:
        raise ValueError(f"Unknown translation provider: {provider}")


__all__ = [
    "BaseTranslator",
    "SarvamTranslator",
    "DictionaryTranslator",
    "build_translator",
]
