"""Bilingual query expansion for BM25 multilingual recall.

Appends English translation terms to non-English queries to improve BM25
keyword matching against the predominantly English dataset.
Only used for BM25 search — dense embeddings handle multilingual natively.
"""

from __future__ import annotations

import re

# Indic term → English equivalents. Curated for the MSMARCO-XI corporate domain.
_TERM_MAP: dict[str, list[str]] = {
    # ── Hindi (Devanagari) ──────────────────────────────────────────────────
    "कंपनी": ["company"],
    "निगम": ["corporation"],
    "कॉर्पोरेशन": ["corporation"],
    "व्यापार": ["business", "trade"],
    "व्यवसाय": ["business"],
    "कारोबार": ["business"],
    "शेयर": ["share", "stock"],
    "शेयरधारक": ["shareholder"],
    "भागधारक": ["shareholder"],
    "निदेशक": ["director"],
    "मंडल": ["board"],
    "कानूनी": ["legal"],
    "संस्था": ["entity", "institution"],
    "सीमित": ["limited"],
    "देयता": ["liability"],
    "प्रबंधन": ["management"],
    "गवर्नेंस": ["governance"],
    "लाभ": ["profit"],
    "पूंजी": ["capital", "equity"],
    "अधिकार": ["right", "authority"],
    "अनुच्छेद": ["article"],
    "कानून": ["law"],
    "राजधानी": ["capital"],
    "संरचना": ["structure"],
    "फायदे": ["benefits"],
    "मालिक": ["owner"],
    "लक्षण": ["characteristics"],
    "गठन": ["formation", "incorporation"],
    "निगमित": ["incorporated"],

    # ── Gujarati ────────────────────────────────────────────────────────────
    "કંપની": ["company"],
    "નિગમ": ["corporation"],
    "કાનૂની": ["legal"],
    "સંસ્થા": ["entity", "institution"],
    "શેરધારક": ["shareholder"],
    "સીમિત": ["limited"],
    "જવાબદારી": ["liability"],
    "ડિરેક્ટર": ["director"],
    "બોર્ડ": ["board"],
    "કૉર્પોરેટ": ["corporate"],
    "ગવર્નન્સ": ["governance"],
    "માળખું": ["structure"],
    "વ્યાપાર": ["business"],
    "ફાયદા": ["benefits"],
    "શેર": ["share", "stock"],
    "કાયદો": ["law"],
    "મૂડી": ["capital"],
    "ખાનગી": ["private"],
    "જાહેર": ["public"],
    "લક્ષણો": ["characteristics"],

    # ── Bengali ─────────────────────────────────────────────────────────────
    "কোম্পানী": ["company"],
    "কোম্পানি": ["company"],
    "নিগম": ["corporation"],
    "শেয়ারধারী": ["shareholder"],
    "সীমিত": ["limited"],
    "দায়": ["liability"],
    "পরিচালনা": ["management", "governance"],
    "পর্ষদ": ["board"],
    "ব্যবসা": ["business"],
    "কাঠামো": ["structure"],
    "সুবিধা": ["benefits"],
    "শেয়ার": ["share", "stock"],
    "আইন": ["law"],
    "মূলধন": ["capital"],
    "বেসরকারি": ["private"],
    "সরকারি": ["public"],
    "কর্পোরেট": ["corporate"],
    "গভর্ন্যান্স": ["governance"],
    "ইক্যুইটি": ["equity"],
    "মালিক": ["owner"],
    "পার্থক্য": ["difference"],

    # ── Assamese (Bengali script + ৰ/ৱ) ─────────────────────────────────────
    "কৰ্পোৰেচন": ["corporation"],
    "কোম্পানী": ["company"],
    "নিগম": ["corporation"],
    "নিগমৰ": ["corporation"],
    "বৈশিষ্ট্য": ["characteristics"],
    "শ্বেয়াৰহোল্ডাৰ": ["shareholder"],
    "সীমিত": ["limited"],
    "দায়বদ্ধতা": ["liability"],
    "পৰিচালনা": ["management", "governance"],
    "সমિતિ": ["board", "committee"],
    "নিগমিতকৰণ": ["incorporation"],
    "ব্যক্তিগত": ["private"],
    "ব্যৱসায়িক": ["business"],
    "ব্যৱসায়": ["business"],
    "সত্তা": ["entity"],
    "সংস্থা": ["entity", "institution"],
    "কৰ্পোৰেট": ["corporate"],
    "ইকুইটি": ["equity"],
    "শ্বেয়াৰ": ["share", "stock"],
    "গাঁথনি": ["structure"],
    "মালিকানা": ["ownership"],
    "ৰাজধানী": ["capital"],

    # ── Tamil ───────────────────────────────────────────────────────────────
    "நிறுவனம்": ["company", "corporation"],
    "கூட்டு": ["corporation", "joint"],
    "பங்குதாரர்": ["shareholder"],
    "வரையறுக்கப்பட்ட": ["limited"],
    "பொறுப்பு": ["liability"],
    "இயக்குநர்": ["director"],
    "குழு": ["board", "group"],
    "வணிகம்": ["business", "commerce"],
    "அமைப்பு": ["structure"],
    "பங்கு": ["share", "stock"],
    "நன்மைகள்": ["benefits"],
    "சட்டம்": ["law"],
    "மூலதனம்": ["capital"],
    "தனியார்": ["private"],
    "பொது": ["public"],
    "கார்ப்பரேட்": ["corporate"],
    "நிர்வாகம்": ["governance", "administration"],
    "உரிமையாளர்": ["owner"],
    "பண்புகள்": ["characteristics"],
    "ஈக்விட்டி": ["equity"],

    # ── Telugu ──────────────────────────────────────────────────────────────
    "కంపెనీ": ["company"],
    "కార్పొరేషన్": ["corporation"],
    "షేర్‌హోల్డర్": ["shareholder"],
    "పరిమిత": ["limited"],
    "బాధ్యత": ["liability"],
    "డైరెక్టర్ల": ["directors"],
    "డైరెక్టర్": ["director"],
    "బోర్డు": ["board"],
    "వ్యాపారం": ["business"],
    "నిర్మాణం": ["structure"],
    "షేర్": ["share", "stock"],
    "చట్టం": ["law"],
    "మూలధనం": ["capital"],
    "ప్రైవేట్": ["private"],
    "పబ్లిక్": ["public"],
    "కార్పొరేట్": ["corporate"],
    "పరిపాలన": ["governance", "administration"],

    # ── Marathi ─────────────────────────────────────────────────────────────
    "कंपनी": ["company"],
    "निगम": ["corporation"],
    "भागधारक": ["shareholder"],
    "मर्यादित": ["limited"],
    "दायित्व": ["liability"],
    "संचालक": ["director"],
    "मंडळ": ["board"],
    "व्यवसाय": ["business"],
    "रचना": ["structure"],
    "शेअर": ["share", "stock"],
    "कायदा": ["law"],
    "भांडवळ": ["capital"],
    "खाजगी": ["private"],
    "सार्वजनिक": ["public"],
    "कॉर्पोरेट": ["corporate"],
    "गव्हर्नन्स": ["governance"],
    "महामंडळ": ["corporation"],

    # ── Malayalam ───────────────────────────────────────────────────────────
    "കമ്പനി": ["company"],
    "നിഗമം": ["corporation"],
    "ഓഹരിയുടമ": ["shareholder"],
    "പരിമിത": ["limited"],
    "ബാധ്യത": ["liability"],
    "ഡയറക്ടർ": ["director"],
    "ബോർഡ്": ["board"],
    "വ്യാപാരം": ["business"],
    "ഓഹരി": ["share", "stock"],
    "നിയമം": ["law"],
    "മൂലധനം": ["capital"],

    # ── Kannada ─────────────────────────────────────────────────────────────
    "ಕಂಪನಿ": ["company"],
    "ನಿಗಮ": ["corporation"],
    "ಷೇರುದಾರ": ["shareholder"],
    "ಸೀಮಿತ": ["limited"],
    "ಹೊಣೆಗಾರಿಕೆ": ["liability"],
    "ನಿರ್ದೇಶಕರ": ["directors"],
    "ನಿರ್ದೇಶಕ": ["director"],
    "ಮಂಡಳಿ": ["board"],
    "ವ್ಯಾಪಾರ": ["business"],
    "ಷೇರು": ["share", "stock"],
    "ಕಾನೂನು": ["law"],
    "ಬಂಡವಾಳ": ["capital"],

    # ── Punjabi (Gurmukhi) ──────────────────────────────────────────────────
    "ਕੰਪਨੀ": ["company"],
    "ਕਾਰਪੋਰੇਸ਼ਨ": ["corporation"],
    "ਸ਼ੇਅਰਧਾਰਕ": ["shareholder"],
    "ਕਾਰੋਬਾਰ": ["business"],
    "ਸੀਮਤ": ["limited"],
    "ਦੇਣਦਾਰੀ": ["liability"],
    "ਡਾਇਰੈਕਟਰ": ["director"],
    "ਬੋਰਡ": ["board"],
    "ਸ਼ੇਅਰ": ["share", "stock"],
    "ਪੂੰਜੀ": ["capital"],

    # ── Odia ────────────────────────────────────────────────────────────────
    "କମ୍ପାନୀ": ["company"],
    "ନିଗମ": ["corporation"],
    "ଅଂଶୀଦାର": ["shareholder"],
    "ବ୍ୟବସାୟ": ["business"],
    "ସୀମିତ": ["limited"],
    "ଦାୟିତ୍ୱ": ["liability"],
    "ନିର୍ଦ୍ଦେଶକ": ["director"],
    "ବୋର୍ଡ": ["board"],
    "ଅଂଶ": ["share", "stock"],
    "ପୁଞ୍ଜି": ["capital"],

    # ── Urdu ────────────────────────────────────────────────────────────────
    "کمپنی": ["company"],
    "کارپوریشن": ["corporation"],
    "حصص_یافتہ": ["shareholder"],
    "محدود": ["limited"],
    "ذمہ_داری": ["liability"],
    "ڈائریکٹر": ["director"],
    "بورڈ": ["board"],
    "کاروبار": ["business"],
    "حصص": ["share", "stock"],
    "سرمایہ": ["capital"],
}

# Precompile word boundary patterns for fast matching
_COMPILED_PATTERNS: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(re.escape(term)), expansions)
    for term, expansions in _TERM_MAP.items()
]


def expand_query(query: str, language: str) -> str:
    """Expand a non-English query with English term translations for BM25.

    Only operates on non-English queries. English queries pass through unchanged.

    Args:
        query: The user's original query.
        language: Detected language code (e.g., 'hi', 'gu').

    Returns:
        Expanded query string with appended English terms.
    """
    if language == "en":
        return query

    if not query or not query.strip():
        return query

    expansions: list[str] = []
    for pattern, english_terms in _COMPILED_PATTERNS:
        if pattern.search(query):
            expansions.extend(english_terms)

    if not expansions:
        return query

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for term in expansions:
        if term not in seen:
            seen.add(term)
            unique.append(term)

    return f"{query} {' '.join(unique)}"


__all__ = ["expand_query"]
