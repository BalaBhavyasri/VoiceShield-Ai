import re
from typing import Any, Dict, List

# ==============================================================================
# VoiceShield AI: Multilingual Rule-Based Scam Tactic Detector
# Covers English, Telugu-English (Tanglish), Hindi-English (Hinglish), and Telugu script.
# Designed for high explainability and low false-positives.
# ==============================================================================

RULES = [
    {
        "category": "OTP or credential theft",
        "severity": "high",
        "patterns": [
            # English
            r"\b(share|tell|send|provide|reveal|confirm|give|enter|read out)\b.{0,45}\b(otp|one[- ]time password|verification code|pin|cvv|password|mpin)\b",
            r"\b(otp|one[- ]time password|verification code|pin|cvv|password|mpin)\b.{0,45}\b(share|tell|send|provide|reveal|give|read out)\b",
            # Telugu-English (Tanglish)
            r"\b(otp|pin|password|verification code)\b.{0,45}\b(cheppandi|ivvandi|pampandi|share cheyandi|enter cheyandi|cheppu)\b",
            r"\b(cheppandi|ivvandi|pampandi|share cheyandi)\b.{0,45}\b(otp|pin|password|mpin)\b",
            r"\b(mee|me|naaku)\b.{0,30}\b(otp|pin|password)\b.{0,30}\b(cheppandi|ivvandi)\b",
            # Hindi-English (Hinglish)
            r"\b(otp|pin|password)\b.{0,40}\b(batao|bataiye|share karo|bhejo|de do|bata do)\b",
            r"\b(batao|bataiye|bhejiye|share kijiye)\b.{0,40}\b(otp|pin|password)\b",
            # Telugu Script
            r"(ఓటీపీ|పిన్|పాస్‌వర్డ్|ఓటిపి).{0,35}(చెప్పండి|ఇవ్వండి|పంపండి|షేర్ చేయండి)",
        ],
        "explanation": (
            "The caller or message asks for a sensitive authentication code (OTP, PIN, or password). "
            "Legitimate organizations will NEVER ask you to reveal your OTP over a call."
        ),
        "recommendation": (
            "Do not share your OTP or PIN with anyone. Disconnect immediately and contact "
            "your bank using the official phone number on your card."
        ),
        "regional_advice": {
            "telugu": "మీ ఓటీపీ లేదా పిన్ ఎవరితోనూ చెప్పకండి. వెంటనే ఫోన్ కట్ చేసి మీ బ్యాంక్ అఫీషియల్ నంబర్‌కు కాల్ చేయండి.",
            "hindi": "अपना ओटीपी या पिन किसी के साथ साझा न करें। तुरंत कॉल काटें और अपने बैंक के आधिकारिक नंबर पर संपर्क करें."
        }
    },
    {
        "category": "Urgent payment pressure",
        "severity": "high",
        "patterns": [
            # English
            r"\b(pay|transfer|send|deposit)\b.{0,50}\b(now|immediately|urgent|urgently|today|within \d+ minutes|instantly)\b",
            r"\b(account|service|connection|sim card|electricity)\b.{0,50}\b(suspend|block|freeze|deactivate|cut|disconnected)\b.{0,60}\b(pay|payment|transfer)\b",
            r"\b(pay|payment|transfer|recharge)\b.{0,60}\b(avoid|prevent|stop)\b.{0,40}\b(suspension|blocking|deactivation|disconnection)\b",
            # Telugu-English (Tanglish)
            r"\b(urgent ga|ventane|ippude)\b.{0,40}\b(dabbu|money|amount|payment|transfer)\b.{0,40}\b(pampandi|cheyandi|kattandi)\b",
            r"\b(account|current bill|electricity|sim)\b.{0,45}\b(block|freeze|disconnect|aipothundi|cut chestharu)\b.{0,45}\b(dabbu|pay|transfer|kattandi)\b",
            r"\b(dabbu|amount|fee)\b.{0,40}\b(kattakapothe|pay cheyakapothe)\b.{0,40}\b(block|cut|cancel)\b",
            # Hindi-English (Hinglish)
            r"\b(turant|abhi|jaldi)\b.{0,40}\b(paise|payment|transfer|bhejo)\b",
            r"\b(account|bijli|sim)\b.{0,45}\b(band|block|kat jayega)\b.{0,45}\b(paise|bharo|transfer)\b",
            # Telugu Script
            r"(ఖాతా|కరెంట్ బిల్లు|సిమ్).{0,35}(బ్లాక్|కట్|నిలిపివేయబడుతుంది).{0,35}(డబ్బులు|చెల్లించండి|కట్టండి)",
        ],
        "explanation": (
            "The speaker is creating artificial urgency, threatening immediate service disconnection "
            "or bank account freezing unless you make an urgent payment."
        ),
        "recommendation": (
            "Never rush into paying. Official utility companies and banks provide formal written notices "
            "and never demand instant UPI transfers over unsolicited calls."
        ),
        "regional_advice": {
            "telugu": "ఆందోళన పడకండి. ఎలాంటి అత్యవసర చెల్లింపులు చేయకండి. అధికారిక వెబ్‌సైట్ ద్వారా మాత్రమే బిల్లులు సరిచూసుకోండి.",
            "hindi": "जल्दबाजी में कोई भुगतान न करें। आधिकारिक ऐप या वेबसाइट के माध्यम से ही स्थिति की जांच करें."
        }
    },
    {
        "category": "Digital Arrest / Law Enforcement Coercion",
        "severity": "high",
        "patterns": [
            # English
            r"\b(digital arrest|arrest warrant|illegal parcel|money laundering|narcotics|customs officer|cbi officer|police station|cyber crime police|supreme court)\b.{0,70}\b(transfer|pay|fine|penalty|stay on call|secret|jail|investigation)\b",
            r"\b(cbi|police|customs|enforcement directorate|ed officer|trai|rbi)\b.{0,60}\b(arrest|court order|frozen|illegal activity|drugs found)\b",
            # Telugu-English (Tanglish)
            r"\b(police case|arrest chestharu|digital arrest|cbi|customs department)\b.{0,60}\b(dabbu|fine|bail|transfer|jail|drugs|illegal)\b",
            r"\b(meeru|mee meeda)\b.{0,40}\b(case pettaru|arrest warrant vachindi|police station ki ravali)\b",
            r"\b(evariki cheppakandi|call cut cheyakandi|secret ga unchandi)\b.{0,40}\b(police|investigation|officer)\b",
            # Hindi-English (Hinglish)
            r"\b(digital arrest|police pakad legi|cbi aayegi|jail bhej denge|arrest warrant|illegal parcel)\b",
            r"\b(kisi ko mat batana|call cut mat karna|kamre me band raho)\b",
            # Telugu Script
            r"(డిజిటల్ అరెస్ట్|పోలీస్ కేసు|అరెస్ట్ చేస్తారు|జైలుకు పంపుతాము)",
        ],
        "explanation": (
            "The caller falsely claims to be from Police, CBI, Customs, or TRAI threatening a 'Digital Arrest' "
            "or linking your Aadhaar/phone to illegal narcotics or money laundering. 'Digital Arrest' does NOT exist in Indian law."
        ),
        "recommendation": (
            "Indian law enforcement agencies NEVER place citizens under 'digital arrest' or demand money via video call. "
            "Hang up and report immediately to the National Cyber Crime Helpline at 1930 or cybercrime.gov.in."
        ),
        "regional_advice": {
            "telugu": "భారతీయ చట్టంలో 'డిజిటల్ అరెస్ట్' అనేదే లేదు. పోలీసులు ఎప్పుడూ ఫోన్ కాల్ ద్వారా డబ్బులు అడగరు. వెంటనే 1930 కి కాల్ చేసి ఫిర్యాదు చేయండి.",
            "hindi": "भारतीय कानून में 'डिजिटल अरेस्ट' का कोई प्रावधान नहीं है। पुलिस कभी फोन या वीडियो कॉल पर पैसे नहीं मांगती। तुरंत 1930 पर शिकायत करें."
        }
    },
    {
        "category": "Remote access request",
        "severity": "high",
        "patterns": [
            # English
            r"\b(install|download|open)\b.{0,40}\b(anydesk|teamviewer|rustdesk|quicksupport|screen share|remote desktop|support apk)\b",
            r"\b(grant|give|allow)\b.{0,40}\b(remote access|screen access|screen sharing|control of your phone|accessibility permission)\b",
            r"\b(microsoft support|bank support|tech support)\b.{0,50}\b(install|anydesk|download|remote)\b",
            # Telugu-English (Tanglish)
            r"\b(anydesk|teamviewer|rustdesk|remote app)\b.{0,40}\b(download cheyandi|install cheyandi|open cheyandi)\b",
            r"\b(screen share cheyandi|mobile control ivvandi)\b",
            # Hindi-English (Hinglish)
            r"\b(anydesk|teamviewer|quicksupport)\b.{0,40}\b(download karo|install karo|chalu karo)\b",
            # Telugu Script
            r"(ఎనీడెస్క్|టీమ్‌వ్యూవర్|యాప్ డౌన్‌లోడ్ చేయండి)",
        ],
        "explanation": (
            "The caller is attempting to convince you to download remote-desktop software like AnyDesk or TeamViewer. "
            "This gives criminals complete access to see your screen, banking apps, and SMS OTPs."
        ),
        "recommendation": (
            "Never install remote screen-sharing tools at the instruction of an unknown caller. If installed, disconnect Wi-Fi/Mobile Data immediately."
        ),
        "regional_advice": {
            "telugu": "ఎనీడెస్క్ (AnyDesk) లేదా స్క్రీన్ షేరింగ్ యాప్‌లను ఎట్టి పరిస్థితుల్లోనూ డౌన్‌లోడ్ చేయవద్దు. మీ బ్యాంక్ వివరాలు దొంగిలించబడతాయి.",
            "hindi": "किसी भी अनजान व्यक्ति के कहने पर AnyDesk या TeamViewer जैसे ऐप डाउनलोड न करें."
        }
    },
    {
        "category": "Impersonation",
        "severity": "medium",
        "patterns": [
            # English
            r"\b(we are|this is|i am calling from|calling on behalf of)\b.{0,40}\b(bank|police|government|tax department|income tax|telecom regulatory|trai|customs|support team|head office)\b",
            r"\b(pretending to be|claim to be|posing as)\b.{0,40}\b(bank|police|government|support|official)\b",
            # Telugu-English (Tanglish)
            r"\b(nenu|memu)\b.{0,30}\b(bank nunchi|police station nunchi|head office nunchi|customs nunchi)\b.{0,30}\b(call chesthunnam|matladuthunnam)\b",
            # Hindi-English (Hinglish)
            r"\b(hum|main)\b.{0,30}\b(bank se|head office se|police station se)\b.{0,30}\b(bol raha hu|call kar raha hu)\b",
        ],
        "explanation": (
            "The caller claims to represent an authoritative body (bank, telecom provider, government department). "
            "Scammers routinely spoof caller ID numbers to appear genuine."
        ),
        "recommendation": (
            "Do not trust caller ID alone. Hang up and verify by calling the official phone number listed on the official entity website."
        ),
        "regional_advice": {
            "telugu": "కాలర్ పేరు లేదా గుర్తింపును గుడ్డిగా నమ్మకండి. అధికారిక వెబ్‌సైట్‌లోని నంబర్‌కు మాత్రమే కాల్ చేసి నిర్ధారించుకోండి.",
            "hindi": "केवल कॉलर आईडी पर भरोसा न करें। आधिकारिक फोन नंबर पर सीधे कॉल करके पुष्टि करें."
        }
    },
    {
        "category": "Prize or reward fee scam",
        "severity": "high",
        "patterns": [
            # English
            r"\b(won|winner|winning|claim|lottery|lucky draw|cash prize)\b.{0,100}\b(fee|payment|processing charge|registration charge|tax|advance)\b",
            r"\b(part[- ]time job|youtube like|telegram task|guaranteed return|daily earning)\b.{0,80}\b(deposit|recharge|investment|transfer)\b",
            # Telugu-English (Tanglish)
            r"\b(lottery|prize|car|gift)\b.{0,50}\b(vachindi|geluchukunnaru)\b.{0,60}\b(fee|tax|charge|kattali|dabbu)\b",
            r"\b(part time job|youtube likes|telegram task)\b.{0,50}\b(daily \d+ earn cheyachu|deposit kattali)\b",
            # Hindi-English (Hinglish)
            r"\b(lottery|inaam|prize)\b.{0,50}\b(nikla hai|jeeta hai)\b.{0,60}\b(tax|charge|processing fee|bhejo)\b",
            # Telugu Script
            r"(లాటరీ|బహుమతి గెలుచుకున్నారు).{0,40}(రుసుము|ఫీజు|డబ్బు చెల్లించండి)",
        ],
        "explanation": (
            "The message promises a lottery prize, gift, or part-time work reward, but requires you to pay an 'advance fee', 'tax', or 'deposit' first."
        ),
        "recommendation": (
            "Legitimate prizes never require an advance payment to claim them. Any request to pay to receive money is a scam."
        ),
        "regional_advice": {
            "telugu": "బహుమతి లేదా లాటరీ పేరుతో ఫీజు అడిగితే అది ఖచ్చితంగా మోసమే. ఎలాంటి రుసుములు చెల్లించవద్దు.",
            "hindi": "इनाम या लॉटरी के नाम पर पैसे मांगना ठगी है। कभी भी पहले पैसे न भेजें."
        }
    },
    {
        "category": "Suspicious parcel release fee",
        "severity": "high",
        "patterns": [
            # English
            r"\b(parcel|package|delivery|shipment|courier)\b.{0,80}\b(held|blocked|customs|delayed|confiscated)\b.{0,100}\b(pay|fee|charge|release)\b",
            r"\b(fedex|dhl|india post)\b.{0,50}\b(illegal items|drugs|customs penalty|release fee)\b",
            # Telugu-English (Tanglish)
            r"\b(parcel|courier|package)\b.{0,50}\b(customs lo agindi|held aindi)\b.{0,60}\b(release cheyyadaniki|charge kattali)\b",
            # Hindi-English (Hinglish)
            r"\b(parcel|courier)\b.{0,50}\b(custom me fasa hai|ruk gaya)\b.{0,60}\b(chhudane ke liye|charges do)\b",
        ],
        "explanation": (
            "The caller claims a courier package in your name is held in customs or contains prohibited substances, demanding a clearance fee."
        ),
        "recommendation": (
            "Track parcels exclusively on the courier's official portal. Never transfer funds directly to individual UPI handles or accounts."
        ),
        "regional_advice": {
            "telugu": "కొరియర్ పేరుతో వచ్చే కాల్స్ పట్ల జాగ్రత్తగా ఉండండి. అధికారిక వెబ్‌సైట్‌లో మాత్రమే ట్రాకింగ్ వివరాలు చూడండి.",
            "hindi": "कूरियर या पार्सल के नाम पर व्यक्तिगत खाते में कभी पैसे ट्रांसफर न करें."
        }
    },
]

SAFE_WARNING_PATTERNS = [
    r"\bnever share your (otp|pin|password)\b",
    r"\bdo not share your (otp|pin|password)\b",
    r"\bnever pay to claim\b",
    r"\bverify through the official (website|app)\b",
    r"\bdo not click suspicious links\b",
    r"\beware of scams\b",
    r"\bcyber crime awareness\b",
]


def is_safety_warning(text: str) -> bool:
    """Return True if the text is clearly an educational scam warning or advisory."""
    lower_text = text.lower()
    return any(re.search(pat, lower_text) for pat in SAFE_WARNING_PATTERNS)


def analyze_transcript(transcript: str) -> Dict[str, Any]:
    """
    Transparent, evidence-linked rule heuristic analysis.
    Highlights specific trigger sentences and matches against 7 tactic categories.
    """
    if not isinstance(transcript, str):
        transcript = str(transcript) if transcript is not None else ""

    clean_text = transcript.strip()

    if not clean_text:
        return {
            "risk_level": "Unknown",
            "risk_score": 0,
            "findings": [],
            "summary": "No transcript was provided for analysis.",
            "disclaimer": "This tool provides heuristic screening, not a definitive determination of fraud."
        }

    if is_safety_warning(clean_text):
        return {
            "risk_level": "No patterns detected",
            "risk_score": 0,
            "findings": [],
            "summary": "The transcript contains standard scam-prevention advice or educational security warnings.",
            "disclaimer": "This tool provides heuristic screening, not a definitive determination of fraud."
        }

    findings: List[Dict[str, Any]] = []
    seen_categories = set()

    for rule in RULES:
        matched_text = None
        start_idx = -1
        end_idx = -1

        for pattern in rule["patterns"]:
            match = re.search(pattern, clean_text, re.IGNORECASE | re.DOTALL)
            if match:
                matched_text = match.group(0)
                start_idx, end_idx = match.span()
                break

        if matched_text and rule["category"] not in seen_categories:
            seen_categories.add(rule["category"])
            findings.append({
                "category": rule["category"],
                "severity": rule["severity"],
                "evidence": matched_text.strip(),
                "start": start_idx,
                "end": end_idx,
                "explanation": rule["explanation"],
                "recommendation": rule["recommendation"],
                "regional_advice": rule.get("regional_advice", {})
            })

    # Severity scoring
    points_map = {"high": 35, "medium": 15, "low": 5}
    raw_score = sum(points_map.get(f["severity"].lower(), 10) for f in findings)
    risk_score = min(100, raw_score)

    has_high = any(f["severity"].lower() == "high" for f in findings)

    if has_high or risk_score >= 60:
        risk_level = "High"
    elif findings or risk_score >= 25:
        risk_level = "Caution"
    else:
        risk_level = "No patterns detected"

    if findings:
        cat_str = ", ".join(f["category"] for f in findings)
        summary = (
            f"Flagged {len(findings)} potentially suspicious tactic(s): {cat_str}. "
            f"Review the evidence below before making any financial decisions."
        )
    else:
        summary = (
            "No known scam phrasing patterns were detected in this transcript. "
            "Please note that absence of triggers does not guarantee that a caller is legitimate."
        )

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "findings": findings,
        "summary": summary,
        "disclaimer": (
            "VoiceShield AI uses heuristic threat intelligence and machine learning. "
            "Scores represent risk indicators, not a calibrated probability of fraud, "
            "and results are labeled as potentially suspicious."
        )
    }