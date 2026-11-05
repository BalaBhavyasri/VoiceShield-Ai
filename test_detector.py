import unittest

from detector import analyze_transcript
from ml_detector import compute_hybrid_assessment, init_ml_classifier, predict_ml_scam


class TestVoiceShieldDetector(unittest.TestCase):

    def test_otp_request_is_flagged(self):
        result = analyze_transcript("Tell me the OTP now to verify your account.")
        self.assertTrue(result["findings"])
        self.assertGreater(result["risk_score"], 0)

    def test_remote_access_request_is_flagged(self):
        result = analyze_transcript("Install AnyDesk and give me remote access to your phone.")
        categories = [f["category"] for f in result["findings"]]
        self.assertIn("Remote access request", categories)

    def test_telugu_english_otp_flagged(self):
        result = analyze_transcript("Mee bank account block aipothundi. Ventane OTP cheppandi.")
        categories = [f["category"] for f in result["findings"]]
        self.assertIn("OTP or credential theft", categories)
        self.assertEqual(result["risk_level"], "High")

    def test_digital_arrest_flagged(self):
        result = analyze_transcript("This is CBI officer. You are under digital arrest for money laundering stay on call.")
        categories = [f["category"] for f in result["findings"]]
        self.assertIn("Digital Arrest / Law Enforcement Coercion", categories)
        self.assertEqual(result["risk_level"], "High")

    def test_safe_otp_warning_is_not_flagged(self):
        result = analyze_transcript("Never share your OTP or password with anyone calling you.")
        categories = [f["category"] for f in result["findings"]]
        self.assertNotIn("OTP or credential theft", categories)
        self.assertEqual(result["risk_level"], "No patterns detected")

    def test_ordinary_conversation_has_no_findings(self):
        result = analyze_transcript("Hello, I am calling to confirm our meeting tomorrow at 10 AM.")
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["risk_level"], "No patterns detected")

    def test_empty_transcript(self):
        result = analyze_transcript("")
        self.assertEqual(result["risk_level"], "Unknown")

    def test_disclaimer_explains_score(self):
        result = analyze_transcript("Tell me the OTP now.")
        self.assertIn("probability", result["disclaimer"].lower())

    def test_hybrid_assessment_combines_scores(self):
        rule_res = analyze_transcript("Your account will be blocked today. Tell me the OTP immediately.")
        ml_res = {"ml_score": 85, "ml_probability": 0.85, "ml_label": "Scam"}
        hybrid = compute_hybrid_assessment(rule_res, ml_res)
        self.assertIn("hybrid_score", hybrid)
        self.assertIn("hybrid_level", hybrid)
        self.assertGreaterEqual(hybrid["hybrid_score"], 60)


if __name__ == "__main__":
    unittest.main(verbosity=2)
