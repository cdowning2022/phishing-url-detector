"""
Tests for the URL feature extractor.

These tests pin down the contract: given a URL, what features come out, and
what values should they have? If someone (future you, a collaborator, a
recruiter cloning the repo) breaks the extractor, these tests catch it.
"""

from __future__ import annotations

import pytest

from src.features import extract_features, features_for_model, normalize_url


# ---------- normalize_url ----------
class TestNormalizeUrl:

    def test_https_url_unchanged(self):
        assert normalize_url("https://example.com") == "https://example.com"

    def test_http_url_unchanged(self):
        assert normalize_url("http://example.com") == "http://example.com"

    def test_missing_scheme_gets_http_prefix(self):
        assert normalize_url("example.com") == "http://example.com"

    def test_strips_whitespace(self):
        assert normalize_url("  https://example.com  ") == "https://example.com"

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            normalize_url("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            normalize_url("   ")

    def test_case_insensitive_scheme_detection(self):
        # HTTPS:// should be recognized — we shouldn't double-prefix
        result = normalize_url("HTTPS://Example.com")
        assert result.lower().startswith("https://")


# ---------- extract_features: structural ----------
class TestExtractFeaturesStructure:

    def test_returns_dict(self):
        features = extract_features("https://example.com")
        assert isinstance(features, dict)

    def test_all_values_are_numeric(self):
        features = extract_features("https://example.com/path?a=1")
        for name, value in features.items():
            assert isinstance(value, (int, float)), f"{name} is {type(value).__name__}"

    def test_returns_consistent_feature_set(self):
        # Different URLs should produce the same set of feature names —
        # otherwise the model can't reliably consume them.
        a = set(extract_features("https://example.com").keys())
        b = set(extract_features("http://192.168.1.1/login?x=1").keys())
        c = set(extract_features("https://very.long.subdomain.example.co.uk/").keys())
        assert a == b == c


# ---------- extract_features: values ----------
class TestExtractFeaturesValues:

    def test_https_flag_set_for_https_url(self):
        assert extract_features("https://example.com")["IsHTTPS"] == 1

    def test_https_flag_unset_for_http_url(self):
        assert extract_features("http://example.com")["IsHTTPS"] == 0

    def test_ip_address_url_flagged(self):
        f = extract_features("http://192.168.1.1/login")
        assert f["IsDomainIP"] == 1

    def test_normal_domain_not_flagged_as_ip(self):
        f = extract_features("https://example.com")
        assert f["IsDomainIP"] == 0

    def test_url_length_matches_string_length(self):
        url = "https://example.com/some/path"
        # Note: extract_features normalizes the URL first, so length may
        # differ if the input was missing a scheme. With a scheme present
        # the length should match.
        assert extract_features(url)["URLLength"] == len(url)

    def test_subdomain_count(self):
        # 'a.b.example.com' → subdomain is 'a.b' → 2 parts
        f = extract_features("https://a.b.example.com")
        assert f["NoOfSubDomain"] == 2

    def test_no_subdomain(self):
        f = extract_features("https://example.com")
        assert f["NoOfSubDomain"] == 0

    def test_obfuscation_detected(self):
        # %20 is URL-encoded space — common in malicious obfuscation
        f = extract_features("https://example.com/path%20with%20spaces")
        assert f["HasObfuscation"] == 1
        assert f["NoOfObfuscatedChar"] >= 2

    def test_no_obfuscation_clean_url(self):
        f = extract_features("https://example.com/normal/path")
        assert f["HasObfuscation"] == 0

    def test_query_string_detected(self):
        assert extract_features("https://example.com/?id=1")["HasQuery"] == 1
        assert extract_features("https://example.com/")["HasQuery"] == 0

    def test_digit_ratio_in_reasonable_range(self):
        f = extract_features("https://192-168-1-1.example.com")
        # Some digits, some letters — should be strictly between 0 and 1
        assert 0 < f["DegitRatioInURL"] < 1


# ---------- features_for_model ----------
class TestFeaturesForModel:

    def test_returns_values_in_requested_order(self):
        expected = ["URLLength", "IsHTTPS", "NoOfDots"]
        values, missing = features_for_model("https://example.com", expected)
        assert len(values) == 3
        # URL "https://example.com" — length 19, https=1, one dot
        assert values[0] == 19
        assert values[1] == 1.0
        assert values[2] == 1
        assert missing == []

    def test_unknown_features_filled_with_zero(self):
        # Asking for a feature the extractor doesn't produce should give
        # back 0 plus a note in the missing list — not a crash.
        expected = ["URLLength", "DomainAgeDays", "WhoisRegistrant"]
        values, missing = features_for_model("https://example.com", expected)
        assert values[1] == 0.0
        assert values[2] == 0.0
        assert "DomainAgeDays" in missing
        assert "WhoisRegistrant" in missing

    def test_all_returned_values_are_floats(self):
        expected = ["URLLength", "IsHTTPS"]
        values, _ = features_for_model("https://example.com", expected)
        assert all(isinstance(v, float) for v in values)


# ---------- Edge cases ----------
class TestEdgeCases:

    def test_handles_url_with_port(self):
        f = extract_features("http://example.com:8080/path")
        # Should not crash; port shouldn't be counted as part of the domain length
        assert f["DomainLength"] == len("example.com")

    def test_handles_very_long_url(self):
        long_url = "https://example.com/" + "a" * 500
        f = extract_features(long_url)
        assert f["URLLength"] > 500

    def test_handles_unicode_domain(self):
        # IDN domains shouldn't crash the extractor
        f = extract_features("https://例え.com")
        assert f["URLLength"] > 0

    def test_handles_url_with_fragment(self):
        f = extract_features("https://example.com/page#section")
        assert f["HasFragment"] == 1
