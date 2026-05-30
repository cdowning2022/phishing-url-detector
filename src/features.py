"""
Extract numeric features from a raw URL string.

The training dataset (PhiUSIIL) provides ~50 pre-extracted features per URL.
For the CLI to classify a fresh URL the user types in, we need to extract
the *same* features from a raw string.

We can't reproduce every feature perfectly (some require WHOIS lookups,
TLS cert inspection, or page-content scraping — out of scope for this
project). What we extract here is the subset that depends only on the
URL string itself. Any feature the model expects but we can't compute
is filled with 0, with a warning.

This is documented as a known limitation in the README.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse


# Top-level domains that commonly appear in legitimate URLs.
# Used as a heuristic for the "TLDLegitimateProb" feature if present.
COMMON_TLDS = {"com", "org", "net", "edu", "gov", "io", "co", "us", "uk", "de", "fr", "jp", "ca"}


def normalize_url(url: str) -> str:
    """Ensure URL has a scheme and strip www. so bare and www-prefixed domains are equivalent."""
    url = url.strip()
    if not url:
        raise ValueError("Empty URL")
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url
    # Strip www. so www.example.com and example.com produce identical features.
    # The training dataset uses www-prefixed legit URLs exclusively; without this
    # normalization, bare domains would never match legit training patterns.
    url = re.sub(r"^(https?://)www\.", r"\1", url, flags=re.IGNORECASE)
    return url


def extract_features(url: str) -> dict[str, float]:
    """
    Extract URL-string-only features.

    Returns a dictionary mapping feature names (matching dataset column names
    where possible) to numeric values. Caller is responsible for selecting
    which of these the model actually needs and filling 0 for any missing ones.
    """
    url = normalize_url(url)
    parsed = urlparse(url)

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path
    query = parsed.query
    fragment = parsed.fragment

    # Separate domain from any port
    domain = netloc.split(":")[0] if ":" in netloc else netloc

    # Subdomain extraction: everything before the last two parts (e.g. "a.b.example.com" → "a.b").
    # "www" is stripped because it carries no signal — www.example.com and example.com are the
    # same site. Without this, the model would learn that 0 subdomains = phishing solely because
    # the training dataset consistently uses www-prefixed URLs for legitimate sites.
    parts = domain.split(".")
    tld = parts[-1] if parts else ""
    subdomain_parts = [p for p in parts[:-2] if p.lower() != "www"]
    subdomain = ".".join(subdomain_parts)

    # Helpful sub-counts
    n_dots      = url.count(".")
    n_hyphens   = url.count("-")
    n_underscore = url.count("_")
    n_slashes   = url.count("/")
    n_question  = url.count("?")
    n_equals    = url.count("=")
    n_at        = url.count("@")
    n_amp       = url.count("&")
    n_digits    = sum(c.isdigit() for c in url)
    n_letters   = sum(c.isalpha() for c in url)
    n_special   = sum(not c.isalnum() and c not in "./:?-=&_" for c in url)

    has_ip = bool(re.match(r"^(\d{1,3}\.){3}\d{1,3}$", domain))

    # Build the feature dict. Names mirror common PhiUSIIL column conventions.
    # The model will pick out only the names that match its training feature list.
    return {
        # Lengths
        "URLLength":            len(url),
        "DomainLength":         len(domain),
        "TLDLength":            len(tld),

        # Counts of structural elements
        "NoOfSubDomain":        len(subdomain.split(".")) if subdomain else 0,
        "NoOfDots":             n_dots,
        "NoOfHyphens":          n_hyphens,
        "NoOfHyphensInURL":     n_hyphens,
        "NoOfUnderscoreInURL":  n_underscore,
        "NoOfSlashInURL":       n_slashes,
        "NoOfQMarkInURL":       n_question,
        "NoOfEqualsInURL":      n_equals,
        "NoOfAtInURL":          n_at,
        "NoOfAmpersandInURL":   n_amp,
        "NoOfDegitsInURL":      n_digits,   # spelling matches dataset typo
        "NoOfLettersInURL":     n_letters,
        "NoOfOtherSpecialCharsInURL": n_special,

        # Character ratios
        "LetterRatioInURL":     n_letters / len(url) if len(url) else 0,
        "DegitRatioInURL":      n_digits  / len(url) if len(url) else 0,  # matches dataset spelling
        "SpecialCharRatioInURL": n_special / len(url) if len(url) else 0,

        # Binary flags
        "IsHTTPS":              1 if scheme == "https" else 0,
        "IsDomainIP":           1 if has_ip else 0,
        "HasObfuscation":       1 if "%" in url else 0,
        "NoOfObfuscatedChar":   url.count("%"),
        "ObfuscationRatio":     url.count("%") / len(url) if len(url) else 0,

        # Query / fragment presence
        "HasQuery":             1 if query else 0,
        "HasFragment":          1 if fragment else 0,

        # TLD heuristic
        "TLDLegitimateProb":    1.0 if tld in COMMON_TLDS else 0.0,
    }


def features_for_model(url: str, expected_feature_names: list[str]) -> tuple[list[float], list[str]]:
    """
    Build a feature vector in the exact order the model expects.

    Returns (values, missing_feature_names). Missing features are filled with 0
    so the model can still produce a prediction — but we report them so the
    CLI can warn the user.
    """
    extracted = extract_features(url)
    values, missing = [], []
    for name in expected_feature_names:
        if name in extracted:
            values.append(float(extracted[name]))
        else:
            values.append(0.0)
            missing.append(name)
    return values, missing
