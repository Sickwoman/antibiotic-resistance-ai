"""Tests for the Version 1.0 result page.

**What these tests can and cannot do.** Continuous integration installs Python only, so the suite does not
execute the page's JavaScript (protocol amendment 7, point 4). Writing tests that *looked* behavioural while
only inspecting source would repeat the failure mode this project has already been bitten by once — a test
that passes while asserting the unsafe behaviour. So the page is written instead so that the unsafe code
cannot exist, and these tests assert that structure on the served HTML:

- the script holds no confidence-label text and no comparison against the returned probability or
  threshold, so there is no branch that could infer a confidence the model did not return;
- the result region is empty in the markup, so nothing can be shown that did not come from a response;
- no storage API and no external origin appear anywhere.

Those are structural facts about the file, and a structural assertion is honest about what it proves.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from src.api.page import PAGE_PATH, render_page, unsubstituted_tokens
from src.api.schemas import FORBIDDEN_IN_RESPONSES
from src.predict import DISCLAIMER
from src.uncertainty import ADVICE, RESISTANT, SUSCEPTIBLE, UNCERTAIN
from tests.test_api import PATH_LIKE, build_app

CONFIDENCE_LABELS = (SUSCEPTIBLE, UNCERTAIN, RESISTANT)

# Storage that would outlive the request. The page must touch none of it: no prediction history, nothing
# left in the browser after a reader closes the tab.
STORAGE_APIS = ("localStorage", "sessionStorage", "indexedDB", "document.cookie", "openDatabase")

# Language that would overstate what the model does. "diagnostic" is deliberately absent from this list:
# the canonical disclaimer uses it in a negating sentence ("not a clinically validated diagnostic").
CLINICAL_OVERCLAIMS = ("diagnosis", "recommended treatment", "confirmed")


@pytest.fixture(scope="module")
def page() -> str:
    """The page as served, with the canonical constants substituted in."""
    return render_page()


def script_of(html: str) -> str:
    match = re.search(r"<script>(.*?)</script>", html, re.S)
    assert match, "the page has no script block"
    return match.group(1)


def asset_targets(html: str) -> list[str]:
    """Every URL the page would fetch: src/href attributes plus every fetch() target."""
    attributes = [m.group(2) for m in re.finditer(r"""\b(src|href)\s*=\s*["']([^"']+)["']""", html)]
    fetches = re.findall(r"""fetch\(\s*["']([^"']+)["']""", html)
    dynamic = re.findall(r"""\?\s*["'](/[^"']+)["']\s*:\s*["'](/[^"']+)["']""", html)
    return attributes + fetches + [u for pair in dynamic for u in pair]


# ------------------------------------------------------------------------------------------------
# Routing
# ------------------------------------------------------------------------------------------------

def test_the_page_is_served_at_the_root(tmp_path, zones_file=None):
    with TestClient(build_app(tmp_path, zones=tmp_path / "absent.json")) as client:
        r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "<!DOCTYPE html>" in r.text


def test_the_page_is_served_even_when_the_service_is_not_ready(tmp_path):
    """A reader arriving at an unready service should be told so by the page, not met with a bare 503."""
    app = build_app(tmp_path, model_path=tmp_path / "absent.joblib")
    with TestClient(app) as client:
        page_response = client.get("/")
        ready = client.get("/ready")
    assert page_response.status_code == 200
    assert ready.status_code == 503, "readiness must still report the truth"
    assert not PATH_LIKE.search(page_response.text)


def test_the_page_has_the_structure_it_needs(page: str):
    for required in ('<header', '<main', '<h1', '<form', 'id="spectrum"', 'id="result"', 'id="status"',
                     'id="about"', 'type="submit"'):
        assert required in page, f"the page is missing {required}"


# ------------------------------------------------------------------------------------------------
# Same origin: a spectrum must not be able to reach a third party
# ------------------------------------------------------------------------------------------------

def test_every_asset_and_request_target_is_same_origin(page: str):
    """Asserted on attribute and fetch values, not on the characters `//`, which every JS comment contains."""
    targets = asset_targets(page)
    assert targets, "no targets were extracted; the check is not covering anything"
    for target in targets:
        assert "://" not in target, f"{target!r} names an external scheme"
        assert not target.startswith("//"), f"{target!r} is protocol-relative, so it would leave the origin"
        assert target.startswith("/") or target.startswith("#"), f"{target!r} is not an absolute local path"


def test_the_page_names_no_external_host(page: str):
    for forbidden in ("http://", "https://", "cdn.", "googleapis", "unpkg", "jsdelivr", "cdnjs",
                      "fonts.", "analytics", "gtag", "googletagmanager"):
        assert forbidden not in page, f"the page references {forbidden!r}"


def test_the_page_requests_only_this_services_own_endpoints(page: str):
    fetches = re.findall(r"""fetch\(\s*["']([^"']+)["']""", page)
    dynamic = re.findall(r"""\?\s*["'](/[^"']+)["']\s*:\s*["'](/[^"']+)["']""", page)
    all_targets = set(fetches) | {u for pair in dynamic for u in pair}
    assert all_targets, "no fetch target was found"
    for target in all_targets:
        assert target.split("?")[0] in ("/predict", "/model-info", "/ready"), f"unexpected target {target!r}"


# ------------------------------------------------------------------------------------------------
# The page makes no scientific decision
# ------------------------------------------------------------------------------------------------

def test_the_script_contains_no_confidence_label(page: str):
    """The decisive one. A script with no label text cannot synthesise a confidence the model did not send.

    The shipped zones have `upper: null`, so a spectrum at p = 0.945 is correctly `Uncertain`. A page that
    inferred a label from the probability would silently overrule that.
    """
    script = script_of(page)
    for label in CONFIDENCE_LABELS:
        assert label not in script, f"the script mentions {label!r}, so it could be synthesising a label"


def test_the_script_never_compares_the_probability_or_the_threshold(page: str):
    """Inferring a confidence requires a comparison. There is none against either field."""
    for line in script_of(page).splitlines():
        if "resistance_probability" in line or "threshold" in line:
            for operator in ("<", ">", "<=", ">=", "Math."):
                assert operator not in line, f"comparison or arithmetic on a model value: {line.strip()!r}"


def test_the_result_region_is_empty_in_the_served_markup(page: str):
    """It is filled only from a response, so the page cannot show a result it was not given."""
    match = re.search(r'<section id="result"[^>]*>(.*?)</section>', page, re.S)
    assert match, "the result region was not found"
    assert match.group(1).strip() == "", f"the result region ships with content: {match.group(1)[:80]!r}"


def test_no_prediction_value_is_hard_coded_in_the_markup(page: str):
    """Scoped to the markup outside the script, per amendment 7 point 3.

    Explanatory copy may name a label; the result region may not, and no probability or cut-off literal
    belongs anywhere in the page.
    """
    markup = page.split("<script>")[0]
    assert "0.9453" not in markup and "0.14261540693905073" not in markup and "0.1426" not in markup
    # The canonical ADVICE constant begins with the uncertain label, and amendment 7 point 3 permits a label
    # in explanatory copy - what it forbids is one in the result region, which its own test covers. So the
    # substituted constants are removed before looking for a stray hard-coded label.
    outside_constants = markup.replace(ADVICE, "").replace(DISCLAIMER, "")
    for label in CONFIDENCE_LABELS:
        assert label not in outside_constants, f"{label!r} is hard-coded in the markup, not read from a response"


def test_the_zone_explanation_comes_from_the_api(page: str):
    """The page states no zone claim of its own: it prints what /model-info says it can return."""
    script = script_of(page)
    assert "possible_labels" in script
    assert "confidence_zones.note" in script or "confidence_zones" in script
    assert 'fetch("/model-info")' in script


# ------------------------------------------------------------------------------------------------
# Privacy
# ------------------------------------------------------------------------------------------------

def test_the_page_persists_nothing(page: str):
    for api in STORAGE_APIS:
        assert api not in page, f"the page references {api}, which would outlive the request"
    assert "history" not in page.lower().replace("historical", ""), "no prediction history may be kept"


def test_the_page_leaks_nothing(page: str):
    for forbidden in FORBIDDEN_IN_RESPONSES:
        assert forbidden not in page, f"the page contains {forbidden!r}"
    found = PATH_LIKE.search(page)
    assert not found, f"the page contains a filesystem path: {found.group(0)!r}"


def test_the_page_never_logs_the_spectrum(page: str):
    script = script_of(page)
    assert "console.log" not in script and "console.debug" not in script
    assert "FileReader" not in script, "the page must not read the spectrum's contents client-side"


# ------------------------------------------------------------------------------------------------
# Medical framing
# ------------------------------------------------------------------------------------------------

def test_the_canonical_disclaimer_is_present(page: str):
    """Asserted by importing the constant, so a paraphrase or a drifted copy fails."""
    assert DISCLAIMER in page
    assert ADVICE in page


def test_no_placeholder_token_survives_rendering(page: str):
    """A surviving token would show a reader a placeholder where the disclaimer belongs."""
    assert unsubstituted_tokens(page) == []
    assert "{{" not in page


def test_the_page_makes_no_clinical_claim(page: str):
    lowered = page.lower()
    for phrase in CLINICAL_OVERCLAIMS:
        assert phrase not in lowered, f"the page uses {phrase!r}"


def test_the_page_names_no_antibiotic_of_its_own(page: str):
    """The species and antibiotic are read from the API; the page must not hard-code a drug name."""
    markup = page.split("<script>")[0].lower()
    for drug in ("ciprofloxacin", "amoxicillin", "ceftriaxone", "meropenem", "colistin", "gentamicin"):
        assert drug not in markup, f"the page hard-codes {drug!r}"


def test_the_probability_is_labelled_as_the_models_own(page: str):
    """Not patient risk, not treatment failure: the model's resistance probability."""
    assert "Model's resistance probability" in page
    for overclaim in ("risk of treatment failure", "patient risk", "chance the antibiotic will fail"):
        assert overclaim not in page.lower()


# ------------------------------------------------------------------------------------------------
# Accessibility (protocol amendment 7, point 1)
# ------------------------------------------------------------------------------------------------

def test_the_file_input_is_labelled(page: str):
    assert re.search(r'<label\s+for="spectrum"', page), "the file input has no programmatic label"
    assert re.search(r'id="spectrum"', page)
    assert 'aria-describedby="spectrum-help"' in page


def test_the_status_region_is_announced_and_focusable(page: str):
    assert 'role="status"' in page
    assert 'aria-live="polite"' in page
    assert 'tabindex="-1"' in page, "the status region cannot receive focus after a render"
    assert "statusBox.focus()" in script_of(page), "focus is never moved to the status region"


def test_the_submit_is_a_real_button(page: str):
    """A <button type=submit> is keyboard-operable by default; a clickable div would not be."""
    assert re.search(r'<button\s+type="submit"', page)
    assert "onclick=" not in page, "behaviour is attached in script, not by inline handlers"


def test_the_page_uses_semantic_landmarks(page: str):
    for element in ("<header", "<main", "<footer", "<h1", "<h2"):
        assert element in page, f"missing semantic element {element}"
    assert 'lang="en"' in page


def test_confidence_is_conveyed_as_text_not_colour(page: str):
    """Colour must never be the only signal, so no style is keyed to a result state at all."""
    assert 'definition(list, "Confidence label", data.confidence)' in script_of(page)
    styles = re.search(r"<style>(.*?)</style>", page, re.S)
    assert styles, "no stylesheet found"
    for state in ("resistant", "susceptible", "uncertain", "danger", "warning", "success"):
        assert state not in styles.group(1).lower(), f"a style is keyed to the result state {state!r}"


# ------------------------------------------------------------------------------------------------
# Failure handling
# ------------------------------------------------------------------------------------------------

def test_a_failed_request_clears_the_previous_result_first(page: str):
    """Otherwise an earlier answer would sit on screen looking like the new upload's."""
    script = script_of(page)
    handler = script[script.index("form.addEventListener"):]
    clear_at = handler.index("clear(resultBox)")
    fetch_at = handler.index("fetch(")
    assert clear_at < fetch_at, "the result region is not cleared before the request is sent"


def test_the_error_message_comes_from_the_api_error_contract(page: str):
    """The 0.9.1 error body carries `message`; the page shows that rather than inventing text."""
    script = script_of(page)
    assert "body.error.message" in script
    assert "response.ok" in script, "the page must distinguish a failure from a result"


def test_the_page_is_deterministic(page: str):
    """render_page is cached and substitutes only constants, so two calls agree byte for byte."""
    assert render_page() == page
    assert PAGE_PATH.is_file()
