"""Browser check for visual playback of the published saved trajectories."""

import re
from pathlib import Path

import pytest


def test_saved_trace_replay_steps_and_resets_without_network() -> None:
    playwright = pytest.importorskip("playwright.sync_api")
    page_path = Path(__file__).resolve().parents[1] / "release/qasper-agent-study/studio.html"
    with playwright.sync_playwright() as runtime:
        try:
            browser = runtime.chromium.launch(headless=True)
        except playwright.Error as exc:
            pytest.skip(f"Chromium unavailable: {exc}")
        try:
            page = browser.new_page()
            requests = []
            errors = []
            page.on("request", lambda request: requests.append(request.url))
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(page_path.as_uri())
            rows = page.locator(".run-table tbody tr")
            assert rows.nth(0).locator("td").nth(4).inner_text() == "Not recorded"
            nemotron = rows.filter(has_text="Nemotron Ultra Nebius").first.locator("td")
            assert nemotron.nth(4).inner_text() == "$2.0354"
            assert [nemotron.nth(index).inner_text() for index in (5, 6, 7)] == [
                "Not reviewed", "Not reviewed", "Not reviewed",
            ]
            bm25 = rows.filter(has_text="BM25 + Qwen v5").first.locator("td")
            assert bm25.nth(5).inner_text() == "Not reviewed"
            assert bm25.nth(7).inner_text() == "20 / 20 · labels"
            assert bm25.nth(8).inner_text() == "No code"
            smoke = rows.nth(5).locator("td")
            assert smoke.nth(3).inner_text() == "Unscored"
            assert smoke.nth(4).inner_text() == "Not recorded"
            assert smoke.nth(5).inner_text() == "Unscored"
            assert smoke.nth(8).inner_text() == "Not tallied"
            page.locator(".run-select", has_text=re.compile(r"^Qwen Base$")).first.click()
            page.locator(".question-select").first.click()
            turns = page.locator("#lanes .turn-entry").count()
            assert turns >= 2

            page.locator("#replay-start").click()
            assert page.locator("#lanes .turn-entry:visible").count() == 0
            page.locator("#replay-next").click()
            assert page.locator("#lanes .turn-entry:visible").count() == 1
            page.locator("#replay-prev").click()
            assert page.locator("#lanes .turn-entry:visible").count() == 0

            page.locator("#replay-play").click()
            page.wait_for_function(
                "document.querySelector('#replay-progress').textContent.startsWith('Step 1 /')"
            )
            assert page.locator("#lanes .turn-entry:visible").count() == 1
            page.locator(".question-select").nth(1).click()
            assert "all visible" in page.locator("#replay-progress").inner_text()
            visible = page.locator("#lanes .turn-entry:visible").count()
            assert visible == page.locator("#lanes .turn-entry").count()
            assert not errors
            assert all(url.startswith("file:") for url in requests)
        finally:
            browser.close()
