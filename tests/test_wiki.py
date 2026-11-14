from __future__ import annotations

from data.wiki import html_to_text

HTML = """
<div class="mw-parser-output">
<table class="infobox"><tr><td><p>Infobox paragraph that must be skipped entirely.</p></td></tr></table>
<p>الرِّيَاض هي عاصمة <b>المملكة العربية السعودية</b><sup class="reference">[1]</sup> وأكبر مدنها، وتقع وسط شبه الجزيرة العربية.</p>
<style>.x{color:red}</style>
<p>short</p>
<h2>التاريخ<span class="mw-editsection">[عدل]</span></h2>
<p>Riyadh was a walled town until the 1940s, and it has grown rapidly ever since then.</p>
</div>
"""


def test_html_to_text_keeps_paragraphs_and_drops_boilerplate() -> None:
    text = html_to_text(HTML, max_chars=10_000)
    paragraphs = text.split("\n\n")
    assert len(paragraphs) == 2
    assert paragraphs[0].startswith("الرِّيَاض هي عاصمة المملكة العربية السعودية وأكبر")
    assert "[1]" not in text and "Infobox" not in text and "color" not in text
    assert "short" not in text and "عدل" not in text


def test_html_to_text_caps_length_at_paragraph_boundary() -> None:
    text = html_to_text(HTML, max_chars=120)
    assert text.startswith("الرِّيَاض")
    assert "Riyadh was" not in text
