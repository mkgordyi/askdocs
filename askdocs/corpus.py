"""Turn the official XHTML of Regulation (EU) 2024/1689 into small, citable chunks.

One chunk is one numbered paragraph of an article. A long paragraph is split by its
points, so "Article 5(1)(a)" can be cited on its own. Annexes are split by point.
"""

import re
import warnings
from dataclasses import asdict, dataclass

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

MAX_CHARS = 1400
BASE_URL = "https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng"
MARKER = re.compile(r"^\(?([0-9]{1,3}|[a-z]{1,4})[).]$")
PARAGRAPH_ID = re.compile(r"^\d{3}\.\d{3}$")


@dataclass(frozen=True)
class Chunk:
    id: str        # stable id, e.g. "art_5.1.a"
    article: str   # "art_5" or "anx_III": the unit the evaluation checks
    ref: str       # human citation, e.g. "Article 5(1)(a)"
    title: str
    text: str
    url: str

    def search_text(self) -> str:
        return f"{self.ref}. {self.title}. {self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.;:)])", r"\1", text)


def _flatten(el: Tag) -> str:
    """All text of an element, with point markers kept inline: "(a) text (b) text"."""
    return _clean(el.get_text(" ", strip=True))


def _units(el: Tag) -> list[tuple[str | None, str]]:
    """Direct content of an element as (marker, text) pairs.

    A plain paragraph gives (None, text). A two-column table whose first cell is a
    marker such as "(a)" or "1." gives (marker, text of the second cell).
    """
    units: list[tuple[str | None, str]] = []
    for child in el.find_all(recursive=False):
        classes = child.get("class") or []
        if child.name == "p":
            if {"oj-ti-art", "oj-doc-ti"} & set(classes):
                continue
            text = _flatten(child)
            if text:
                units.append((None, text))
        elif child.name == "table":
            for row in child.find_all("tr"):
                if row.find_parent("table") is not child:
                    continue  # rows of nested tables are read with their parent cell
                cells = row.find_all("td", recursive=False)
                first = _flatten(cells[0]) if cells else ""
                if len(cells) == 2 and MARKER.match(first):
                    units.append((first, _flatten(cells[1])))
                else:
                    text = _flatten(row)
                    if text:
                        units.append((None, text))
        elif child.name == "div" and "eli-title" not in classes:
            units.extend(_units(child))
    return units


def _split_long(text: str, limit: int = MAX_CHARS) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, current = [], ""
    for sentence in re.split(r"(?<=[.;])\s+", text):
        if current and len(current) + len(sentence) + 1 > limit:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts


def _bare(marker: str) -> str:
    return marker.strip("().")


def _paragraph_chunks(units, base_id, base_ref, article, title, url) -> list[Chunk]:
    """Chunks for one paragraph (or one unnumbered article)."""
    full = " ".join(f"{m} {t}" if m else t for m, t in units)
    has_points = any(m for m, _ in units)
    if len(full) <= MAX_CHARS or not has_points:
        return [
            Chunk(f"{base_id}.{i}" if i else base_id, article, base_ref, title, part, url)
            for i, part in enumerate(_split_long(full))
        ]

    chunks: list[Chunk] = []
    first_point = next(i for i, (m, _) in enumerate(units) if m)
    intro = " ".join(t for _, t in units[:first_point])
    seen: dict[str, int] = {}
    tail: list[str] = []
    for marker, text in units[first_point:]:
        if not marker:
            tail.append(text)
            continue
        key = _bare(marker)
        seen[key] = seen.get(key, 0) + 1
        suffix = key if seen[key] == 1 else f"{key}-{seen[key]}"
        body = f"{intro} {marker} {text}".strip()
        for i, part in enumerate(_split_long(body)):
            chunk_id = f"{base_id}.{suffix}" + (f".{i}" if i else "")
            chunks.append(Chunk(chunk_id, article, f"{base_ref}({key})", title, part, url))
    if tail:
        for i, part in enumerate(_split_long(" ".join(tail))):
            chunks.append(Chunk(f"{base_id}.tail{i or ''}", article, base_ref, title, part, url))
    return chunks


def _article_chunks(div: Tag) -> list[Chunk]:
    art_id = div["id"]
    number = art_id.split("_")[1]
    title_el = div.find(class_="eli-title")
    title = _flatten(title_el) if title_el else ""
    url = f"{BASE_URL}#{art_id}"
    paragraphs = [c for c in div.find_all("div", recursive=False) if PARAGRAPH_ID.match(c.get("id", ""))]
    if not paragraphs:
        return _paragraph_chunks(_units(div), art_id, f"Article {number}", art_id, title, url)

    chunks: list[Chunk] = []
    for para in paragraphs:
        n = int(para["id"].split(".")[1])
        units = _units(para)
        if units and units[0][0] is None:
            units[0] = (None, re.sub(rf"^{n}\.\s*", "", units[0][1]))
        chunks.extend(
            _paragraph_chunks(units, f"{art_id}.{n}", f"Article {number}({n})", art_id, title, url)
        )
    return chunks


def _annex_chunks(div: Tag) -> list[Chunk]:
    anx_id = div["id"]
    number = anx_id.split("_")[1]
    headings = [_flatten(p) for p in div.find_all("p", class_="oj-doc-ti", recursive=False)]
    title = headings[1] if len(headings) > 1 else f"Annex {number}"
    url = f"{BASE_URL}#{anx_id}"
    chunks: list[Chunk] = []
    section, loose, counter = "", [], 0

    def flush():
        nonlocal loose, counter
        if loose:
            ref = f"Annex {number}" + (f", {section}" if section else "")
            for part in _split_long(" ".join(loose)):
                counter += 1
                chunks.append(Chunk(f"{anx_id}.c{counter}", anx_id, ref, title, part, url))
            loose = []

    for marker, text in _units(div):
        if marker is None:
            match = re.match(r"^(Section [A-Z0-9]+)\b", text)
            if match:
                flush()
                section = match.group(1)
            loose.append(text)
            continue
        flush()
        ref = f"Annex {number}" + (f", {section}" if section else "") + f", point {_bare(marker)}"
        for part in _split_long(f"{marker} {text}"):
            counter += 1
            chunks.append(Chunk(f"{anx_id}.c{counter}", anx_id, ref, title, part, url))
    flush()
    return chunks


def parse(xhtml: str) -> list[Chunk]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)  # the source is XHTML, the HTML parser reads it fine
        soup = BeautifulSoup(xhtml, "lxml")
    chunks: list[Chunk] = []
    for div in soup.find_all("div", id=re.compile(r"^art_\d+$")):
        chunks.extend(_article_chunks(div))
    for div in soup.find_all("div", id=re.compile(r"^anx_[IVX]+$")):
        chunks.extend(_annex_chunks(div))
    return [c for c in chunks if len(c.text) >= 20]
