"""Build the SYNERGY systematic-review data from its labels, PubMed and OpenAlex.

SYNERGY (De Bruin et al. 2023, CC0) is 26 systematic reviews. For each one it
publishes the candidate pool returned by the review's search, every record
screened, with `label_included` = 1 for the studies the review included. Each
pool's answer set is complete, so each review is a find-all query over its own
pool. The review's eligibility criteria, also published, are the query.

The repository holds only identifiers and labels. Records are selected as in
the official release script (scripts/release.py at the pinned commit): read
`{key}_ids.csv`, drop rows with no OpenAlex id, deduplicate on the id keeping an
inclusion. Text then comes from:

  PubMed    title and abstract, for every record with a PMID (83% of records;
            no key needed, 3 requests per second)
  OpenAlex  title and abstract for the rest, and the abstract for PubMed
            records that have none. OpenAlex now gives anonymous
            users a small daily budget per IP address, so set OPENALEX_API_KEY
            (free) for anything large. Merged ids are fetched one at a time,
            which follows the merge.

Text already fetched from OpenAlex in an earlier build is reused, so a rebuild
spends no OpenAlex requests on it. If the OpenAlex budget runs out, reviews that
still need it are skipped and named; rerun later to finish them. `reviews.json`
records the counts, including records left title-only.

Needs network access, so it runs locally, not on CHTC:

    python -m src.synergy_build            # writes data/synergy/
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

COMMIT = "dc2dadfdbb98eb1b4259604789abd640aa3b693e"
RAW = f"https://raw.githubusercontent.com/asreview/synergy-dataset/{COMMIT}"
TREE = f"https://api.github.com/repos/asreview/synergy-dataset/git/trees/{COMMIT}?recursive=1"
OPENALEX = "https://api.openalex.org/works"
PAGE = 50          # ids per OpenAlex request, its maximum for an OR filter
PAUSE = 0.15
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_PAGE = 200
PUBMED_PAUSE = 0.4  # NCBI allows 3 requests per second without an API key
AGENT = "ThinkFast-QueryFaster/0.1 (course research project)"
SELECT = "id,title,abstract_inverted_index,publication_year"


class OutOfBudget(Exception):
    """OpenAlex refused for lack of budget; retrying today will not help."""


def get(url: str, tries: int = 5, headers: dict | None = None) -> bytes:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": AGENT, **(headers or {})})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and "openalex" in url and b"budget" in e.read().lower():
                raise OutOfBudget() from None
            if e.code == 404 or attempt == tries - 1:
                raise
            print(f"  retry {attempt + 1} after HTTP {e.code}", flush=True)
        except Exception as e:  # network hiccups: back off and retry
            if attempt == tries - 1:
                raise
            print(f"  retry {attempt + 1} after {type(e).__name__}", flush=True)
        time.sleep(2 ** attempt)
    raise AssertionError


def openalex(url: str, tries: int = 5) -> dict:
    # The key goes in a header, so it never appears in a URL or an error message.
    key = os.environ.get("OPENALEX_API_KEY", "").strip()
    return json.loads(get(url, tries, {"Authorization": f"Bearer {key}"} if key else None))


def abstract(inverted: dict | None) -> str:
    """OpenAlex stores abstracts as word -> positions; put the words back in order."""
    if not inverted:
        return ""
    words = [(p, w) for w, ps in inverted.items() for p in ps]
    return " ".join(w for _, w in sorted(words))


def short(openalex_id: str) -> str:
    return openalex_id.rstrip("/").rsplit("/", 1)[-1]


def labels(path: str) -> tuple[dict[str, int], dict[str, str]]:
    """OpenAlex id -> label, as release.py: drop missing ids, keep an inclusion.
    Also OpenAlex id -> PMID where the row has one."""
    out: dict[str, int] = {}
    pmids: dict[str, str] = {}
    for row in csv.DictReader(io.StringIO(get(f"{RAW}/{path}").decode("utf-8"))):
        oid = (row.get("openalex_id") or "").strip()
        if not oid:
            continue
        oid = short(oid)
        out[oid] = max(out.get(oid, 0), int(float(row["label_included"])))
        pmid = (row.get("pmid") or "").strip().rstrip("/").rsplit("/", 1)[-1]
        if pmid.isdigit():
            pmids.setdefault(oid, pmid)
    return out, pmids


def openalex_works(ids: list[str]) -> dict[str, dict]:
    out = {}
    for i in range(0, len(ids), PAGE):
        chunk = ids[i: i + PAGE]
        q = urllib.parse.urlencode({"filter": "openalex_id:" + "|".join(chunk),
                                    "select": SELECT, "per-page": PAGE})
        for w in openalex(f"{OPENALEX}?{q}")["results"]:
            out[short(w["id"])] = w
        time.sleep(PAUSE)
        if (i // PAGE) % 100 == 0:
            print(f"  openalex {min(i + PAGE, len(ids))}/{len(ids)}", flush=True)
    for oid in [i for i in ids if i not in out]:
        # A merged id is absent from filter results, but a direct lookup redirects.
        try:
            out[oid] = openalex(f"{OPENALEX}/{oid}?select={SELECT}", tries=2)
        except OutOfBudget:
            raise
        except Exception:
            pass
        time.sleep(PAUSE)
    return out


def pubmed(pmids: list[str]) -> dict[str, tuple[str, str]]:
    """PMID -> (title, abstract) from NCBI E-utilities."""
    import xml.etree.ElementTree as ET
    out = {}
    for i in range(0, len(pmids), PUBMED_PAGE):
        q = urllib.parse.urlencode({"db": "pubmed", "retmode": "xml",
                                    "tool": "ThinkFastQueryFaster",
                                    "id": ",".join(pmids[i: i + PUBMED_PAGE])})
        root = ET.fromstring(get(f"{EFETCH}?{q}"))
        for art in root.iter("PubmedArticle"):
            pmid = art.findtext(".//MedlineCitation/PMID")
            title_el = art.find(".//ArticleTitle")
            title = "".join(title_el.itertext()).strip() if title_el is not None else ""
            parts = ["".join(a.itertext()).strip() for a in art.iter("AbstractText")]
            if pmid and title:
                out[pmid] = (title, " ".join(p for p in parts if p))
        time.sleep(PUBMED_PAUSE)
        if (i // PUBMED_PAGE) % 50 == 0:
            print(f"  pubmed {min(i + PUBMED_PAGE, len(pmids))}/{len(pmids)}", flush=True)
    return out


def previous(path: Path) -> dict[str, dict]:
    """Rows from an earlier build, to reuse text already fetched from OpenAlex."""
    if not path.exists():
        return {}
    with gzip.open(path, "rt") as fh:
        return {r["id"]: r for r in map(json.loads, fh)}


def review_title(doi: str | None) -> str:
    """The review's own title, from Crossref (no key needed)."""
    if not doi:
        return ""
    doi = doi.removeprefix("https://doi.org/")
    try:
        msg = json.loads(get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"))
        return " ".join((msg["message"].get("title") or [""])[0].split())
    except Exception:
        return ""


def build(ds: dict, path: str, out: Path, index: dict) -> None:
    key = ds["key"]
    pub = ds.get("publication", {})
    labelled, pmids = labels(path)
    old = previous(out / f"{key}.jsonl.gz")
    from_pm = pubmed(sorted(set(pmids.values())))
    # Earlier builds took titles from OpenAlex, so any earlier row can stand in;
    # a PubMed record without an abstract can borrow an earlier row's abstract.
    def covered(o):
        pm = from_pm.get(pmids.get(o))
        return (pm is not None and pm[1]) or o in old
    need = sorted(o for o in labelled if not covered(o))
    # Ids an earlier build looked up and OpenAlex did not have. The first build
    # stored only a count, so a count at least as large as `need` also counts.
    tried = index.get(key, {})
    if set(need) <= set(tried.get("not_found_ids", [])) or (
            old and len(need) <= tried.get("n_not_in_openalex", tried.get("n_not_found", 0))):
        need = []
    fetched = openalex_works(need) if need else {}
    title = index.get(key, {}).get("title") or review_title(pub.get("doi"))

    rows, missing = [], 0
    for oid, label in labelled.items():
        if pmids.get(oid) in from_pm:
            t, a = from_pm[pmids[oid]]
            source = "pubmed"
            if not a:
                if oid in fetched:
                    a = abstract(fetched[oid].get("abstract_inverted_index"))
                elif oid in old:
                    a = old[oid]["abstract"]
                source = "pubmed+openalex" if a else "pubmed"
        elif oid in fetched:
            w = fetched[oid]
            t, a, source = w.get("title") or "", abstract(w.get("abstract_inverted_index")), "openalex"
        elif oid in old:
            t, a = old[oid]["title"], old[oid]["abstract"]
            source = old[oid].get("text_source") or "openalex"
        else:
            missing += 1
            continue
        rows.append({"id": oid, "title": t, "abstract": a, "text_source": source, "label": label})
    with gzip.open(out / f"{key}.jsonl.gz", "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    index[key] = {
        "title": title, "doi": pub.get("doi") or pub.get("url"),
        "criteria": (pub.get("eligibility_criteria") or "").strip(),
        "n": len(rows), "n_included": sum(r["label"] for r in rows),
        "n_from_pubmed": sum(r["text_source"].startswith("pubmed") for r in rows),
        "n_title_only": sum(not r["abstract"] for r in rows),
        "n_title_only_included": sum(1 for r in rows if r["label"] and not r["abstract"]),
        "n_not_found": missing,
        "not_found_ids": sorted(o for o in labelled if not any(r["id"] == o for r in rows)),
        "commit": COMMIT}
    if not title:
        print("  WARNING: no review title", flush=True)
    print(f"  n={len(rows)} included={index[key]['n_included']} "
          f"pubmed={index[key]['n_from_pubmed']} title-only={index[key]['n_title_only']} "
          f"not-found={missing}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/synergy")
    p.add_argument("--only", default="", help="comma-separated review keys")
    args = p.parse_args(argv)

    try:
        import tomllib as toml
    except ImportError:          # Python < 3.11
        import tomli as toml

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = toml.loads(get(f"{RAW}/datasets.toml").decode("utf-8"))
    paths = {Path(t["path"]).name: t["path"] for t in json.loads(get(TREE))["tree"]}
    only = {k for k in args.only.split(",") if k}
    index_path = out / "reviews.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    print("OpenAlex key:", "set" if os.environ.get("OPENALEX_API_KEY") else "not set", flush=True)

    skipped = []
    for ds in config["datasets"]:
        key = ds["key"]
        if not ds.get("active", True) or (only and key not in only):
            continue
        print(f"[synergy] {key}", flush=True)
        try:
            build(ds, paths[f"{key}_ids.csv"], out, index)
        except OutOfBudget:
            print("  skipped: OpenAlex budget used up", flush=True)
            skipped.append(key)
            continue
        index_path.write_text(json.dumps(index, indent=1))
    if skipped:
        print("\nNot built, rerun with OPENALEX_API_KEY set or after midnight UTC:",
              ",".join(skipped))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
