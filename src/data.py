"""Corpus loading and query sets.

Datasets with complete ground truth for "find all matching documents":

  20 Newsgroups  -- 18,846 docs, 20 categories. Every doc carries a category
                    label, so a query like "documents about computer graphics"
                    has an exact, complete answer set. Recall is unambiguous.
                    Ships with scikit-learn; no download gate.

  NFCorpus       -- 3,633 docs, 323 test queries, ~38 relevant docs per query.
                    Real IR relevance judgments. Needs `datasets`.

  SYNERGY        -- 26 systematic reviews, one dataset each (synergy_<review>):
                    the screened candidate pool and the review's inclusions.
                    Built locally by src.synergy_build into data/synergy/.

Both expose the same interface:

    corpus  : list[Doc]            -- id, text
    queries : list[Query]          -- id, text, gold (set of doc ids)
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Doc:
    id: str
    text: str


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    gold: frozenset[str]


@dataclass
class Corpus:
    name: str
    docs: list[Doc]
    queries: list[Query]
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.docs)

    def doc_ids(self) -> list[str]:
        return [d.id for d in self.docs]

    def subsample(self, n: int, seed: int = 0) -> "Corpus":
        """Take n documents at random and restrict every gold set to survivors.

        Queries left with no relevant documents are dropped, since recall is
        undefined for them.
        """
        if n >= len(self.docs):
            return self
        rng = random.Random(seed)
        keep = rng.sample(self.docs, n)
        kept_ids = {d.id for d in keep}
        qs = []
        for q in self.queries:
            g = q.gold & kept_ids
            if g:
                qs.append(Query(q.id, q.text, frozenset(g)))
        return Corpus(
            name=self.name,
            docs=keep,
            queries=qs,
            meta={**self.meta, "subsampled_from": len(self.docs), "seed": seed},
        )


# --------------------------------------------------------------------------
# 20 Newsgroups
# --------------------------------------------------------------------------

# One natural-language query per category. Phrased as a retrieval request so
# the decision model sees a semantic predicate, not a label name. The gold set
# is every document carrying that category label.
NEWSGROUP_QUERIES: dict[str, str] = {
    "alt.atheism":
        "documents discussing atheism, religious skepticism, or arguments "
        "for and against the existence of God",
    "comp.graphics":
        "documents about computer graphics: image processing, 3D rendering, "
        "graphics file formats, or graphics software and libraries",
    "comp.os.ms-windows.misc":
        "documents about the Microsoft Windows operating system: drivers, "
        "configuration, Windows software, or troubleshooting Windows",
    "comp.sys.ibm.pc.hardware":
        "documents about IBM PC hardware: motherboards, disk controllers, "
        "expansion cards, CPUs, or PC component compatibility",
    "comp.sys.mac.hardware":
        "documents about Apple Macintosh hardware: Mac models, upgrades, "
        "peripherals, or Mac-specific hardware problems",
    "comp.windows.x":
        "documents about the X Window System: X11 servers, widget toolkits, "
        "window managers, or X application development on Unix",
    "misc.forsale":
        "documents offering items for sale or wanted, including asking "
        "prices, shipping terms, or contact details for a private sale",
    "rec.autos":
        "documents about cars and driving: car models, engines, maintenance, "
        "buying a car, or road experiences",
    "rec.motorcycles":
        "documents about motorcycles: bike models, riding experiences, "
        "motorcycle maintenance, helmets, or road safety for riders",
    "rec.sport.baseball":
        "documents about baseball: teams, players, games, statistics, "
        "pitching, batting, or the baseball season",
    "rec.sport.hockey":
        "documents about ice hockey: NHL teams, players, games, playoffs, "
        "goaltending, or hockey season results",
    "sci.crypt":
        "documents about cryptography and encryption: ciphers, key escrow, "
        "the Clipper chip, digital signatures, or cryptographic policy",
    "sci.electronics":
        "documents about electronics: circuits, components, amplifiers, "
        "signal processing hardware, or electronic design and repair",
    "sci.med":
        "documents about medicine and health: diseases, symptoms, "
        "treatments, drugs, clinical evidence, or medical practice",
    "sci.space":
        "documents about space and astronomy: spacecraft, orbits, NASA "
        "missions, planets, launch vehicles, or space exploration policy",
    "soc.religion.christian":
        "documents discussing Christianity: scripture, theology, church "
        "practice, Christian doctrine, or faith and worship",
    "talk.politics.guns":
        "documents about firearms policy and gun politics: gun control "
        "legislation, the Second Amendment, or firearms regulation debates",
    "talk.politics.mideast":
        "documents about Middle East politics: Israel, Palestine, Arab "
        "states, regional conflict, or Middle Eastern foreign policy",
    "talk.politics.misc":
        "documents about general political debate: government policy, "
        "elections, taxation, civil liberties, or political ideology",
    "talk.religion.misc":
        "documents on general religious discussion and debate across "
        "traditions, including morality, belief, and religious criticism",
}

_WS = re.compile(r"\s+")


def _clean(text: str, max_chars: int) -> str:
    text = _WS.sub(" ", text).strip()
    return text[:max_chars]


def load_20newsgroups(max_chars: int = 2000, strip_metadata: bool = True) -> Corpus:
    """18,846 documents in 20 categories, with complete category labels.

    strip_metadata removes headers, quoted replies and signature blocks. Keep it
    on: the headers contain the newsgroup name, which makes the task trivial and
    the measurement meaningless.
    """
    from sklearn.datasets import fetch_20newsgroups

    remove = ("headers", "footers", "quotes") if strip_metadata else ()
    raw = fetch_20newsgroups(subset="all", remove=remove, shuffle=False)

    docs: list[Doc] = []
    by_cat: dict[str, set[str]] = {c: set() for c in raw.target_names}
    for i, (text, tgt) in enumerate(zip(raw.data, raw.target)):
        cleaned = _clean(text, max_chars)
        if len(cleaned) < 40:
            continue  # empty once headers are stripped
        did = f"ng{i:05d}"
        docs.append(Doc(did, cleaned))
        by_cat[raw.target_names[tgt]].add(did)

    queries = [
        Query(f"q_{cat}", NEWSGROUP_QUERIES[cat], frozenset(by_cat[cat]))
        for cat in raw.target_names
        if by_cat[cat]
    ]
    return Corpus(
        name="20newsgroups",
        docs=docs,
        queries=queries,
        meta={"max_chars": max_chars, "strip_metadata": strip_metadata},
    )


# Query shapes beyond one semantic category, on the same documents. Every gold
# set is still complete, because it comes from category labels and from exact
# word matches on the text the classifier sees.
KEYWORDS = ("encryption", "nasa", "bible", "helmet", "monitor", "insurance",
            "israel", "goalie", "compiler", "battery")
EITHER = (("rec.sport.baseball", "rec.sport.hockey"), ("rec.autos", "rec.motorcycles"),
          ("comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware"),
          ("alt.atheism", "soc.religion.christian"),
          ("talk.politics.guns", "talk.politics.mideast"))
BOTH = (("rec.autos", "insurance"), ("sci.space", "shuttle"), ("sci.crypt", "clipper"),
        ("sci.med", "cancer"), ("talk.politics.guns", "nra"), ("rec.sport.hockey", "toronto"))
BUT_NOT = (("rec.sport.baseball", "yankees"), ("sci.space", "nasa"),
           ("comp.graphics", "jpeg"), ("sci.crypt", "clipper"))


def _has_word(word: str):
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    return lambda text: bool(pattern.search(text))


def load_20newsgroups_logic(max_chars: int = 2000) -> Corpus:
    """The same documents, with keyword and multi-condition queries."""
    base = load_20newsgroups(max_chars=max_chars)
    cats = {q.id.removeprefix("q_"): q for q in base.queries}

    def containing(word: str) -> frozenset[str]:
        has = _has_word(word)
        return frozenset(d.id for d in base.docs if has(d.text))

    queries = [Query(f"kw_{w}", f'documents that contain the word "{w}"', containing(w))
               for w in KEYWORDS]
    queries += [Query(f"or_{a}+{b}", f"{cats[a].text}; or {cats[b].text}",
                      cats[a].gold | cats[b].gold) for a, b in EITHER]
    queries += [Query(f"and_{c}+{w}", f'{cats[c].text}, and that contain the word "{w}"',
                      cats[c].gold & containing(w)) for c, w in BOTH]
    queries += [Query(f"not_{c}+{w}", f'{cats[c].text}, but that do not contain the word "{w}"',
                      cats[c].gold - containing(w)) for c, w in BUT_NOT]
    return Corpus(name="20newsgroups_logic", docs=base.docs, queries=queries, meta=base.meta)


# --------------------------------------------------------------------------
# NFCorpus (BEIR)
# --------------------------------------------------------------------------

def load_nfcorpus(max_chars: int = 2000, split: str = "test") -> Corpus:
    """3,633 medical documents, 323 test queries, ~38 relevant docs per query."""
    from datasets import load_dataset

    corpus_ds = load_dataset("BeIR/nfcorpus", "corpus")["corpus"]
    queries_ds = load_dataset("BeIR/nfcorpus", "queries")["queries"]
    qrels = load_dataset("BeIR/nfcorpus-qrels")[split]

    docs = []
    for row in corpus_ds:
        text = _clean(f"{row['title']} {row['text']}", max_chars)
        if len(text) < 40:
            continue
        docs.append(Doc(str(row["_id"]), text))
    present = {d.id for d in docs}

    gold: dict[str, set[str]] = {}
    for row in qrels:
        if int(row["score"]) <= 0:
            continue
        did = str(row["corpus-id"])
        if did in present:
            gold.setdefault(str(row["query-id"]), set()).add(did)

    qtext = {str(r["_id"]): r["text"] for r in queries_ds}
    queries = [
        Query(qid, qtext[qid], frozenset(g))
        for qid, g in sorted(gold.items())
        if qid in qtext and g
    ]
    return Corpus(
        name="nfcorpus",
        docs=docs,
        queries=queries,
        meta={"max_chars": max_chars, "split": split},
    )


# --------------------------------------------------------------------------
# Synthetic corpus (offline smoke tests)
# --------------------------------------------------------------------------

_TOPICS = {
    "astronomy": "telescope orbit galaxy nebula spectroscopy redshift planet comet",
    "cooking": "saute braise simmer dough yeast marinade skillet seasoning",
    "finance": "equity dividend portfolio hedge liquidity yield amortize ledger",
    "gardening": "perennial mulch compost pruning trellis loam germination",
    "networking": "router packet latency subnet protocol firewall bandwidth",
}
_FILLER = (
    "the and for with that this from have been would could about which there "
    "their other into than them these those where when what some more very"
).split()


def load_synthetic(n_docs: int = 2000, doc_words: int = 120, seed: int = 0) -> Corpus:
    """Topic-labelled random text. No network, no download, instant.

    Not a scientific instrument. Its only job is to exercise the full pipeline
    end to end so that a CHTC run fails on the science rather than on a typo.
    """
    rng = random.Random(seed)
    topics = list(_TOPICS)
    docs, by_topic = [], {t: set() for t in topics}
    for i in range(n_docs):
        t = topics[i % len(topics)]
        signal = _TOPICS[t].split()
        words = [
            rng.choice(signal) if rng.random() < 0.35 else rng.choice(_FILLER)
            for _ in range(doc_words)
        ]
        did = f"syn{i:05d}"
        docs.append(Doc(did, " ".join(words)))
        by_topic[t].add(did)

    queries = [
        Query(f"q_{t}", f"documents about {t}: {_TOPICS[t]}", frozenset(by_topic[t]))
        for t in topics
    ]
    return Corpus(name="synthetic", docs=docs, queries=queries,
                  meta={"n_docs": n_docs, "doc_words": doc_words, "seed": seed})


# --------------------------------------------------------------------------
# SYNERGY systematic reviews (build first with `python -m src.synergy_build`)
# --------------------------------------------------------------------------

SYNERGY_KEYS = (
    "Appenzeller-Herzog_2019", "Bos_2018", "Brouwer_2019", "Chou_2003", "Chou_2004",
    "Donners_2021", "Hall_2012", "Jeyaraman_2020", "Leenaars_2019", "Leenaars_2020",
    "Meijboom_2021", "Menon_2022", "Moran_2021", "Muthu_2021", "Nelson_2002",
    "Oud_2018", "Radjenovic_2013", "Sep_2021", "Smid_2020", "van_de_Schoot_2018",
    "van_der_Valk_2021", "van_der_Waal_2022", "van_Dis_2020", "Walker_2018",
    "Wassenaar_2017", "Wolters_2018",
)


def synergy_dir() -> "Path":
    import os
    from pathlib import Path
    return Path(os.environ.get("SYNERGY_DIR",
                               Path(__file__).resolve().parent.parent / "data" / "synergy"))


def load_synergy(review: str, max_chars: int = 2000, query: str = "criteria") -> Corpus:
    """One systematic review: its screened candidate pool, one find-all query.

    Every record in the pool was screened, so the review's inclusions are the
    complete answer set for that pool. Pools are never merged, because a record
    in one review's pool was never screened against another review's criteria.
    Title-only records stay in: screeners saw them too.

    query = "criteria"  the review's published eligibility criteria (default)
            "title"     only the review's title
    """
    import gzip
    import json

    root = synergy_dir()
    meta = json.loads((root / "reviews.json").read_text())[review]
    docs, gold = [], set()
    with gzip.open(root / f"{review}.jsonl.gz", "rt") as fh:
        for line in fh:
            r = json.loads(line)
            text = _clean(f"{r['title']}\n\n{r['abstract']}", max_chars) or "(no title)"
            docs.append(Doc(r["id"], text))
            if r["label"]:
                gold.add(r["id"])
    # Source files often list inclusions first; no arm may see label order.
    random.Random(0).shuffle(docs)
    ask = f'Studies to include in the systematic review "{meta["title"]}".'
    if query == "criteria":
        ask += f"\nEligibility criteria:\n{meta['criteria']}"
    elif query != "title":
        raise ValueError(f"query must be 'criteria' or 'title', not {query!r}")
    suffix = "" if query == "criteria" else f"_{query}"
    return Corpus(
        name=f"synergy_{review}{suffix}",
        docs=docs,
        queries=[Query(f"syn_{review}{suffix}", ask, frozenset(gold))],
        meta={"max_chars": max_chars, "query": query, "review": review,
              "commit": meta.get("commit")},
    )


# --------------------------------------------------------------------------
# arXiv metadata (CC0), for corpora of 10K to 1M documents
# --------------------------------------------------------------------------

ARXIV_REPO = "librarian-bots/arxiv-metadata-snapshot"
ARXIV_REVISION = "c428226b689a0e17db2cd55265133b9eca15d97f"
ARXIV_SAMPLE = 1_000_000

# Category -> request. A paper is in the answer set if the category is any of
# its own categories, cross-lists included, so every answer set is complete.
ARXIV_QUERIES = {
    "quant-ph": "papers about quantum physics: quantum information, quantum computing, entanglement, quantum optics and foundations",
    "astro-ph.GA": "papers about astrophysics of galaxies: galaxy formation and evolution, the interstellar medium, the Milky Way",
    "gr-qc": "papers about general relativity and quantum cosmology: black holes, gravitational waves, cosmological models of gravity",
    "math.AG": "papers about algebraic geometry: varieties, schemes, moduli spaces, sheaves and their cohomology",
    "math.PR": "papers about probability theory: stochastic processes, random walks, limit theorems, martingales",
    "cond-mat.supr-con": "papers about superconductivity: superconducting materials, pairing mechanisms, vortices, Josephson effects",
    "physics.optics": "papers about optics: lasers, photonics, nonlinear optics, optical materials and imaging",
    "hep-ex": "papers about experimental high energy physics: results and detectors from particle collider and accelerator experiments",
    "cs.CR": "papers about computer security and cryptography: attacks, defenses, privacy, cryptographic protocols",
    "cs.RO": "papers about robotics: robot control, motion planning, manipulation, localization and mapping",
    "stat.ME": "papers about statistical methodology: new estimation, testing and modelling methods for data analysis",
    "eess.SP": "papers about signal processing: filtering, estimation, communications signals, sensing and audio or radar signals",
    "cs.LO": "papers about logic in computer science: formal verification, type theory, automata, semantics of programs",
    "cs.IR": "papers about information retrieval: search engines, ranking, recommender systems, retrieval evaluation",
    "cs.DB": "papers about databases: query processing and optimization, data management, transactions, data integration",
    "q-bio.NC": "papers about neurons and cognition: computational neuroscience, neural coding, brain networks, cognitive models",
    "nlin.CD": "papers about chaotic dynamics: chaos, bifurcations, dynamical systems, turbulence and synchronization",
    "physics.bio-ph": "papers about biological physics: the physics of proteins, membranes, cells and biological systems",
    "econ.EM": "papers about econometrics: statistical methods for economic data, causal inference and forecasting in economics",
    "q-fin.ST": "papers about statistical finance: empirical properties of financial markets, price and volatility statistics",
}


def arxiv_dir() -> "Path":
    import os
    from pathlib import Path
    return Path(os.environ.get("ARXIV_DIR",
                               Path(__file__).resolve().parent.parent / "data" / "arxiv"))


def _arxiv_sample() -> "Path":
    """The fixed 1M-paper sample: the first 1M arXiv ids ordered by MD5 hash.

    Hashing makes the sample independent of file order and the same on every
    machine. Built once from the pinned snapshot (2.7 GB of parquet).
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    out = arxiv_dir() / "arxiv_1m.parquet"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in HfApi().list_repo_files(ARXIV_REPO, repo_type="dataset",
                                                      revision=ARXIV_REVISION)
                   if f.startswith("data/") and f.endswith(".parquet"))
    tables = []
    for f in files:
        path = hf_hub_download(ARXIV_REPO, f, repo_type="dataset", revision=ARXIV_REVISION)
        tables.append(pq.read_table(path, columns=["id", "title", "abstract", "categories"]))
    # The snapshot repeats a few ids; sample unique ids and keep one row each.
    ids = sorted({i for t in tables for i in t.column("id").to_pylist()})
    keep = sorted(ids, key=lambda i: hashlib.md5(i.encode()).hexdigest())[:ARXIV_SAMPLE]
    # Filter each shard on its own: one 2.7M-row string column overflows Arrow's
    # 2 GB offsets, while the kept rows fit.
    keep_arr = pa.array(sorted(keep))
    kept = pa.concat_tables([t.filter(pc.is_in(t.column("id"), value_set=keep_arr))
                             for t in tables])
    df = kept.to_pandas().drop_duplicates("id", keep="first").sort_values("id")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out)
    return out


def load_arxiv(max_chars: int = 2000) -> Corpus:
    """1M arXiv papers (title and abstract) and 20 category queries."""
    import pyarrow.parquet as pq

    t = pq.read_table(_arxiv_sample()).to_pydict()
    docs, gold = [], {c: set() for c in ARXIV_QUERIES}
    for pid, title, abstract, cats in zip(t["id"], t["title"], t["abstract"], t["categories"]):
        did = f"arx_{pid}"
        docs.append(Doc(did, _clean(f"{title}\n\n{abstract}", max_chars)))
        for c in cats.split():
            if c in gold:
                gold[c].add(did)
    queries = [Query(f"arx_{c}", text, frozenset(gold[c])) for c, text in ARXIV_QUERIES.items()]
    return Corpus(name="arxiv", docs=docs, queries=queries,
                  meta={"max_chars": max_chars, "revision": ARXIV_REVISION})


def _synergy_loader(review: str, query: str):
    def loader(max_chars: int = 2000) -> Corpus:
        return load_synergy(review, max_chars=max_chars, query=query)
    return loader


LOADERS = {
    "20newsgroups": load_20newsgroups,
    "20newsgroups_logic": load_20newsgroups_logic,
    "nfcorpus": load_nfcorpus,
    "synthetic": load_synthetic,
    "arxiv": load_arxiv,
}
for _key in SYNERGY_KEYS:
    LOADERS[f"synergy_{_key}"] = _synergy_loader(_key, "criteria")
    LOADERS[f"synergy_{_key}_title"] = _synergy_loader(_key, "title")


def expand(names: str) -> list[str]:
    """Comma-separated dataset names; "synergy" means every review, criteria queries."""
    out = []
    for name in (n.strip() for n in names.split(",")):
        if name == "synergy":
            out += [f"synergy_{k}" for k in SYNERGY_KEYS]
        elif name:
            out.append(name)
    return out


def subsets(names: str, sizes: str, max_chars: int = 2000, seed: int = 0):
    """Yield each dataset at each corpus size, as the runner does.

    A size above a pool's size uses the whole pool, once: SYNERGY pools range
    from 257 to 48K records, so sizes=100000 runs every review in full.
    """
    ns = [int(x) for x in sizes.split(",") if x.strip()]
    for name in expand(names):
        full = load(name, **({} if name == "synthetic" else {"max_chars": max_chars}))
        for n in ns:
            if any(len(full.docs) <= m < n for m in ns):
                continue
            yield full.subsample(n, seed=seed)


def load(name: str, **kw) -> Corpus:
    if name not in LOADERS:
        raise ValueError(f"unknown dataset {name!r}; have {sorted(LOADERS)}")
    return LOADERS[name](**kw)


def fingerprint(corpus: Corpus) -> str:
    """Stable hash of corpus contents, recorded with every result row."""
    h = hashlib.sha256()
    h.update(corpus.name.encode())
    for d in corpus.docs:
        h.update(d.id.encode())
    for q in corpus.queries:
        h.update(q.id.encode())
        h.update(str(sorted(q.gold)).encode())
    return h.hexdigest()[:12]


if __name__ == "__main__":
    import sys

    c = load(sys.argv[1] if len(sys.argv) > 1 else "20newsgroups")
    print(f"{c.name}: {len(c.docs)} docs, {len(c.queries)} queries  [{fingerprint(c)}]")
    rel = [len(q.gold) for q in c.queries]
    print(f"relevant per query: min={min(rel)} median={sorted(rel)[len(rel)//2]} max={max(rel)}")
    tot = sum(len(d.text) for d in c.docs)
    print(f"mean doc chars: {tot/len(c.docs):.0f}  (~{tot/len(c.docs)/4:.0f} tokens)")
    for n in (1000, 5000, 10000):
        s = c.subsample(n)
        r = [len(q.gold) for q in s.queries]
        print(f"  subsample {n}: {len(s.docs)} docs, {len(s.queries)} queries, "
              f"median gold {sorted(r)[len(r)//2]}")
