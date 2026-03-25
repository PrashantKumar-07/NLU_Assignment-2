"""
Task 1 - Dataset Preparation

Scrapes IIT Jodhpur web pages and extracts text from regulation PDFs
to build a clean corpus for Word2Vec training.

Sources:
  - Department pages (9 departments)
  - Academic program pages (BTech, MTech, MSc, PhD, etc.)
  - Announcements, circulars, curriculum, research highlights
  - Faculty members listing
  - Two regulation PDFs (PG 2022 onwards, Academic 2019)

Run this first — all downstream tasks depend on cleaned_corpus.txt.
"""

import re
import time
import json
import logging
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse
from collections import Counter

import fitz
import nltk
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent
CORPUS_DIR = BASE_DIR / "corpus"
OUT_DIR    = BASE_DIR / "outputs"
CORPUS_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

CLEANED_CORPUS_FILE = CORPUS_DIR / "corpus.txt"
STATS_FILE          = OUT_DIR / "corpus_stats.json"

ALLOWED_DOMAIN     = "iitj.ac.in"
DELAY_SEC          = 0.8
MAX_DEPTH          = 1    # depth=1 means seed + its direct links only; 
MAX_PAGES_PER_SEED = 15   # keeps each seed fast; 11 seeds × 15 pages ≈ 165 pages total

# Each entry: (seed_url, allowed_path_prefixes).
# Child links are only followed if their path starts with one of the listed prefixes,
# keeping each crawl strictly inside its own section of the site.
# Seeds with an empty prefix list scrape only the seed page itself.
SEED_CONFIG = [
    (
        "https://iitj.ac.in/m/Index/main-departments?lg=en",
        [
            "/bioscience-and-bioengineering",
            "/chemistry",
            "/chemical-engineering",
            "/civil-and-infrastructure-engineering",
            "/computer-science-engineering",
            "/electrical-engineering",
            "/mathematics",
            "/mechanical-engineering",
            "/materials-engineering",
            "/physics",
            "/humanities-and-social-sciences",
            "/m/Index/main-departments",
        ],
    ),
    (
        # Program slugs are Title-Case on this site (e.g. /Bachelor-of-Technology),
        # unlike the department slugs which are all lowercase.
        "https://iitj.ac.in/m/Index/main-programs?lg=en",
        [
            "/Bachelor-of-Technology",
            "/Master-of-Technology",
            "/Master-of-Science",
            "/Doctor-of-Philosophy",
            "/itep",                         # Integrated Teacher Education Programme
            "/office-of-executive-education",
            "/school-of-design",
            "/school-of-management",         # MBA 
            "/suraj",                        # Summer Undergraduate Research at Jodhpur
            "/office-of-academics/en/bs",
            "/m/Index/main-programs",
        ],
    ),
    (
        "https://iitj.ac.in/admission-postgraduate-programs/en/Admission-to-Postgraduate-Programs",
        [],
    ),
    (
        "https://iitj.ac.in/main/en/all-announcement",
        ["/main/en/all-announcement", "/main/en/announcement"],
    ),
    (
        "https://iitj.ac.in/office-of-academics/en/circulars",
        ["/office-of-academics/en/circular"],
    ),
    (
        "https://iitj.ac.in/office-of-academics/en/curriculum",
        ["/office-of-academics/en/curriculum", "/office-of-academics/en/course"],
    ),
    (
        "https://www.iitj.ac.in/main/en/research-highlight",
        ["/main/en/research"],
    ),
    (
        "https://www.iitj.ac.in/main/en/research-areas-removed",
        ["/main/en/research"],
    ),
    (
        "https://iitj.ac.in/main/en/important-links",
        [],
    ),
    (
        "https://www.iitj.ac.in/main/en/faculty-members",
        [],
    ),
    (
        "https://iitj.ac.in/",
        [],
    ),
]

SKIP_PATTERNS = [
    r"/login", r"/logout", r"/cart", r"/search\?",
    r"javascript:", r"mailto:", r"#",
    r"/hi/",
    r"[?&]lg=hi",
    r"\.(pdf|jpg|jpeg|png|gif|svg|css|js|zip|doc|docx|xls|xlsx)$",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AcademicCrawler/1.0; "
        "+https://iitj.ac.in) Python-requests"
    )
}


def should_skip(url):
    return any(re.search(pat, url, re.IGNORECASE) for pat in SKIP_PATTERNS)


def is_internal(url):
    return ALLOWED_DOMAIN in urlparse(url).netloc


def in_allowed_zone(url, allowed_prefixes):
    # empty prefix list means "seed page only, follow nothing"
    if not allowed_prefixes:
        return False
    return any(urlparse(url).path.startswith(p) for p in allowed_prefixes)


def fetch(url, timeout=12):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200 and "text/html" in r.headers.get("Content-Type", ""):
            return r.text
    except Exception as e:
        log.debug(f"fetch failed {url}: {e}")
    return None


def extract_links(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for tag in soup.find_all("a", href=True):
        full = urljoin(base_url, tag["href"].strip()).split("#")[0].rstrip("/")
        if is_internal(full) and not should_skip(full):
            links.append(full)
    return links


def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "noscript", "iframe", "form"]):
        tag.decompose()
    return soup.get_text(separator=" ")


def crawl_seed(seed_url, allowed_prefixes):
    # Standard BFS — queue holds (url, depth) tuples.
    # We stop expanding at MAX_DEPTH so the crawl stays shallow and predictable.
    visited = set()
    queue   = [(seed_url, 0)]
    pages   = []

    while queue and len(visited) < MAX_PAGES_PER_SEED:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        log.info(f"  [{len(visited)}/{MAX_PAGES_PER_SEED}] depth={depth}  {url}")
        html = fetch(url)
        if not html:
            time.sleep(DELAY_SEC)
            continue

        text = html_to_text(html)
        if len(text.split()) > 30:
            pages.append(text)

        if depth < MAX_DEPTH:
            for link in extract_links(html, url):
                if link not in visited and in_allowed_zone(link, allowed_prefixes):
                    queue.append((link, depth + 1))

        time.sleep(DELAY_SEC)

    return pages


def crawl():
    all_pages = []
    for seed_url, allowed_prefixes in SEED_CONFIG:
        log.info(f"\n── Crawling seed: {seed_url}")
        log.info(f"   Allowed prefixes: {allowed_prefixes or ['(seed only)']}")
        pages = crawl_seed(seed_url, allowed_prefixes)
        all_pages.extend(pages)
        log.info(f"   Got {len(pages)} pages from this seed.")
    log.info(f"\nCrawl complete. Total pages: {len(all_pages)}")
    return all_pages


PDF_PATHS = [
    BASE_DIR / "4_Regulation_PG_2022-onwards_20022023.pdf",
    BASE_DIR / "Academic_Regulations_Final_03_09_2019.pdf",
]


def extract_pdf(path):
    try:
        doc  = fitz.open(str(path))
        text = " ".join(page.get_text() for page in doc)
        doc.close()
        log.info(f"PDF extracted: {path.name}  ({len(text.split())} words)")
        return text
    except Exception as e:
        log.warning(f"PDF extraction failed for {path}: {e}")
        return ""


_MULTI_SPACE  = re.compile(r"\s+")
_NON_ASCII    = re.compile(r"[^\x00-\x7F]+")
_URL_PAT      = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_PAT    = re.compile(r"\S+@\S+\.\S+")
_PUNCT_HEAVY  = re.compile(r"[^a-z\s\.\,\!\?\;\:\'\-]")  # digits removed — numbers add noise
_NUMBER_TOKEN = re.compile(r"\b\d+\b")                          # catch any remaining standalone numbers

# Normalise dotted abbreviations *before* the punctuation stripper runs.
# Without this, "B.Tech." → "b tech" after dot removal, and "b" gets
# dropped by the length filter in task2, losing the term entirely.
_ABBREV_MAP = [
    (re.compile(r"b\.tech\.?", re.IGNORECASE), "btech"),
    (re.compile(r"m\.tech\.?", re.IGNORECASE), "mtech"),
    (re.compile(r"m\.sc\.?",   re.IGNORECASE), "msc"),
    (re.compile(r"m\.b\.a\.?", re.IGNORECASE), "mba"),
    (re.compile(r"ph\.d\.?",   re.IGNORECASE), "phd"),
    (re.compile(r"b\.sc\.?",   re.IGNORECASE), "bsc"),
]


def clean_text(raw):
    text = raw.lower()
    for pat, replacement in _ABBREV_MAP:
        text = pat.sub(replacement, text)
    text = _NON_ASCII.sub(" ", text)
    text = _URL_PAT.sub(" ", text)
    text = _EMAIL_PAT.sub(" ", text)
    text = _PUNCT_HEAVY.sub(" ", text)
    text = _NUMBER_TOKEN.sub(" ", text)   # remove phone numbers, years, roll numbers etc.
    return _MULTI_SPACE.sub(" ", text).strip()


def is_mostly_english(text, threshold=0.7):
    total  = sum(1 for c in text if c.isalpha())
    if total == 0:
        return False
    ascii_ = sum(1 for c in text if c.isascii() and c.isalpha())
    return ascii_ / total >= threshold


def compute_stats(sentences):
    tokens = [t for s in sentences for t in word_tokenize(s)]
    freq   = Counter(tokens)
    return {
        "total_documents": len(sentences),
        "total_tokens"   : len(tokens),
        "vocabulary_size": len(set(tokens)),
        "top_50_words"   : freq.most_common(50),
    }, tokens


def save_wordcloud(tokens):
    from nltk.corpus import stopwords
    noise = set(stopwords.words("english")) | {
        "iit", "jodhpur", "iitj", "institute", "shall", "may", "also",
        "per", "one", "two", "three", "four", "five",
        "section", "table", "course", "courses", "program", "programs",
        "will", "would", "please", "click", "link", "links", "page",
        "website", "portal", "contact", "email", "phone", "address",
    }
    # also exclude any remaining numeric tokens that slipped through
    freq = Counter(
        t for t in tokens
        if t not in noise and len(t) > 3 and not t.isdigit() and not re.search(r'\d', t)
    )
    wc   = WordCloud(
        width=1400, height=700, background_color="white",
        colormap="Blues", max_words=150, collocations=False,
    ).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Most Frequent Words in IIT Jodhpur Corpus", fontsize=16, pad=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wordcloud.png", dpi=150)
    plt.close(fig)
    log.info("Word cloud saved.")



def deduplicate(docs, threshold=0.7):
    """
    Remove near-duplicate documents using Jaccard similarity on word trigrams.
    Web pages share large chunks of boilerplate (nav menus, footers, sidebars)
    that survive html_to_text; this step prevents the same content appearing
    dozens of times in the corpus and inflating term frequencies artificially.
    threshold=0.7 means two docs that share >70% of their trigrams are considered
    duplicates and the second occurrence is dropped.
    """
    def trigrams(text):
        words = text.split()
        return set(zip(words, words[1:], words[2:])) if len(words) >= 3 else set(words)

    unique = []
    seen_grams = []
    for doc in docs:
        grams = trigrams(doc)
        if not grams:
            continue
        is_dup = False
        for existing in seen_grams:
            if not existing:
                continue
            overlap = len(grams & existing) / max(len(grams | existing), 1)
            if overlap >= threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(doc)
            seen_grams.append(grams)
    return unique


def main():
    raw_docs = []

    log.info("Starting web crawl …")
    raw_docs.extend(crawl())

    log.info("Extracting regulation PDFs …")
    for p in PDF_PATHS:
        if p.exists():
            raw_docs.append(extract_pdf(p))
        else:
            log.warning(f"PDF not found: {p}")

    log.info(f"Total raw documents: {len(raw_docs)}")

    cleaned = [
        c for doc in raw_docs
        for c in [clean_text(doc)]
        if is_mostly_english(c) and len(c.split()) > 20
    ]
    log.info(f"Documents after cleaning: {len(cleaned)}")

    # remove near-duplicate pages (same boilerplate content from multiple URLs)
    cleaned = deduplicate(cleaned, threshold=0.7)
    log.info(f"Documents after deduplication: {len(cleaned)}")

    with open(CLEANED_CORPUS_FILE, "w", encoding="utf-8") as f:
        f.writelines(doc + "\n" for doc in cleaned)
    log.info(f"Cleaned corpus saved → {CLEANED_CORPUS_FILE}")

    stats, tokens = compute_stats(cleaned)
    log.info(
        f"Stats → docs={stats['total_documents']}  "
        f"tokens={stats['total_tokens']}  "
        f"vocab={stats['vocabulary_size']}"
    )

    serialisable = dict(stats)
    with open(STATS_FILE, "w") as f:
        json.dump(serialisable, f, indent=2)

    print("\n" + "=" * 55)
    print("CORPUS STATISTICS")
    print("=" * 55)
    print(f"  Total documents : {stats['total_documents']}")
    print(f"  Total tokens    : {stats['total_tokens']}")
    print(f"  Vocabulary size : {stats['vocabulary_size']}")
    print("=" * 55 + "\n")

    save_wordcloud(tokens)


if __name__ == "__main__":
    main()