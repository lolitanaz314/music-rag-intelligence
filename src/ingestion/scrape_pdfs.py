import os
import time
import requests
from bs4 import BeautifulSoup
import argparse
from urllib.parse import urljoin, urlparse

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PDFScraper/1.0; +https://github.com/yourrepo)"
}

# -----------------------------------
# Helpers
# -----------------------------------

def safe_filename(url):
    """Generate a safe filename from a URL."""
    parsed = urlparse(url)
    name = os.path.basename(parsed.path)
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name.replace("%20", "_")

def download_pdf(url, output_dir):
    """Download a single PDF file."""
    try:
        filename = safe_filename(url)
        save_path = os.path.join(output_dir, filename)

        if os.path.exists(save_path):
            print(f"[SKIP] Already downloaded: {filename}")
            return

        print(f"[DOWNLOAD] {url}")
        resp = requests.get(url, headers=HEADERS, timeout=20)

        if resp.status_code == 200 and resp.headers.get("Content-Type", "").lower().startswith("application/pdf"):
            with open(save_path, "wb") as f:
                f.write(resp.content)
            print(f"[SAVED] {filename}")
        else:
            print(f"[ERROR] Could not download {url} (status: {resp.status_code})")
        time.sleep(1)  # rate limit
    except Exception as e:
        print(f"[FAILED] {url} — {e}")

def extract_pdf_links(page_url):
    """Extract all PDF links from a webpage."""
    print(f"[CRAWL] {page_url}")
    try:
        resp = requests.get(page_url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        pdf_links = []

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf"):
                full_url = urljoin(page_url, href)
                pdf_links.append(full_url)

        print(f"[FOUND] {len(pdf_links)} PDF links on {page_url}")
        return pdf_links
    except Exception as e:
        print(f"[ERROR] Could not crawl {page_url}: {e}")
        return []

# -----------------------------------
# Site-specific Crawlers
# -----------------------------------

def crawl_musiclawcontracts():
    base = "https://www.musiclawcontracts.com"
    pages = [
        base,
        base + "/recording-contracts",
        base + "/publishing-contracts",
        base + "/management-contracts",
        base + "/producer-contracts",
        base + "/licensing-contracts",
    ]
    links = []
    for p in pages:
        links.extend(extract_pdf_links(p))
    return links

def crawl_ifpi():
    # IFPI changes URLs often — this captures common PDF directories.
    pages = [
        "https://ifpi.org/resources",
        "https://ifpi.org/news",
        "https://ifpi.org/reports"
    ]
    links = []
    for p in pages:
        links.extend(extract_pdf_links(p))
    return links

def crawl_ucla_contracts():
    # UCLA often has PDFs in repositories, this is a common index.
    pages = [
        "https://guides.library.ucla.edu/musicindustry/contracts",
    ]
    links = []
    for p in pages:
        links.extend(extract_pdf_links(p))
    return links

def crawl_ascap():
    return extract_pdf_links("https://www.ascap.com/help")

def crawl_bmi():
    return extract_pdf_links("https://www.bmi.com/help")

def crawl_sesac():
    return extract_pdf_links("https://www.sesac.com")

def crawl_copyright_office():
    pages = [
        "https://www.copyright.gov/circs/",
        "https://www.copyright.gov/rulemaking/",
    ]
    links = []
    for p in pages:
        links.extend(extract_pdf_links(p))
    return links

# -----------------------------------
# Google Scholar Manual Loader
# -----------------------------------

def load_scholar_urls(file_path):
    """Load manually provided Google Scholar PDF URLs."""
    if not os.path.exists(file_path):
        print("[INFO] No scholar_urls.txt file found.")
        return []
    with open(file_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    print(f"[SCHOLAR] Loaded {len(urls)} manual Scholar URLs.")
    return urls

# -----------------------------------
# Master Scrape Function
# -----------------------------------

def scrape_all(output_dir, scholar_file=None):
    os.makedirs(output_dir, exist_ok=True)

    all_links = []

    print("\n=== Crawling MusicLawContracts ===")
    all_links += crawl_musiclawcontracts()

    print("\n=== Crawling IFPI ===")
    all_links += crawl_ifpi()

    print("\n=== Crawling UCLA Contracts ===")
    all_links += crawl_ucla_contracts()

    print("\n=== Crawling ASCAP ===")
    all_links += crawl_ascap()

    print("\n=== Crawling BMI ===")
    all_links += crawl_bmi()

    print("\n=== Crawling SESAC ===")
    all_links += crawl_sesac()

    print("\n=== Crawling U.S. Copyright Office ===")
    all_links += crawl_copyright_office()

    print("\n=== Loading Manual Google Scholar URLs ===")
    if scholar_file:
        all_links += load_scholar_urls(scholar_file)

    print(f"\n[TOTAL FOUND] {len(all_links)} PDFs before de-duplication.\n")

    # Remove duplicates
    all_links = list(set(all_links))
    print(f"[UNIQUE] {len(all_links)} PDFs\n")

    # Download all
    for url in all_links:
        download_pdf(url, output_dir)

    print("\n[DONE] All sites scraped.\n")

# -----------------------------------
# CLI
# -----------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape music industry PDFs")
    parser.add_argument("--output", default="data/raw_pdfs", help="Directory to save PDFs")
    parser.add_argument("--scholar_file", default="scholar_urls.txt",
                        help="Optional file with manual Google Scholar PDF URLs")
    args = parser.parse_args()

    scrape_all(args.output, args.scholar_file)
