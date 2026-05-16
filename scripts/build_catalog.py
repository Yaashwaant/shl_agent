"""
SHL Product Catalog Scraper — Production Version
Scrapes https://www.shl.com/solutions/products/product-catalog/
specifically targeting Individual Test Solutions.

Run: python scripts/build_catalog.py
"""
import json
import time
import logging
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOG_BASE = f"{BASE_URL}/solutions/products/product-catalog/"
PRODUCT_BASE = f"{BASE_URL}/products/product-catalog/view/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

TEST_TYPE_CODES = {
    "Ability": "A",
    "Aptitude": "A",
    "Biodata": "B",
    "Situational": "B",
    "Competenc": "C",
    "Development": "D",
    "360": "D",
    "Exercise": "E",
    "Knowledge": "K",
    "Skills": "K",
    "Personality": "P",
    "Behavior": "P",
    "Behaviour": "P",
    "Simulation": "S",
}


def get_html(url: str, max_retries: int = 3) -> Optional[str]:
    """Fetch HTML with retry and polite delay."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=20)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.warning(f"[Attempt {attempt+1}/{max_retries}] Failed to fetch {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    return None


def extract_test_types_from_text(text: str) -> List[str]:
    """Extract test type codes from descriptive text."""
    codes = set()
    for keyword, code in TEST_TYPE_CODES.items():
        if keyword.lower() in text.lower():
            codes.add(code)
    return sorted(codes)


def parse_catalog_page(html: str) -> Tuple[List[Dict], bool]:
    """
    Parse catalog page.
    Returns: (list of products found, has_next_page)
    """
    soup = BeautifulSoup(html, "html.parser")
    products = []

    # SHL uses a custom web component / table for the catalog
    # Try multiple selectors
    product_links = soup.select(
        "a[href*='/product-catalog/view/'], "
        "a[href*='/products/product-catalog/view/']"
    )

    # Also check for table rows
    table = soup.find("table")
    if table:
        rows = table.select("tbody tr")
        for row in rows:
            link = row.find("a", href=True)
            if not link:
                continue

            href = link["href"]
            if not href.startswith("http"):
                href = BASE_URL + href

            name = link.get_text(strip=True)
            if not name:
                continue

            # Get all cells
            cells = row.find_all("td")
            test_types = []

            # SHL catalog columns: Name | A | B | C | D | E | K | P | S | Duration | Remote | Adaptive
            type_cols = ["A", "B", "C", "D", "E", "K", "P", "S"]
            for col_idx, code in enumerate(type_cols, start=1):
                if col_idx < len(cells):
                    cell = cells[col_idx]
                    # Check for checkmark, filled circle, or any visible content
                    content = cell.get_text(strip=True)
                    img = cell.find("img")
                    span = cell.find("span", class_=lambda c: c and ("check" in c or "yes" in c or "active" in c))
                    if content and content not in ["-", "–", "", " "] or img or span:
                        test_types.append(code)

            # Get duration from second-to-last cell
            duration = ""
            if len(cells) >= 3:
                dur_cell = cells[-3] if len(cells) > 3 else cells[-1]
                dur_text = dur_cell.get_text(strip=True)
                if re.search(r"\d+.*min|variable|—|-", dur_text, re.IGNORECASE):
                    duration = dur_text

            products.append({
                "name": name,
                "url": href,
                "test_types": test_types,
                "duration": duration,
                "description": "",
                "languages": "",
                "remote_testing": "unknown",
                "adaptive_irt": "unknown",
            })

    # Fallback: just collect links
    if not products and product_links:
        seen_urls = set()
        for link in product_links:
            href = link["href"]
            if not href.startswith("http"):
                href = BASE_URL + href
            if href in seen_urls:
                continue
            seen_urls.add(href)

            name = link.get_text(strip=True)
            if name and len(name) > 3:
                products.append({
                    "name": name,
                    "url": href,
                    "test_types": [],
                    "duration": "",
                    "description": "",
                    "languages": "",
                    "remote_testing": "unknown",
                    "adaptive_irt": "unknown",
                })

    # Check for next page
    next_btn = soup.find(
        lambda tag: tag.name in ["a", "button"] and
        any(x in (tag.get("aria-label", "") + tag.get("class", [""])[0] if tag.get("class") else "").lower()
            for x in ["next", "forward", "›", "»"])
    )
    has_next = bool(next_btn)

    logger.info(f"Parsed {len(products)} products, has_next={has_next}")
    return products, has_next


def enrich_product(product: Dict) -> Dict:
    """Fetch individual product page and enrich metadata."""
    url = product["url"]
    html = get_html(url)
    if not html:
        return product

    soup = BeautifulSoup(html, "html.parser")
    full_text = soup.get_text(" ", strip=True)

    # Description — try multiple selectors
    desc = ""
    for sel in [".product-description", "[class*='description']", "[class*='intro']",
                ".content-block p", "main p", "article p"]:
        el = soup.select_one(sel)
        if el:
            desc = el.get_text(strip=True)
            if len(desc) > 50:
                break

    if not desc:
        # Get first substantial paragraph
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 80:
                desc = text[:500]
                break

    product["description"] = desc

    # Duration
    if not product.get("duration"):
        dur_match = re.search(r"(\d+)\s*(?:–|to|-)\s*(\d+)?\s*minutes?", full_text, re.IGNORECASE)
        if dur_match:
            product["duration"] = dur_match.group(0).strip()
        elif re.search(r"variable", full_text, re.IGNORECASE):
            product["duration"] = "Variable"

    # Test types from description if not set
    if not product.get("test_types"):
        product["test_types"] = extract_test_types_from_text(desc + full_text[:2000])

    # Languages
    lang_patterns = [
        r"(?:available in|supported languages?|languages?)[\s:]+([^.]+(?:English|French|Spanish|German|Chinese|Japanese|Arabic|Portuguese)[^.]*)",
    ]
    for pattern in lang_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            product["languages"] = match.group(1)[:300].strip()
            break

    # Remote testing
    if re.search(r"remote\s*testing\s*[:\s]*yes", full_text, re.IGNORECASE):
        product["remote_testing"] = "yes"
    elif re.search(r"remote\s*testing\s*[:\s]*no", full_text, re.IGNORECASE):
        product["remote_testing"] = "no"

    # Adaptive/IRT
    if re.search(r"adaptive\s*[:\s]*yes|irt\s*[:\s]*yes|computer\s+adaptive", full_text, re.IGNORECASE):
        product["adaptive_irt"] = "yes"
    elif re.search(r"adaptive\s*[:\s]*no", full_text, re.IGNORECASE):
        product["adaptive_irt"] = "no"

    return product


def scrape_all_pages() -> List[Dict]:
    """Scrape all catalog pages with pagination."""
    all_products = []
    page = 1
    start = 0
    page_size = 12  # SHL shows 12 items per page

    while True:
        if page == 1:
            url = CATALOG_BASE
        else:
            url = f"{CATALOG_BASE}?start={start}&type=1"  # type=1 = Individual Test Solutions

        logger.info(f"Fetching page {page}: {url}")
        html = get_html(url)
        if not html:
            logger.error(f"Failed to fetch page {page}")
            break

        products, has_next = parse_catalog_page(html)

        if not products:
            logger.info(f"No products on page {page} — stopping")
            break

        all_products.extend(products)
        logger.info(f"Total so far: {len(all_products)} products")

        if not has_next or len(products) < page_size:
            logger.info("Reached last page")
            break

        page += 1
        start += page_size
        time.sleep(2)  # Polite delay

    return all_products


def main():
    output_path = Path("data/shl_catalog.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SHL Catalog Scraper — Starting")
    logger.info("=" * 60)

    # Step 1: Get all product listings
    products = scrape_all_pages()
    logger.info(f"Scraped {len(products)} product listings")

    if not products:
        logger.error("No products scraped! Check the website structure.")
        sys.exit(1)

    # Step 2: Enrich each product
    enriched = []
    for i, product in enumerate(products):
        logger.info(f"Enriching {i+1}/{len(products)}: {product['name']}")
        enriched_product = enrich_product(product)
        enriched.append(enriched_product)
        time.sleep(1.5)  # Polite delay between product pages

    # Step 3: Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Saved {len(enriched)} products to {output_path}")
    logger.info("Done! Run 'python main.py' to start the server.")


if __name__ == "__main__":
    main()
