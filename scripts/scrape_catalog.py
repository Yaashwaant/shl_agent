"""
SHL Assessment Catalog Scraper
Scrapes the SHL product catalog and persists data as JSON + builds ChromaDB vector index.
"""
import json
import time
import logging
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOG_URL = f"{BASE_URL}/solutions/products/product-catalog/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# SHL test type codes
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgment",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def fetch_page(url: str, retries: int = 3, delay: float = 2.0) -> Optional[str]:
    """Fetch a page with retries."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    return None


def parse_catalog_listing(html: str) -> List[Dict]:
    """Parse the main catalog listing page for all product links."""
    soup = BeautifulSoup(html, "html.parser")
    products = []

    # SHL catalog uses a table or grid — find product rows
    # Try table approach first
    rows = soup.select("table tbody tr")
    if not rows:
        # Try card/grid approach
        rows = soup.select(".product-catalogue-training-calendar__row, [data-course-id]")

    logger.info(f"Found {len(rows)} product rows on listing page")

    for row in rows:
        try:
            # Extract name and URL
            link = row.select_one("a[href]")
            if not link:
                continue
            name = link.get_text(strip=True)
            href = link.get("href", "")
            if not href.startswith("http"):
                href = BASE_URL + href

            # Extract test type letters
            type_cells = row.select("td")
            test_types = []
            for cell in type_cells:
                text = cell.get_text(strip=True)
                if text in TEST_TYPE_MAP:
                    test_types.append(text)

            # Try to get duration from row
            duration = ""
            for cell in type_cells:
                t = cell.get_text(strip=True)
                if "min" in t.lower() or re.match(r"\d+", t):
                    duration = t
                    break

            products.append({
                "name": name,
                "url": href,
                "test_types": test_types,
                "duration": duration,
            })
        except Exception as e:
            logger.debug(f"Error parsing row: {e}")

    return products


def parse_product_page(html: str, url: str) -> Dict:
    """Parse an individual product page for detailed metadata."""
    soup = BeautifulSoup(html, "html.parser")
    data = {"url": url}

    # Title / name
    title_el = soup.select_one("h1, .product-title, [class*='title']")
    data["name"] = title_el.get_text(strip=True) if title_el else ""

    # Description
    desc_el = soup.select_one(
        ".product-description, [class*='description'], [class*='intro'], .hero-description, p"
    )
    data["description"] = desc_el.get_text(strip=True) if desc_el else ""

    # Duration — look for patterns like "25 minutes", "Variable"
    full_text = soup.get_text(" ", strip=True)
    dur_match = re.search(r"(\d+)\s*(?:–|-|to)?\s*(\d+)?\s*minutes?", full_text, re.IGNORECASE)
    if dur_match:
        data["duration"] = dur_match.group(0).strip()
    elif "variable" in full_text.lower():
        data["duration"] = "Variable"
    else:
        data["duration"] = ""

    # Remote/Supervised flags
    data["remote_testing"] = "yes" if re.search(r"remote\s*testing.*yes", full_text, re.IGNORECASE) else "unknown"
    data["adaptive_irt"] = "yes" if re.search(r"adaptive.*yes|irt.*yes", full_text, re.IGNORECASE) else "unknown"

    # Languages
    lang_section = soup.find(string=re.compile(r"languages?", re.IGNORECASE))
    if lang_section:
        parent = lang_section.find_parent()
        if parent:
            langs_text = parent.get_text(strip=True)
            data["languages"] = langs_text
        else:
            data["languages"] = ""
    else:
        data["languages"] = ""

    # Test type codes from page
    type_codes = re.findall(r"\b([ABCDEKPS])\b\s*[-–]\s*(?:Ability|Biodata|Competenc|Development|Assessment|Knowledge|Personality|Simulation)", full_text)
    data["test_types"] = list(set(type_codes))

    return data


def scrape_catalog(output_path: str = "data/shl_catalog.json") -> List[Dict]:
    """Main scraping routine."""
    logger.info(f"Starting SHL catalog scrape from {CATALOG_URL}")

    # Step 1: Get listing page (try with pagination)
    all_products = []
    page = 1
    while True:
        url = f"{CATALOG_URL}?start={(page-1)*12}" if page > 1 else CATALOG_URL
        logger.info(f"Fetching catalog page {page}: {url}")
        html = fetch_page(url)
        if not html:
            break

        soup = BeautifulSoup(html, "html.parser")

        # Check for "Individual Test Solutions" section specifically
        products_on_page = []

        # SHL catalog table structure
        table = soup.select_one("table.product-catalogue__table, table")
        if table:
            rows = table.select("tbody tr")
            for row in rows:
                cells = row.select("td")
                link = row.select_one("a[href]")
                if not link:
                    continue
                name = link.get_text(strip=True)
                href = link.get("href", "")
                if not href.startswith("http"):
                    href = BASE_URL + href

                # Parse test type checkmarks — SHL uses ✓ or filled circles
                test_types = []
                # Columns typically: Name, A, B, C, D, E, K, P, S, Duration, Remote, Adaptive
                type_order = ["A", "B", "C", "D", "E", "K", "P", "S"]
                for i, code in enumerate(type_order):
                    if i + 1 < len(cells):
                        cell_text = cells[i + 1].get_text(strip=True)
                        img = cells[i + 1].select_one("img, svg, [class*='check'], [class*='yes']")
                        if img or cell_text in ["●", "✓", "✔", "1"]:
                            test_types.append(code)

                # Duration
                duration = cells[-3].get_text(strip=True) if len(cells) >= 3 else ""

                products_on_page.append({
                    "name": name,
                    "url": href,
                    "test_types": test_types,
                    "duration": duration,
                    "description": "",
                    "languages": "",
                    "remote_testing": "unknown",
                    "adaptive_irt": "unknown",
                })

        if not products_on_page:
            # Try alternative selectors
            for link in soup.select("a[href*='/product-catalog/view/'], a[href*='/products/product-catalog/view/']"):
                href = link.get("href", "")
                if not href.startswith("http"):
                    href = BASE_URL + href
                name = link.get_text(strip=True)
                if name and href:
                    products_on_page.append({
                        "name": name,
                        "url": href,
                        "test_types": [],
                        "duration": "",
                        "description": "",
                        "languages": "",
                        "remote_testing": "unknown",
                        "adaptive_irt": "unknown",
                    })

        if not products_on_page:
            logger.info("No more products found — stopping pagination")
            break

        all_products.extend(products_on_page)
        logger.info(f"Page {page}: found {len(products_on_page)} products (total: {len(all_products)})")

        # Check for next page
        next_link = soup.select_one("a[rel='next'], .pagination__next, a[aria-label='Next']")
        if not next_link:
            # Try to check if current page result count < 12
            if len(products_on_page) < 12:
                break
            page += 1
        else:
            page += 1

        time.sleep(1.5)  # Polite crawling

    logger.info(f"Found {len(all_products)} total products from catalog listing")

    # Step 2: Fetch individual product pages for enrichment (batch to avoid hammering)
    enriched = []
    for i, product in enumerate(all_products):
        logger.info(f"Enriching {i+1}/{len(all_products)}: {product['name']}")
        detail_html = fetch_page(product["url"])
        if detail_html:
            detail = parse_product_page(detail_html, product["url"])
            # Merge: prefer listing data for test_types if available
            product.update({
                "description": detail.get("description", ""),
                "languages": detail.get("languages", ""),
                "remote_testing": detail.get("remote_testing", "unknown"),
                "adaptive_irt": detail.get("adaptive_irt", "unknown"),
            })
            if not product["test_types"] and detail.get("test_types"):
                product["test_types"] = detail["test_types"]
            if not product["duration"] and detail.get("duration"):
                product["duration"] = detail["duration"]

        enriched.append(product)
        time.sleep(1.0)  # Polite crawling

    # Save to JSON
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(enriched)} products to {output_path}")
    return enriched


if __name__ == "__main__":
    scrape_catalog()
