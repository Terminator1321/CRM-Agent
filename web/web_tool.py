"""
web/web_tool.py

Generic internet-access tools for the agent.

Five tools:
  - web_search          : Tavily (primary) / SearXNG fallback
  - web_fetch_page      : fetch one URL → readable text
  - web_crawl           : multi-page crawler
  - web_company_search  : find candidate official URLs for a company name
  - web_company_extract : deep extraction of email, phone, description,
                          social profiles from a confirmed company URL

web_company_extract improvements v2:
  ✅ Regex email scan  (plain-text mentions, not just mailto: links)
  ✅ Regex phone scan  (Indian +91, international, 10-digit formats)
  ✅ JSON-LD / schema.org structured data (modern company sites)
  ✅ WhatsApp wa.me link detection
  ✅ Social profile extraction (LinkedIn, Twitter/X, Instagram, YouTube, Facebook)
  ✅ Auto-crawl up to 3 subpages (Contact, About, Team) before giving up
  ✅ 2-3 sentence company description (meta → JSON-LD → first paragraph)
  ✅ Person / lead name extraction from schema.org Person objects
"""

import logging
import os
import re
import json
import time
from typing import Optional
from urllib.parse import urljoin, urlparse
import urllib.robotparser as robotparser

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

logger = logging.getLogger("web-tools")

# ---------------------------------------------------------------------------
# Tavily client
# ---------------------------------------------------------------------------
_TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
_tavily_client = None
if _TAVILY_API_KEY:
    try:
        from tavily import TavilyClient
        _tavily_client = TavilyClient(api_key=_TAVILY_API_KEY)
        logger.info("Tavily search client initialized.")
    except Exception as _e:
        logger.warning("Tavily import failed (%s) — falling back to SearXNG.", _e)
else:
    logger.info("TAVILY_API_KEY not set — using SearXNG.")

_SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8080")

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 MagmaAssistant/1.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_REQUEST_TIMEOUT    = 12
_MAX_SEARCH_RESULTS = 8
_MAX_PAGE_CHARS     = 8000
_MAX_CRAWL_PAGES    = 10
_CRAWL_PAGE_CHARS   = 2000

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE
)

_PHONE_RE = re.compile(
    r"(?:"
    r"\+91[\s\-]?\d{5}[\s\-]?\d{5}"
    r"|\+91[\s\-]?\d{10}"
    r"|\+\d{1,3}[\s\-]?\(?\d{1,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}"
    r"|0\d{2,4}[\s\-]\d{6,8}"
    r"|\b[6-9]\d{9}\b"
    r")"
)

_GENERIC_EMAIL_PREFIXES = {
    "noreply", "no-reply", "donotreply", "webmaster", "postmaster", "abuse",
    "mailer-daemon", "newsletter", "updates", "notifications"
}

_CONTACT_LINK_HINTS = (
    "contact", "reach us", "get in touch", "contact-us", "contactus",
    "about", "about-us", "aboutus", "team", "our-team", "ourteam",
    "connect", "reach", "talk to us"
)

_NON_OFFICIAL_HOSTS = {
    "wikipedia.org", "wikimedia.org", "linkedin.com", "facebook.com",
    "twitter.com", "x.com", "instagram.com", "youtube.com",
    "crunchbase.com", "bloomberg.com", "reuters.com", "indiamart.com",
    "glassdoor.com", "glassdoor.co.in", "zoominfo.com", "owler.com",
    "craft.co", "medium.com", "quora.com", "reddit.com", "yelp.com",
    "tradeindia.com", "justdial.com", "opencorporates.com", "dnb.com",
    "google.com", "bing.com", "duckduckgo.com", "britannica.com",
    "forbes.com", "investopedia.com", "wsj.com", "ambitionbox.com",
    "tracxn.com", "tofler.in", "vccircle.com", "moneycontrol.com",
}

_SOCIAL_PATTERNS = {
    "linkedin":  re.compile(r"linkedin\.com/company/", re.I),
    "twitter":   re.compile(r"(?:twitter|x)\.com/(?!share|intent)[a-zA-Z0-9_]{1,50}", re.I),
    "instagram": re.compile(r"instagram\.com/[a-zA-Z0-9_.]{1,50}", re.I),
    "youtube":   re.compile(r"youtube\.com/(?:c/|channel/|user/|@)[a-zA-Z0-9_\-]{1,100}", re.I),
    "facebook":  re.compile(r"facebook\.com/(?!sharer|share)[a-zA-Z0-9.]{1,100}", re.I),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_call(action, fn):
    try:
        return fn()
    except requests.exceptions.Timeout:
        return f"Timed out while trying to {action}."
    except requests.exceptions.RequestException as exc:
        return f"Could not {action}: {exc}"
    except Exception as exc:
        logger.exception("web tool failed: %s", action)
        return f"Could not {action}: {exc}"


def _clean_text(soup):
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]):
        tag.decompose()
    lines = [l.strip() for l in soup.get_text(separator="\n").splitlines()]
    return "\n".join(l for l in lines if l)


def _page_title(soup, fallback):
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    return fallback


def _allowed_by_robots(url):
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(_HEADERS["User-Agent"], url)
    except Exception:
        return True


def _is_non_official_host(netloc):
    host = netloc.lower()
    return any(host == h or host.endswith("." + h) for h in _NON_OFFICIAL_HOSTS)


def _fetch_soup(url):
    # 1. Attempt Tavily Extract to bypass bot protection and render JS
    if _tavily_client:
        try:
            res = _tavily_client.extract(urls=[url])
            results = res.get("results", [])
            if results and results[0].get("raw_content"):
                markdown = results[0]["raw_content"]
                # Convert markdown links [text](url) to HTML <a href="url">text</a>
                html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', markdown)
                # Ensure the text is parseable by beautifulsoup
                html_doc = f"<html><body><p>{html}</p></body></html>"
                return BeautifulSoup(html_doc, "html.parser")
        except Exception as e:
            logger.warning("Tavily extract failed for %s: %s", url, e)

    # 2. Fallback to standard requests if Tavily is unavailable or fails
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        if "html" not in ct and "text" not in ct:
            return None
        return BeautifulSoup(resp.text, "html.parser")
    except Exception:
        return None


def _extract_from_schema(item, emails, phones, socials, addresses):
    if not isinstance(item, dict):
        return
        
    # Address extraction
    if item.get("@type") == "PostalAddress":
        addr_parts = []
        for k in ("streetAddress", "addressLocality", "addressRegion", "postalCode", "addressCountry"):
            val = item.get(k)
            if val and isinstance(val, str):
                addr_parts.append(val)
        if addr_parts:
            full_addr = ", ".join(addr_parts)
            if full_addr not in addresses:
                addresses.append(full_addr)
    else:
        addr = item.get("address")
        if isinstance(addr, str) and addr not in addresses:
            addresses.append(addr)

    for field in ("email", "contactEmail"):
        v = item.get(field, "")
        if v and "@" in v and v.lower() not in emails:
            emails.append(v.lower())
    for field in ("telephone", "faxNumber", "contactTelephone"):
        v = item.get(field, "")
        if v and v not in phones:
            phones.append(v)
    same_as = item.get("sameAs", [])
    if isinstance(same_as, str):
        same_as = [same_as]
    for url in same_as:
        for platform, pattern in _SOCIAL_PATTERNS.items():
            if platform not in socials and pattern.search(url or ""):
                socials[platform] = url.split("?")[0].rstrip("/")
    for key in ("contactPoint", "address", "founder", "employee", "member"):
        child = item.get(key)
        if isinstance(child, dict):
            _extract_from_schema(child, emails, phones, socials, addresses)
        elif isinstance(child, list):
            for c in child:
                _extract_from_schema(c, emails, phones, socials, addresses)


def _extract_contacts_from_soup(soup):
    emails, phones, whatsapp, socials, addresses = [], [], [], {}, []
    page_text = soup.get_text(" ", strip=True)

    # 1. mailto:/tel:/wa.me links
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().startswith("mailto:"):
            addr = href.split(":", 1)[1].split("?")[0].strip().lower()
            local = addr.split("@")[0]
            if addr and local not in _GENERIC_EMAIL_PREFIXES and addr not in emails:
                emails.append(addr)
        elif href.lower().startswith("tel:"):
            num = re.sub(r"\s", "", href.split(":", 1)[1].strip())
            if num and num not in phones:
                phones.append(num)
        elif "wa.me" in href.lower() or "api.whatsapp.com/send" in href.lower():
            num = re.sub(r"\D", "", href)[-12:]
            if num and ("+" + num) not in whatsapp:
                whatsapp.append("+" + num)
        for platform, pattern in _SOCIAL_PATTERNS.items():
            if platform not in socials and pattern.search(href):
                full = urljoin("https://x.com", href) if href.startswith("/") else href
                socials[platform] = full.split("?")[0].rstrip("/")

    # 2. Regex scan of page text
    for match in _EMAIL_RE.finditer(page_text):
        addr = match.group().lower()
        local = addr.split("@")[0]
        if local not in _GENERIC_EMAIL_PREFIXES and addr not in emails:
            emails.append(addr)
    for match in _PHONE_RE.finditer(page_text):
        num = re.sub(r"[\s\-]", "", match.group())
        if num not in phones and len(num) >= 7:
            phones.append(num)

    # 3. JSON-LD / schema.org
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            items = data if isinstance(data, list) else [data]
            for item in items:
                _extract_from_schema(item, emails, phones, socials, addresses)
        except Exception:
            pass

    # HTML <address> tags
    for tag in soup.find_all("address"):
        text = tag.get_text(", ", strip=True)
        if text and len(text) < 150 and text not in addresses:
            addresses.append(text)

    return {
        "emails":   list(dict.fromkeys(emails))[:5],
        "phones":   list(dict.fromkeys(phones))[:5],
        "whatsapp": list(dict.fromkeys(whatsapp))[:3],
        "socials":  socials,
        "addresses": list(dict.fromkeys(addresses))[:3]
    }


def _extract_description(soup):
    """Gathers up to ~1000 chars of descriptive text from the page.
    The LLM will summarize this into a concise 2-3 sentences."""
    snippets = []
    
    # 1. Meta descriptions
    for selector in [
        {"property": "og:description"},
        {"name": "description"},
    ]:
        tag = soup.find("meta", attrs=selector)
        if tag and tag.get("content", "").strip():
            desc = tag["content"].strip()
            if len(desc) > 40 and desc not in snippets:
                snippets.append(desc)
                
    # 2. JSON-LD description
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            items = data if isinstance(data, list) else [data]
            for item in items:
                desc = item.get("description", "")
                if desc and len(desc) > 40 and desc not in snippets:
                    snippets.append(desc)
        except Exception:
            pass
            
    # 3. Substantial paragraphs
    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True)
        if len(text) > 80 and text not in snippets:
            snippets.append(text)
        if len(snippets) >= 5:  # Cap at 5 snippets
            break
            
    return " | ".join(snippets)[:1000]


def _extract_person_names(soup):
    names = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict) and item.get("@type") in ("Person", "Employee"):
                    name = item.get("name", "").strip()
                    if name and name not in names:
                        names.append(name)
        except Exception:
            pass
    return names[:5]


def _find_subpage_links(soup, base_url, max_links=5):
    base_domain = urlparse(base_url).netloc
    found, seen = [], set()
    for a in soup.find_all("a", href=True):
        label = (a.get_text() or "").strip().lower()
        href = a["href"].lower()
        if any(hint in label or hint in href for hint in _CONTACT_LINK_HINTS):
            full_url = urljoin(base_url, a["href"]).split("#")[0]
            if (full_url.startswith("http")
                    and urlparse(full_url).netloc == base_domain
                    and full_url not in seen
                    and full_url != base_url):
                seen.add(full_url)
                found.append(full_url)
                if len(found) >= max_links:
                    break
    return found


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Searches the public internet for up-to-date information — news, product
    specs, company lookups, documentation, etc. Uses Tavily (AI-optimized) as
    primary; falls back to local SearXNG. Returns numbered results with title,
    URL, snippet. Use web_fetch_page afterwards for full page content."""
    n = max(1, min(int(max_results or 5), _MAX_SEARCH_RESULTS))

    def run():
        if _tavily_client:
            try:
                resp = _tavily_client.search(query=query, max_results=n,
                    include_answer=True, include_raw_content=False, search_depth="advanced")
                results = resp.get("results", [])
                answer = resp.get("answer", "")
                if not results:
                    return f"No web results found for '{query}'."
                lines = [f"Web search results for '{query}':"]
                if answer:
                    lines.append(f"\n📋 Direct answer: {answer}\n")
                for i, r in enumerate(results, 1):
                    lines.append(f"{i}. {r.get('title','(no title)')}\n   {r.get('url','')}\n   {(r.get('content') or '').strip()[:400]}")
                return "\n".join(lines)
            except Exception as exc:
                logger.warning("Tavily failed (%s) — falling back to SearXNG.", exc)
        try:
            resp = requests.get(f"{_SEARXNG_URL}/search",
                params={"q": query, "format": "json"}, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            results = resp.json().get("results", [])[:n]
        except Exception:
            return f"Web search unavailable for '{query}'."
        if not results:
            return f"No web results found for '{query}'."
        lines = [f"Web search results for '{query}' (SearXNG):"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title','(no title)')}\n   {r.get('url','')}\n   {(r.get('content') or '').strip()}")
        return "\n".join(lines)

    return _safe_call(f"search the web for '{query}'", run)


@tool
def web_fetch_page(url: str, max_chars: int = 4000) -> str:
    """Fetches one specific web page and returns its readable text (scripts/styles
    stripped), truncated to max_chars. Provide a full http(s) URL."""
    if not url or not re.match(r"^https?://", url.strip(), re.IGNORECASE):
        return "Please provide a full http:// or https:// URL."
    limit = max(500, min(int(max_chars or 4000), _MAX_PAGE_CHARS))
    url = url.strip()

    def run():
        resp = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        if "html" not in ct and "text" not in ct:
            return f"'{url}' is not a readable page (Content-Type: {ct})."
        soup = BeautifulSoup(resp.text, "html.parser")
        title = _page_title(soup, url)
        text = _clean_text(soup)
        if not text:
            return f"'{url}' loaded but had no readable text."
        return f"Content of '{title}' ({url}):\n\n{text[:limit]}{'... [truncated]' if len(text) > limit else ''}"

    return _safe_call(f"fetch '{url}'", run)


@tool
def web_crawl(start_url: str, max_pages: int = 5, same_domain_only: bool = True) -> str:
    """Crawls outward from start_url, following links to gather info from several
    related pages. Fetches up to max_pages total, respects robots.txt. For a
    single known page use web_fetch_page instead."""
    if not start_url or not re.match(r"^https?://", start_url.strip(), re.IGNORECASE):
        return "Please provide a full http:// or https:// starting URL."
    start_url = start_url.strip()
    n = max(1, min(int(max_pages or 5), _MAX_CRAWL_PAGES))

    def run():
        start_domain = urlparse(start_url).netloc
        visited, queue, pages_out = set(), [start_url], []
        while queue and len(visited) < n:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            if not _allowed_by_robots(url):
                pages_out.append(f"--- {url} ---\n[Skipped: robots.txt]")
                continue
            try:
                resp = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
                resp.raise_for_status()
            except requests.exceptions.RequestException as exc:
                pages_out.append(f"--- {url} ---\n[Error: {exc}]")
                continue
            ct = resp.headers.get("Content-Type", "")
            if "html" not in ct:
                pages_out.append(f"--- {url} ---\n[Skipped: not HTML]")
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            if len(visited) < n:
                for a in soup.find_all("a", href=True):
                    link = urljoin(url, a["href"]).split("#")[0]
                    if not link.startswith("http"):
                        continue
                    if same_domain_only and urlparse(link).netloc != start_domain:
                        continue
                    if link not in visited and link not in queue:
                        queue.append(link)
            title = _page_title(soup, url)
            text = _clean_text(soup)
            snippet = text[:_CRAWL_PAGE_CHARS] + ("\n... [truncated]" if len(text) > _CRAWL_PAGE_CHARS else "")
            pages_out.append(f"--- {title} ({url}) ---\n{snippet or '[no readable text]'}")
            time.sleep(0.3)
        if not pages_out:
            return f"Could not gather any content from '{start_url}'."
        return f"Crawled {len(visited)} page(s) from '{start_url}':\n\n" + "\n\n".join(pages_out)

    return _safe_call(f"crawl '{start_url}'", run)


@tool
def web_company_search(company_name: str, search_hint: Optional[str] = None) -> str:
    """Find candidate official websites for a company by name.
    Returns a ranked shortlist (directories/social sites filtered out).
    ALWAYS present options to the user for confirmation before calling web_company_extract.
    Use search_hint to disambiguate (e.g. 'Mumbai', 'software', 'textile')."""
    name = (company_name or "").strip()
    if not name:
        return "Please provide a company name."
    hint = (search_hint or "").strip()
    query = f"{name} {hint} official website".strip()

    results = []
    if _tavily_client:
        try:
            resp = _tavily_client.search(query=query, max_results=7, search_depth="basic")
            results = resp.get("results", [])
        except Exception:
            pass
    if not results:
        try:
            resp = requests.get(f"{_SEARXNG_URL}/search",
                params={"q": query, "format": "json"}, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            results = resp.json().get("results", [])[:7]
        except Exception:
            return "Web search unavailable."
    if not results:
        return f"No websites found for '{name}'."

    lines = [f"Top candidate websites for '{name}':"]
    count = 0
    for r in results:
        url = r.get("url", "")
        if _is_non_official_host(urlparse(url).netloc):
            continue
        count += 1
        lines.append(f"{count}. {r.get('title','')}\n   URL: {url}\n   {(r.get('content') or '').strip()[:150]}...")
        if count >= 5:
            break
    if count == 0:
        return f"Only directory/social results found for '{name}'. Add an industry or city hint."
    lines.append("\nConfirm the correct URL with the user, then call `web_company_extract` on it.")
    return "\n".join(lines)


@tool
def web_company_extract(url: str, company_name: Optional[str] = None) -> str:
    """Deep extraction of contact details and company profile from a confirmed website.

    Extracts:
    - Email (mailto: links + plain-text regex scan + JSON-LD schema.org)
    - Phone (tel: links + Indian/international phone regex + JSON-LD)
    - WhatsApp (wa.me links)
    - Social profiles (LinkedIn, Twitter/X, Instagram, YouTube, Facebook)
    - Company description (2-3 sentences: meta tags → JSON-LD → first paragraph)
    - Person/lead names (schema.org Person objects)

    Auto-crawls up to 3 subpages (Contact, About, Team) if homepage yields nothing.
    Call ONLY after user has confirmed the correct URL from web_company_search."""

    url = (url or "").strip()
    if not url or not re.match(r"^https?://", url, re.IGNORECASE):
        return "Please provide a valid http:// or https:// URL."

    def run():
        soup = _fetch_soup(url)
        if not soup:
            return f"Could not reach or parse {url}."

        contacts = _extract_contacts_from_soup(soup)
        description = _extract_description(soup)
        person_names = _extract_person_names(soup)
        pages_tried = [url]

        # Auto-crawl subpages if contacts are incomplete
        if not contacts["emails"] or not contacts["phones"]:
            subpages = _find_subpage_links(soup, url, max_links=5)
            for sub_url in subpages[:3]:
                if sub_url in pages_tried:
                    continue
                pages_tried.append(sub_url)
                sub_soup = _fetch_soup(sub_url)
                if not sub_soup:
                    continue
                sub_contacts = _extract_contacts_from_soup(sub_soup)
                for key in ("emails", "phones", "whatsapp", "addresses"):
                    for val in sub_contacts[key]:
                        if val not in contacts[key]:
                            contacts[key].append(val)
                for platform, link in sub_contacts["socials"].items():
                    if platform not in contacts["socials"]:
                        contacts["socials"][platform] = link
                if not description:
                    description = _extract_description(sub_soup)
                if not person_names:
                    person_names = _extract_person_names(sub_soup)
                if contacts["emails"] and contacts["phones"]:
                    break
                time.sleep(0.3)

        # Fallback: If no email found on the official site, execute a broad web search automatically
        fallback_emails = []
        if not contacts["emails"]:
            search_query = f"{company_name or urlparse(url).netloc.replace('www.', '')} contact email address"
            try:
                # Use Tavily if available
                if _tavily_client:
                    resp = _tavily_client.search(query=search_query, max_results=10)
                    results = resp.get("results", [])
                else:
                    resp = requests.get(f"{_SEARXNG_URL}/search", params={"q": search_query, "format": "json"}, timeout=_REQUEST_TIMEOUT)
                    results = resp.json().get("results", [])
                    
                # Scan search snippets for emails
                combined_text = " ".join([r.get("content", "") + " " + r.get("title", "") for r in results])
                for match in _EMAIL_RE.finditer(combined_text):
                    addr = match.group().lower()
                    local = addr.split("@")[0]
                    if local not in _GENERIC_EMAIL_PREFIXES and addr not in fallback_emails:
                        fallback_emails.append(addr)
            except Exception as e:
                logger.warning(f"Fallback email search failed: {e}")

        # Build output
        lines = [f"Contact extraction for: {url}",
                 f"Pages scanned: {', '.join(pages_tried)}\n"]

        primary_email = "NOT FOUND"
        if contacts["emails"]:
            primary_email = contacts["emails"][0]
            lines.append(f"Email:       {primary_email}")
            if len(contacts["emails"]) > 1:
                lines.append(f"  (also: {', '.join(contacts['emails'][1:])})")
        elif fallback_emails:
            primary_email = fallback_emails[0]
            lines.append(f"Email:       {primary_email} (found via broad web search)")
            if len(fallback_emails) > 1:
                lines.append(f"  (also: {', '.join(fallback_emails[1:])})")
        else:
            lines.append(f"Email:       NOT FOUND")

        primary_phone = contacts["phones"][0] if contacts["phones"] else "NOT FOUND"
        lines.append(f"Phone:       {primary_phone}")
        if len(contacts["phones"]) > 1:
            lines.append(f"  (also: {', '.join(contacts['phones'][1:])})")

        if contacts["whatsapp"]:
            lines.append(f"WhatsApp:    {contacts['whatsapp'][0]}")

        if contacts["addresses"]:
            lines.append(f"\nAddress:     {contacts['addresses'][0]}")
            if len(contacts["addresses"]) > 1:
                lines.append(f"  (also: {', '.join(contacts['addresses'][1:])})")

        if contacts["socials"]:
            lines.append("\nSocial Profiles:")
            for platform, link in contacts["socials"].items():
                lines.append(f"  {platform.capitalize()}: {link}")

        lines.append(f"\nDescription: {description if description else 'NOT FOUND'}")

        if person_names:
            lines.append(f"\nPerson(s) found: {', '.join(person_names)}")

        if primary_email == "NOT FOUND" and primary_phone == "NOT FOUND":
            lines.append(
                "\n⚠️  No direct contact details found. Company may use a contact form. "
                "Try web_fetch_page on their /contact page."
            )

        return "\n".join(lines)

    return _safe_call(f"extract contacts from '{url}'", run)


WEB_TOOLS = [web_search, web_fetch_page, web_crawl, web_company_search, web_company_extract]
