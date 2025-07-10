import os
import argparse
import json
import asyncio
import uuid
import re
import logging
import pickle
from pathlib import Path
from urllib.parse import urljoin, urlparse
from datetime import datetime
from typing import Dict, List, Any
import openai
from markdownify import markdownify as md
from dotenv import load_dotenv
from pydantic import BaseModel
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from bs4.element import Tag, PageElement
import bm25s
from research_agent import enrich_description

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SummaryAndKeywords(BaseModel):
    """Structured response for description and keywords generation"""

    summary: str
    keywords: List[str]


class PageEvaluation(BaseModel):
    """Structured response for page content evaluation"""

    keep_page: bool
    reason: str
    summary: str = ""


class EuroBioImagingWebscraper:
    def __init__(
        self,
        base_url: str = "https://www.eurobioimaging.eu",
        output_dir: str = "eubio_website",
    ):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.html_dir = self.output_dir / "html"
        self.markdown_dir = self.output_dir / "markdown"
        self.visited_urls = set()
        self.to_visit = []
        self.pages_data = []
        self.pages_fetched = 0
        self.pages_kept = 0
        self.pages_discarded = 0

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.html_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)

        # Configure request headers
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; EuroBioImaging-Indexer/1.0; Research purposes)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        # URL patterns to exclude
        self.exclude_patterns = [
            r"\.pdf$",
            r"\.doc$",
            r"\.docx$",
            r"\.ppt$",
            r"\.pptx$",
            r"\.xls$",
            r"\.xlsx$",
            r"\.zip$",
            r"\.tar$",
            r"\.gz$",
            r"\.jpg$",
            r"\.jpeg$",
            r"\.png$",
            r"\.gif$",
            r"\.css$",
            r"\.js$",
            r"\.ico$",
            r"\.svg$",
            r"\.woff$",
            r"\.ttf$",
            r"/wp-admin/",
            r"/wp-content/uploads/",
            r"/feed/",
            r"/rss/",
            r"#",
            r"mailto:",
            r"tel:",
            r"javascript:",
        ]

        # Domains to stay within
        self.allowed_domains = ["eurobioimaging.eu", "www.eurobioimaging.eu"]

    def should_crawl_url(self, url: str) -> bool:
        """Check if URL should be crawled"""
        if not url or url in self.visited_urls:
            return False

        # Check URL patterns to exclude
        for pattern in self.exclude_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return False

        # Check if within allowed domains
        parsed = urlparse(url)
        if parsed.netloc and parsed.netloc not in self.allowed_domains:
            return False

        return True

    def normalize_url(self, url: str, base_url: str) -> str | None:
        """Normalize and validate URL"""
        try:
            # Join relative URLs with base URL
            full_url = urljoin(base_url, url)

            # Parse and clean URL
            parsed = urlparse(full_url)

            # Remove fragment (anchor)
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                clean_url += f"?{parsed.query}"

            return clean_url
        except Exception as e:
            logger.warning(f"Error normalizing URL {url}: {e}")
            return None

    def generate_page_id(self, url: str) -> str:
        """Generate unique short page ID"""
        return str(uuid.uuid4())[:8]

    def clean_html_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Clean HTML content by removing unnecessary elements"""
        # Remove script and style elements
        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        # Remove navigation, footer, sidebar elements
        for selector in [
            "nav",
            "footer",
            ".nav",
            ".navigation",
            ".sidebar",
            ".menu",
            ".header",
            ".breadcrumb",
            ".social-media",
            ".cookie-notice",
            "#header",
            "#footer",
            "#sidebar",
            "#navigation",
        ]:
            for element in soup.select(selector):
                element.decompose()

        # Remove WordPress-specific elements
        for class_name in ["wp-admin-bar", "wp-toolbar", "admin-bar"]:
            for element in soup.find_all(class_=class_name):
                element.decompose()

        return soup

    def extract_page_content(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract key content from page"""
        content = {"title": "", "description": "", "main_content": "", "headings": []}

        # Extract title
        title_tag = soup.find("title")
        if title_tag:
            content["title"] = title_tag.get_text().strip()

        # Extract meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc:
            assert isinstance(meta_desc, Tag), "meta_desc should be a Tag"
            meta_desc_content = meta_desc.get("content", "")
            assert isinstance(
                meta_desc_content, str
            ), "meta_desc_content should be a string"
            content["description"] = meta_desc_content.strip()

        # Extract main content areas
        main_selectors = [
            "main",
            ".main",
            "#main",
            ".content",
            "#content",
            ".post-content",
            ".entry-content",
            "article",
            ".article",
        ]

        main_content = None
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break

        # If no main content found, use body
        if not main_content:
            main_content = soup.find("body")

        if main_content:
            # Clean the main content
            main_content = self.clean_html_content(
                BeautifulSoup(str(main_content), "html.parser")
            )
            content["main_content"] = main_content.get_text(" ", strip=True)

            # Extract headings
            for heading in main_content.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
                heading_text = heading.get_text().strip()
                if heading_text:
                    assert not isinstance(
                        heading, PageElement
                    ), "heading should not be a PageElement"
                    content["headings"].append(
                        {"level": heading.name, "text": heading_text}
                    )

        return content

    def extract_links(self, soup: BeautifulSoup, current_url: str) -> List[str]:
        """Extract all valid links from page"""
        links = []

        for link in soup.find_all("a", href=True):
            assert isinstance(link, Tag), "link should be a Tag"
            href = link.get("href")
            assert isinstance(href, str), "href should be a string"
            if href:
                normalized_url = self.normalize_url(href, current_url)
                if normalized_url and self.should_crawl_url(normalized_url):
                    links.append(normalized_url)

        return list(set(links))  # Remove duplicates

    def clean_markdown_content(self, markdown_content: str) -> str:
        """Clean up markdown content by removing excessive whitespace and formatting issues"""
        # Split into lines
        lines = markdown_content.split("\n")

        # Remove leading and trailing empty lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        # Reduce multiple consecutive empty lines to at most 2
        cleaned_lines = []
        empty_line_count = 0

        for line in lines:
            if not line.strip():
                empty_line_count += 1
                if empty_line_count <= 2:  # Allow max 2 consecutive empty lines
                    cleaned_lines.append(line)
            else:
                empty_line_count = 0
                cleaned_lines.append(line)

        # Join back and clean up common markdown issues
        cleaned_content = "\n".join(cleaned_lines)

        # Remove excessive spaces
        cleaned_content = re.sub(r" +", " ", cleaned_content)

        # Clean up image references (remove broken image syntax)
        cleaned_content = re.sub(r"!\[\]\([^)]*\)\s*", "", cleaned_content)

        # Clean up empty links
        cleaned_content = re.sub(r"\[\]\([^)]*\)", "", cleaned_content)

        # Remove standalone copyright symbols and fix spacing
        cleaned_content = re.sub(r"\n©\s*\n", "\n\n", cleaned_content)

        # Fix multiple consecutive spaces in lines
        lines = cleaned_content.split("\n")
        cleaned_lines = [re.sub(r"\s+", " ", line).strip() for line in lines]

        return "\n".join(cleaned_lines)

    async def fetch_page(
        self, session: aiohttp.ClientSession, url: str
    ) -> Dict[str, Any] | None:
        """Fetch and process a single page"""
        try:
            logger.info(f"Fetching: {url}")

            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None

                content_type = response.headers.get("content-type", "").lower()
                if "text/html" not in content_type:
                    logger.info(f"Skipping non-HTML content: {url}")
                    return None

                html_content = await response.text()

                # Parse HTML
                soup = BeautifulSoup(html_content, "html.parser")

                # Generate page ID
                page_id = self.generate_page_id(url)

                # Extract content
                page_content = self.extract_page_content(soup)

                # Extract links for further crawling
                links = self.extract_links(soup, url)

                # Save raw HTML
                html_file = self.html_dir / f"{page_id}.html"
                async with aiofiles.open(html_file, "w", encoding="utf-8") as f:
                    await f.write(html_content)

                # Convert to markdown
                clean_soup = self.clean_html_content(soup)
                markdown_content = md(str(clean_soup), heading_style="ATX")

                # Clean up the markdown content
                markdown_content = self.clean_markdown_content(markdown_content)

                # Save markdown
                markdown_file = self.markdown_dir / f"{page_id}.md"
                async with aiofiles.open(markdown_file, "w", encoding="utf-8") as f:
                    await f.write(markdown_content)

                # Prepare page data
                page_data = {
                    "id": page_id,
                    "url": url,
                    "title": page_content["title"],
                    "meta_description": page_content["description"],
                    "main_content": page_content["main_content"][
                        :5000
                    ],  # Limit content length
                    "full_content": page_content["main_content"],
                    "headings": page_content["headings"],
                    "markdown_file": str(markdown_file.name),
                    "html_file": str(html_file.name),
                    "crawled_at": datetime.now().isoformat(),
                    "links_found": len(links),
                }

                # Add new links to crawling queue
                for link in links:
                    if link not in self.visited_urls and link not in self.to_visit:
                        self.to_visit.append(link)

                return page_data

        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    async def evaluate_page_content(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate if page content is worth keeping and generate summary"""
        try:
            content_to_evaluate = page_data.get("main_content", "")[
                :3000
            ]  # Limit input length

            if not content_to_evaluate.strip():
                return {
                    "keep_page": False,
                    "reason": "No meaningful content",
                    "summary": "",
                }

            prompt = f"""
            Evaluate webpage for LLM search index table of contents. Pack maximum critical info to differentiate content.
            
            Rules:
            - Skip obvious words (page, website, Euro-BioImaging) unless differentiating  
            - Include specific services, procedures, contact info, unique offerings
            - Focus on actionable content and distinct value
            - Maximum accuracy in minimal space
            
            Provide:
            1. keep_page: true/false
            2. reason: Brief explanation (one sentence)
            3. summary: If keeping, concise description (max 100 chars, title excluded, pack critical info)
            
            Title: {page_data.get('title', 'N/A')}
            Content: {content_to_evaluate}
            
            KEEP if contains: services, facilities, access procedures, technologies, training, news, contact info, tools
            DISCARD if contains: navigation menus, errors, boilerplate, minimal content
            
            Good examples:
            "Application forms, funding requirements, eligibility criteria"
            "Workshop schedules: STED, STORM, sample prep, Sept 2024"
            "Node contacts: Prague, Brno facilities, email, phone numbers"
            "Grant deadlines, submission process, evaluation criteria"
            
            Bad examples (too generic):
            "Information about services" "Training opportunities" "Contact details"
            """

            response = await client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format=PageEvaluation,
                max_tokens=300,
                temperature=0.3,
            )

            result = response.choices[0].message.parsed

            assert result is not None, "Result should not be None"

            return {
                "keep_page": result.keep_page,
                "reason": result.reason,
                "summary": result.summary if result.keep_page else "",
            }

        except Exception as e:
            logger.warning(
                f"Failed to evaluate content for {page_data.get('url', 'unknown')}: {e}"
            )
            # On error, keep the page but with minimal summary
            return {
                "keep_page": True,
                "reason": "Evaluation failed - keeping by default",
                "summary": page_data.get("meta_description", "No summary available"),
            }

    async def crawl_website(self, max_pages: int = 50, delay: float = 1.0):
        """Main crawling function"""
        logger.info(f"Starting crawl of {self.base_url}")

        # Start with base URL
        self.to_visit.append(self.base_url)

        # Track statistics
        self.pages_fetched = 0
        self.pages_kept = 0
        self.pages_discarded = 0

        # Create aiohttp session
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=30)

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout, headers=self.headers
        ) as session:

            while self.to_visit and self.pages_kept < max_pages:
                # Get next URL to visit
                current_url = self.to_visit.pop(0)

                if current_url in self.visited_urls:
                    continue

                self.visited_urls.add(current_url)

                # Fetch page
                page_data = await self.fetch_page(session, current_url)

                if page_data:
                    self.pages_fetched += 1

                    # Evaluate page content and decide whether to keep it
                    logger.info(f"Evaluating content for: {page_data['title']}")
                    evaluation = await self.evaluate_page_content(page_data)

                    if evaluation["keep_page"]:
                        page_data["ai_summary"] = evaluation["summary"]
                        page_data["evaluation_reason"] = evaluation["reason"]
                        self.pages_data.append(page_data)
                        self.pages_kept += 1
                        logger.info(
                            f"✅ Kept page: {page_data['title']} - {evaluation['reason']}"
                        )
                    else:
                        self.pages_discarded += 1
                        logger.info(
                            f"❌ Discarded page: {page_data['title']} - {evaluation['reason']}"
                        )

                    logger.info(
                        f"Stats: {self.pages_kept} kept, {self.pages_discarded} discarded, {len(self.to_visit)} in queue"
                    )

                # Rate limiting
                await asyncio.sleep(delay)

        logger.info(
            f"Crawling completed. Fetched: {self.pages_fetched}, Kept: {self.pages_kept}, Discarded: {self.pages_discarded}"
        )

    def classify_page_type(self, page_data: Dict[str, Any]) -> str:
        """Classify the type of page based on URL and content"""
        url = page_data.get("url", "").lower()
        title = page_data.get("title", "").lower()

        # Page type classification
        if "/about" in url or "about" in title:
            return "about"
        elif "/service" in url or "service" in title:
            return "services"
        elif "/training" in url or "training" in title:
            return "training"
        elif "/access" in url or "access" in title:
            return "access"
        elif "/news" in url or "/post" in url or "news" in title:
            return "news"
        elif "/contact" in url or "contact" in title:
            return "contact"
        elif "/node" in url or "node" in title:
            return "nodes"
        elif "/technology" in url or "technology" in title:
            return "technology"
        elif url.endswith("/") or "home" in url:
            return "homepage"
        else:
            return "general"


def generate_short_uuid():
    """Generate a short UUID for indexing"""
    return str(uuid.uuid4())[:8]


def get_description_text(entry: Dict[str, Any]) -> str:
    """Extract description text from an entry, handling both description and long_description fields"""
    desc = (entry.get("description") or "").strip()
    long_desc = (entry.get("long_description") or "").strip()

    # Determine what content to use
    if long_desc and desc:
        # If both exist and are different/complementary
        if desc not in long_desc and long_desc not in desc:
            # Both are complementary - combine them
            return f"{desc}\n\n{long_desc}"
        else:
            # Long description contains or expands the short one - use long
            return long_desc
    elif long_desc:
        # Only long description available
        return long_desc
    elif desc:
        # Only description available
        return desc
    else:
        return ""


def convert_html_to_markdown(html_content: str) -> str:
    """Convert HTML description to clean markdown"""
    if not html_content:
        return ""

    # Convert HTML to markdown
    markdown_desc = md(html_content, heading_style="ATX")

    # Clean up the markdown - remove excessive whitespace
    lines = markdown_desc.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if line:  # Only keep non-empty lines
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


class SummaryOnly(BaseModel):
    """Structured response for summary generation only"""

    summary: str


async def generate_summary_only(name: str, description: str) -> str:
    """Generate AI summary for nodes and website pages using structured output"""
    try:
        prompt = f"""
        Create an index entry for LLM search table of contents. Pack maximum critical info to differentiate from other entries.
        
        Rules:
        - Skip obvious words (technique, imaging, microscopy) unless differentiating
        - Include specific capabilities, resolution, applications, unique features
        - Focus on what makes this item distinct and useful
        - Maximum accuracy in minimal space
        
        Provide:
        1. Concise description (max 80 chars, name excluded, pack critical differentiating info)
        
        Name: {name}
        Full description: {description[:800]}
        
        Good examples:
        "Sub-100nm resolution, live cells, photobleaching recovery analysis"
        "German multi-modal: STED, STORM, cryo-EM, training programs" 
        "Open-source platform: algorithm challenges, cloud computing, leaderboards"
        "Austria 8-site: microCT, microPET, correlative workflows, cyclotron"
        
        Bad examples (too generic):
        "Advanced imaging technique" "Microscopy facility" "Training courses"
        """

        response = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format=SummaryOnly,
            max_tokens=200,
            temperature=0.3,
        )

        result = response.choices[0].message.parsed
        assert result is not None
        return result.summary

    except Exception as e:
        print(f"Error generating summary for {name}: {e}")
        # Fallback to simple extraction
        return description[:150] + "..." if len(description) > 150 else description


def build_tech_to_nodes_mapping(nodes_data: List[Dict]) -> Dict[str, List[str]]:
    """Build a mapping of technology IDs to node IDs that offer them"""
    tech_to_nodes = {}

    for node in nodes_data:
        node_id = node.get("id", "")
        technologies = node.get("technologies", [])

        for tech_id in technologies:
            if tech_id not in tech_to_nodes:
                tech_to_nodes[tech_id] = []
            tech_to_nodes[tech_id].append(node_id)

    return tech_to_nodes


async def process_tech_data(
    tech_data: List[Dict], tech_to_nodes: Dict[str, List[str]]
) -> tuple[List[Dict], Dict[str, str]]:
    """Process technology data and generate indexed entries using research agent"""
    indexed_tech = []

    # Create mapping from original_id to short_id for technologies
    tech_id_mapping = {}
    for tech in tech_data:
        short_id = generate_short_uuid()
        tech_id_mapping[tech["id"]] = short_id

    print(f"Processing {len(tech_data)} technology entries with research agent...")

    for i, tech in enumerate(tech_data, 1):
        print(f"Processing tech {i}/{len(tech_data)}: {tech.get('name', 'Unnamed')}")

        name = tech.get("name", "")
        original_description = tech.get("description", "")
        original_long_description = tech.get("long_description", "")

        # Prepare tech for enrichment - force enrichment by making description short
        tech_for_enrichment = {
            "name": name,
            "description": original_description,
            "long_description": "",  # Force enrichment
        }

        try:
            # Use research agent to enrich the technology
            enriched_tech = await enrich_description(tech_for_enrichment, "technology")

            # Check enrichment status
            enrichment_meta: Dict[str, Any] = enriched_tech.get(
                "enrichment_metadata", {}
            )

            if not enriched_tech:
                # Enrichment failed completely
                print(f"  ⚠️  Enrichment failed for {name}, using original content")
                ai_description = original_description
                ai_documentation = ""
                enrichment_meta = {"enriched": False}
            elif enrichment_meta.get("skipped_reason") == "sufficient_description":
                # Enrichment was skipped because description was already sufficient
                print(f"  📝 Description sufficient for {name}, no enhancement needed")
                ai_description = (
                    enriched_tech.get("description", original_description)
                    or original_description
                )
                ai_documentation = ""
            elif enrichment_meta.get("enriched", False):
                # Enrichment was successful
                ai_description = (
                    enriched_tech.get("description", original_description)
                    or original_description
                )
                ai_documentation = enriched_tech.get("long_description", "") or ""
            else:
                # Enrichment failed or returned without proper metadata
                print(f"  ⚠️  Enrichment failed for {name}, using original content")
                ai_description = original_description
                ai_documentation = ""
                enrichment_meta = {"enriched": False}

            # Build the final documentation with proper structure
            final_documentation = ""

            # Add original long description if it exists
            if original_long_description and original_long_description.strip():
                final_documentation += convert_html_to_markdown(
                    original_long_description
                )
                final_documentation += "\n\n"

            # Only add AI sections if enrichment was actually performed (not skipped)
            if enrichment_meta.get("enriched", False):
                # Add AI Generated Documentation section
                if ai_documentation and ai_documentation.strip():
                    final_documentation += "## AI Generated Documentation\n\n"
                    final_documentation += ai_documentation
                    final_documentation += "\n\n"

                # Add References section
                sources = enrichment_meta.get("sources_used", [])
                if sources:
                    final_documentation += "## References\n\n"
                    for i, source in enumerate(sources, 1):
                        final_documentation += f"{i}. {source}\n"
                    final_documentation += "\n"

                # Add confidence score
                confidence = enrichment_meta.get("confidence_score", 0.0)
                final_documentation += (
                    f"**AI Enhancement Confidence Score:** {confidence:.2f}\n"
                )

                print(
                    f"  ✅ Enhanced with confidence {confidence:.2f}, sources: {len(sources)}"
                )
            elif enrichment_meta.get("skipped_reason") == "sufficient_description":
                # Don't print anything here - already printed above
                pass
            else:
                print("  📝 Using original content only")

        except Exception as e:
            print(f"  ⚠️  Enhancement failed for {name}, using original content")
            logger.warning(f"Enhancement failed for {name}: {e}")
            # Fallback to original content with safe defaults
            ai_description = original_description or ""
            final_documentation = (
                convert_html_to_markdown(get_description_text(tech)) or ""
            )

        # Convert provider_node_ids to use short node IDs (we'll need to create a node mapping)
        provider_node_ids_original = tech_to_nodes.get(tech["id"], [])

        # Create the indexed entry with short ID
        short_id = tech_id_mapping[tech["id"]]
        indexed_entry = {
            "id": short_id,
            "name": name,
            "original_id": tech["id"],
            "description": ai_description,  # Use AI-generated concise description
            "documentation": final_documentation,  # Use enhanced documentation
            "provider_node_ids": provider_node_ids_original,  # Will be converted to short IDs later
            # Preserve essential original fields
            "abbr": tech.get("abbr", ""),
            "category": tech.get("category", {}),
        }

        indexed_tech.append(indexed_entry)

    return indexed_tech, tech_id_mapping


async def process_nodes_data(
    nodes_data: List[Dict],
) -> tuple[List[Dict], Dict[str, str]]:
    """Process nodes data and generate indexed entries"""
    indexed_nodes = []

    # Create mapping from original_id to short_id for nodes
    node_id_mapping = {}
    for node in nodes_data:
        short_id = generate_short_uuid()
        node_id_mapping[node["id"]] = short_id

    print(f"Processing {len(nodes_data)} node entries...")

    for i, node in enumerate(nodes_data, 1):
        print(f"Processing node {i}/{len(nodes_data)}: {node.get('name', 'Unnamed')}")

        # Generate AI summary only (no keywords for nodes)
        name = node.get("name", "")
        description_text = get_description_text(node)

        try:
            summary = await generate_summary_only(name, description_text)
        except Exception as e:
            print(f"Warning: Failed to generate AI content for {name}: {e}")
            summary = (
                description_text[:150] + "..."
                if len(description_text) > 150
                else description_text
            )

        # Convert offer_technology_ids list to use short tech IDs (will be done later)
        offer_technology_ids_original = node.get("technologies", [])

        # Create the indexed entry with short ID
        short_id = node_id_mapping[node["id"]]
        indexed_entry = {
            "id": short_id,
            "name": name,
            "original_id": node["id"],
            "description": summary,
            "documentation": convert_html_to_markdown(get_description_text(node)),
            "offer_technology_ids": offer_technology_ids_original,  # Will be converted to short IDs later
            # Preserve essential original fields
            "country": node.get("country", {}),
        }

        indexed_nodes.append(indexed_entry)

    return indexed_nodes, node_id_mapping


async def process_website_data(
    max_pages: int = 50, delay: float = 1.0, output_dir: str = "eubio_website"
) -> List[Dict]:
    """Process website data by crawling and indexing"""
    print(f"Processing website data - crawling up to {max_pages} pages...")

    # Create scraper
    scraper = EuroBioImagingWebscraper(output_dir=output_dir)

    # Crawl website
    await scraper.crawl_website(max_pages=max_pages, delay=delay)

    # Create simplified index for search integration
    simplified_index = []
    for page in scraper.pages_data:
        simplified_page = {
            "id": page["id"],
            "url": page["url"],
            "title": page["title"],
            "description": page.get("ai_summary", page.get("meta_description", "")),
            "documentation": page["main_content"] if page["main_content"] else "",
            "headings": [h["text"] for h in page.get("headings", [])],
            "page_type": scraper.classify_page_type(page),
        }
        simplified_index.append(simplified_page)

    # Save website index files
    index_data = {
        "metadata": {
            "crawled_at": datetime.now().isoformat(),
            "base_url": scraper.base_url,
            "total_pages": len(scraper.pages_data),
            "crawled_pages": len(scraper.visited_urls),
            "pages_fetched": getattr(
                scraper, "pages_fetched", len(scraper.visited_urls)
            ),
            "pages_kept": getattr(scraper, "pages_kept", len(scraper.pages_data)),
            "pages_discarded": getattr(scraper, "pages_discarded", 0),
        },
        "pages": scraper.pages_data,
    }

    # Save complete index
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    complete_index_file = output_path / "eubio_website_index.json"
    with open(complete_index_file, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    # Save search index
    search_index_file = output_path / "eubio_website_search_index.json"
    with open(search_index_file, "w", encoding="utf-8") as f:
        json.dump(simplified_index, f, indent=2, ensure_ascii=False)

    print("Website processing completed:")
    print(f"  📊 Pages fetched: {getattr(scraper, 'pages_fetched', 0)}")
    print(f"  ✅ Pages kept: {getattr(scraper, 'pages_kept', len(scraper.pages_data))}")
    print(f"  ❌ Pages discarded: {getattr(scraper, 'pages_discarded', 0)}")
    print(f"  💾 Files saved in: {output_path}")

    return simplified_index


def build_bm25_index(
    tech_data: List[Dict],
    nodes_data: List[Dict],
    website_data: List[Dict],
    output_dir: Path,
):
    """Build BM25 index for full-text search"""
    print("🔍 Building BM25 index...")

    # Prepare corpus and metadata
    corpus = []
    metadata = []

    # Add technologies (no keywords for technologies anymore)
    for tech in tech_data:
        doc = f"{tech.get('name', '')} {tech.get('description', '')} {tech.get('documentation', '')}"
        corpus.append(doc)
        metadata.append({"type": "tech", "id": tech["id"]})

    # Add nodes (no keywords for nodes either)
    for node in nodes_data:
        doc = f"{node.get('name', '')} {node.get('description', '')} {node.get('documentation', '')} {node.get('country', {}).get('name', '')}"
        corpus.append(doc)
        metadata.append({"type": "node", "id": node["id"]})

    # Add website pages (no keywords - only title, description, documentation, and headings)
    for page in website_data:
        doc = f"{page.get('title', '')} {page.get('description', '')} {page.get('documentation', '')} {' '.join(page.get('headings', []))}"
        corpus.append(doc)
        metadata.append({"type": "page", "id": page["id"]})

    # Tokenize corpus
    print("📝 Tokenizing corpus...")
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en")

    # Create and index BM25 model
    print("📊 Creating BM25 index...")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    bm25_filename = "eurobioimaging_bm25_index.pkl"
    # Save BM25 index as pickle
    bm25_file = output_dir / bm25_filename
    print(f"💾 Saving BM25 index to {bm25_file}...")
    with open(bm25_file, "wb") as f:
        pickle.dump(retriever, f)

    print(f"✅ BM25 index built successfully with {len(corpus)} documents")
    return bm25_filename, metadata


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Build search indexes for Euro-BioImaging data"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Build test indexes with first 10 items only",
    )
    parser.add_argument(
        "--max-pages", type=int, default=50, help="Maximum website pages to crawl"
    )
    parser.add_argument(
        "--crawl-delay",
        type=float,
        default=1.0,
        help="Delay between website requests (seconds)",
    )
    parser.add_argument(
        "--skip-website", action="store_true", help="Skip website crawling"
    )
    parser.add_argument(
        "--data-dir",
        default="euro-bioimaging-index",
        help="Directory to store all generated files (default: euro-bioimaging-index)",
    )
    parser.add_argument(
        "--new-db",
        action="store_true",
        help="Use local EB-1-Nodes-Technologies submodule JSON files instead of fetching from API",
    )
    args = parser.parse_args()

    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.test:
        args.max_pages = 10
        args.crawl_delay = 0.5

    print("🚀 Starting parallel indexing of Euro-BioImaging data...")
    print(f"📁 Data directory: {data_dir}")
    print(
        f"🌐 Website: {'Skipped' if args.skip_website else f'Up to {args.max_pages} pages'}"
    )
    print(f"📊 Dataset: {'Test (10 items)' if args.test else 'Full dataset'}")
    print("=" * 60)

    # Option to use EB-1-Nodes-Technologies from GitHub zip if --new-db is set
    if args.new_db:
        print(
            "📖 Downloading and parsing data from EB-1-Nodes-Technologies GitHub zip..."
        )
        import re, uuid, tempfile, zipfile, requests, shutil

        # Download the zip
        url = "https://github.com/Hylcke/EB-1-Nodes-Technologies/archive/refs/heads/main.zip"
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "eb1.zip"
            print(f"  ⬇️  Downloading zip from {url}")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            # Extract zip
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)
            # Find the extracted folder
            extracted = next(Path(tmpdir).glob("EB-1-Nodes-Technologies-*/"))
            node_files = list(extracted.glob("RREXP *.txt"))
            if not node_files:
                print(f"  ❌ No RREXP *.txt files found in {extracted}")
                return
            print(f"  📄 Found {len(node_files)} node files")

            def parse_node_file(file_path):
                content = file_path.read_text(encoding="utf-8")
                filename = file_path.name
                country_match = re.search(r"RREXP\s+([A-Z\s]+?)\s+", filename)
                country = country_match.group(1).strip() if country_match else "Unknown"
                name_match = re.search(r"RREXP\s+[A-Z\s]+?\s+(.+?)\.txt$", filename)
                name = (
                    name_match.group(1).strip()
                    if name_match
                    else filename.replace(".txt", "")
                )
                description_match = re.search(
                    r"# Description\s*\n\n(.*?)(?=\n##|\n#|\Z)", content, re.DOTALL
                )
                description = (
                    description_match.group(1).strip() if description_match else ""
                )
                description = re.sub(r"\n+", " ", description)
                description = re.sub(r"\*\*(.*?)\*\*", r"\1", description)
                description = re.sub(r"\s+", " ", description).strip()
                technologies = []
                tech_table_match = re.search(
                    r"\| Technologies \|.*?\n(.*?)(?=\n##|\n#|\Z)", content, re.DOTALL
                )
                if tech_table_match:
                    table_content = tech_table_match.group(1)
                    for line in table_content.split("\n"):
                        if "|" in line and not line.strip().startswith("|---"):
                            tech_match = re.search(r"\|\s*([^|]+?)\s*\|", line)
                            if tech_match:
                                tech_name = tech_match.group(1).strip()
                                if tech_name and tech_name != "Technologies":
                                    technologies.append(tech_name)
                node_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_DNS, f"eurobioimaging.node.{country}.{name}"
                    )
                )
                return {
                    "id": node_id,
                    "name": name,
                    "description": description,
                    "country": {"name": country},
                    "technologies": technologies,
                    "entity_type": "node",
                }

            def extract_technologies_from_files(file_paths):
                all_technologies = set()
                for file_path in file_paths:
                    content = file_path.read_text(encoding="utf-8")
                    tech_table_match = re.search(
                        r"\| Technologies \|.*?\n(.*?)(?=\n##|\n#|\Z)",
                        content,
                        re.DOTALL,
                    )
                    if tech_table_match:
                        table_content = tech_table_match.group(1)
                        for line in table_content.split("\n"):
                            if "|" in line and not line.strip().startswith("|---"):
                                tech_match = re.search(r"\|\s*([^|]+?)\s*\|", line)
                                if tech_match:
                                    tech_name = tech_match.group(1).strip()
                                    if tech_name and tech_name != "Technologies":
                                        all_technologies.add(tech_name)
                technologies = []
                for tech_name in sorted(all_technologies):
                    tech_id = str(
                        uuid.uuid5(
                            uuid.NAMESPACE_DNS, f"eurobioimaging.technology.{tech_name}"
                        )
                    )
                    category = "Unknown"
                    if any(
                        keyword in tech_name.lower()
                        for keyword in ["microscopy", "imaging", "scan"]
                    ):
                        category = "Microscopy"
                    elif any(
                        keyword in tech_name.lower()
                        for keyword in ["spectroscopy", "raman"]
                    ):
                        category = "Spectroscopy"
                    elif any(
                        keyword in tech_name.lower()
                        for keyword in ["tomography", "tem", "sem"]
                    ):
                        category = "Electron Microscopy"
                    elif any(
                        keyword in tech_name.lower()
                        for keyword in ["clearing", "expansion"]
                    ):
                        category = "Sample Preparation"
                    technologies.append(
                        {
                            "id": tech_id,
                            "name": tech_name,
                            "description": f"Bioimaging technology: {tech_name}",
                            "category": {"name": category},
                            "entity_type": "technology",
                        }
                    )
                return technologies

            nodes_data = [parse_node_file(f) for f in node_files]
            tech_data = extract_technologies_from_files(node_files)
            print(
                f"  ✅ Parsed {len(nodes_data)} nodes and {len(tech_data)} unique technologies from downloaded text files"
            )
    else:
        # Fetch the JSON data from APIs
        print("📖 Fetching data from APIs...")
        async with aiohttp.ClientSession() as session:
            # Fetch technologies data
            try:
                print("  🔬 Fetching technologies from API...")
                async with session.get(
                    "https://eb-api.aashvi.net/v1/technologies/?format=json"
                ) as response:
                    if response.status == 200:
                        tech_data = await response.json()
                        print(f"  ✅ Loaded {len(tech_data)} technologies from API")
                    else:
                        print(
                            f"  ❌ Failed to fetch technologies: HTTP {response.status}"
                        )
                        tech_data = []
            except Exception as e:
                print(f"  ❌ Error fetching technologies: {e}")
                tech_data = []

            # Fetch nodes data
            try:
                print("  🏢 Fetching nodes from API...")
                async with session.get(
                    "https://eb-api.aashvi.net/v1/nodes/?format=json"
                ) as response:
                    if response.status == 200:
                        nodes_data = await response.json()
                        print(f"  ✅ Loaded {len(nodes_data)} nodes from API")
                    else:
                        print(f"  ❌ Failed to fetch nodes: HTTP {response.status}")
                        nodes_data = []
            except Exception as e:
                print(f"  ❌ Error fetching nodes: {e}")
                nodes_data = []

    # Use test or production data based on argument
    if args.test:
        tech_data_to_process = tech_data[:10]
        nodes_data_to_process = nodes_data[:10]
        output_filename = "test_eurobioimaging_index.json"
    else:
        tech_data_to_process = tech_data
        nodes_data_to_process = nodes_data
        output_filename = "eurobioimaging_index.json"

    # Build technology to nodes mapping (using full nodes data for accurate cross-references)
    print("🔗 Building technology-to-nodes mapping...")
    tech_to_nodes = build_tech_to_nodes_mapping(nodes_data)
    print(f"  ✅ Mapped {len(tech_to_nodes)} technologies to nodes")

    # Create parallel tasks
    tasks = []

    # Task 1: Process tech data
    tech_task = process_tech_data(tech_data_to_process, tech_to_nodes)
    tasks.append(tech_task)

    # Task 2: Process nodes data
    nodes_task = process_nodes_data(nodes_data_to_process)
    tasks.append(nodes_task)

    # Task 3: Process website data (if not skipped)
    if not args.skip_website:
        website_output_dir = data_dir / "website"
        website_task = process_website_data(
            max_pages=args.max_pages,
            delay=args.crawl_delay,
            output_dir=str(website_output_dir),
        )
        tasks.append(website_task)

    # Run all tasks in parallel
    print(f"⚡ Running {len(tasks)} indexing tasks in parallel...")
    start_time = asyncio.get_event_loop().time()

    try:
        if args.skip_website:
            # Only tech and nodes
            tech_result, nodes_result = await asyncio.gather(*tasks)
            website_index = []
        else:
            # All three tasks
            tech_result, nodes_result, website_index = await asyncio.gather(*tasks)

        tech_index, tech_id_mapping = tech_result
        nodes_index, node_id_mapping = nodes_result

    except Exception as e:
        print(f"❌ Error during parallel processing: {e}")
        return

    end_time = asyncio.get_event_loop().time()
    print(f"✅ Parallel processing completed in {end_time - start_time:.1f} seconds")

    # Convert cross-references to use short IDs
    print("🔄 Converting cross-references to short IDs...")

    # Update provider_node_ids in tech_index to use short node IDs
    for tech in tech_index:
        tech["provider_node_ids"] = [
            node_id_mapping.get(node_id, node_id)
            for node_id in tech["provider_node_ids"]
            if node_id in node_id_mapping
        ]

    # Update offer_technology_ids in nodes_index to use short tech IDs
    for node in nodes_index:
        node["offer_technology_ids"] = [
            tech_id_mapping.get(tech_id, tech_id)
            for tech_id in node["offer_technology_ids"]
            if tech_id in tech_id_mapping
        ]

    print("🔄 Building BM25 index...")
    # Build BM25 index
    bm25_file_name, bm25_metadata = build_bm25_index(
        tech_index, nodes_index, website_index, data_dir
    )

    # Create combined index
    print("🔄 Creating combined index...")
    combined_index = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "version": "1.0",
            "description": "Euro-BioImaging combined search index",
            "dataset_type": "test" if args.test else "full",
            "statistics": {
                "technologies": len(tech_index),
                "nodes": len(nodes_index),
                "website_pages": len(website_index),
                "total_entries": len(tech_index)
                + len(nodes_index)
                + len(website_index),
            },
            "bm25_file": bm25_file_name,
        },
        "technologies": tech_index,
        "nodes": nodes_index,
        "website_pages": website_index,
        "bm25_metadata": bm25_metadata,
    }

    # Save the combined index
    print("💾 Saving combined index...")
    output_file = data_dir / output_filename
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_index, f, indent=2, ensure_ascii=False)

    print("\n🎉 Indexing completed successfully!")
    print(f"📊 Combined index saved to: {output_file}")
    print(f"🔍 BM25 index saved as: {bm25_file_name}")
    print("📈 Statistics:")
    print(f"  🔬 Technologies: {len(tech_index)} entries")
    print(f"  🏢 Nodes: {len(nodes_index)} entries")

    if not args.skip_website:
        print(f"  🌐 Website pages: {len(website_index)} entries")
        print(f"  📁 Website files: {data_dir}/website/")

    print(
        f"  📦 Total entries: {len(tech_index) + len(nodes_index) + len(website_index)}"
    )

    # Print some statistics
    print("\n📈 Technology statistics:")
    print(
        f"  🔗 Technologies with provider nodes: {sum(1 for t in tech_index if t['provider_node_ids'])}"
    )
    if tech_index:
        print(
            f"  📊 Average nodes per technology: {sum(len(t['provider_node_ids']) for t in tech_index) / len(tech_index):.1f}"
        )

    print("\n🏢 Node statistics:")
    print(
        f"  🔧 Nodes with technologies: {sum(1 for n in nodes_index if n['offer_technology_ids'])}"
    )
    if nodes_index:
        print(
            f"  📊 Average technologies per node: {sum(len(n['offer_technology_ids']) for n in nodes_index) / len(nodes_index):.1f}"
        )

    # Show sample entries
    print("\n📝 Sample technology entry:")
    if tech_index:
        sample_tech = tech_index[0]
        print(f"  🆔 ID: {sample_tech['id']} (was {sample_tech['original_id'][:8]}...)")
        print(f"  📛 Name: {sample_tech['name']}")
        print(f"  📄 Description: {sample_tech['description']}")
        print(
            f"  🏢 Available in {len(sample_tech['provider_node_ids'])} nodes: {sample_tech['provider_node_ids']}"
        )
        print(f"  📂 Category: {sample_tech['category'].get('name', 'N/A')}")
        print("  🤖 AI-Enhanced: Yes")

    print("\n🏢 Sample node entry:")
    if nodes_index:
        sample_node = nodes_index[0]
        print(f"  🆔 ID: {sample_node['id']} (was {sample_node['original_id'][:8]}...)")
        print(f"  📛 Name: {sample_node['name']}")
        print(f"  📄 Description: {sample_node['description']}")
        print(f"  🌍 Country: {sample_node['country'].get('name', 'N/A')}")
        print(
            f"  🔧 Technologies: {len(sample_node['offer_technology_ids'])} - {sample_node['offer_technology_ids'][:5]}{'...' if len(sample_node['offer_technology_ids']) > 5 else ''}"
        )

    if not args.skip_website and website_index:
        print("\n🌐 Sample website page:")
        sample_page = website_index[0]
        print(f"  🆔 ID: {sample_page['id']}")
        print(f"  📛 Title: {sample_page['title']}")
        print(f"  📄 Description: {sample_page['description']}")
        print(f"  📑 Documentation length: {len(sample_page.get('documentation', ''))}")
        print(f"  🔗 URL: {sample_page['url']}")
        print(f"  📂 Page Type: {sample_page['page_type']}")


if __name__ == "__main__":
    asyncio.run(main())
