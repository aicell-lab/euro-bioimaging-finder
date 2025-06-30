import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import aiohttp
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv
import os

# Try to import duckduckgo search API
try:
    from ddg import Duckduckgo
except ImportError:
    print("Warning: duckduckgo-search-api not installed. Run: pip install duckduckgo-search-api")
    Duckduckgo = None

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnrichedDocumentation(BaseModel):
    """Structured response for enriched description"""
    enriched_description: str  # One-sentence concise description to replace the auto-generated description
    enriched_documentation: str  # Full detailed documentation (200-400 words)
    confidence_score: float  # 0.0 to 1.0
    sources_used: List[str]

class SearchResult(BaseModel):
    """Search result from DuckDuckGo"""
    title: str
    url: str
    description: str

class WebPageContent(BaseModel):
    """Content extracted from web page"""
    title: str
    content: str
    url: str

class AgenticSearcher:
    """Agentic searcher that uses DuckDuckGo and web scraping to enrich descriptions"""
    
    def __init__(self):
        self.ddg_api = Duckduckgo() if Duckduckgo else None
        self.session = None
        self.max_search_results = 5
        self.max_pages_to_read = 3
        self.max_content_length = 2000  # Limit content length per page
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User-Agent': 'Mozilla/5.0 (compatible; EuroBioImaging-Enricher/1.0; Research purposes)'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def search_duckduckgo(self, query: str) -> List[SearchResult]:
        """Search DuckDuckGo for the given query"""
        try:
            if not self.ddg_api:
                logger.warning("DuckDuckGo API not available")
                return []
            
            logger.info(f"Searching DuckDuckGo for: {query}")
            
            # Search using DuckDuckGo API
            search_response = self.ddg_api.search(query)
            
            if not search_response.get("success", False):
                logger.warning(f"DuckDuckGo search failed: {search_response.get('message', 'Unknown error')}")
                return []
            
            results = []
            data = search_response.get("data", [])
            
            for item in data[:self.max_search_results]:
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    description=item.get("description", "")
                )
                results.append(result)
            
            logger.info(f"Found {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching DuckDuckGo: {e}")
            return []

    async def read_web_page(self, url: str) -> Optional[WebPageContent]:
        """Read and extract content from a web page"""
        try:
            logger.info(f"Reading web page: {url}")
            
            if not self.session:
                logger.error("Session not initialized")
                return None
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None
                
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    logger.info(f"Skipping non-HTML content: {url}")
                    return None
                
                html_content = await response.text()
                
                # Parse HTML
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Extract title
                title_tag = soup.find('title')
                title = title_tag.get_text().strip() if title_tag else ""
                
                # Remove script, style, and other non-content elements
                for element in soup(['script', 'style', 'noscript', 'nav', 'footer', 'header']):
                    element.decompose()
                
                # Extract main content
                main_selectors = [
                    'main', '.main', '#main', '.content', '#content', 
                    '.post-content', '.entry-content', 'article', '.article'
                ]
                
                main_content = None
                for selector in main_selectors:
                    main_content = soup.select_one(selector)
                    if main_content:
                        break
                
                if not main_content:
                    main_content = soup.find('body')
                
                if main_content:
                    # Get text content
                    text_content = main_content.get_text(' ', strip=True)
                    
                    # Limit content length
                    if len(text_content) > self.max_content_length:
                        text_content = text_content[:self.max_content_length] + "..."
                    
                    return WebPageContent(
                        title=title,
                        content=text_content,
                        url=url
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Error reading web page {url}: {e}")
            return None

    async def acall(self, query: str, entity_name: str, entity_type: str = "technology") -> EnrichedDocumentation:
        """
        Agentic call function that performs search and web reading in a ReAct loop
        to enrich description for a given entity (technology or node)
        """
        logger.info(f"Starting agentic enrichment for {entity_type}: {entity_name}")
        
        # Define tools for OpenAI function calling
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_duckduckgo",
                    "description": "Search DuckDuckGo for information about the given query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query to find relevant information"
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_web_page",
                    "description": "Read and extract content from a specific web page URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL of the web page to read"
                            }
                        },
                        "required": ["url"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        ]
        
        # Initialize conversation
        messages = [
            {
                "role": "system",
                "content": f"""You are an expert researcher specializing in bioimaging technologies and research infrastructure. 
Your task is to enrich the description of a {entity_type} called "{entity_name}" by gathering comprehensive information from web sources.

Instructions:
1. Search for information about "{entity_name}" focusing on scientific/technical details
2. Read relevant web pages to gather detailed information
3. Focus on: technical specifications, applications, capabilities, research uses, advantages
4. Aim to create a comprehensive description of 200-400 words
5. Include specific technical details, not generic descriptions
6. Prioritize official sources, research papers, manufacturer documentation

Search Strategy:
- Start with broad search about the technology/facility
- Follow up with specific searches for technical details
- Read 2-3 most relevant pages for detailed information

Stop when you have enough information to write a comprehensive description."""
            },
            {
                "role": "user",
                "content": f"Please research and gather detailed information about the {entity_type} '{entity_name}' to create an enriched description. Start by searching for relevant information."
            }
        ]
        
        # Track information gathered
        search_results = []
        web_content = []
        sources_used = []
        max_iterations = 8  # Limit to prevent infinite loops
        iteration = 0
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        while iteration < max_iterations and consecutive_errors < max_consecutive_errors:
            iteration += 1
            logger.info(f"ReAct iteration {iteration}")
            
            try:
                # Call OpenAI with current conversation
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=8000,  # Reduced to fit within gpt-4o-mini limits
                    temperature=0.3
                )
                
                # Reset error counter on success
                consecutive_errors = 0
                
                assistant_message = response.choices[0].message
                messages.append({
                    "role": "assistant", 
                    "content": assistant_message.content,
                    "tool_calls": assistant_message.tool_calls
                })
                
                # Check if the model wants to call tools
                if assistant_message.tool_calls:
                    # Execute tool calls
                    for tool_call in assistant_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        logger.info(f"Executing tool: {function_name} with args: {function_args}")
                        
                        if function_name == "search_duckduckgo":
                            results = await self.search_duckduckgo(function_args["query"])
                            search_results.extend(results)
                            
                            # Format results for the model
                            results_text = "\n".join([
                                f"Title: {r.title}\nURL: {r.url}\nDescription: {r.description}\n"
                                for r in results
                            ])
                            
                            messages.append({
                                "role": "tool",
                                "content": f"Search results:\n{results_text}",
                                "tool_call_id": tool_call.id
                            })
                            
                        elif function_name == "read_web_page":
                            content = await self.read_web_page(function_args["url"])
                            if content:
                                web_content.append(content)
                                sources_used.append(content.url)
                                
                                messages.append({
                                    "role": "tool",
                                    "content": f"Web page content from {content.url}:\nTitle: {content.title}\nContent: {content.content}",
                                    "tool_call_id": tool_call.id
                                })
                            else:
                                messages.append({
                                    "role": "tool",
                                    "content": f"Failed to read content from {function_args['url']}",
                                    "tool_call_id": tool_call.id
                                })
                else:
                    # Model has finished and provided a response
                    break
                    
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error in ReAct iteration {iteration}: {e}")
                
                # If we have too many consecutive errors, give up
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive errors ({consecutive_errors}), stopping enrichment")
                    break
                
                # Wait a bit before retrying
                await asyncio.sleep(1.0 * consecutive_errors)  # Simple backoff
        
        # Final call to generate enriched description
        final_prompt = f"""Based on all the research you've conducted, please create both a concise description and detailed documentation for the {entity_type} '{entity_name}'.

Requirements for enriched_description (one sentence, max 80 chars):
- Pack maximum critical info to differentiate from other entries
- Skip obvious words unless differentiating 
- Include specific capabilities, resolution, applications, unique features
- Focus on what makes this item distinct and useful

Requirements for enriched_documentation (200-400 words):
- Focus on technical capabilities, applications, and unique features
- Include specific details that differentiate this from other similar technologies
- Be factual and technical, not marketing language
- Structure: overview, key capabilities, applications, advantages

Examples of good enriched_description:
"Sub-100nm resolution, live cells, photobleaching recovery analysis"
"German multi-modal: STED, STORM, cryo-EM, training programs"
"Austria 8-site: microCT, microPET, correlative workflows, cyclotron"

Provide both the concise description and the detailed documentation."""

        messages.append({
            "role": "user",
            "content": final_prompt
        })
        
        try:
            final_response = await client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=messages,
                response_format=EnrichedDocumentation,
                max_tokens=8000,  # Reduced to fit within gpt-4o-mini limits
                temperature=0.3
            )
            
            result = final_response.choices[0].message.parsed
            result.sources_used = list(set(sources_used))  # Remove duplicates
            
            logger.info(f"Enrichment completed for {entity_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating final enriched description: {e}")
            
            # Try a simplified fallback without web research - just basic enhancement
            try:
                logger.info(f"Attempting simplified enrichment for {entity_name}")
                simplified_prompt = f"""Create a concise description and brief documentation for the {entity_type} '{entity_name}'.

Requirements for enriched_description (one sentence, max 80 chars):
- Focus on key technical capabilities and unique features
- Be specific and differentiating

Requirements for enriched_documentation (100-200 words):
- Provide basic technical overview
- Focus on applications and capabilities
- Be factual and informative

If you don't have specific information, provide a generic but accurate description based on the technology name and type."""

                simplified_response = await client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": simplified_prompt}],
                    response_format=EnrichedDocumentation,
                    max_tokens=1000,
                    temperature=0.5
                )
                
                result = simplified_response.choices[0].message.parsed
                result.sources_used = []  # No sources in simplified mode
                result.confidence_score = 0.3  # Lower confidence for simplified enrichment
                
                logger.info(f"Simplified enrichment completed for {entity_name}")
                return result
                
            except Exception as e2:
                logger.error(f"Simplified enrichment also failed for {entity_name}: {e2}")
                # Return None to indicate enrichment failed completely
                return None

async def enrich_description(entity: Dict[str, Any], entity_type: str = "technology") -> Dict[str, Any]:
    """
    Enrich description for a technology or node if the current description is too short
    
    Args:
        entity: Technology or node dictionary
        entity_type: "technology" or "node"
    
    Returns:
        Updated entity with enriched description
    """
    name = entity.get('name', '')
    current_desc = entity.get('description', '')
    current_long_desc = entity.get('long_description', '')
    
    # Determine if enrichment is needed
    main_desc = current_long_desc if current_long_desc else current_desc
    
    if len(main_desc) >= 100:
        logger.info(f"Description for {name} is sufficient ({len(main_desc)} chars), skipping enrichment")
        # Add metadata to indicate enrichment was skipped (successful no-op)
        entity_with_metadata = entity.copy()
        entity_with_metadata['enrichment_metadata'] = {
            'enriched': False,  # No AI enhancement was performed
            'skipped_reason': 'sufficient_description',
            'confidence_score': 0.0,
            'sources_used': [],
            'original_description_length': len(current_desc or ""),
            'original_documentation_length': len(main_desc or ""),
            'enriched_description_length': 0,
            'enriched_documentation_length': 0
        }
        return entity_with_metadata
    
    logger.info(f"Description for {name} is too short ({len(main_desc)} chars), enriching...")
    
    # Create search query
    search_query = f"{name}"
    if entity_type == "technology":
        search_query += " bioimaging microscopy technology"
    else:
        search_query += " bioimaging facility research infrastructure"
    
    try:
        async with AgenticSearcher() as searcher:
            enriched = await searcher.acall(
                query=search_query,
                entity_name=name,
                entity_type=entity_type
            )
            
            # Check if enrichment succeeded
            if enriched is None:
                logger.warning(f"Enrichment failed for {name}, skipping AI enhancement")
                return entity
            
            # Update entity with enriched description (with safe defaults)
            updated_entity = entity.copy()
            updated_entity['description'] = enriched.enriched_description or current_desc or ""
            updated_entity['long_description'] = enriched.enriched_documentation or ""
            updated_entity['enrichment_metadata'] = {
                'enriched': True,
                'confidence_score': enriched.confidence_score or 0.0,
                'sources_used': enriched.sources_used or [],
                'original_description_length': len(current_desc or ""),
                'original_documentation_length': len(main_desc or ""),
                'enriched_description_length': len(enriched.enriched_description or ""),
                'enriched_documentation_length': len(enriched.enriched_documentation or "")
            }
            
            logger.info(f"Successfully enriched description for {name} (confidence: {enriched.confidence_score:.2f})")
            return updated_entity
            
    except Exception as e:
        logger.error(f"Failed to enrich description for {name}: {e}")
        logger.info(f"Skipping AI enhancement for {name}, using original content")
        return entity

# Test function
async def test_enrichment():
    """Test the enrichment functionality"""
    
    # Test technology
    test_tech = {
        'id': 'test_tech',
        'name': 'STED Microscopy',
        'description': 'Super resolution technique',
        'long_description': ''
    }
    
    print("Testing technology enrichment...")
    enriched_tech = await enrich_description(test_tech, "technology")
    print(f"Original description: {test_tech.get('description', '')}")
    print(f"Enriched description: {enriched_tech.get('description', '')}")
    print(f"Enriched documentation: {enriched_tech.get('long_description', '')[:200]}...")
    print(f"Metadata: {enriched_tech.get('enrichment_metadata', {})}")
    
    # Test node
    test_node = {
        'id': 'test_node',
        'name': 'Euro-BioImaging EMBL',
        'description': 'Imaging facility',
        'long_description': ''
    }
    
    print("\nTesting node enrichment...")
    enriched_node = await enrich_description(test_node, "node")
    print(f"Original description: {test_node.get('description', '')}")
    print(f"Enriched description: {enriched_node.get('description', '')}")
    print(f"Enriched documentation: {enriched_node.get('long_description', '')[:200]}...")
    print(f"Metadata: {enriched_node.get('enrichment_metadata', {})}")

if __name__ == "__main__":
    asyncio.run(test_enrichment()) 