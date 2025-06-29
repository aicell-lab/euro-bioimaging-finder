#!/usr/bin/env python3
import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class TechDetail(BaseModel):
    """Details of a specific technology"""
    id: str
    name: str
    original_id: str
    description: str
    keywords: List[str]
    documentation: str
    category: Dict[str, Any]
    provider_node_ids: List[str]
    # Include essential original fields
    abbr: str

class NodeDetail(BaseModel):
    """Details of a specific node"""
    id: str
    name: str
    original_id: str
    description: str
    keywords: List[str]
    documentation: str
    country: Dict[str, Any]
    offer_technology_ids: List[str]

class WebsitePageDetail(BaseModel):
    """Details of a Euro-BioImaging website page"""
    id: str
    url: str
    title: str
    description: str
    keywords: List[str]
    documentation: str
    headings: List[str]
    page_type: str

class SearchResponse(BaseModel):
    """Structured search response"""
    answer: str
    relevant_technologies: List[str] = []
    relevant_nodes: List[str] = []
    relevant_pages: List[str] = []
    summary: str = ""

class ToolCallRequest(BaseModel):
    """Request for tool calling"""
    action: str = Field(description="The action to take: 'get_tech_details', 'get_node_details', 'get_nodes_by_country', or 'get_website_pages'")
    tech_ids: List[str] = Field(default=[], description="List of technology IDs to get details for")
    node_ids: List[str] = Field(default=[], description="List of node IDs to get details for")
    country_codes: List[str] = Field(default=[], description="List of ISO country codes (e.g., 'PL', 'SE') to get all nodes for")
    page_ids: List[str] = Field(default=[], description="List of website page IDs to get details for")
    reasoning: str = Field(description="Reasoning for why these specific items were selected")
    query_type: str = Field(description="Type of query: 'geographic', 'technique', 'general', 'website_info', or 'listing'")
    coverage_strategy: str = Field(description="Strategy for comprehensive coverage: 'all_nodes_in_region', 'all_techs_for_technique', 'website_pages', 'selective', or 'comprehensive_listing'")

# Global data storage
tech_data: List[Dict[str, Any]] = []
nodes_data: List[Dict[str, Any]] = []
website_data: List[Dict[str, Any]] = []

async def stream_print(message: str, end: str = "\n", flush: bool = True):
    """Print message with streaming effect"""
    print(message, end=end, flush=flush)
    await asyncio.sleep(0.01)  # Small delay for better UX

async def show_progress(message: str):
    """Show progress indicator"""
    await stream_print(f"🔍 {message}...")

async def show_completion(message: str):
    """Show completion indicator"""
    await stream_print(f"✅ {message}")

async def show_info(message: str):
    """Show information message"""
    await stream_print(f"ℹ️  {message}")

async def show_section_header(title: str):
    """Show section header"""
    await stream_print(f"\n{'='*50}")
    await stream_print(f"📋 {title}")
    await stream_print(f"{'='*50}")

def load_data(test_mode: bool = False, data_dir: str = "euro-bioimaging-index"):
    """Load the indexed data from combined index file"""
    global tech_data, nodes_data, website_data
    
    data_path = Path(data_dir)
    
    if test_mode:
        index_file = data_path / 'test_eurobioimaging_index.json'
    else:
        index_file = data_path / 'eurobioimaging_index.json'
    
    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            combined_data = json.load(f)
        
        # Extract data from combined index
        tech_data = combined_data.get('technologies', [])
        nodes_data = combined_data.get('nodes', [])
        website_data = combined_data.get('website_pages', [])
        
        # Print metadata if available
        metadata = combined_data.get('metadata', {})
        dataset_type = metadata.get('dataset_type', 'unknown')
        created_at = metadata.get('created_at', 'unknown')
        
        print(f"Loaded combined index from {index_file}")
        print(f"  📊 Dataset: {dataset_type}")
        print(f"  📅 Created: {created_at}")
        print(f"  🔬 Technologies: {len(tech_data)}")
        print(f"  🏢 Nodes: {len(nodes_data)}")
        print(f"  🌐 Website pages: {len(website_data)}")
        print(f"  📦 Total entries: {len(tech_data) + len(nodes_data) + len(website_data)}")
        
    except FileNotFoundError:
        print(f"Warning: {index_file} not found. All data will be empty.")
        tech_data = []
        nodes_data = []
        website_data = []

def read_tech_details(tech_id: str) -> Optional[TechDetail]:
    """Read details of a specific technology by ID"""
    for tech in tech_data:
        if tech['id'] == tech_id:
            return TechDetail(**tech)
    return None

def read_node_details(node_id: str) -> Optional[NodeDetail]:
    """Read details of a specific node by ID"""
    for node in nodes_data:
        if node['id'] == node_id:
            return NodeDetail(**node)
    return None

def read_nodes_by_country(country_code: str) -> List[NodeDetail]:
    """Read all nodes in a specific country by ISO country code (e.g., 'PL', 'SE')"""
    country_nodes = []
    for node in nodes_data:
        node_country = node.get('country', {})
        if node_country.get('iso_a2', '').upper() == country_code.upper():
            try:
                country_nodes.append(NodeDetail(**node))
            except Exception as e:
                print(f"Warning: Could not parse node {node.get('name', 'Unknown')}: {e}")
    return country_nodes

def read_website_page_details(page_id: str) -> Optional[WebsitePageDetail]:
    """Read details of a specific website page by ID"""
    for page in website_data:
        if page['id'] == page_id:
            return WebsitePageDetail(**page)
    return None

def find_website_pages_by_keywords(keywords: List[str]) -> List[str]:
    """Find website page IDs that match the given keywords with relevance scoring"""
    page_scores = []
    
    for page in website_data:
        page_title = page.get('title', '').lower()
        page_description = page.get('description', '').lower()
        page_keywords = [kw.lower() for kw in page.get('keywords', [])]
        page_content = page.get('documentation', '').lower()
        page_type = page.get('page_type', '').lower()
        
        score = 0
        matched_keywords = 0
        
        # Check keyword matches with scoring
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in page_title:
                score += 5  # Title matches are most important
                matched_keywords += 1
            elif keyword_lower in page_description:
                score += 3  # Description matches are important
                matched_keywords += 1
            elif any(keyword_lower in kw for kw in page_keywords):
                score += 2  # Keyword matches are good
                matched_keywords += 1
            elif keyword_lower in page_content:
                score += 1  # Content matches are least important
                matched_keywords += 1
        
        # Boost score for official/important page types
        if 'about' in page_type or 'service' in page_type or 'access' in page_type:
            score += 2
        
        # Only include pages with meaningful matches
        if matched_keywords > 0 and score >= 2:
            page_scores.append((page['id'], score, matched_keywords))
    
    # Sort by score (descending) and number of matched keywords
    page_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    # Return top 10 most relevant page IDs
    return [page_id for page_id, _, _ in page_scores[:10]]

def get_country_codes() -> Dict[str, str]:
    """Get mapping of country names to ISO codes from the data"""
    country_mapping = {}
    for node in nodes_data:
        country = node.get('country', {})
        country_name = country.get('name', '')
        country_code = country.get('iso_a2', '')
        if country_name and country_code:
            country_mapping[country_name.lower()] = country_code.upper()
    return country_mapping

def find_nodes_by_technique(technique_keywords: List[str]) -> List[str]:
    """Find node IDs that offer technologies matching the given technique keywords with comprehensive matching"""
    matching_node_ids = set()
    
    # Search through all technologies to find matches
    for tech in tech_data:
        tech_name = tech.get('name', '').lower()
        tech_keywords = [kw.lower() for kw in tech.get('keywords', [])]
        tech_category = tech.get('category', {}).get('name', '').lower()
        tech_abbr = tech.get('abbr', '').lower()
        tech_description = tech.get('documentation', '').lower()
        
        # Score this technology based on keyword matches
        match_score = 0
        matched_keywords = 0
        
        for keyword in technique_keywords:
            keyword_lower = keyword.lower()
            
            # Exact matches in name or abbreviation (highest priority)
            if keyword_lower == tech_name or keyword_lower == tech_abbr:
                match_score += 10
                matched_keywords += 1
            # Partial matches in name or abbreviation
            elif keyword_lower in tech_name or keyword_lower in tech_abbr:
                match_score += 5
                matched_keywords += 1
            # Matches in category
            elif keyword_lower in tech_category:
                match_score += 3
                matched_keywords += 1
            # Matches in keywords
            elif any(keyword_lower in kw or kw in keyword_lower for kw in tech_keywords):
                match_score += 2
                matched_keywords += 1
            # Matches in description (lower priority)
            elif keyword_lower in tech_description:
                match_score += 1
                matched_keywords += 1
        
        # Include technology if it has meaningful matches
        if match_score >= 2 or matched_keywords >= 2:
            # Add all nodes that offer this technology
            matching_node_ids.update(tech.get('provider_node_ids', []))
    
    return list(matching_node_ids)

def get_comprehensive_node_coverage(query_type: str, specific_items: List[str]) -> List[str]:
    """Get comprehensive node coverage based on query type and specific requirements"""
    if query_type == 'geographic':
        # For geographic queries, get all nodes in mentioned countries
        all_nodes = []
        for item in specific_items:
            # Try to match country names or codes
            for node in nodes_data:
                country = node.get('country', {})
                country_name = country.get('name', '').lower()
                country_code = country.get('iso_a2', '').upper()
                
                if (item.lower() in country_name or 
                    item.upper() == country_code):
                    all_nodes.append(node['id'])
        return all_nodes
    
    elif query_type == 'technique':
        # For technique queries, find all nodes offering the technique
        return find_nodes_by_technique(specific_items)
    
    elif query_type == 'listing':
        # For listing queries, return all nodes
        return [node['id'] for node in nodes_data]
    
    return []

def create_search_prompt(query: str) -> str:
    """Create the search prompt with all available data"""
    
    # Create technology index with more details
    tech_index = []
    for tech in tech_data:
        available_count = len(tech.get('provider_node_ids', []))
        tech_index.append(f"{tech['id']}: {tech['name']} (available at {available_count} nodes)")
    
    # Create nodes index with geographic grouping and country codes
    nodes_by_country = {}
    country_codes_info = {}
    for node in nodes_data:
        country = node['country']['name']
        country_code = node['country'].get('iso_a2', 'XX')
        if country not in nodes_by_country:
            nodes_by_country[country] = []
            country_codes_info[country] = country_code
        nodes_by_country[country].append(f"{node['id']}: {node['name']}")
    
    nodes_index = []
    for country, country_nodes in sorted(nodes_by_country.items()):
        country_code = country_codes_info[country]
        nodes_index.append(f"\n**{country} ({country_code}):**")
        nodes_index.extend([f"  - {node}" for node in country_nodes])
    
    # Create website pages index
    website_index = []
    for page in website_data:
        website_index.append(f"{page['id']}: {page['title']} - {page['page_type']}")
    
    prompt = f"""You are a search assistant for Euro-BioImaging infrastructure.
## Available General Website Pages:
{chr(10).join(website_index)}

## Available Technologies: 
{chr(10).join(tech_index)}

## Available Nodes by Country (with ISO codes):
{chr(10).join(nodes_index)}


**AVAILABLE ACTIONS:**
- get_tech_details: Get specific technology details
- get_node_details: Get specific node details  
- get_nodes_by_country: Get ALL nodes in specific countries (use ISO codes)
- get_website_pages: Get website page details for general information


## User Query:
```
{query}
```

**IMPORTANT INSTRUCTIONS:**
1. For GEOGRAPHIC queries (asking about specific countries/regions): 
   - Use country_codes field with ISO codes (e.g., ['PL'] for Poland, ['SE'] for Sweden)
   - Include ALL nodes from that region
2. For TECHNIQUE queries: Map techniques to specific nodes that offer them
3. For GENERAL/INFORMATIONAL queries about Euro-BioImaging: Use website pages
4. Always consider cross-references between technologies, nodes, and website content
5. Prioritize comprehensive coverage over partial listings
6. For country queries, use get_nodes_by_country action with appropriate country codes

Analyze the user query and identify the most relevant technologies, nodes, and/or website pages. Be thorough in geographic coverage."""

    return prompt

async def search_eubio(query: str) -> SearchResponse:
    """Main search function using structured outputs and tool calling"""
    
    # Create the search prompt
    prompt = create_search_prompt(query)
    
    try:
        # First, get the AI to identify relevant items using structured output
        await show_progress("Analyzing query and identifying relevant items")
        identify_response = await client.beta.chat.completions.parse(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": """You are an expert at identifying relevant biomedical imaging technologies and facilities based on user queries.

CRITICAL COVERAGE RULES:
1. GEOGRAPHIC queries (asking about specific countries/regions): Use country_codes field to get ALL nodes from that region
2. TECHNIQUE queries: Find ALL nodes that offer the requested technique, not just a few examples
3. LISTING queries: Be comprehensive and systematic
4. For technique-to-node mapping: Check cross-references carefully and be thorough

COMPREHENSIVE COVERAGE STRATEGY:
- For geographic queries: Always use get_nodes_by_country with appropriate ISO codes
- For technique queries: Include ALL relevant node_ids that offer the technique
- For mixed queries: Combine both approaches
- Always prioritize completeness over brevity

QUERY ANALYSIS:
- Identify query_type: 'geographic', 'technique', 'general', or 'listing'
- Choose coverage_strategy: 'all_nodes_in_region', 'all_techs_for_technique', 'selective', or 'comprehensive_listing'
- Be thorough in your selection to avoid missing relevant items
- For technique queries, consider synonyms and related terms"""},
                {"role": "user", "content": prompt}
            ],
            response_format=ToolCallRequest,
            max_tokens=32768,
            temperature=0.05
        )
        
        tool_request = identify_response.choices[0].message.parsed
        
        # Validate and limit page_ids to prevent excessive processing
        if len(tool_request.page_ids) > 5:
            await show_info(f"Limiting website pages from {len(tool_request.page_ids)} to 5 most relevant")
            tool_request.page_ids = tool_request.page_ids[:5]
        

        
        await show_completion(f"Query classified as: {tool_request.query_type}")
        await show_info(f"Coverage strategy: {tool_request.coverage_strategy}")
        
        # Validate and enhance coverage based on query type
        await show_progress("Validating and enhancing coverage")
        if tool_request.query_type == 'geographic' and not tool_request.country_codes:
            # Extract country information from the query for geographic queries
            query_lower = query.lower()
            country_codes_map = get_country_codes()
            for country_name, country_code in country_codes_map.items():
                if country_name in query_lower:
                    tool_request.country_codes.append(country_code)
        
        elif tool_request.query_type == 'technique':
            # For technique queries, ensure we have comprehensive node coverage
            technique_keywords = []
            query_words = query.lower().split()
            # Extract potential technique keywords from query
            technique_keywords.extend([word for word in query_words if len(word) > 3])
            
            # Find additional nodes that might offer the technique
            additional_nodes = find_nodes_by_technique(technique_keywords)
            for node_id in additional_nodes:
                if node_id not in tool_request.node_ids:
                    tool_request.node_ids.append(node_id)
        
        elif tool_request.query_type == 'listing':
            # For listing queries, ensure all relevant nodes are included
            if 'all' in query.lower() and 'node' in query.lower():
                # Get all nodes for comprehensive listing
                all_node_ids = [node['id'] for node in nodes_data]
                for node_id in all_node_ids:
                    if node_id not in tool_request.node_ids:
                        tool_request.node_ids.append(node_id)
        
        elif tool_request.query_type == 'general' or 'eurobioimaging' in query.lower():
            # For general queries about Euro-BioImaging, include relevant website pages
            query_keywords = [word for word in query.lower().split() if len(word) > 3]
            matching_pages = find_website_pages_by_keywords(query_keywords)
            for page_id in matching_pages[:5]:  # Limit to top 5 relevant pages
                if page_id not in tool_request.page_ids:
                    tool_request.page_ids.append(page_id)
        
        # GENERAL COVERAGE ENHANCEMENT - Improve coverage based on query analysis
        query_lower = query.lower()
        
        # For technique queries, ensure comprehensive node coverage by finding all relevant techniques
        if tool_request.query_type == 'technique':
            # Extract meaningful keywords from the query for technique matching
            query_words = [word for word in query_lower.split() if len(word) > 3]
            
            # Find additional nodes that might offer related techniques
            additional_nodes = find_nodes_by_technique(query_words)
            for node_id in additional_nodes:
                if node_id not in tool_request.node_ids:
                    tool_request.node_ids.append(node_id)
        
        # For geographic queries, ensure comprehensive country coverage
        if tool_request.query_type == 'geographic':
            # Extract all mentioned countries and get their nodes
            country_codes_map = get_country_codes()
            for country_name, country_code in country_codes_map.items():
                if country_name in query_lower or country_code.lower() in query_lower:
                    if country_code not in tool_request.country_codes:
                        tool_request.country_codes.append(country_code)
        
        # Final validation and limit page_ids after all processing
        if len(tool_request.page_ids) > 5:
            await show_info(f"Final page limit: reducing from {len(tool_request.page_ids)} to 5 pages")
            tool_request.page_ids = tool_request.page_ids[:5]
        
        # Gather detailed information
        await show_progress("Gathering detailed information")
        detailed_info = []
        
        # Show what we're collecting
        total_items = len(tool_request.tech_ids) + len(tool_request.node_ids) + len(tool_request.country_codes) + len(tool_request.page_ids)
        if total_items > 0:
            await show_info(f"Collecting details for {len(tool_request.tech_ids)} technologies, {len(tool_request.node_ids)} nodes, {len(tool_request.country_codes)} countries, and {len(tool_request.page_ids)} website pages")
        
        # Get technology details
        for i, tech_id in enumerate(tool_request.tech_ids, 1):
            if len(tool_request.tech_ids) > 3:  # Only show progress for larger collections
                await stream_print(f"  📊 Processing technology {i}/{len(tool_request.tech_ids)}", end="\r")
            tech_detail = read_tech_details(tech_id)
            if tech_detail:
                # Get names of nodes where this tech is available
                node_names = []
                for node_id in tech_detail.provider_node_ids:
                    node_detail = read_node_details(node_id)
                    if node_detail:
                        node_names.append(f"{node_detail.name} ({node_detail.country.get('name', 'N/A')})")
                
                detailed_info.append(f"Technology: {tech_detail.name}\n"
                                   f"Description: {tech_detail.description}\n"
                                   f"Keywords: {', '.join(tech_detail.keywords)}\n"
                                   f"Category: {tech_detail.category.get('name', 'N/A')}\n"
                                   f"Available at nodes: {'; '.join(node_names) if node_names else 'None listed'}")
        
        # Get node details
        for i, node_id in enumerate(tool_request.node_ids, 1):
            if len(tool_request.node_ids) > 3:  # Only show progress for larger collections
                await stream_print(f"  🏢 Processing node {i}/{len(tool_request.node_ids)}", end="\r")
            node_detail = read_node_details(node_id)
            if node_detail:
                # Get names of technologies available at this node
                tech_names = []
                for tech_id in node_detail.offer_technology_ids:
                    tech_detail = read_tech_details(tech_id)
                    if tech_detail:
                        tech_names.append(f"{tech_detail.name} ({tech_detail.category.get('name', 'N/A')})")
                
                detailed_info.append(f"Node: {node_detail.name}\n"
                                   f"Country: {node_detail.country.get('name', 'N/A')}\n"
                                   f"Description: {node_detail.description}\n"
                                   f"Keywords: {', '.join(node_detail.keywords)}\n"
                                   f"Available technologies: {'; '.join(tech_names) if tech_names else 'None listed'}")
        
        # Get nodes by country
        for i, country_code in enumerate(tool_request.country_codes, 1):
            if len(tool_request.country_codes) > 1:
                await stream_print(f"  🌍 Processing country {i}/{len(tool_request.country_codes)}: {country_code}")
            country_nodes = read_nodes_by_country(country_code)
            for node_detail in country_nodes:
                # Get names of technologies available at this node
                tech_names = []
                for tech_id in node_detail.offer_technology_ids:
                    tech_detail = read_tech_details(tech_id)
                    if tech_detail:
                        tech_names.append(f"{tech_detail.name} ({tech_detail.category.get('name', 'N/A')})")
                
                detailed_info.append(f"Node: {node_detail.name}\n"
                                   f"Country: {node_detail.country.get('name', 'N/A')} ({country_code.upper()})\n"
                                   f"Description: {node_detail.description}\n"
                                   f"Keywords: {', '.join(node_detail.keywords)}\n"
                                   f"Available technologies: {'; '.join(tech_names) if tech_names else 'None listed'}")
            
            # Add summary info for the country
            if country_nodes:
                detailed_info.append(f"Country Summary for {country_code.upper()}: {len(country_nodes)} nodes found")
            else:
                detailed_info.append(f"Country Summary for {country_code.upper()}: No nodes found")
        
        # Get website page details
        for i, page_id in enumerate(tool_request.page_ids, 1):
            if len(tool_request.page_ids) > 3:  # Only show progress for larger collections
                await stream_print(f"  📄 Processing page {i}/{len(tool_request.page_ids)}", end="\r")
            page_detail = read_website_page_details(page_id)
            if page_detail:
                detailed_info.append(f"Website Page: {page_detail.title}\n"
                                   f"URL: {page_detail.url}\n"
                                   f"Page Type: {page_detail.page_type}\n"
                                   f"Description: {page_detail.description}\n"
                                   f"Keywords: {', '.join(page_detail.keywords)}\n"
                                   f"Content: {page_detail.documentation}")
        
        if total_items > 3:
            await stream_print("")  # Clear the progress line
        
        # Generate final response with detailed information
        await show_progress("Generating comprehensive response")
        final_prompt = f"""Based on the user query:
```
{query}
```

Query Analysis:
- Query Type: {tool_request.query_type}
- Coverage Strategy: {tool_request.coverage_strategy}
- Selection Reasoning: {tool_request.reasoning}

Coverage Summary:
- Technologies selected: {len(tool_request.tech_ids)}
- Nodes selected: {len(tool_request.node_ids)}
- Countries selected: {len(tool_request.country_codes)}

Detailed information:
```
{chr(10).join(detailed_info)}
```

**RESPONSE REQUIREMENTS:**
1. Provide a comprehensive answer addressing the user's specific question
2. Include ACTIONABLE next steps with specific guidance:
    - How to contact relevant nodes (mention Euro-BioImaging website)
    - How to apply for access to services
    - What information to prepare before contacting nodes
3. For geographic queries: Ensure ALL relevant nodes in the region are mentioned
4. For technique queries: Map specific techniques to ALL available nodes offering them
5. Include practical guidance for researchers with specific next steps
6. Reference Euro-BioImaging infrastructure explicitly
7. Provide contact pathways and access procedures
8. If listing nodes, organize by country/region for clarity
9. Include brief specialty descriptions for each node when relevant
10. Validate comprehensive coverage based on the query type
11. If coverage is comprehensive, mention this explicitly
12. For geographic queries, state "All [country] nodes are included" when applicable
13. Always include specific node names and countries, not just generic statements

Format your response to be maximally helpful, specific, and actionable for researchers seeking to use Euro-BioImaging resources.
"""
        
        final_response = await client.beta.chat.completions.parse(
            model="gpt-4.1",
            messages=[{"role": "user", "content": final_prompt}],
            response_format=SearchResponse,
            max_tokens=32768,
            temperature=0.3
        )
        
        result = final_response.choices[0].message.parsed
        result.relevant_technologies = tool_request.tech_ids
        result.relevant_nodes = tool_request.node_ids
        result.relevant_pages = tool_request.page_ids
        
        # Add country-based nodes to relevant_nodes for comprehensive tracking
        for country_code in tool_request.country_codes:
            country_nodes = read_nodes_by_country(country_code)
            for node in country_nodes:
                if node.id not in result.relevant_nodes:
                    result.relevant_nodes.append(node.id)
        
        await show_completion("Search completed successfully")
        return result
        
    except Exception as e:
        await stream_print(f"❌ Error in search: {e}")
        await show_completion("Search failed due to technical error")
        return SearchResponse(
            answer=f"Sorry, I encountered an error while searching: {e}",
            summary="Search failed due to technical error"
        )

async def search_eubio_simple(query: str) -> SearchResponse:
    """Simple search function without streaming for programmatic use"""
    
    # Create the search prompt
    prompt = create_search_prompt(query)
    
    try:
        # Get the AI to identify relevant items using structured output
        identify_response = await client.beta.chat.completions.parse(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": """You are an expert at identifying relevant biomedical imaging technologies and facilities based on user queries.

CRITICAL COVERAGE RULES:
1. GEOGRAPHIC queries (asking about specific countries/regions): Use country_codes field to get ALL nodes from that region
2. TECHNIQUE queries: Find ALL nodes that offer the requested technique, not just a few examples
3. LISTING queries: Be comprehensive and systematic
4. For technique-to-node mapping: Check cross-references carefully and be thorough

COMPREHENSIVE COVERAGE STRATEGY:
- For geographic queries: Always use get_nodes_by_country with appropriate ISO codes
- For technique queries: Include ALL relevant node_ids that offer the technique
- For mixed queries: Combine both approaches
- Always prioritize completeness over brevity

QUERY ANALYSIS:
- Identify query_type: 'geographic', 'technique', 'general', or 'listing'
- Choose coverage_strategy: 'all_nodes_in_region', 'all_techs_for_technique', 'selective', or 'comprehensive_listing'
- Be thorough in your selection to avoid missing relevant items
- For technique queries, consider synonyms and related terms"""},
                {"role": "user", "content": prompt}
            ],
            response_format=ToolCallRequest,
            max_tokens=32768,
            temperature=0.05
        )
        
        tool_request = identify_response.choices[0].message.parsed
        
        # Validate and limit page_ids to prevent excessive processing
        if len(tool_request.page_ids) > 5:
            tool_request.page_ids = tool_request.page_ids[:5]
        
        # Apply the same validation and enhancement logic as the streaming version
        # (Same logic as in search_eubio function but without streaming calls)
        
        # Validate and enhance coverage based on query type
        if tool_request.query_type == 'geographic' and not tool_request.country_codes:
            query_lower = query.lower()
            country_codes_map = get_country_codes()
            for country_name, country_code in country_codes_map.items():
                if country_name in query_lower:
                    tool_request.country_codes.append(country_code)
        
        elif tool_request.query_type == 'technique':
            technique_keywords = []
            query_words = query.lower().split()
            technique_keywords.extend([word for word in query_words if len(word) > 3])
            additional_nodes = find_nodes_by_technique(technique_keywords)
            for node_id in additional_nodes:
                if node_id not in tool_request.node_ids:
                    tool_request.node_ids.append(node_id)
        
        elif tool_request.query_type == 'listing':
            if 'all' in query.lower() and 'node' in query.lower():
                all_node_ids = [node['id'] for node in nodes_data]
                for node_id in all_node_ids:
                    if node_id not in tool_request.node_ids:
                        tool_request.node_ids.append(node_id)
        
        # Apply same coverage enhancements as streaming version
        query_lower = query.lower()
        
        # For technique queries, ensure comprehensive node coverage
        if tool_request.query_type == 'technique':
            # Extract meaningful keywords from the query for technique matching
            query_words = [word for word in query_lower.split() if len(word) > 3]
            
            # Find additional nodes that might offer related techniques
            additional_nodes = find_nodes_by_technique(query_words)
            for node_id in additional_nodes:
                if node_id not in tool_request.node_ids:
                    tool_request.node_ids.append(node_id)
        
        # For geographic queries, ensure comprehensive country coverage
        if tool_request.query_type == 'geographic':
            country_codes_map = get_country_codes()
            for country_name, country_code in country_codes_map.items():
                if country_name in query_lower or country_code.lower() in query_lower:
                    if country_code not in tool_request.country_codes:
                        tool_request.country_codes.append(country_code)
        
        # Final validation and limit page_ids after all processing
        if len(tool_request.page_ids) > 5:
            tool_request.page_ids = tool_request.page_ids[:5]
        
        # Gather detailed information (same logic as streaming version)
        detailed_info = []
        
        # Get technology details
        for tech_id in tool_request.tech_ids:
            tech_detail = read_tech_details(tech_id)
            if tech_detail:
                node_names = []
                for node_id in tech_detail.provider_node_ids:
                    node_detail = read_node_details(node_id)
                    if node_detail:
                        node_names.append(f"{node_detail.name} ({node_detail.country.get('name', 'N/A')})")
                
                detailed_info.append(f"Technology: {tech_detail.name}\n"
                                   f"Description: {tech_detail.description}\n"
                                   f"Keywords: {', '.join(tech_detail.keywords)}\n"
                                   f"Category: {tech_detail.category.get('name', 'N/A')}\n"
                                   f"Available at nodes: {'; '.join(node_names) if node_names else 'None listed'}")
        
        # Get node details
        for node_id in tool_request.node_ids:
            node_detail = read_node_details(node_id)
            if node_detail:
                tech_names = []
                for tech_id in node_detail.offer_technology_ids:
                    tech_detail = read_tech_details(tech_id)
                    if tech_detail:
                        tech_names.append(f"{tech_detail.name} ({tech_detail.category.get('name', 'N/A')})")
                
                detailed_info.append(f"Node: {node_detail.name}\n"
                                   f"Country: {node_detail.country.get('name', 'N/A')}\n"
                                   f"Description: {node_detail.description}\n"
                                   f"Keywords: {', '.join(node_detail.keywords)}\n"
                                   f"Available technologies: {'; '.join(tech_names) if tech_names else 'None listed'}")
        
        # Get nodes by country
        for country_code in tool_request.country_codes:
            country_nodes = read_nodes_by_country(country_code)
            for node_detail in country_nodes:
                tech_names = []
                for tech_id in node_detail.offer_technology_ids:
                    tech_detail = read_tech_details(tech_id)
                    if tech_detail:
                        tech_names.append(f"{tech_detail.name} ({tech_detail.category.get('name', 'N/A')})")
                
                detailed_info.append(f"Node: {node_detail.name}\n"
                                   f"Country: {node_detail.country.get('name', 'N/A')} ({country_code.upper()})\n"
                                   f"Description: {node_detail.description}\n"
                                   f"Keywords: {', '.join(node_detail.keywords)}\n"
                                   f"Available technologies: {'; '.join(tech_names) if tech_names else 'None listed'}")
            
            if country_nodes:
                detailed_info.append(f"Country Summary for {country_code.upper()}: {len(country_nodes)} nodes found")
            else:
                detailed_info.append(f"Country Summary for {country_code.upper()}: No nodes found")
        
        # Generate final response
        final_prompt = f"""Based on the user query:
```
{query}
```

Query Analysis:
- Query Type: {tool_request.query_type}
- Coverage Strategy: {tool_request.coverage_strategy}
- Selection Reasoning: {tool_request.reasoning}

Coverage Summary:
- Technologies selected: {len(tool_request.tech_ids)}
- Nodes selected: {len(tool_request.node_ids)}
- Countries selected: {len(tool_request.country_codes)}

Detailed information:
```
{chr(10).join(detailed_info)}
```

**RESPONSE REQUIREMENTS:**
1. Provide a comprehensive answer addressing the user's specific question
2. Include ACTIONABLE next steps with specific guidance:
    - How to contact relevant nodes (mention Euro-BioImaging website)
    - How to apply for access to services
    - What information to prepare before contacting nodes
3. For geographic queries: Ensure ALL relevant nodes in the region are mentioned
4. For technique queries: Map specific techniques to ALL available nodes offering them
5. Include practical guidance for researchers with specific next steps
6. Reference Euro-BioImaging infrastructure explicitly
7. Provide contact pathways and access procedures
8. If listing nodes, organize by country/region for clarity
9. Include brief specialty descriptions for each node when relevant
10. Validate comprehensive coverage based on the query type
11. If coverage is comprehensive, mention this explicitly
12. For geographic queries, state "All [country] nodes are included" when applicable
13. Always include specific node names and countries, not just generic statements

Format your response to be maximally helpful, specific, and actionable for researchers seeking to use Euro-BioImaging resources.
"""
        
        final_response = await client.beta.chat.completions.parse(
            model="gpt-4.1",
            messages=[{"role": "user", "content": final_prompt}],
            response_format=SearchResponse,
            max_tokens=32768,
            temperature=0.3
        )
        
        result = final_response.choices[0].message.parsed
        result.relevant_technologies = tool_request.tech_ids
        result.relevant_nodes = tool_request.node_ids
        result.relevant_pages = tool_request.page_ids
        
        # Add country-based nodes to relevant_nodes
        for country_code in tool_request.country_codes:
            country_nodes = read_nodes_by_country(country_code)
            for node in country_nodes:
                if node.id not in result.relevant_nodes:
                    result.relevant_nodes.append(node.id)
        
        return result
        
    except Exception as e:
        return SearchResponse(
            answer=f"Sorry, I encountered an error while searching: {e}",
            summary="Search failed due to technical error"
        )

async def main():
    parser = argparse.ArgumentParser(description='Search Euro-BioImaging technologies and nodes')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--test', action='store_true', help='Use test data instead of full dataset')
    parser.add_argument('--no-stream', action='store_true', help='Disable streaming output for cleaner logs')
    parser.add_argument('--data-dir', default='euro-bioimaging-index', help='Directory containing indexed data files (default: euro-bioimaging-index)')
    
    args = parser.parse_args()
    
    # Load the data
    await show_progress("Loading Euro-BioImaging data")
    load_data(test_mode=args.test, data_dir=args.data_dir)
    
    if not tech_data and not nodes_data:
        await stream_print("❌ No data loaded. Please ensure the index files exist.")
        return
    
    dataset_type = 'test' if args.test else 'full'
    await show_completion(f"Data loaded successfully ({dataset_type} dataset)")
    await show_info(f"Technologies: {len(tech_data)}, Nodes: {len(nodes_data)}")
    
    # Perform the search
    await show_section_header(f"Searching for: {args.query}")
    
    # Show search initiation
    if not args.no_stream:
        await show_progress("Initializing search system")
        await asyncio.sleep(0.1)  # Brief pause for better UX
    
    # Choose search function based on streaming preference
    if args.no_stream:
        result = await search_eubio_simple(args.query)
    else:
        result = await search_eubio(args.query)
    
    # Display results with streaming
    if not args.no_stream:
        await show_section_header("Search Results")
        
        # Stream the answer
        if result.answer:
            await stream_print("📄 **Response:**")
            await stream_print("")
            
            # Split answer into paragraphs for better streaming
            paragraphs = result.answer.split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    await stream_print(paragraph.strip())
                    await stream_print("")
                    await asyncio.sleep(0.1)  # Small delay between paragraphs
        
        # Show metadata
        if result.relevant_technologies or result.relevant_nodes:
            await show_section_header("Search Metadata")
            
            if result.relevant_technologies:
                await stream_print(f"🔬 **Relevant Technologies:** {len(result.relevant_technologies)}")
                for tech_id in result.relevant_technologies[:5]:  # Show first 5
                    tech_detail = read_tech_details(tech_id)
                    if tech_detail:
                        await stream_print(f"  • {tech_detail.name}")
                if len(result.relevant_technologies) > 5:
                    await stream_print(f"  • ... and {len(result.relevant_technologies) - 5} more")
                await stream_print("")
            
            if result.relevant_nodes:
                await stream_print(f"🏢 **Relevant Nodes:** {len(result.relevant_nodes)}")
                for node_id in result.relevant_nodes[:5]:  # Show first 5
                    node_detail = read_node_details(node_id)
                    if node_detail:
                        await stream_print(f"  • {node_detail.name} ({node_detail.country.get('name', 'N/A')})")
                if len(result.relevant_nodes) > 5:
                    await stream_print(f"  • ... and {len(result.relevant_nodes) - 5} more")
                await stream_print("")
        
        # Final completion message
        await show_section_header("Search Complete")
        await show_completion(f"Search completed using {dataset_type} dataset")
        await show_info("For more information, visit: https://www.eurobioimaging.eu/")
        
        # Performance summary
        total_resources = len(result.relevant_technologies) + len(result.relevant_nodes)
        if total_resources > 0:
            await show_info(f"Found {total_resources} relevant resources across Euro-BioImaging infrastructure")
    else:
        # Non-streaming output
        print("\nSearch Results:")
        print("=" * 50)
        print(result.answer)
        print("\n" + "=" * 50)
        print(f"Search completed using {dataset_type} dataset")
        total_resources = len(result.relevant_technologies) + len(result.relevant_nodes)
        if total_resources > 0:
            print(f"Found {total_resources} relevant resources across Euro-BioImaging infrastructure")

if __name__ == "__main__":
    asyncio.run(main()) 