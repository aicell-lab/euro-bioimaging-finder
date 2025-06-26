import micropip
await micropip.install(["httpx", "pydantic"])

import httpx
from typing import Dict, List, Any
from pydantic import BaseModel

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
    content_preview: str
    headings: List[str]
    page_type: str

class SearchResponse(BaseModel):
    """Structured search response"""
    answer: str
    relevant_technologies: List[str] = []
    relevant_nodes: List[str] = []
    relevant_pages: List[str] = []
    summary: str = ""

# Global data storage
tech_data: List[Dict[str, Any]] = []
nodes_data: List[Dict[str, Any]] = []
website_data: List[Dict[str, Any]] = []

async def load_eurobioimaging_data():
    """
    Load Euro-BioImaging data from remote index file.
    This function fetches the combined index from the remote URL and populates the global data structures.
    
    Returns:
    - dict: Metadata about the loaded dataset including counts and creation info
    """
    global tech_data, nodes_data, website_data
    
    url = "https://oeway.github.io/euro-bioimaging/eurobioimaging_index.json"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            combined_data = response.json()
        
        # Extract data from combined index
        tech_data = combined_data.get('technologies', [])
        nodes_data = combined_data.get('nodes', [])
        website_data = combined_data.get('website_pages', [])
        
        # Get metadata
        metadata = combined_data.get('metadata', {})
        dataset_info = {
            'dataset_type': metadata.get('dataset_type', 'unknown'),
            'created_at': metadata.get('created_at', 'unknown'),
            'technologies_count': len(tech_data),
            'nodes_count': len(nodes_data),
            'website_pages_count': len(website_data),
            'total_entries': len(tech_data) + len(nodes_data) + len(website_data)
        }
        
        return dataset_info
        
    except Exception as e:
        print(f"Error loading data: {e}")
        tech_data = []
        nodes_data = []
        website_data = []
        return {'error': str(e)}

def read_tech_details(tech_id: str):
    """
    Read details of a specific technology by ID.
    
    Parameters:
    - tech_id (str): The technology ID
    
    Returns:
    - Optional[TechDetail]: Technology details or None if not found
    """
    for tech in tech_data:
        if tech['id'] == tech_id:
            return TechDetail(**tech)
    return None

def read_node_details(node_id: str):
    """
    Read details of a specific node by ID.
    
    Parameters:
    - node_id (str): The node ID
    
    Returns:
    - Optional[NodeDetail]: Node details or None if not found
    """
    for node in nodes_data:
        if node['id'] == node_id:
            return NodeDetail(**node)
    return None

def read_nodes_by_country(country_code: str):
    """
    Read all nodes in a specific country by ISO country code.
    
    Parameters:
    - country_code (str): ISO country code (e.g., 'PL', 'SE', 'DE')
    
    Returns:
    - List[NodeDetail]: List of all nodes in the specified country
    """
    country_nodes = []
    for node in nodes_data:
        node_country = node.get('country', {})
        if node_country.get('iso_a2', '').upper() == country_code.upper():
            try:
                country_nodes.append(NodeDetail(**node))
            except Exception as e:
                print(f"Warning: Could not parse node {node.get('name', 'Unknown')}: {e}")
    return country_nodes

def read_website_page_details(page_id: str):
    """
    Read details of a specific website page by ID.
    
    Parameters:
    - page_id (str): The website page ID
    
    Returns:
    - Optional[WebsitePageDetail]: Website page details or None if not found
    """
    for page in website_data:
        if page['id'] == page_id:
            return WebsitePageDetail(**page)
    return None

def get_country_codes():
    """
    Get mapping of country names to ISO codes from the data.
    
    Returns:
    - Dict[str, str]: Mapping of country names to ISO codes
    """
    country_mapping = {}
    for node in nodes_data:
        country = node.get('country', {})
        country_name = country.get('name', '')
        country_code = country.get('iso_a2', '')
        if country_name and country_code:
            country_mapping[country_name.lower()] = country_code.upper()
    return country_mapping

def find_nodes_by_technique(technique_keywords: List[str]):
    """
    Find node IDs that offer technologies matching the given technique keywords.
    
    Parameters:
    - technique_keywords (List[str]): List of technique-related keywords
    
    Returns:
    - List[str]: List of node IDs offering the specified techniques
    """
    matching_node_ids = set()
    
    # Search through all technologies to find matches
    for tech in tech_data:
        tech_name = tech.get('name', '').lower()
        tech_keywords = [kw.lower() for kw in tech.get('keywords', [])]
        tech_category = tech.get('category', {}).get('name', '').lower()
        tech_abbr = tech.get('abbr', '').lower()
        tech_description = tech.get('documentation', '').lower()
        
        match_score = 0
        matched_keywords = 0
        
        for keyword in technique_keywords:
            keyword_lower = keyword.lower()
            
            if keyword_lower == tech_name or keyword_lower == tech_abbr:
                match_score += 10
                matched_keywords += 1
            elif keyword_lower in tech_name or keyword_lower in tech_abbr:
                match_score += 5
                matched_keywords += 1
            elif keyword_lower in tech_category:
                match_score += 3
                matched_keywords += 1
            elif any(keyword_lower in kw or kw in keyword_lower for kw in tech_keywords):
                match_score += 2
                matched_keywords += 1
            elif keyword_lower in tech_description:
                match_score += 1
                matched_keywords += 1
        
        if match_score >= 2 or matched_keywords >= 2:
            matching_node_ids.update(tech.get('provider_node_ids', []))
    
    return list(matching_node_ids)

def find_website_pages_by_keywords(keywords: List[str]):
    """
    Find website page IDs that match the given keywords with relevance scoring.
    
    Parameters:
    - keywords (List[str]): List of keywords to search for
    
    Returns:
    - List[str]: List of website page IDs matching the keywords
    """
    page_scores = []
    
    for page in website_data:
        page_title = page.get('title', '').lower()
        page_description = page.get('description', '').lower()
        page_keywords = [kw.lower() for kw in page.get('keywords', [])]
        page_content = page.get('content_preview', '').lower()
        page_type = page.get('page_type', '').lower()
        
        score = 0
        matched_keywords = 0
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in page_title:
                score += 5
                matched_keywords += 1
            elif keyword_lower in page_description:
                score += 3
                matched_keywords += 1
            elif any(keyword_lower in kw for kw in page_keywords):
                score += 2
                matched_keywords += 1
            elif keyword_lower in page_content:
                score += 1
                matched_keywords += 1
        
        if 'about' in page_type or 'service' in page_type or 'access' in page_type:
            score += 2
        
        if matched_keywords > 0 and score >= 2:
            page_scores.append((page['id'], score, matched_keywords))
    
    page_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [page_id for page_id, _, _ in page_scores[:10]]

def create_search_prompt():
    """Create the search prompt with all available data indexes"""
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
    
    prompt = f"""
Here is a list of all the available resources associated with their IDs.

## Available General Website Pages:
{chr(10).join(website_index)}

## Available Technologies: 
{chr(10).join(tech_index)}

## Available Nodes by Country (with ISO codes):
{chr(10).join(nodes_index)}

**IMPORTANT**: Each entry in the indexes above follows the format `ID: Summary/Description`

- For **Technologies**: `tech_id: Technology Name (available at X nodes)`
- For **Nodes**: `node_id: Node Name` 
- For **Website Pages**: `page_id: Page Title - page_type`

For any matched topics, you can call the utility functions to get more details by the ID.

**To use the utility functions**: Extract the ID (the part before the colon) and pass it to the appropriate function:
- Use `tech_id` with `read_tech_details(tech_id)`
- Use `node_id` with `read_node_details(node_id)` 
- Use `page_id` with `read_website_page_details(page_id)`

These utilities are available under the current execution context, and no need to use await to call them. For example:

```python
# Example: Get details of a specific technology
tech_details = read_tech_details("02e4459a")
print(f"Technology: {{tech_details.name}}")
print(f"Description: {{tech_details.description}}")
```

Imprtantly, they are not under api.* namespace, so you can call them directly.

"""

    return prompt

# Initialize data and create the search prompt
data_loaded = await load_eurobioimaging_data()

if not data_loaded:
    print("WARNING: ❌ Failed to load Euro-BioImaging data, none of the following functions will work properly.")

print("""
You are an AI assistant specialized in helping users discover imaging technologies, instruments, and services provided by the Euro-BioImaging network.

You have access to several utility functions for retrieving specific information about Euro-BioImaging facilities. Below are all the available resources in the Euro-BioImaging network:
""")

# Print the search context with all available data
search_context = create_search_prompt()
print(search_context)

print("""## 🛠️ Available Utility Functions

**IMPORTANT**: Call these functions directly - NO `await`, NO `api.` prefix, NO keyword arguments!

### Detail Retrieval Functions
- `read_tech_details(tech_id)` → `TechDetail(id, name, original_id, description, keywords, documentation, category, provider_node_ids, abbr)` or `None`
- `read_node_details(node_id)` → `NodeDetail(id, name, original_id, description, keywords, documentation, country, offer_technology_ids)` or `None`  
- `read_nodes_by_country(country_code)` → `List[NodeDetail(id, name, original_id, description, keywords, documentation, country, offer_technology_ids)]`
- `read_website_page_details(page_id)` → `WebsitePageDetail(id, url, title, description, keywords, content_preview, headings, page_type)` or `None`
The returned objects are pydantic models, so you can access the attributes directly. For example: tech_details.name, node_details.country['name'], etc.

### Search Helper Functions
- `find_nodes_by_technique(technique_keywords)` → `List[str]` (node IDs)
- `find_website_pages_by_keywords(keywords)` → `List[str]` (page IDs)
- `get_country_codes()` → `Dict[str, str]` (country_name → ISO_code)


## ✅ Correct Function Call Examples

```python
# CORRECT: Get node details
node_details = read_node_details("48c78d41")
print(f"Node: {node_details.name}")
print(f"Country: {node_details.country['name']}")
print(f"Description: {node_details.description}")

# CORRECT: Get technology details  
tech_details = read_tech_details("02e4459a")
print(f"Technology: {tech_details.name}")
print(f"Category: {tech_details.category['name']}")
print(f"Available at {len(tech_details.provider_node_ids)} nodes")

# CORRECT: Get all nodes in Germany
german_nodes = read_nodes_by_country("DE")
for node in german_nodes:
    print(f"Node: {node.name}")
    print(f"Technologies: {len(node.offer_technology_ids)} available")

# CORRECT: Find nodes by technique
node_ids = find_nodes_by_technique(["microscopy", "imaging"])
for node_id in node_ids:
    details = read_node_details(node_id)
    print(f"Node: {details.name} in {details.country['name']}")
```

## ❌ WRONG Function Call Examples

```python
# WRONG: Don't use await
node_details = await read_node_details("48c78d41")  # ❌

# WRONG: Don't use api prefix
node_details = api.read_node_details("48c78d41")  # ❌

# WRONG: Don't use keyword arguments
node_details = read_node_details(node_id="48c78d41")  # ❌

# WRONG: Don't combine await + api
node_details = await api.read_node_details(node_id="48c78d41")  # ❌
```

## 🧠 How to Answer User Queries

Follow this systematic approach:

### 1. **Analyze the Query Type**
- **Geographic queries** (e.g., "facilities in Germany"): Focus on specific countries/regions
- **Technology queries** (e.g., "super-resolution microscopy"): Focus on specific techniques or instruments  
- **General information** (e.g., "how to access services"): Focus on website pages and general guidance
- **Listing queries** (e.g., "all available nodes"): Provide comprehensive listings

### 2. **Select Relevant IDs**
From the indexes above, identify the specific IDs you need by looking at the summaries:
- **Technology IDs**: Look for relevant techniques in the technology list, extract the ID before the colon
- **Node IDs**: Look for relevant facilities in the node list, extract the ID before the colon
- **Country codes**: Use the ISO codes in parentheses (e.g., DE, SE, FR)
- **Page IDs**: Look for relevant information pages, extract the ID before the colon

**Example**: If you see `tech_super_res_microscopy: Super-Resolution Microscopy (available at 15 nodes)`, use `tech_super_res_microscopy` as the ID.

### 3. **Retrieve Detailed Information**
Use the utility functions systematically:

**For Technology Queries:**
```python
# Example: "Where can I find MINFLUX?"
# Step 1: Get the specific technology details
minflux_details = read_tech_details("4bf1cfdd")  # Extract ID from index
print(f"Technology: {minflux_details.name}")

# Step 2: Get nodes where it's available (don't list all their technologies)
for node_id in minflux_details.provider_node_ids:
    node_details = read_node_details(node_id)
    print(f"Available at: {node_details.name} in {node_details.country['name']}")
```

**For Geographic Queries:**
```python
# Example: "What imaging facilities are in Germany?"
country_nodes = read_nodes_by_country("DE")  # Germany
for node in country_nodes:
    print(f"Node: {node.name}")
    print(f"Description: {node.description}")
    print(f"Technologies available: {len(node.offer_technology_ids)} total")
    # Only list technologies if specifically asked, don't retrieve all details
```

**For General Information:**
```python
# Find relevant website pages
page_ids = find_website_pages_by_keywords(["access", "services"])
for page_id in page_ids:
    page_details = read_website_page_details(page_id)
    print(f"Page: {page_details.title}")
    print(f"URL: {page_details.url}")
    print(f"Description: {page_details.description}")
```

### 4. **Be Focused and Efficient**
- Only retrieve information that directly answers the user's question
- Don't list all technologies at a node unless specifically asked
- Don't retrieve all details unless needed for the specific query
- Answer the question, don't provide unnecessary comprehensive listings

### 5. **Reason Across Results**
- Integrate information from multiple search results if needed
- Draw conclusions, extract availability details, instrument capabilities, node locations, and relevant contact information

### 6. **Respond Factually and Comprehensively**
- Ensure your final reply contains all key findings and actionable information
- Always include all relevant details in your final answer, the intermediate script output won't be seen by the user
- Avoid vague summaries—be precise and specific based on the returned context

## 📋 Query Pattern Examples

**Geographic Query**: "What imaging facilities are available in Germany?"
1. Use `read_nodes_by_country("DE")` to get all German nodes
2. For each node, use `read_node_details()` to get full information
3. For each technology at the nodes, use `read_tech_details()` to understand capabilities

**Technology Query**: "Where can I access super-resolution microscopy?"
1. Use `find_nodes_by_technique(["super-resolution", "microscopy"])` to find relevant nodes
2. Use `read_node_details()` for each node to get contact information
3. Use `read_tech_details()` to understand specific capabilities

**General Information**: "How do I access Euro-BioImaging services?"
1. Use `find_website_pages_by_keywords(["access", "services", "application"])` to find relevant pages
2. Use `read_website_page_details()` for each page to get detailed information

## 🎯 Response Quality Guidelines
0. When generating the thoughts, limit each thought to 5 words maximum.
1. **Use the indexes above** to identify relevant resources by their summaries
2. **Extract IDs correctly** (the part before the colon) from the index entries
3. **Call functions directly** without await, api prefix, or keyword arguments
4. **Be systematic** in retrieving details for all relevant items
5. **Include geographic context** (country names and codes)
6. **Provide comprehensive information** which corresponds to the user query, try to be helpful and comprehensive.
7. Since we are targeting non-technical users, unless asked, you can generate and execute scripts, but no need to commit code during the final response.

Remember: Scan the indexes for relevant entries, extract the correct IDs, then call the utility functions directly to get detailed information.
""") 
