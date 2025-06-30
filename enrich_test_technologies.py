#!/usr/bin/env python3
"""
Script to enrich the first 3 technologies from test_eurobioimaging_index.json
using the research agent and save results as markdown files for comparison.
"""

import asyncio
import json
import os
from pathlib import Path
from research_agent import enrich_description

async def enrich_and_save_technologies():
    """Load test data, enrich first 3 technologies, and save as markdown files"""
    
    # Load the test index
    with open('euro-bioimaging-index/test_eurobioimaging_index.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    technologies = data['technologies'][:3]  # First 3 technologies
    
    # Create output directory if it doesn't exist
    output_dir = Path('euro-bioimaging-index')
    output_dir.mkdir(exist_ok=True)
    
    for i, tech in enumerate(technologies, 1):
        print(f"\n{'='*60}")
        print(f"ENRICHING TECHNOLOGY {i}/3: {tech['name']}")
        print(f"{'='*60}")
        
        # Current description info
        current_desc = tech.get('description', '')
        current_long_desc = tech.get('documentation', '')
        
        print(f"Current description length: {len(current_desc)} chars")
        print(f"Current documentation length: {len(current_long_desc)} chars")
        
        # Create a simplified tech object for enrichment
        tech_for_enrichment = {
            'name': tech['name'],
            'description': current_desc,
            'long_description': ''  # Force enrichment by making this empty
        }
        
        try:
            # Enrich the technology
            enriched_tech = await enrich_description(tech_for_enrichment, "technology")
            
            # Get enriched descriptions
            enriched_short_desc = enriched_tech.get('description', '')
            enriched_long_desc = enriched_tech.get('long_description', '')
            enrichment_meta = enriched_tech.get('enrichment_metadata', {})
            
            print(f"Enriched short description length: {len(enriched_short_desc)} chars")
            print(f"Enriched long description length: {len(enriched_long_desc)} chars")
            print(f"Confidence score: {enrichment_meta.get('confidence_score', 'N/A')}")
            
            # Create markdown content
            markdown_content = f"""# {tech['name']}

## Technology Information
- **ID**: {tech['id']}
- **Category**: {tech.get('category', {}).get('name', 'N/A')}

## Original Description
{current_desc}

## Original Documentation
{current_long_desc}

## Enriched Description (Concise)
{enriched_short_desc}

## Enriched Documentation (Detailed)
{enriched_long_desc}

## Enrichment Metadata
- **Enriched**: {enrichment_meta.get('enriched', False)}
- **Confidence Score**: {enrichment_meta.get('confidence_score', 'N/A')}
- **Sources Used**: {', '.join(enrichment_meta.get('sources_used', []))}
- **Original Description Length**: {enrichment_meta.get('original_description_length', 'N/A')} chars
- **Original Documentation Length**: {enrichment_meta.get('original_documentation_length', 'N/A')} chars
- **Enriched Description Length**: {enrichment_meta.get('enriched_description_length', 'N/A')} chars
- **Enriched Documentation Length**: {enrichment_meta.get('enriched_documentation_length', 'N/A')} chars

## Keywords
{', '.join(tech.get('keywords', []))}
"""
            
            # Save as markdown file
            filename = f"enriched_{tech['id']}_{tech['name'].replace('/', '_').replace('*', '').replace('(', '').replace(')', '').replace(' ', '_')}.md"
            filepath = output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            print(f"✅ Saved enriched documentation to: {filepath}")
            
        except Exception as e:
            print(f"❌ Error enriching {tech['name']}: {e}")
            
            # Create a simple markdown file with error info
            error_content = f"""# {tech['name']} - ENRICHMENT FAILED

## Error
{str(e)}

## Original Description
{current_desc}

## Original Documentation
{current_long_desc}
"""
            filename = f"error_{tech['id']}_{tech['name'].replace('/', '_').replace('*', '').replace('(', '').replace(')', '').replace(' ', '_')}.md"
            filepath = output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(error_content)
            
            print(f"⚠️  Saved error info to: {filepath}")
    
    print(f"\n{'='*60}")
    print("ENRICHMENT COMPLETE!")
    print(f"Check the '{output_dir}' directory for the markdown files.")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(enrich_and_save_technologies()) 