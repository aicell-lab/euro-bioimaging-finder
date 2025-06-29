#!/usr/bin/env python3
"""
Publish Euro-BioImaging index to GitHub Pages

This script publishes the generated index to the gh-pages branch for GitHub Pages hosting.
It creates a static website with the index data and API endpoints.
"""

import json
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import tempfile
import os

def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr and check:
        print(f"Error: {result.stderr.strip()}")
    return result

def check_git_status():
    """Check if git working directory is clean"""
    try:
        result = run_command(['git', 'status', '--porcelain'], check=False)
        return len(result.stdout.strip()) == 0
    except subprocess.CalledProcessError:
        return False

def get_current_branch():
    """Get the current git branch"""
    result = run_command(['git', 'branch', '--show-current'])
    return result.stdout.strip()

def get_stable_timestamp(index_file):
    """Get a stable timestamp based on the index file modification time to avoid conflicts"""
    if index_file.exists():
        # Use the index file's modification time for stable timestamps
        mtime = index_file.stat().st_mtime
        return datetime.fromtimestamp(mtime)
    else:
        # Fallback to current time, but only use date (not time) to reduce conflicts
        return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

def create_index_html(index_file, output_dir):
    """Create index.html for GitHub Pages"""
    
    # Get stable timestamp
    timestamp = get_stable_timestamp(index_file)
    
    # Load index data for stats
    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        stats = {
            'technologies': len(data.get('technologies', [])),
            'nodes': len(data.get('nodes', [])),
            'website_pages': len(data.get('website_pages', [])),
            'total_entries': len(data.get('technologies', [])) + len(data.get('nodes', [])) + len(data.get('website_pages', []))
        }
        
        metadata = data.get('metadata', {})
        dataset_type = metadata.get('dataset_type', 'unknown')
        created_at = timestamp.strftime('%Y-%m-%d')  # Only date, not time
        
    except Exception as e:
        print(f"Warning: Could not load index data for stats: {e}")
        stats = {'technologies': 'N/A', 'nodes': 'N/A', 'website_pages': 'N/A', 'total_entries': 'N/A'}
        dataset_type = 'unknown'
        created_at = timestamp.strftime('%Y-%m-%d')
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Euro-BioImaging Search Index</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 2rem;
            background: #f8f9fa;
        }}
        .header {{
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            color: #007bff;
            margin-bottom: 0.5rem;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }}
        .stat-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-number {{
            font-size: 2rem;
            font-weight: bold;
            color: #007bff;
        }}
        .download-section, .api-section {{
            background: white;
            padding: 2rem;
            margin: 2rem 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .download-link {{
            display: inline-block;
            background: #007bff;
            color: white;
            padding: 1rem 2rem;
            text-decoration: none;
            border-radius: 5px;
            margin: 1rem 0;
            font-weight: bold;
        }}
        .download-link:hover {{
            background: #0056b3;
        }}
        .code-block {{
            background: #f1f3f4;
            padding: 1rem;
            border-radius: 4px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9rem;
            border-left: 4px solid #007bff;
            margin: 1rem 0;
            overflow-x: auto;
            white-space: pre;
        }}
        
        .data-structure {{
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Euro-BioImaging Search Index</h1>
        <p>Comprehensive search index for Euro-BioImaging technologies, nodes, and resources</p>
    </div>
    
    <div class="stats">
        <div class="stat-card">
            <div class="stat-number">{stats.get('technologies', 'N/A')}</div>
            <div>Technologies</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{stats.get('nodes', 'N/A')}</div>
            <div>Nodes</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{stats.get('website_pages', 'N/A')}</div>
            <div>Website Pages</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{stats.get('total_entries', 'N/A')}</div>
            <div>Total Entries</div>
        </div>
    </div>
    
    <div class="download-section">
        <h2>📥 Download Index</h2>
        <p>Download the complete Euro-BioImaging search index in JSON format:</p>
        <a href="eurobioimaging_index.json" class="download-link">📄 Download JSON Index</a>
        <p><strong>Dataset:</strong> {dataset_type.title()}<br>
        <strong>Last Updated:</strong> {created_at}</p>
    </div>
    
    <div class="api-section">
        <h2>🔗 API Access</h2>
        <p>Access the index programmatically via HTTPS:</p>
        <div class="code-block">
# Main index with all data
https://oeway.github.io/euro-bioimaging-finder/eurobioimaging_index.json

# BM25 index file (required for full-text search)
https://oeway.github.io/euro-bioimaging-finder/eurobioimaging_bm25_index.pkl</div>
    </div>
    
    <div class="api-section">
        <h2>📋 Data Structure</h2>
        <p>The JSON index contains the following structure:</p>
        <div class="data-structure">
            <h3>Combined Index (eurobioimaging_index.json)</h3>
            <div class="code-block">
{{
    "metadata": {{
        "created_at": "ISO timestamp",
        "version": "1.0",
        "description": "Euro-BioImaging combined search index",
        "dataset_type": "full|test",
        "statistics": {{ ... }}
    }},
    "technologies": [
        {{
            "id": "unique_id",
            "name": "Technology Name",
            "description": "Description",
            "keywords": ["keyword1", "keyword2"],
            "documentation": "Full documentation text",
            "category": {{ "name": "Category Name", ... }},
            "provider_node_ids": ["node_id1", "node_id2"]
        }}
    ],
    "nodes": [
        {{
            "id": "unique_id",
            "name": "Node Name",
            "description": "Description",
            "keywords": ["keyword1", "keyword2"],
            "documentation": "Full documentation text",
            "country": {{ "name": "Country Name", "iso_a2": "ISO Code" }},
            "offer_technology_ids": ["tech_id1", "tech_id2"]
        }}
    ],
    "website_pages": [
        {{
            "id": "unique_id",
            "url": "Page URL",
            "title": "Page Title",
            "description": "Description",
            "keywords": ["keyword1", "keyword2"],
            "content_preview": "Content preview text",
            "headings": ["heading1", "heading2"],
            "page_type": "about|services|nodes|etc"
        }}
    ]
}}</div>
        </div>
        
        <div class="data-structure">
            <h3>BM25 Full-Text Search</h3>
            <p>The BM25 index is stored in a pickle file that contains:</p>
            <ul>
                <li><strong>retriever</strong>: The BM25 retriever object for performing searches</li>
            </ul>
            <p>The document metadata for mapping search results is stored in the main JSON index under <code>bm25_metadata</code>.</p>
        </div>
        
        <h3>Example Usage</h3>
        <p>Python with bm25s package:</p>
        <div class="code-block">
import pickle
import bm25s
import httpx
import json

# Download and load main index with metadata
response = httpx.get('https://oeway.github.io/euro-bioimaging-finder/eurobioimaging_index.json')
combined_data = response.json()
bm25_metadata = combined_data.get('bm25_metadata', [])

# Download and load BM25 retriever
response = httpx.get('https://oeway.github.io/euro-bioimaging-finder/eurobioimaging_bm25_index.pkl')
with open('eurobioimaging_bm25_index.pkl', 'wb') as f:
    f.write(response.content)

with open('eurobioimaging_bm25_index.pkl', 'rb') as f:
    retriever = pickle.load(f)

def fulltext_search(query, k=5):
    query_tokens = bm25s.tokenize(query)
    results, scores = retriever.retrieve(query_tokens, k=k)
    hits = []
    for i in range(results.shape[1]):
        doc_idx = results[0, i]
        score = scores[0, i]
        metadata = bm25_metadata[doc_idx]
        hit = metadata.copy()
        hit["score"] = float(score)
        hits.append(hit)
    return hits

# Example search
results = fulltext_search("super resolution microscopy", k=5)
for hit in results:
    print(f"Score: {hit['score']:.2f} - Type: {hit['type']} - ID: {hit['id']}")</div>
    </div>
    
    <footer style="text-align: center; margin-top: 3rem; padding-top: 2rem; border-top: 1px solid #e0e0e0; color: #666;">
        <p>🔬 Euro-BioImaging Search Index | Updated on {created_at}</p>
        <p><a href="https://www.eurobioimaging.eu/">Visit Euro-BioImaging</a></p>
    </footer>
</body>
</html>"""
    
    # Save HTML file
    html_file = output_dir / 'index.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Created index.html at {html_file}")

def check_remote_origin():
    """Check if remote origin is configured"""
    try:
        result = run_command(['git', 'remote', 'get-url', 'origin'], check=False)
        if result.returncode == 0:
            remote_url = result.stdout.strip()
            print(f"🌐 Remote origin: {remote_url}")
            return True
        else:
            print("❌ No remote origin configured")
            return False
    except Exception:
        print("❌ Error checking remote origin")
        return False

def publish_to_gh_pages(data_dir, index_filename, force=False, message=None):
    """Publish the index to gh-pages branch"""
    
    data_path = Path(data_dir)
    index_file = data_path / index_filename
    bm25_file = data_path / "eurobioimaging_bm25_index.pkl"
    
    if not index_file.exists():
        print(f"❌ Index file not found: {index_file}")
        return False
    
    if not bm25_file.exists():
        print(f"❌ BM25 index file not found: {bm25_file}")
        return False
    
    # Check if we have a remote origin
    if not check_remote_origin():
        print("❌ No remote repository configured. Cannot publish to GitHub Pages.")
        return False
    
    # Check git status
    if not force and not check_git_status():
        print("❌ Working directory is not clean. Use --force to proceed anyway.")
        return False
    
    # Get current branch
    current_branch = get_current_branch()
    print(f"📍 Current branch: {current_branch}")
    
    # Get stable timestamp based on index file
    stable_timestamp = get_stable_timestamp(index_file)
    
    # Create temporary directory for gh-pages content
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        gh_pages_dir = temp_path / 'gh-pages'
        
        print("🔄 Setting up gh-pages branch...")
        
        # Get the remote origin URL first
        result = run_command(['git', 'remote', 'get-url', 'origin'])
        remote_url = result.stdout.strip()
        print(f"🌐 Using remote URL: {remote_url}")
        
        # Clone from the actual remote, not local directory
        run_command(['git', 'clone', remote_url, str(gh_pages_dir)])
        
        # Switch to gh-pages branch (create if doesn't exist)
        try:
            run_command(['git', 'checkout', 'gh-pages'], cwd=gh_pages_dir)
        except subprocess.CalledProcessError:
            print("📝 Creating new gh-pages branch...")
            run_command(['git', 'checkout', '--orphan', 'gh-pages'], cwd=gh_pages_dir)
            # Remove all files from the orphan branch
            run_command(['git', 'rm', '-rf', '.'], cwd=gh_pages_dir, check=False)
        
        # Before making changes, pull latest to avoid conflicts
        try:
            run_command(['git', 'pull', 'origin', 'gh-pages'], cwd=gh_pages_dir, check=False)
        except subprocess.CalledProcessError:
            print("📝 No existing gh-pages to pull from")
        
        # Copy index file to gh-pages directory
        target_index = gh_pages_dir / 'eurobioimaging_index.json'
        shutil.copy2(index_file, target_index)
        print(f"📄 Copied {index_file} to {target_index}")
        
        # Copy .nojekyll file if it exists
        nojekyll_file = Path('.nojekyll')
        if nojekyll_file.exists():
            target_nojekyll = gh_pages_dir / '.nojekyll'
            shutil.copy2(nojekyll_file, target_nojekyll)
            print(f"📄 Copied .nojekyll to disable Jekyll processing")
        
        # Copy BM25 pickle file
        target_bm25_file = gh_pages_dir / 'eurobioimaging_bm25_index.pkl'
        shutil.copy2(bm25_file, target_bm25_file)
        print(f"📄 Copied BM25 index to {target_bm25_file}")
        
        # Create index.html with updated documentation
        create_index_html(index_file, gh_pages_dir)
        
        # Create README.md for gh-pages with stable timestamp
        readme_content = f"""# Euro-BioImaging Search Index

This repository hosts the Euro-BioImaging search index for public access.

## 📥 Access the Index

- **JSON Index**: [eurobioimaging_index.json](eurobioimaging_index.json)
- **BM25 Index**: [eurobioimaging_bm25_index.pkl](eurobioimaging_bm25_index.pkl)
- **Web Interface**: [index.html](index.html)

## 🔗 Direct Links

- JSON API: `https://oeway.github.io/euro-bioimaging-finder/eurobioimaging_index.json`
- BM25 Index: `https://oeway.github.io/euro-bioimaging-finder/eurobioimaging_bm25_index.pkl`
- Web Interface: `https://oeway.github.io/euro-bioimaging-finder/`

## 📊 Example Usage

```python
import pickle
import bm25s
import httpx
import json

# Download and load main index with metadata
response = httpx.get('https://oeway.github.io/euro-bioimaging-finder/eurobioimaging_index.json')
combined_data = response.json()
bm25_metadata = combined_data.get('bm25_metadata', [])

# Download and load BM25 retriever
response = httpx.get('https://oeway.github.io/euro-bioimaging-finder/eurobioimaging_bm25_index.pkl')
with open('eurobioimaging_bm25_index.pkl', 'wb') as f:
    f.write(response.content)

with open('eurobioimaging_bm25_index.pkl', 'rb') as f:
    retriever = pickle.load(f)

def fulltext_search(query, k=5):
    query_tokens = bm25s.tokenize(query)
    results, scores = retriever.retrieve(query_tokens, k=k)
    hits = []
    for i in range(results.shape[1]):
        doc_idx = results[0, i]
        score = scores[0, i]
        metadata = bm25_metadata[doc_idx]
        hit = metadata.copy()
        hit["score"] = float(score)
        hits.append(hit)
    return hits

# Example search
results = fulltext_search("super resolution microscopy", k=5)
for hit in results:
    print(f"Score: {hit['score']:.2f} - Type: {hit['type']} - ID: {hit['id']}")
```

## 📊 Current Statistics

Last updated: {stable_timestamp.strftime('%Y-%m-%d')}

## 🔬 About Euro-BioImaging

Euro-BioImaging is the European research infrastructure for biological and biomedical imaging.

Visit: [https://www.eurobioimaging.eu/](https://www.eurobioimaging.eu/)
"""
        
        readme_file = gh_pages_dir / 'README.md'
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # Add files to git
        run_command(['git', 'add', '.'], cwd=gh_pages_dir)
        
        # Check if there are changes to commit
        result = run_command(['git', 'status', '--porcelain'], cwd=gh_pages_dir, check=False)
        if not result.stdout.strip():
            print("✅ No changes to publish")
            return True
        
        # Commit changes with stable timestamp in message
        if message:
            commit_message = message
        else:
            commit_message = f"Update Euro-BioImaging index - {stable_timestamp.strftime('%Y-%m-%d')}"
        
        run_command(['git', 'commit', '-m', commit_message], cwd=gh_pages_dir)
        
        # Push to remote origin
        print("🚀 Publishing to remote gh-pages...")
        try:
            # First try to push to remote
            run_command(['git', 'push', 'origin', 'gh-pages'], cwd=gh_pages_dir)
            print("✅ Successfully pushed to remote gh-pages branch")
        except subprocess.CalledProcessError as e:
            # If it fails, try to set upstream and push
            print("🔄 Branch doesn't exist on remote, setting upstream and pushing...")
            try:
                run_command(['git', 'push', '--set-upstream', 'origin', 'gh-pages'], cwd=gh_pages_dir)
                print("✅ Successfully created and pushed gh-pages branch to remote")
            except subprocess.CalledProcessError as e2:
                print(f"❌ Failed to push to remote: {e2}")
                print("💡 You may need to check your GitHub repository permissions")
                return False
        
        print("✅ Successfully published to remote gh-pages!")
        print("🌐 Your index will be available at:")
        print("   https://oeway.github.io/euro-bioimaging-finder/")
        print("   https://oeway.github.io/euro-bioimaging-finder/eurobioimaging_index.json")
        print("   https://oeway.github.io/euro-bioimaging-finder/eurobioimaging_bm25_index.pkl")
        print("📋 Note: It may take a few minutes for GitHub Pages to update")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Publish Euro-BioImaging index to GitHub Pages')
    parser.add_argument('--data-dir', default='euro-bioimaging-index', 
                       help='Directory containing the index file (default: euro-bioimaging-index)')
    parser.add_argument('--test', action='store_true', 
                       help='Publish test index instead of full index')
    parser.add_argument('--force', action='store_true', 
                       help='Force publish even if working directory is not clean')
    parser.add_argument('--message', '-m', 
                       help='Custom commit message')
    
    args = parser.parse_args()
    
    # Determine index filename
    if args.test:
        index_filename = 'test_eurobioimaging_index.json'
        print("📊 Publishing test index...")
    else:
        index_filename = 'eurobioimaging_index.json'
        print("📊 Publishing full index...")
    
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📄 Index file: {index_filename}")
    print("="*60)
    
    # Publish to gh-pages
    success = publish_to_gh_pages(
        data_dir=args.data_dir,
        index_filename=index_filename,
        force=args.force,
        message=args.message
    )
    
    if success:
        print("\n🎉 Publication completed successfully!")
        print("\n📋 Next steps:")
        print("1. Enable GitHub Pages in your repository settings")
        print("2. Set source to 'gh-pages' branch")
        print("3. Your index will be available at the GitHub Pages URL")
    else:
        print("\n❌ Publication failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 