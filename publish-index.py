#!/usr/bin/env python3
"""
Publish Euro-BioImaging Index to GitHub Pages

This script publishes the generated combined index to the gh-pages branch
for public access and distribution.
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
    """Run a shell command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result

def check_git_status():
    """Check if git repository is clean"""
    result = run_command(['git', 'status', '--porcelain'], check=False)
    if result.stdout.strip():
        print("⚠️  Warning: Working directory has uncommitted changes:")
        print(result.stdout)
        return False
    return True

def get_current_branch():
    """Get the current git branch"""
    result = run_command(['git', 'branch', '--show-current'])
    return result.stdout.strip()

def create_index_html(index_file, output_dir):
    """Create a simple HTML page to serve the index"""
    
    # Load index metadata
    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        metadata = data.get('metadata', {})
    except Exception as e:
        print(f"Warning: Could not load index metadata: {e}")
        metadata = {}
    
    created_at = metadata.get('created_at', 'Unknown')
    dataset_type = metadata.get('dataset_type', 'Unknown')
    stats = metadata.get('statistics', {})
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Euro-BioImaging Search Index</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            line-height: 1.6;
            color: #333;
        }}
        .header {{
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #e0e0e0;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-number {{
            font-size: 2rem;
            font-weight: bold;
            color: #007bff;
        }}
        .download-section {{
            background: #e8f4fd;
            padding: 1.5rem;
            border-radius: 8px;
            margin: 2rem 0;
        }}
        .download-link {{
            display: inline-block;
            background: #007bff;
            color: white;
            padding: 0.75rem 1.5rem;
            text-decoration: none;
            border-radius: 4px;
            margin: 0.5rem;
        }}
        .download-link:hover {{
            background: #0056b3;
        }}
        .api-section {{
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            margin: 2rem 0;
        }}
        code {{
            background: #e9ecef;
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            font-family: 'Monaco', 'Consolas', monospace;
        }}
        .code-block {{
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 4px;
            border-left: 4px solid #007bff;
            margin: 1rem 0;
            overflow-x: auto;
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
            <code>https://oeway.github.io/euro-bioimaging/eurobioimaging_index.json</code>
        </div>
        
        <h3>Example Usage</h3>
        <p>JavaScript:</p>
        <div class="code-block">
            <code>
fetch('https://oeway.github.io/euro-bioimaging/eurobioimaging_index.json')<br>
&nbsp;&nbsp;.then(response => response.json())<br>
&nbsp;&nbsp;.then(data => console.log(data));
            </code>
        </div>
        
        <p>Python:</p>
        <div class="code-block">
            <code>
import requests<br>
response = requests.get('https://oeway.github.io/euro-bioimaging/eurobioimaging_index.json')<br>
data = response.json()
            </code>
        </div>
        
        <p>curl:</p>
        <div class="code-block">
            <code>curl -s https://oeway.github.io/euro-bioimaging/eurobioimaging_index.json</code>
        </div>
    </div>
    
    <div class="api-section">
        <h2>📋 Index Structure</h2>
        <p>The JSON index contains the following structure:</p>
        <div class="code-block">
            <code>
{{<br>
&nbsp;&nbsp;"metadata": {{ ... }},<br>
&nbsp;&nbsp;"technologies": [ ... ],<br>
&nbsp;&nbsp;"nodes": [ ... ],<br>
&nbsp;&nbsp;"website_pages": [ ... ]<br>
}}
            </code>
        </div>
    </div>
    
    <footer style="text-align: center; margin-top: 3rem; padding-top: 2rem; border-top: 1px solid #e0e0e0; color: #666;">
        <p>🔬 Euro-BioImaging Search Index | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><a href="https://www.eurobioimaging.eu/">Visit Euro-BioImaging</a></p>
    </footer>
</body>
</html>"""
    
    # Save HTML file
    html_file = output_dir / 'index.html'
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Created index.html at {html_file}")

def publish_to_gh_pages(data_dir, index_filename, force=False, message=None):
    """Publish the index to gh-pages branch"""
    
    data_path = Path(data_dir)
    index_file = data_path / index_filename
    
    if not index_file.exists():
        print(f"❌ Index file not found: {index_file}")
        return False
    
    # Check git status
    if not force and not check_git_status():
        print("❌ Working directory is not clean. Use --force to proceed anyway.")
        return False
    
    # Get current branch
    current_branch = get_current_branch()
    print(f"📍 Current branch: {current_branch}")
    
    # Create temporary directory for gh-pages content
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        gh_pages_dir = temp_path / 'gh-pages'
        
        print("🔄 Setting up gh-pages branch...")
        
        # Clone current repo to temp directory
        run_command(['git', 'clone', '.', str(gh_pages_dir)])
        
        # Switch to gh-pages branch (create if doesn't exist)
        try:
            run_command(['git', 'checkout', 'gh-pages'], cwd=gh_pages_dir)
        except subprocess.CalledProcessError:
            print("📝 Creating new gh-pages branch...")
            run_command(['git', 'checkout', '--orphan', 'gh-pages'], cwd=gh_pages_dir)
            # Remove all files from the orphan branch
            run_command(['git', 'rm', '-rf', '.'], cwd=gh_pages_dir, check=False)
        
        # Copy index file to gh-pages directory
        target_index = gh_pages_dir / 'eurobioimaging_index.json'
        shutil.copy2(index_file, target_index)
        print(f"📄 Copied {index_file} to {target_index}")
        
        # Create index.html
        create_index_html(index_file, gh_pages_dir)
        
        # Create README.md for gh-pages
        readme_content = f"""# Euro-BioImaging Search Index

This repository hosts the Euro-BioImaging search index for public access.

## 📥 Access the Index

- **JSON Index**: [eurobioimaging_index.json](eurobioimaging_index.json)
- **Web Interface**: [index.html](index.html)

## 🔗 Direct Links

- JSON API: `https://oeway.github.io/euro-bioimaging/eurobioimaging_index.json`
- Web Interface: `https://oeway.github.io/euro-bioimaging/`

## 📊 Current Statistics

Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

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
        
        # Commit changes
        commit_message = message or f"Update Euro-BioImaging index - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        run_command(['git', 'commit', '-m', commit_message], cwd=gh_pages_dir)
        
        # Push to origin
        print("🚀 Publishing to gh-pages...")
        run_command(['git', 'push', 'origin', 'gh-pages'], cwd=gh_pages_dir)
        
        print("✅ Successfully published to gh-pages!")
        print("🌐 Your index will be available at:")
        print("   https://oeway.github.io/euro-bioimaging/")
        print("   https://oeway.github.io/euro-bioimaging/eurobioimaging_index.json")
        
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