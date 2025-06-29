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
    
    # Load HTML template
    template_file = Path('index_template.html')
    if not template_file.exists():
        print(f"Warning: Template file {template_file} not found, creating basic template")
        html_content = f"""<!DOCTYPE html>
<html><head><title>Euro-BioImaging Search Index</title></head>
<body><h1>Euro-BioImaging Search Index</h1>
<p>Technologies: {stats['technologies']}</p>
<p>Nodes: {stats['nodes']}</p>
<p>Website Pages: {stats['website_pages']}</p>
<p>Dataset: {dataset_type.title()}</p>
<p>Last Updated: {created_at}</p>
</body></html>"""
    else:
        # Read template and format with data
        with open(template_file, 'r', encoding='utf-8') as f:
            template_content = f.read()
        
        html_content = template_content % {
            'technologies_count': stats['technologies'],
            'nodes_count': stats['nodes'],
            'website_pages_count': stats['website_pages'],
            'total_entries': stats['total_entries'],
            'dataset_type': dataset_type.title(),
            'created_at': created_at
        }
    
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