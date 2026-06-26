# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

# Site URL — override with DOCS_SITE_URL env var (adaptive deployment).
SITE_URL = os.environ.get('DOCS_SITE_URL', 'https://www.torchsla.com/').rstrip('/') + '/'

# -- Autodoc configuration ---------------------------------------------------
# Mock optional dependencies that may not be installed
autodoc_mock_imports = [
    'scikits',
    'scikits.umfpack',
]

# -- Project information -----------------------------------------------------
project = 'torch-sla'
copyright = '2024-2026, Mingyuan Chi, Shizheng Wen'
author = 'Mingyuan Chi, Shizheng Wen'
version = '0.1.2'
release = '0.1.2'

# -- SEO & Metadata ----------------------------------------------------------
# SEO-optimized title with high-value keywords
html_short_title = 'torch-sla - PyTorch Sparse Linear Algebra'
html_baseurl = SITE_URL

# Open Graph metadata for social sharing
ogp_site_url = SITE_URL
ogp_site_name = 'torch-sla: PyTorch Sparse Linear Algebra'
ogp_image = SITE_URL + '_static/logo.jpg'
ogp_description_length = 300
ogp_type = 'website'
ogp_description = 'torch-sla: PyTorch Sparse Linear Algebra library with GPU acceleration. Differentiable sparse solvers with autograd support for torch.sparse tensors.'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',  # Link to external docs
    'sphinx.ext.graphviz',     # For vector diagrams
    'sphinx_sitemap',          # Generate sitemap.xml for SEO
]

# Graphviz settings
graphviz_output_format = 'svg'

# Intersphinx mapping for cross-references
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
}

templates_path = ['_templates']
# 'zh' holds the Chinese docs, which are built as a separate Sphinx project
# (see source/zh/conf.py, built via `sphinx-build -b html source/zh ...`).
# Exclude it from the English build so its pages are not processed twice
# (which otherwise produces duplicate-label / orphan-toctree warnings).
exclude_patterns = ['setup.py', '__init__.py', 'zh']

# -- Language settings -------------------------------------------------------
language = 'en'
locale_dirs = ['locale/']
gettext_compact = False
gettext_uuid = True

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['custom.css', 'recommend.css']
html_js_files = ['recommend.js']
autodoc_member_order = 'bysource'

# Furo theme options with SEO enhancements
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#64288C",
        "color-brand-content": "#F15213",
    },
    "dark_css_variables": {
        "color-brand-primary": "#EE9525",
        "color-brand-content": "#8E53A2",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/walkerchi/torch-sla",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
        {
            "name": "arXiv",
            "url": "https://arxiv.org/abs/2601.13994",
            "html": """
                <svg viewBox="0 0 24 24" fill="currentColor" stroke="none">
                    <path d="M4 2h16a2 2 0 0 1 2 2v16a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2zm2 4v12h12V6H6zm2 2h8v2H8V8zm0 4h8v2H8v-2zm0 4h5v2H8v-2z"/>
                </svg>
            """,
            "class": "",
        },
    ],
}

# Logo and title - SEO optimized with high-value keywords
html_logo = "_static/logo.jpg"
html_title = "torch-sla: PyTorch Sparse Linear Algebra"
html_favicon = "_static/logo.jpg"

# Additional SEO meta tags - targeting common search queries
html_context = {
    # Primary description targeting key search terms
    'description': 'torch-sla: PyTorch Sparse Linear Algebra library. GPU-accelerated sparse solvers with autograd support. Works with torch.sparse tensors, COO/CSR formats. pip install torch-sla.',
    # Comprehensive keywords covering all search variations
    'keywords': 'torch sparse, torch sparse matrix, torch sparse tensor, pytorch sparse, pytorch sparse matrix, pytorch sparse solver, sparse linear algebra pytorch, torch.sparse, sparse linear algebra, GPU sparse solver, CUDA sparse, cuSOLVER, cuDSS, differentiable sparse solver, autograd sparse, scipy sparse pytorch, sparse COO, sparse CSR, FEM pytorch, CFD pytorch, spsolve pytorch',
    'author': 'Mingyuan Chi, Shizheng Wen',
    'og_title': 'torch-sla: PyTorch Sparse Linear Algebra with GPU Acceleration',
    'og_description': 'PyTorch Sparse Linear Algebra library. Solve Ax=b with GPU acceleration via cuDSS. Full autograd support for differentiable sparse operations. pip install torch-sla.',
    'og_image': SITE_URL + '_static/logo.jpg',
    'site_url': SITE_URL,
    'twitter_card': 'summary_large_image',
    'google_site_verification': '',  # Add your Google Search Console verification code here
    # Language selector
    'languages': [
        ('en', 'English', './'),
        ('zh', '中文', './zh/'),
    ],
    'current_language': 'en',
}

# Sitemap for search engines (requires sphinx-sitemap)
sitemap_url_scheme = '{link}'
sitemap_locales = ['en', 'zh']

# Search engine optimization
html_use_index = True
html_split_index = False
html_copy_source = True
html_show_sourcelink = True
html_show_sphinx = False
html_show_copyright = True


def setup(app):
    """Emit robots.txt with current SITE_URL into the build output."""
    def write_robots(app, exception):
        if exception is not None:
            return
        path = os.path.join(app.outdir, 'robots.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(
                "User-agent: *\n"
                "Allow: /\n\n"
                f"Sitemap: {SITE_URL}sitemap.xml\n"
                "Crawl-delay: 1\n"
            )
    app.connect('build-finished', write_robots)
