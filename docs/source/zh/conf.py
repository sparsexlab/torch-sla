# Chinese documentation configuration
# Inherits from parent conf.py

import sys
import os
sys.path.insert(0, os.path.abspath('../'))
from conf import *

# Override templates path for Chinese
templates_path = ['_templates', '../_templates']

# Override language
language = 'zh_CN'

# Set master doc for Chinese
master_doc = 'index'

# Override paths for static files
html_static_path = ['../_static']

# Override context for Chinese
html_context = {
    'description': 'torch-sla: PyTorch 稀疏线性代数库。GPU加速的稀疏求解器，支持自动微分。',
    'keywords': 'torch稀疏, pytorch稀疏矩阵, 稀疏线性代数, GPU稀疏求解, CUDA稀疏, 可微分稀疏求解器, 有限元, CFD, FEM, 神经网络',
    'author': 'Walker Chi',
    'og_title': 'torch-sla: PyTorch 稀疏线性代数库',
    'og_description': 'PyTorch 稀疏线性代数库。通过 CuPy/cuDSS 实现 GPU 加速求解 Ax=b。完整支持自动微分。',
    'og_image': SITE_URL + '_static/logo.jpg',
    'site_url': SITE_URL,
    'twitter_card': 'summary_large_image',
    'languages': [
        ('en', 'English', '../'),
        ('zh', '中文', './'),
    ],
    'current_language': 'zh',
}

# Override title
html_title = "torch-sla: PyTorch 稀疏线性代数"

# Update announcement for Chinese
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

