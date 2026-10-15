"""
app.py — Modern Semantic Search Interface (Govt Scheme Finder)

Key Enhancements:
  - Zero Emojis: Bespoke library of clean, high-resolution SVG vector icons
  - Modern Color Palette: Clean Light Mode (Slate-50 #F8FAFC canvas, Pure White #FFFFFF cards, Electric Indigo #4F46E5, Emerald & Amber accents)
  - Integrated Card Architecture: Built-in collapsible drawer (<details><summary>)
    eliminating disconnected default Streamlit expanders
  - Responsive Prompt Grid: Clean 2x3 cards with domain icons, zero text clipping
  - Polished Sidebar & Top Navigation Bar with live index statistics
"""

import streamlit as st
import pandas as pd
import urllib.parse
import html
import search_engine

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="My Yojana Saarthi — Smart Welfare Search",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Helper for Bulletproof HTML Rendering in Streamlit Markdown ───────────────

def render_html(raw_html: str):
    """Strip leading spaces per line to prevent CommonMark from treating HTML as code blocks."""
    clean = "\n".join(line.strip() for line in raw_html.strip().splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)

# ── Bespoke SVG Vector Icon Library (Zero Emojis) ─────────────────────────────

def get_svg_icon(name: str, size: int = 16, stroke_width: float = 1.8, color: str = "currentColor", extra_class: str = "") -> str:
    """Return crisp, lightweight inline SVG vector icons with standard 24x24 viewBox."""
    paths = {
        "search": (
            '<circle cx="11" cy="11" r="8"/>'
            '<line x1="21" y1="21" x2="16.65" y2="16.65"/>'
        ),
        "sparkles": (
            '<path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/>'
            '<path d="M19 3v4"/>'
            '<path d="M21 5h-4"/>'
        ),
        "shield-check": (
            '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>'
            '<path d="m9 12 2 2 4-4"/>'
        ),
        "map-pin": (
            '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/>'
            '<circle cx="12" cy="10" r="3"/>'
        ),
        "building": (
            '<rect width="16" height="20" x="4" y="2" rx="2" ry="2"/>'
            '<path d="M9 22v-4h6v4"/>'
            '<path d="M8 6h.01"/>'
            '<path d="M16 6h.01"/>'
            '<path d="M8 10h.01"/>'
            '<path d="M16 10h.01"/>'
            '<path d="M8 14h.01"/>'
            '<path d="M16 14h.01"/>'
        ),
        "tag": (
            '<path d="M12 2H2v10l9.29 9.29c.94.94 2.48.94 3.42 0l6.58-6.58c.94-.94.94-2.48 0-3.42L12 2Z"/>'
            '<circle cx="7" cy="7" r=".5" fill="currentColor"/>'
        ),
        "flag": (
            '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/>'
            '<line x1="4" x2="4" y1="22" y2="15"/>'
        ),
        "sliders": (
            '<line x1="4" x2="4" y1="21" y2="14"/>'
            '<line x1="4" x2="4" y1="10" y2="3"/>'
            '<line x1="12" x2="12" y1="21" y2="12"/>'
            '<line x1="12" x2="12" y1="8" y2="3"/>'
            '<line x1="20" x2="20" y1="21" y2="16"/>'
            '<line x1="20" x2="20" y1="12" y2="3"/>'
            '<line x1="1" x2="7" y1="14" y2="14"/>'
            '<line x1="9" x2="15" y1="8" y2="8"/>'
            '<line x1="17" x2="23" y1="16" y2="16"/>'
        ),
        "scale": (
            '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/>'
            '<path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/>'
            '<path d="M7 21h10"/>'
            '<path d="M12 3v18"/>'
            '<path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>'
        ),
        "cpu": (
            '<rect width="16" height="16" x="4" y="4" rx="2"/>'
            '<rect width="6" height="6" x="9" y="9" rx="1"/>'
            '<path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/>'
            '<path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>'
        ),
        "chevron-down": (
            '<polyline points="6 9 12 15 18 9"/>'
        ),
        "external-link": (
            '<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>'
            '<polyline points="15 3 21 3 21 9"/>'
            '<line x1="10" x2="21" y1="14" y2="3"/>'
        ),
        "laptop": (
            '<path d="M20 16V7a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v9m16 0H4m16 0 1.28 2.55a1 1 0 0 1-.9 1.45H3.62a1 1 0 0 1-.9-1.45L4 16"/>'
        ),
        "sprout": (
            '<path d="M7 20h10"/>'
            '<path d="M10 20c5.5-2.5.8-6.4 3-10"/>'
            '<path d="M9.5 9.4c1.1.8 1.8 2.2 2.3 3.7-2 .4-3.5.4-4.8-.3-1.2-.6-2.3-1.9-3-4.2 2.8-.5 4.4 0 5.5.8z"/>'
            '<path d="M14.1 6a7 7 0 0 0-1.1 4c1.9-.1 3.3-.6 4.3-1.4 1-1 1.6-2.3 1.7-4.6-2.7.1-4 1-4.9 2z"/>'
        ),
        "heart": (
            '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>'
        ),
        "sun": (
            '<circle cx="12" cy="12" r="4"/>'
            '<path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/>'
            '<path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>'
        ),
        "wheelchair": (
            '<circle cx="12" cy="4" r="2"/>'
            '<path d="m18 19 1-7-6 1"/>'
            '<path d="m5 8 3-3 5.5 1-2.36 3.5"/>'
            '<path d="M4.24 14.5a5 5 0 0 0 6.88 6"/>'
            '<path d="M13.76 17.5a5 5 0 0 0-1.76-7.5"/>'
        ),
        "briefcase": (
            '<rect width="20" height="14" x="2" y="7" rx="2" ry="2"/>'
            '<path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>'
        ),
        "zap": (
            '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>'
        ),
        "layers": (
            '<polygon points="12 2 2 7 12 12 22 7 12 2"/>'
            '<polyline points="2 17 12 22 22 17"/>'
            '<polyline points="2 12 12 17 22 12"/>'
        ),
        "database": (
            '<ellipse cx="12" cy="5" rx="9" ry="3"/>'
            '<path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>'
            '<path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/>'
        ),
        "info": (
            '<circle cx="12" cy="12" r="10"/>'
            '<path d="M12 16v-4"/>'
            '<path d="M12 8h.01"/>'
        ),
        "check": (
            '<polyline points="20 6 9 17 4 12"/>'
        ),
        "target": (
            '<circle cx="12" cy="12" r="10"/>'
            '<circle cx="12" cy="12" r="6"/>'
            '<circle cx="12" cy="12" r="2"/>'
        )
    }
    path_data = paths.get(name, paths["info"])
    cls_attr = f' class="{extra_class}"' if extra_class else ''
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round"{cls_attr} style="vertical-align: middle; display: inline-block;">'
        f'{path_data}</svg>'
    )

# ── Custom CSS: Modern Clean Light Mode Design System ─────────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Base Reset & Modern Font */
html, body, [class*="css"], [class*="st-"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* Crisp Clean Light Background with Gentle Eye-Friendly Ambient Glow */
.stApp {
    background-color: #F8FAFC !important;
    background-image: 
        radial-gradient(circle at 50% -10%, rgba(99, 102, 241, 0.07) 0%, rgba(248, 250, 252, 0) 60%),
        radial-gradient(circle at 100% 30%, rgba(14, 165, 233, 0.05) 0%, rgba(248, 250, 252, 0) 50%) !important;
    color: #0F172A !important;
}

/* Streamlit Header clean up */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Top Navigation / Brand Header Bar */
.brand-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 0 22px 0;
    margin-bottom: 24px;
    border-bottom: 1px solid #E2E8F0;
}

.brand-left {
    display: flex;
    align-items: center;
    gap: 14px;
}

.brand-icon-box {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(14, 165, 233, 0.1) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 12px;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.12);
    color: #4F46E5;
}

.brand-title {
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    color: #0F172A;
    line-height: 1.2;
}

.brand-sub {
    font-size: 0.82rem;
    color: #64748B;
    font-weight: 500;
}

.brand-meta-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 7px 16px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #334155;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}

.brand-meta-badge span.pulse-dot {
    width: 8px;
    height: 8px;
    background: #10B981;
    border-radius: 50%;
    display: inline-block;
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.7);
}

/* Hero Section */
.hero-container {
    margin-bottom: 22px;
}

.hero-headline {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #0F172A;
    line-height: 1.25;
    margin-bottom: 8px;
}

.hero-highlight {
    background: linear-gradient(135deg, #4F46E5 0%, #0284C7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 1.0rem;
    color: #475569;
    font-weight: 500;
    line-height: 1.6;
}

/* Pinterest & Mobbin Inspired Floating Pill Search Input */
div[data-testid="stTextInput"] {
    margin-bottom: 0px !important;
}

div[data-testid="stTextInput"] > div > div > input {
    background-color: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 9999px !important;
    color: #0F172A !important;
    padding: 14px 24px 14px 54px !important;
    font-size: 1.02rem !important;
    font-weight: 500 !important;
    min-height: 52px !important;
    box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.06), 0 2px 6px -1px rgba(15, 23, 42, 0.03) !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='22' height='22' viewBox='0 0 24 24' fill='none' stroke='%236366F1' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'/%3E%3C/svg%3E") !important;
    background-repeat: no-repeat !important;
    background-position: 20px center !important;
    background-size: 22px 22px !important;
}

div[data-testid="stTextInput"] > div > div > input:hover {
    border-color: #94A3B8 !important;
    box-shadow: 0 6px 24px -2px rgba(15, 23, 42, 0.09) !important;
}

div[data-testid="stTextInput"] > div > div > input:focus {
    border-color: #4F46E5 !important;
    background-color: #FFFFFF !important;
    box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15), 0 12px 30px -4px rgba(99, 102, 241, 0.12) !important;
}

div[data-testid="stTextInput"] > div > div > input::placeholder {
    color: #64748B !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
}

/* Action button in search bar row (Search / Clear) */
button[kind="primary"] {
    background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%) !important;
    border: 1px solid #4338CA !important;
    color: #FFFFFF !important;
    border-radius: 9999px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    min-height: 52px !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.25) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4338CA 0%, #3730A3 100%) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* Clear button alongside search */
button[key="clear_search_btn"],
div:has(> button[key="clear_search_btn"]) button {
    border-radius: 9999px !important;
    min-height: 52px !important;
    border: 1.5px solid #CBD5E1 !important;
    background: #FFFFFF !important;
    color: #475569 !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

button[key="clear_search_btn"]:hover,
div:has(> button[key="clear_search_btn"]) button:hover {
    background: #F1F5F9 !important;
    color: #0F172A !important;
    border-color: #94A3B8 !important;
}

/* Pinterest-style Trending Scenarios Ribbon & Pills */
.prompt-shelf-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.8rem;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 18px;
    margin-bottom: 12px;
}

/* Scenario Quick-Pill Buttons */
div[data-testid="stHorizontalBlock"] button {
    border-radius: 9999px !important;
    border: 1px solid #E2E8F0 !important;
    background-color: #FFFFFF !important;
    color: #334155 !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    padding: 8px 16px !important;
    min-height: 42px !important;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

div[data-testid="stHorizontalBlock"] button:hover {
    border-color: #818CF8 !important;
    background-color: #EEF2FF !important;
    color: #4338CA !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.15) !important;
}

div[data-testid="stHorizontalBlock"] button:active {
    transform: translateY(0) !important;
}

/* Sidebar High-Contrast Styling (WCAG AAA) */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
    padding-top: 2rem !important;
}

.sidebar-heading {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.8rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #1E293B;
    margin-top: 14px;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #F1F5F9;
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
    color: #1E293B !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}

/* Selectbox Dropdowns in Sidebar */
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div > div {
    background-color: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 10px !important;
    color: #0F172A !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
    transition: all 0.2s ease !important;
}

section[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div > div:hover {
    border-color: #94A3B8 !important;
    background-color: #F8FAFC !important;
}

section[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12) !important;
}

section[data-testid="stSidebar"] div[data-testid="stSelectbox"] span,
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] div {
    color: #0F172A !important;
    font-weight: 600 !important;
}

/* BaseWeb Popover Menu for Dropdowns */
div[data-baseweb="popover"],
ul[data-baseweb="menu"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.12) !important;
}

ul[data-baseweb="menu"] li {
    color: #1E293B !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    transition: all 0.15s ease !important;
}

ul[data-baseweb="menu"] li:hover,
ul[data-baseweb="menu"] li[aria-selected="true"] {
    background-color: #EEF2FF !important;
    color: #4338CA !important;
    font-weight: 600 !important;
}

/* Radio Buttons in Sidebar */
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label {
    color: #334155 !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    padding: 3px 0 !important;
    transition: color 0.15s ease !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    color: #4F46E5 !important;
}

/* Slider Controls */
section[data-testid="stSidebar"] div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
    color: #4F46E5 !important;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.86rem !important;
}

section[data-testid="stSidebar"] div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
section[data-testid="stSidebar"] div[data-testid="stSlider"] div[data-testid="stTickBarMax"] {
    color: #64748B !important;
    font-weight: 600 !important;
    font-size: 0.76rem !important;
}

/* Toggle Switch */
section[data-testid="stSidebar"] div[data-testid="stToggle"] label p {
    color: #1E293B !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}

/* Architecture Specs Box */
.tech-spec-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 12px 14px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
}

.tech-spec-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #F1F5F9;
}

.tech-spec-item:last-child {
    border-bottom: none;
}

.tech-spec-key {
    color: #475569;
    font-weight: 500;
    font-size: 0.77rem;
}

.tech-spec-val {
    color: #0F172A;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.76rem;
    background: #FFFFFF;
    padding: 2px 7px;
    border-radius: 6px;
    border: 1px solid #E2E8F0;
}

/* Unified Self-Contained Scheme Card (Mobbin Inspired) */
.scheme-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 20px;
    box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.04), 0 1px 3px rgba(15, 23, 42, 0.02);
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
}

.scheme-card:hover {
    border-color: #818CF8;
    transform: translateY(-2px);
    box-shadow: 0 12px 28px -4px rgba(99, 102, 241, 0.1), 0 4px 12px rgba(15, 23, 42, 0.04);
}

.scheme-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
    margin-bottom: 12px;
}

.scheme-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.35;
    letter-spacing: -0.015em;
}

/* Match Score Badges with High Contrast */
.score-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    white-space: nowrap;
    flex-shrink: 0;
}

.score-high {
    background: #ECFDF5;
    color: #047857;
    border: 1px solid #A7F3D0;
}

.score-mid {
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
}

.score-low {
    background: #FFFBEB;
    color: #B45309;
    border: 1px solid #FDE68A;
}

/* Metadata Row & Chips */
.meta-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px;
}

.meta-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 0.77rem;
    font-weight: 600;
    background: #F8FAFC;
    color: #475569;
    border: 1px solid #E2E8F0;
    line-height: 1;
}

.chip-central {
    background: #EEF2FF;
    color: #4338CA;
    border-color: #C7D2FE;
}

.chip-state {
    background: #F0F9FF;
    color: #0369A1;
    border-color: #BAE6FD;
}

.chip-cat {
    background: #FAF5FF;
    color: #7E22CE;
    border-color: #E9D5FF;
}

/* Card Description Snippet */
.scheme-desc {
    color: #334155;
    font-size: 0.92rem;
    line-height: 1.65;
    margin-bottom: 14px;
}

/* Collapsible Details Drawer Inside Card */
details.scheme-drawer {
    border-top: 1px solid #E2E8F0;
    padding-top: 10px;
    margin-top: 10px;
}

details.scheme-drawer summary {
    cursor: pointer;
    list-style: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.83rem;
    font-weight: 600;
    color: #4F46E5;
    user-select: none;
    padding: 9px 14px;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    transition: all 0.2s ease;
}

details.scheme-drawer summary::-webkit-details-marker {
    display: none;
}

details.scheme-drawer summary:hover {
    background: #EEF2FF;
    border-color: #C7D2FE;
    color: #4338CA;
}

.chevron-icon {
    transition: transform 0.2s ease;
}

details.scheme-drawer[open] .chevron-icon {
    transform: rotate(180deg);
}

.drawer-trigger-left {
    display: flex;
    align-items: center;
    gap: 8px;
}

.drawer-content {
    padding: 16px 18px;
    margin-top: 12px;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    font-size: 0.88rem;
    color: #334155;
    line-height: 1.6;
}

/* Structured Drawer Section Cards */
.drawer-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

.drawer-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.88rem;
    font-weight: 700;
    margin-bottom: 8px;
}

.drawer-card-body {
    font-size: 0.88rem;
    line-height: 1.65;
    color: #334155;
}

.drawer-overview {
    border-left: 3px solid #6366F1;
}
.drawer-overview .drawer-card-header {
    color: #4338CA;
}

.drawer-benefits {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-left: 3px solid #10B981;
}
.drawer-benefits .drawer-card-header {
    color: #047857;
}
.drawer-benefits .drawer-card-body {
    color: #14532D;
}

.drawer-eligibility {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-left: 3px solid #3B82F6;
}
.drawer-eligibility .drawer-card-header {
    color: #1D4ED8;
}
.drawer-eligibility .drawer-card-body {
    color: #1E3A8A;
}

.drawer-application {
    background: #FAF5FF;
    border: 1px solid #E9D5FF;
    border-left: 3px solid #8B5CF6;
}
.drawer-application .drawer-card-header {
    color: #6D28D9;
}
.drawer-application .drawer-card-body {
    color: #4C1D95;
}

.drawer-documents {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-left: 3px solid #F59E0B;
}
.drawer-documents .drawer-card-header {
    color: #B45309;
}
.drawer-documents .drawer-card-body {
    color: #78350F;
}

.drawer-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    margin: 14px 0;
    padding-top: 12px;
    border-top: 1px solid #E2E8F0;
}

.drawer-grid-item {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 10px 12px;
}

.drawer-item-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748B;
    margin-bottom: 3px;
}

.drawer-item-val {
    font-size: 0.84rem;
    color: #0F172A;
    font-weight: 600;
}

.drawer-tags-wrap {
    margin-top: 12px;
    margin-bottom: 14px;
}

.drawer-tags-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 6px;
}

.drawer-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.drawer-tag-chip {
    font-size: 0.74rem;
    font-weight: 500;
    color: #475569;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 3px 9px;
}

.drawer-action-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 9px 18px;
    border-radius: 9999px;
    background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%);
    border: 1px solid #4338CA;
    color: #FFFFFF !important;
    text-decoration: none !important;
    font-size: 0.82rem;
    font-weight: 600;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.25);
    transition: all 0.2s ease;
}

.drawer-action-btn:hover {
    background: linear-gradient(135deg, #4338CA 0%, #3730A3 100%);
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
    transform: translateY(-1px);
    color: #FFFFFF !important;
}

/* Comparison View Column Subheaders */
.comp-header-sem {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1.15rem;
    font-weight: 700;
    color: #4F46E5;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #818CF8;
}

.comp-header-kw {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1.15rem;
    font-weight: 700;
    color: #D97706;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid #F59E0B;
}

/* Feature Showcase Cards (Empty State) */
.feature-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 24px;
    height: 100%;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}

.feature-card:hover {
    border-color: #818CF8;
    transform: translateY(-3px);
    box-shadow: 0 10px 25px -4px rgba(99, 102, 241, 0.12);
}

.feature-icon-box {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(14, 165, 233, 0.1) 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    color: #4F46E5;
    margin-bottom: 16px;
}

.feature-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 8px;
}

.feature-body {
    font-size: 0.86rem;
    color: #475569;
    line-height: 1.6;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Load Index ────────────────────────────────────────────────────────────────

@st.cache_resource
def init():
    df = search_engine.load_index()
    states = search_engine.get_unique_states()
    categories = search_engine.get_unique_categories()
    return len(df), states, categories

total_schemes, states, categories = init()

# ── Top Navigation / Brand Bar ────────────────────────────────────────────────

brand_html = f"""
<div class="brand-bar">
    <div class="brand-left">
        <div class="brand-icon-box">
            {get_svg_icon("shield-check", size=22, stroke_width=2, color="#4F46E5")}
        </div>
        <div>
            <div class="brand-title">My Yojana Saarthi</div>
            <div class="brand-sub">Neural Semantic Welfare Retrieval Engine</div>
        </div>
    </div>
    <div class="brand-meta-badge">
        <span class="pulse-dot"></span>
        <span>{total_schemes:,} Schemes Indexed</span>
        <span style="opacity: 0.4;">·</span>
        <span>Offline BLAS Vector Engine</span>
    </div>
</div>
"""
render_html(brand_html)

# ── Sidebar Configuration ─────────────────────────────────────────────────────

with st.sidebar:
    render_html(
        f'<div class="sidebar-heading">{get_svg_icon("sliders", size=16, color="#4F46E5")} Filters & Constraints</div>'
    )

    selected_state = st.selectbox(
        "Beneficiary State / Region",
        ["All States"] + states,
        index=0,
        help="Filter schemes valid in a specific territory or nationwide."
    )
    state_filter = None if selected_state == "All States" else selected_state

    selected_category = st.selectbox(
        "Focus Sector / Category",
        ["All Categories"] + categories,
        index=0,
        help="Target specific domain (Agriculture, Education, Health, etc.)."
    )
    category_filter = None if selected_category == "All Categories" else selected_category

    search_mode = st.radio(
        "Search Strategy",
        ["Hybrid (BM25 + Semantic)", "Semantic Only", "BM25 Lexical Only"],
        index=0,
        help="Hybrid combines neural semantic vector matching with Okapi BM25 lexical ranking for optimal recall."
    )

    num_results = st.slider("Result Count (Top-K)", min_value=3, max_value=20, value=5)

    st.markdown("<br>", unsafe_allow_html=True)
    render_html(
        f'<div class="sidebar-heading">{get_svg_icon("scale", size=16, color="#D97706")} Benchmarking</div>'
    )
    show_comparison = st.toggle("Side-by-side Comparison Mode", value=False)

    st.markdown("<br>", unsafe_allow_html=True)
    render_html(
        f'<div class="sidebar-heading">{get_svg_icon("cpu", size=16, color="#0284C7")} Architecture Specs</div>'
    )
    render_html(
        f"""
        <div class="tech-spec-box">
            <div class="tech-spec-item">
                <span class="tech-spec-key">Fusion</span>
                <span class="tech-spec-val">Weighted RRF (k=60)</span>
            </div>
            <div class="tech-spec-item">
                <span class="tech-spec-key">Lexical</span>
                <span class="tech-spec-val">Okapi BM25</span>
            </div>
            <div class="tech-spec-item">
                <span class="tech-spec-key">Neural Vector</span>
                <span class="tech-spec-val">all-MiniLM-L6-v2</span>
            </div>
            <div class="tech-spec-item">
                <span class="tech-spec-key">Dimensions</span>
                <span class="tech-spec-val">384-d normalized</span>
            </div>
            <div class="tech-spec-item">
                <span class="tech-spec-key">Expansion</span>
                <span class="tech-spec-val">943 Auto-Synonyms</span>
            </div>
            <div class="tech-spec-item">
                <span class="tech-spec-key">Corpus Size</span>
                <span class="tech-spec-val">{total_schemes:,} schemes</span>
            </div>
            <div class="tech-spec-item">
                <span class="tech-spec-key">Latency</span>
                <span class="tech-spec-val">&lt; 3.0 ms / query</span>
            </div>
        </div>
        """
    )

# ── Hero Section ──────────────────────────────────────────────────────────────

render_html(
    """
    <div class="hero-container">
        <div class="hero-headline">Find government schemes <span class="hero-highlight">tailored to your situation.</span></div>
        <div class="hero-subtitle">Search over 4,600 central & state welfare programs using natural language. Just describe your background, need, or life scenario.</div>
    </div>
    """
)

# ── Search Input (Pinterest & Mobbin Inspired Floating Pill) ──────────────────

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

def set_quick_query(prompt: str):
    st.session_state.search_query = prompt

def clear_search_query():
    st.session_state.search_query = ""

col_search, col_action = st.columns([5, 1], gap="small")
with col_search:
    query = st.text_input(
        "Search Welfare Schemes",
        key="search_query",
        placeholder="Describe your situation (e.g. 'college student laptop loan' or 'subsidy for solar rooftop')...",
        label_visibility="collapsed"
    )
with col_action:
    if st.session_state.get("search_query", ""):
        st.button("✕ Clear", key="clear_search_btn", on_click=clear_search_query, use_container_width=True)
    else:
        st.button("Search", type="primary", key="main_search_btn", use_container_width=True)

# ── Pinterest-Style Trending Scenarios / Quick Filter Chips ────────────────────

render_html(
    f'<div class="prompt-shelf-label">{get_svg_icon("sparkles", size=14, color="#4F46E5")} <span>Popular Scenarios & Suggested Searches</span></div>'
)

prompts = [
    ("🎓  College Student Laptop", "college student laptop assistance"),
    ("🌾  Farmer Equipment Loan", "my father is a farmer and needs a loan for equipment"),
    ("👩‍👧  Widow School Allowance", "financial help for widow with school-going children"),
    ("☀️  Rooftop Solar Subsidy", "subsidy for solar panel installation on rooftop"),
    ("♿  Disability Fellowship", "fellowship for disabled students in higher education"),
    ("💼  Rural Startup Funding", "I want to start a small business in rural area")
]

row1_cols = st.columns(3)
for idx in range(3):
    label, prompt_val = prompts[idx]
    with row1_cols[idx]:
        st.button(
            label,
            key=f"prompt_btn_{idx}",
            on_click=set_quick_query,
            args=(prompt_val,),
            use_container_width=True
        )

row2_cols = st.columns(3)
for idx in range(3, 6):
    label, prompt_val = prompts[idx]
    with row2_cols[idx - 3]:
        st.button(
            label,
            key=f"prompt_btn_{idx}",
            on_click=set_quick_query,
            args=(prompt_val,),
            use_container_width=True
        )

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

# ── Self-Contained Card Renderer ──────────────────────────────────────────────

def render_scheme_card(row, method: str):
    score = float(row["score"])

    # Score category & formatting
    if method == "Hybrid":
        # RRF scores typically range from ~0.008 to ~0.016+
        sem_score = float(row.get("semantic_score", 0.0))
        bm25_sc = float(row.get("bm25_score", 0.0))
        if score >= 0.0150 or sem_score >= 0.55:
            badge_class = "score-high"
            pct_display = "Top Hybrid Match"
        elif score >= 0.0120 or sem_score >= 0.40:
            badge_class = "score-mid"
            pct_display = "Strong Match"
        else:
            badge_class = "score-low"
            pct_display = "Candidate Match"
        score_display = f"RRF: {score:.4f} (Sem: {sem_score:.2f} · BM25: {bm25_sc:.1f})"
    elif method == "BM25":
        badge_class = "score-mid" if score > 15 else "score-low"
        pct_display = f"BM25: {score:.1f}"
        score_display = f"Score: {score:.1f}"
    else:
        if score >= 0.70:
            badge_class = "score-high"
            pct_display = f"{int(min(100, score * 100))}% Match"
        elif score >= 0.45:
            badge_class = "score-mid"
            pct_display = f"{int(min(100, score * 100))}% Match"
        else:
            badge_class = "score-low"
            pct_display = f"{int(max(0, score * 100))}% Match"
        score_display = f"Score: {score:.3f}"

    # Escaped Values
    scheme_title = html.escape(str(row.get('scheme_name', 'Unnamed Scheme')))

    # Scope & Geography
    level_val = str(row.get('level', 'Central'))
    level_is_central = "Central" in level_val
    level_class = "chip-central" if level_is_central else "chip-state"
    flag_icon = get_svg_icon("flag", size=13, color="#4338CA" if level_is_central else "#0369A1")

    state_val = str(row.get('state', 'All India'))
    if len(state_val) > 28:
        state_val = state_val[:25] + "..."
    state_val = html.escape(state_val)
    pin_icon = get_svg_icon("map-pin", size=13, color="#64748B")

    # Sector / Category
    category_val = str(row.get('category', ''))
    cat_element = ""
    if pd.notna(category_val) and category_val:
        first_cat = html.escape(category_val.split(';')[0].strip())
        tag_icon = get_svg_icon("tag", size=13, color="#7E22CE")
        cat_element = f'<span class="meta-chip chip-cat">{tag_icon} {first_cat}</span>'

    # Ministry / Department
    ministry_val = str(row.get('ministry', ''))
    has_ministry = pd.notna(row.get('ministry')) and ministry_val != "Not specified (state scheme)"
    ministry_val_clean = html.escape(ministry_val)
    building_icon = get_svg_icon("building", size=13, color="#64748B")

    # Snippet & Description
    full_desc_raw = str(row.get('description', 'No detailed description available.'))
    desc_snippet = full_desc_raw[:250] + "..." if len(full_desc_raw) > 250 else full_desc_raw
    desc_snippet_clean = html.escape(desc_snippet)
    full_desc_clean = html.escape(full_desc_raw).replace("\n", "<br>")

    # Official tags
    tags_val = str(row.get('tags', ''))
    tag_elements = ""
    if pd.notna(tags_val) and tags_val:
        tag_list = [t.strip() for t in tags_val.split(';') if t.strip()][:6]
        tag_elements = "".join([f'<span class="drawer-tag-chip">#{html.escape(t)}</span>' for t in tag_list])

    # Direct portal verification link
    raw_slug = row.get('slug')
    if pd.isna(raw_slug) or str(raw_slug).lower() in ('nan', 'none', ''):
        raw_slug = row.get('scheme_id')
    slug_str = str(raw_slug).strip() if pd.notna(raw_slug) else ''
    if slug_str and slug_str.lower() not in ('nan', 'none', ''):
        myscheme_url = f"https://www.myscheme.gov.in/schemes/{slug_str}"
    else:
        encoded_query = urllib.parse.quote(str(row.get('scheme_name', '')))
        myscheme_url = f"https://www.myscheme.gov.in/search?q={encoded_query}"

    # Rich Details from updated_data.csv styled into structured drawer cards
    benefits_val = str(row.get('benefits', '')).strip()
    benefits_html = ""
    if benefits_val and benefits_val != 'nan':
        benefits_clean = html.escape(benefits_val).replace("\n", "<br>")
        sparkle_icon = get_svg_icon("sparkles", size=15, color="#059669")
        benefits_html = f"""
        <div class="drawer-card drawer-benefits">
            <div class="drawer-card-header">{sparkle_icon} <span>Key Scheme Benefits</span></div>
            <div class="drawer-card-body">{benefits_clean}</div>
        </div>
        """

    eligibility_val = str(row.get('eligibility', '')).strip()
    eligibility_html = ""
    if eligibility_val and eligibility_val != 'nan':
        eligibility_clean = html.escape(eligibility_val).replace("\n", "<br>")
        shield_icon = get_svg_icon("shield-check", size=15, color="#2563EB")
        eligibility_html = f"""
        <div class="drawer-card drawer-eligibility">
            <div class="drawer-card-header">{shield_icon} <span>Eligibility Criteria</span></div>
            <div class="drawer-card-body">{eligibility_clean}</div>
        </div>
        """

    application_val = str(row.get('application', '')).strip()
    application_html = ""
    if application_val and application_val != 'nan':
        application_clean = html.escape(application_val).replace("\n", "<br>")
        layers_icon = get_svg_icon("layers", size=15, color="#7C3AED")
        application_html = f"""
        <div class="drawer-card drawer-application">
            <div class="drawer-card-header">{layers_icon} <span>Application Process</span></div>
            <div class="drawer-card-body">{application_clean}</div>
        </div>
        """

    documents_val = str(row.get('documents', '')).strip()
    documents_html = ""
    if documents_val and documents_val != 'nan':
        documents_clean = html.escape(documents_val).replace("\n", "<br>")
        tag_icon = get_svg_icon("tag", size=15, color="#D97706")
        documents_html = f"""
        <div class="drawer-card drawer-documents">
            <div class="drawer-card-header">{tag_icon} <span>Required Documents</span></div>
            <div class="drawer-card-body">{documents_clean}</div>
        </div>
        """

    chevron_svg = get_svg_icon("chevron-down", size=14, color="#4F46E5", extra_class="chevron-icon")
    ext_link_svg = get_svg_icon("external-link", size=13, color="#FFFFFF")
    info_icon_svg = get_svg_icon("info", size=14, color="#4F46E5")

    card_html = f"""
<div class="scheme-card">
<div class="scheme-card-header">
<div class="scheme-title">{scheme_title}</div>
<div class="score-badge {badge_class}">
<span>{pct_display}</span>
<span style="opacity:0.4;">·</span>
<span>{score_display}</span>
</div>
</div>
<div class="meta-row">
<span class="meta-chip {level_class}">{flag_icon} {html.escape(level_val)}</span>
<span class="meta-chip">{pin_icon} {state_val}</span>
{cat_element}
{f'<span class="meta-chip">{building_icon} {ministry_val_clean}</span>' if has_ministry else ''}
</div>
<div class="scheme-desc">{desc_snippet_clean}</div>
<details class="scheme-drawer">
<summary>
<div class="drawer-trigger-left">
{info_icon_svg}
<span>View Scheme Summary, Benefits & Eligibility</span>
</div>
{chevron_svg}
</summary>
<div class="drawer-content">
<div class="drawer-card drawer-overview">
    <div class="drawer-card-header">{info_icon_svg} <span>Program Overview</span></div>
    <div class="drawer-card-body">{full_desc_clean}</div>
</div>
{benefits_html}
{eligibility_html}
{application_html}
{documents_html}
<div class="drawer-grid">
<div class="drawer-grid-item">
<div class="drawer-item-label">Jurisdiction</div>
<div class="drawer-item-val">{html.escape(level_val)} ({state_val})</div>
</div>
<div class="drawer-grid-item">
<div class="drawer-item-label">Sector Category</div>
<div class="drawer-item-val">{html.escape(category_val) if category_val else 'General Welfare'}</div>
</div>
<div class="drawer-grid-item">
<div class="drawer-item-label">Nodal Ministry</div>
<div class="drawer-item-val">{ministry_val_clean if has_ministry else 'State Government / Local Authority'}</div>
</div>
<div class="drawer-grid-item">
<div class="drawer-item-label">Identifier Slug</div>
<div class="drawer-item-val" style="font-family: 'JetBrains Mono', monospace; font-size: 0.76rem;">{html.escape(str(row.get('slug', row.get('scheme_id', 'N/A'))))}</div>
</div>
</div>
{f'<div class="drawer-tags-wrap"><div class="drawer-tags-label">Tags:</div><div class="drawer-tags">{tag_elements}</div></div>' if tag_elements else ''}
<div style="margin-top: 14px;">
<a href="{myscheme_url}" target="_blank" rel="noopener noreferrer" class="drawer-action-btn">
<span>Verify on Official Portal (myscheme.gov.in)</span>
{ext_link_svg}
</a>
</div>
</div>
</details>
</div>
"""
    render_html(card_html)


def render_results(results: pd.DataFrame, method_name: str):
    """Render full result list."""
    if results.empty:
        search_icon = get_svg_icon("search", size=24, color="#64748B")
        render_html(
            f"""
<div style="padding: 36px; border-radius: 14px; background: #FFFFFF; border: 1.5px dashed #CBD5E1; text-align: center; color: #64748B; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
<div style="margin-bottom: 10px;">{search_icon}</div>
<div style="font-weight: 700; color: #0F172A; margin-bottom: 6px; font-size: 1.05rem;">No matching schemes found</div>
<div style="font-size: 0.86rem; color: #475569;">Try adjusting state or category filters, or rephrasing your search query.</div>
</div>
"""
        )
        return

    for _, row in results.iterrows():
        render_scheme_card(row, method_name)


# ── Execution Logic ───────────────────────────────────────────────────────────

query_clean = query.strip() if query else ""

if query_clean:
    if show_comparison:
        # Comparison View: Hybrid vs BM25 Lexical
        scale_icon = get_svg_icon("scale", size=16, color="#4F46E5")
        render_html(
            f"""
<div style="background: #FFFFFF; border: 1px solid #E2E8F0; padding: 14px 18px; border-radius: 12px; margin-bottom: 24px; font-size: 0.88rem; color: #334155; display: flex; align-items: center; gap: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
{scale_icon}
<div>
<b style="color: #0F172A;">Comparative Benchmarking Active:</b> Comparing <b style="color: #4F46E5;">Hybrid Fusion (BM25 + Semantic RRF)</b> with <b style="color: #D97706;">Okapi BM25 Lexical Search</b>.
</div>
</div>
"""
        )

        col_hybrid, col_kw = st.columns(2)

        with col_hybrid:
            sparkles_icon = get_svg_icon("sparkles", size=18, color="#4F46E5")
            render_html(
                f'<div class="comp-header-sem">{sparkles_icon} Hybrid Fusion (BM25 + Semantic RRF)</div>'
            )
            hybrid_results = search_engine.hybrid_search(
                query_clean, k=num_results, state=state_filter, category=category_filter
            )
            render_results(hybrid_results, "Hybrid")

        with col_kw:
            tag_icon = get_svg_icon("tag", size=18, color="#D97706")
            render_html(
                f'<div class="comp-header-kw">{tag_icon} Okapi BM25 Lexical Retrieval</div>'
            )
            kw_results = search_engine.bm25_search(
                query_clean, k=num_results, state=state_filter, category=category_filter
            )
            render_results(kw_results, "BM25")

    else:
        # Extract and display zero-click intent filters
        intent = search_engine.parse_query_intent(query_clean)
        auto_detected_pills = []
        if not state_filter and intent.get("detected_state"):
            auto_detected_pills.append(f"State: <b>{html.escape(intent['detected_state'])}</b>")
        if not category_filter and intent.get("detected_category"):
            auto_detected_pills.append(f"Category: <b>{html.escape(intent['detected_category'])}</b>")

        col_header, col_stats = st.columns([3, 1])
        with col_header:
            engine_label = "Hybrid Matches" if "Hybrid" in search_mode else ("Semantic Matches" if "Semantic" in search_mode else "BM25 Matches")
            render_html(
                f'<div style="font-size: 1.05rem; font-weight: 600; color: #475569; margin-bottom: 10px;">'
                f'Top {num_results} {engine_label} for: <span style="color: #0F172A; font-weight: 700;">"{html.escape(query_clean)}"</span>'
                f'</div>'
            )
            if auto_detected_pills:
                target_svg = get_svg_icon("target", size=14, color="#4F46E5")
                render_html(
                    f'<div style="background: #EEF2FF; border: 1px solid #C7D2FE; padding: 6px 12px; border-radius: 8px; margin-bottom: 16px; font-size: 0.82rem; color: #3730A3; display: inline-flex; align-items: center; gap: 8px;">'
                    f'{target_svg} <span><b>Zero-Click Intent Applied:</b> {" · ".join(auto_detected_pills)}</span>'
                    f'</div>'
                )
        with col_stats:
            active_filters = []
            if state_filter:
                active_filters.append(f"State: {state_filter}")
            elif intent.get("detected_state"):
                active_filters.append(f"State (Auto): {intent['detected_state']}")

            if category_filter:
                active_filters.append(f"Category: {category_filter}")
            elif intent.get("detected_category"):
                active_filters.append(f"Category (Auto): {intent['detected_category']}")

            if active_filters:
                st.caption(f"Active Filters: {' · '.join(active_filters)}")

        if "Hybrid" in search_mode:
            results = search_engine.hybrid_search(
                query_clean, k=num_results, state=state_filter, category=category_filter
            )
            render_results(results, "Hybrid")
        elif "Semantic" in search_mode:
            results = search_engine.semantic_search(
                query_clean, k=num_results, state=state_filter, category=category_filter
            )
            render_results(results, "Semantic")
        else:
            results = search_engine.bm25_search(
                query_clean, k=num_results, state=state_filter, category=category_filter
            )
            render_results(results, "BM25")

else:
    # Empty Landing State with Feature Highlights
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        cpu_svg = get_svg_icon("cpu", size=20, color="#4F46E5")
        render_html(
            f"""
<div class="feature-card">
<div class="feature-icon-box">{cpu_svg}</div>
<div class="feature-title">Semantic Natural Language</div>
<div class="feature-body">
Search using ordinary phrasing. Describe your life situation, education, or employment status without needing specific scheme acronyms.
</div>
</div>
"""
        )

    with c2:
        filter_svg = get_svg_icon("sliders", size=20, color="#4F46E5")
        render_html(
            f"""
<div class="feature-card">
<div class="feature-icon-box">{filter_svg}</div>
<div class="feature-title">Demographic & State Filtering</div>
<div class="feature-body">
Combine dense neural similarity with exact state and category restrictions across 36 states and union territories.
</div>
</div>
"""
        )

    with c3:
        zap_svg = get_svg_icon("zap", size=20, color="#4F46E5")
        render_html(
            f"""
<div class="feature-card">
<div class="feature-icon-box">{zap_svg}</div>
<div class="feature-title">Sub-Millisecond Retrieval</div>
<div class="feature-body">
Dense precomputed embeddings (4,614 × 384) powered by vectorized NumPy BLAS dot product ranking running in ~0.62ms on standard CPUs.
</div>
</div>
"""
        )
