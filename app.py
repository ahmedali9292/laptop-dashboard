import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Laptop Price EDA Dashboard",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0f1628 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1525 0%, #0a0e1a 100%);
        border-right: 1px solid rgba(0, 212, 255, 0.2);
    }

    [data-testid="stSidebar"] * {
        color: #8ba3c7 !important;
    }

    /* Radio buttons in sidebar */
    [data-testid="stSidebar"] .stRadio label {
        font-size: 1rem !important;
        font-weight: 600 !important;
        padding: 6px 0 !important;
    }

    /* KPI metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #131d35, #0d1525);
        border: 1px solid rgba(0, 212, 255, 0.25);
        border-radius: 14px;
        padding: 18px 20px !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.08);
    }

    [data-testid="stMetricLabel"] {
        color: #8ba3c7 !important;
        font-size: 0.82rem !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }

    [data-testid="stMetricValue"] {
        color: #00ffcc !important;
        font-size: 1.7rem !important;
        font-weight: 800 !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: #131d35;
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 10px;
    }

    /* Success message */
    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Divider */
    hr {
        border-color: rgba(0, 212, 255, 0.15) !important;
    }

    /* Headers */
    h1, h2, h3 {
        color: #00d4ff !important;
    }

    /* All text */
    p, li {
        color: #c8d8ec !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB STYLE
# ─────────────────────────────────────────────
COLORS = {
    'bg':       '#0f1628',
    'card':     '#131d35',
    'cyan':     '#00ffcc',
    'blue':     '#00d4ff',
    'purple':   '#7b5ea7',
    'orange':   '#ff6b35',
    'gold':     '#ffd700',
    'pink':     '#ff6b9d',
    'green':    '#00ff88',
    'text':     '#c8d8ec',
    'muted':    '#5a7a9a',
    'grid':     '#1a2640',
}

BRAND_COLORS = [COLORS['cyan'], COLORS['blue'], COLORS['orange'], COLORS['purple'], COLORS['gold']]

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(COLORS['card'])
    ax.tick_params(colors=COLORS['text'], labelsize=9)
    ax.xaxis.label.set_color(COLORS['text'])
    ax.yaxis.label.set_color(COLORS['text'])
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS['grid'])
    ax.grid(axis='y', color=COLORS['grid'], linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=COLORS['cyan'], fontsize=11, fontweight='bold',
                     pad=10, fontfamily='monospace')
    if xlabel:
        ax.set_xlabel(xlabel, color=COLORS['muted'], fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=COLORS['muted'], fontsize=9)

def make_fig(w=12, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(COLORS['bg'])
    return fig, ax

def make_figs(rows, cols, w=14, h=5):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h))
    fig.patch.set_facecolor(COLORS['bg'])
    return fig, axes

# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/laptop_price.csv")
    return df

df = load_data()

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <div style='font-size:2.5rem; filter: drop-shadow(0 0 12px #00d4ff);'>💻</div>
        <div style='font-family:monospace; font-size:0.9rem; font-weight:900;
                    color:#00d4ff; letter-spacing:1px; line-height:1.5;
                    text-shadow: 0 0 12px rgba(0,212,255,0.7);'>
            LAPTOP PRICE<br>EDA DASHBOARD
        </div>
    </div>
    <hr style='border-color: rgba(0,212,255,0.2); margin-bottom:16px;'>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-family:monospace; font-size:0.65rem; letter-spacing:3px; color:#5a7a9a;'>⬡ NAVIGATION</p>", unsafe_allow_html=True)

    page = st.radio(
        label="",
        options=[
            "🏠  Overview",
            "💰  Price Analysis",
            "🏷️  Brand Analysis",
            "⚙️  Specs Analysis",
            "📐  Size & Weight",
            "🔗  Correlation",
        ],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color: rgba(0,212,255,0.15); margin-top:20px;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='padding: 10px 0;'>
        <p style='font-family:monospace; font-size:0.62rem; letter-spacing:2px; color:#5a7a9a; margin-bottom:8px;'>
            🎨 DASHBOARD THEME
        </p>
        <div style='background: linear-gradient(135deg,#131d35,#0d1525);
                    border:1px solid rgba(0,212,255,0.25); border-radius:8px;
                    padding:10px 14px; font-size:0.9rem; font-weight:700;
                    color:#00ffcc; display:flex; justify-content:space-between;'>
            <span>Cyber Matrix</span><span>▼</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SECTION HEADER HELPER
# ─────────────────────────────────────────────
def section_header(icon, title):
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:14px; margin-bottom:24px;'>
        <span style='font-size:2.4rem; filter: drop-shadow(0 0 14px rgba(0,212,255,0.8));'>{icon}</span>
        <h1 style='font-family:monospace; font-size:1.8rem; font-weight:900;
                   background:linear-gradient(135deg,#00d4ff,#00ffcc);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                   background-clip:text; margin:0;'>{title}</h1>
    </div>
    """, unsafe_allow_html=True)

def sub_title(icon, text):
    st.markdown(f"""
    <div style='font-family:monospace; font-size:0.78rem; font-weight:700;
                color:#00d4ff; letter-spacing:2px; text-transform:uppercase;
                margin: 20px 0 10px 0; display:flex; align-items:center; gap:8px;'>
        {icon} {text}
        <div style='flex:1; height:1px; background:linear-gradient(90deg,rgba(0,212,255,0.4),transparent); margin-left:8px;'></div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PAGE 1: OVERVIEW
# ─────────────────────────────────────────────
if page == "🏠  Overview":
    section_header("💻", "Laptop Price Dashboard")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🖥️ Total Laptops",    f"{len(df):,}")
    col2.metric("💰 Avg Price",        f"₨ {df['Price'].mean():,.0f}")
    col3.metric("🏆 Max Price",        f"₨ {df['Price'].max():,.0f}")
    col4.metric("🏷️ Unique Brands",    f"{df['Brand'].nunique()}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Raw Data Preview
    with st.expander("🗂️  Preview Raw Data"):
        st.dataframe(
            df.head(10).style.set_properties(**{
                'background-color': '#131d35',
                'color': '#c8d8ec',
                'border': '1px solid rgba(0,212,255,0.1)'
            }),
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset Shape & Stats
    col_l, col_r = st.columns(2)

    with col_l:
        sub_title("📋", "Dataset Shape & Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Dtype': [str(d) for d in df.dtypes],
            'Non-Null': [df[c].notna().sum() for c in df.columns],
            'Nulls': [df[c].isna().sum() for c in df.columns]
        }).reset_index(drop=True)
        st.dataframe(dtype_df, use_container_width=True, hide_index=False)

    with col_r:
        sub_title("📊", "Statistical Summary")
        st.dataframe(
            df.describe().round(3).T,
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if df.isnull().sum().sum() == 0:
        st.success("✅  No missing values found in the dataset!")
    else:
        st.warning(f"⚠️  Missing values detected: {df.isnull().sum().sum()} total")

# ─────────────────────────────────────────────
#  PAGE 2: PRICE ANALYSIS
# ─────────────────────────────────────────────
elif page == "💰  Price Analysis":
    section_header("💰", "Price Analysis")

    # Mini stat cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Min",    f"₨ {df['Price'].min():,.0f}")
    c2.metric("Avg",    f"₨ {df['Price'].mean():,.0f}")
    c3.metric("Median", f"₨ {df['Price'].median():,.0f}")
    c4.metric("Max",    f"₨ {df['Price'].max():,.0f}")
    c5.metric("Std Dev",f"₨ {df['Price'].std():,.0f}")
    c6.metric("Range",  f"₨ {(df['Price'].max()-df['Price'].min()):,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Price Distribution + Brand Avg Price
    sub_title("📊", "Price Distribution & Brand Comparison")
    fig, (ax1, ax2) = make_figs(1, 2, w=14, h=5)

    # Price range bins
    bins   = [0, 10000, 15000, 20000, 25000, 30000, 35000]
    labels = ['Under 10k', '10k-15k', '15k-20k', '20k-25k', '25k-30k', 'Above 30k']
    df['PriceRange'] = pd.cut(df['Price'], bins=bins, labels=labels)
    pr = df['PriceRange'].value_counts().reindex(labels)
    bar_colors = [COLORS['cyan'], COLORS['blue'], COLORS['purple'],
                  COLORS['orange'], COLORS['gold'], COLORS['green']]
    bars = ax1.bar(pr.index, pr.values, color=[c+'bb' for c in bar_colors],
                   edgecolor=bar_colors, linewidth=1.5, width=0.6)
    for bar, val in zip(bars, pr.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 str(val), ha='center', va='bottom', color=COLORS['text'], fontsize=9)
    style_ax(ax1, "PRICE DISTRIBUTION", "Price Range", "Number of Laptops")
    ax1.tick_params(axis='x', rotation=15)

    # Brand Avg Price
    brand_avg = df.groupby('Brand')['Price'].mean().sort_values(ascending=False)
    bars2 = ax2.bar(brand_avg.index, brand_avg.values,
                    color=[c+'bb' for c in BRAND_COLORS],
                    edgecolor=BRAND_COLORS, linewidth=1.5, width=0.5)
    for bar, val in zip(bars2, brand_avg.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                 f'₨{val:,.0f}', ha='center', va='bottom', color=COLORS['text'], fontsize=8)
    style_ax(ax2, "AVG PRICE BY BRAND", "Brand", "Avg Price (₨)")

    plt.tight_layout(pad=2)
    st.pyplot(fig)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # Storage vs Price
    sub_title("💾", "Storage Capacity vs Avg Price")
    fig2, ax3 = make_fig(w=12, h=4.5)
    storage_avg = df.groupby('Storage_Capacity')['Price'].mean()
    storage_colors = [COLORS['blue'], COLORS['cyan'], COLORS['orange']]
    bars3 = ax3.bar([f"{s} GB" for s in storage_avg.index], storage_avg.values,
                    color=[c+'bb' for c in storage_colors],
                    edgecolor=storage_colors, linewidth=2, width=0.4)
    for bar, val in zip(bars3, storage_avg.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
                 f'₨{val:,.0f}', ha='center', va='bottom', color=COLORS['cyan'],
                 fontsize=11, fontweight='bold')
    style_ax(ax3, "AVERAGE PRICE BY STORAGE CAPACITY", "Storage (GB)", "Avg Price (₨)")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ─────────────────────────────────────────────
#  PAGE 3: BRAND ANALYSIS
# ─────────────────────────────────────────────
elif page == "🏷️  Brand Analysis":
    section_header("🏷️", "Brand Analysis")

    sub_title("🥧", "Brand Distribution & Counts")
    fig, (ax1, ax2) = make_figs(1, 2, w=14, h=5)

    brand_counts = df['Brand'].value_counts()

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        brand_counts.values,
        labels=brand_counts.index,
        autopct='%1.1f%%',
        colors=[c+'cc' for c in BRAND_COLORS],
        startangle=140,
        wedgeprops=dict(edgecolor='#0f1628', linewidth=2)
    )
    for t in texts: t.set_color(COLORS['text']); t.set_fontsize(10)
    for at in autotexts: at.set_color(COLORS['bg']); at.set_fontweight('bold'); at.set_fontsize(9)
    ax1.set_facecolor(COLORS['card'])
    ax1.set_title("BRAND DISTRIBUTION", color=COLORS['cyan'], fontsize=11,
                  fontweight='bold', fontfamily='monospace')

    # Bar chart
    bars = ax2.bar(brand_counts.index, brand_counts.values,
                   color=[c+'bb' for c in BRAND_COLORS],
                   edgecolor=BRAND_COLORS, linewidth=1.5, width=0.5)
    for bar, val in zip(bars, brand_counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha='center', va='bottom', color=COLORS['text'], fontsize=10,
                 fontweight='bold')
    style_ax(ax2, "LAPTOP COUNT BY BRAND", "Brand", "Count")

    fig.patch.set_facecolor(COLORS['bg'])
    plt.tight_layout(pad=2)
    st.pyplot(fig)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # Min / Avg / Max by brand
    sub_title("📈", "Min / Avg / Max Price by Brand")
    fig2, ax3 = make_fig(w=13, h=5)
    brands   = df['Brand'].unique()
    x        = np.arange(len(brands))
    w        = 0.25
    mins     = [df[df['Brand']==b]['Price'].min()  for b in brands]
    avgs     = [df[df['Brand']==b]['Price'].mean() for b in brands]
    maxs     = [df[df['Brand']==b]['Price'].max()  for b in brands]

    ax3.bar(x - w, mins, width=w, label='Min Price', color=COLORS['blue']+'99',   edgecolor=COLORS['blue'],   linewidth=1.5)
    ax3.bar(x,     avgs, width=w, label='Avg Price', color=COLORS['cyan']+'99',   edgecolor=COLORS['cyan'],   linewidth=1.5)
    ax3.bar(x + w, maxs, width=w, label='Max Price', color=COLORS['orange']+'99', edgecolor=COLORS['orange'], linewidth=1.5)

    ax3.set_xticks(x)
    ax3.set_xticklabels(brands)
    legend = ax3.legend(facecolor=COLORS['card'], edgecolor=COLORS['grid'],
                        labelcolor=COLORS['text'], fontsize=9)
    style_ax(ax3, "MIN / AVG / MAX PRICE BY BRAND", "Brand", "Price (₨)")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ─────────────────────────────────────────────
#  PAGE 4: SPECS ANALYSIS
# ─────────────────────────────────────────────
elif page == "⚙️  Specs Analysis":
    section_header("⚙️", "Specs Analysis")

    sub_title("🧠💾⚡", "RAM · Storage · Processor Distribution")
    fig, axes = make_figs(1, 3, w=16, h=5)

    # RAM Distribution (donut-style pie)
    ram_counts = df['RAM_Size'].value_counts().sort_index()
    ram_colors = [COLORS['blue'], COLORS['cyan'], COLORS['purple'], COLORS['orange']]
    wedges, texts, autotexts = axes[0].pie(
        ram_counts.values, labels=[f"{r} GB" for r in ram_counts.index],
        autopct='%1.1f%%', colors=[c+'cc' for c in ram_colors],
        startangle=90, wedgeprops=dict(edgecolor='#0f1628', linewidth=2, width=0.6)
    )
    for t in texts: t.set_color(COLORS['text']); t.set_fontsize(9)
    for at in autotexts: at.set_color(COLORS['bg']); at.set_fontsize(8); at.set_fontweight('bold')
    axes[0].set_facecolor(COLORS['card'])
    axes[0].set_title("RAM DISTRIBUTION", color=COLORS['cyan'], fontsize=10,
                      fontweight='bold', fontfamily='monospace')

    # Storage Distribution
    st_counts = df['Storage_Capacity'].value_counts().sort_index()
    st_colors = [COLORS['gold'], COLORS['green'], COLORS['pink']]
    wedges2, texts2, autotexts2 = axes[1].pie(
        st_counts.values, labels=[f"{s} GB" for s in st_counts.index],
        autopct='%1.1f%%', colors=[c+'cc' for c in st_colors],
        startangle=90, wedgeprops=dict(edgecolor='#0f1628', linewidth=2, width=0.6)
    )
    for t in texts2: t.set_color(COLORS['text']); t.set_fontsize(9)
    for at in autotexts2: at.set_color(COLORS['bg']); at.set_fontsize(8); at.set_fontweight('bold')
    axes[1].set_facecolor(COLORS['card'])
    axes[1].set_title("STORAGE DISTRIBUTION", color=COLORS['cyan'], fontsize=10,
                      fontweight='bold', fontfamily='monospace')

    # Processor Speed Bins
    proc_bins   = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    proc_labels = ['1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0-3.5', '3.5-4.0']
    df['ProcBin'] = pd.cut(df['Processor_Speed'], bins=proc_bins, labels=proc_labels)
    proc_counts  = df['ProcBin'].value_counts().reindex(proc_labels)
    bars = axes[2].bar(proc_counts.index, proc_counts.values,
                       color=COLORS['purple']+'99', edgecolor=COLORS['purple'],
                       linewidth=1.5, width=0.6)
    for bar, val in zip(bars, proc_counts.values):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     str(val), ha='center', va='bottom', color=COLORS['text'], fontsize=9)
    style_ax(axes[2], "PROCESSOR SPEED BINS", "GHz Range", "Count")
    axes[2].tick_params(axis='x', rotation=15)

    fig.patch.set_facecolor(COLORS['bg'])
    plt.tight_layout(pad=2)
    st.pyplot(fig)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # RAM Avg Price + Storage Count
    sub_title("💰", "Avg Price by RAM & Storage Count")
    fig2, (ax4, ax5) = make_figs(1, 2, w=14, h=5)

    ram_avg = df.groupby('RAM_Size')['Price'].mean().sort_index()
    bars4   = ax4.bar([f"{r} GB" for r in ram_avg.index], ram_avg.values,
                      color=[c+'99' for c in ram_colors],
                      edgecolor=ram_colors, linewidth=1.5, width=0.5)
    for bar, val in zip(bars4, ram_avg.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                 f'₨{val:,.0f}', ha='center', va='bottom', color=COLORS['text'], fontsize=8)
    style_ax(ax4, "AVG PRICE BY RAM SIZE", "RAM (GB)", "Avg Price (₨)")

    st_laptop = df['Storage_Capacity'].value_counts().sort_index()
    bars5 = ax5.bar([f"{s} GB" for s in st_laptop.index], st_laptop.values,
                    color=[c+'99' for c in st_colors],
                    edgecolor=st_colors, linewidth=1.5, width=0.4)
    for bar, val in zip(bars5, st_laptop.values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha='center', va='bottom', color=COLORS['text'], fontsize=10,
                 fontweight='bold')
    style_ax(ax5, "LAPTOPS PER STORAGE TYPE", "Storage (GB)", "Count")

    fig2.patch.set_facecolor(COLORS['bg'])
    plt.tight_layout(pad=2)
    st.pyplot(fig2)
    plt.close()

# ─────────────────────────────────────────────
#  PAGE 5: SIZE & WEIGHT
# ─────────────────────────────────────────────
elif page == "📐  Size & Weight":
    section_header("📐", "Size & Weight")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Min Screen",  f'{df["Screen_Size"].min():.2f}"')
    c2.metric("Avg Screen",  f'{df["Screen_Size"].mean():.2f}"')
    c3.metric("Max Screen",  f'{df["Screen_Size"].max():.2f}"')
    c4.metric("Min Weight",  f'{df["Weight"].min():.2f} kg')
    c5.metric("Avg Weight",  f'{df["Weight"].mean():.2f} kg')
    c6.metric("Max Weight",  f'{df["Weight"].max():.2f} kg')

    st.markdown("<br>", unsafe_allow_html=True)

    sub_title("📏⚖️", "Screen Size & Weight Distribution")
    fig, (ax1, ax2) = make_figs(1, 2, w=14, h=5)

    # Screen size bins
    screen_bins   = [11, 12, 13, 14, 15, 16, 17]
    screen_labels = ['11-12"', '12-13"', '13-14"', '14-15"', '15-16"', '16-17"']
    df['ScreenBin'] = pd.cut(df['Screen_Size'], bins=screen_bins, labels=screen_labels)
    scr = df['ScreenBin'].value_counts().reindex(screen_labels)
    bars1 = ax1.bar(scr.index, scr.values, color=COLORS['cyan']+'99',
                    edgecolor=COLORS['cyan'], linewidth=1.5, width=0.6)
    for bar, val in zip(bars1, scr.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha='center', va='bottom', color=COLORS['text'], fontsize=9)
    style_ax(ax1, "SCREEN SIZE DISTRIBUTION", "Screen Size", "Count")

    # Weight bins
    weight_bins   = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    weight_labels = ['2.0-2.5', '2.5-3.0', '3.0-3.5', '3.5-4.0', '4.0-4.5', '4.5-5.0']
    df['WeightBin'] = pd.cut(df['Weight'], bins=weight_bins, labels=weight_labels)
    wt = df['WeightBin'].value_counts().reindex(weight_labels)
    bars2 = ax2.bar(wt.index, wt.values, color=COLORS['orange']+'99',
                    edgecolor=COLORS['orange'], linewidth=1.5, width=0.6)
    for bar, val in zip(bars2, wt.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(val), ha='center', va='bottom', color=COLORS['text'], fontsize=9)
    style_ax(ax2, "WEIGHT DISTRIBUTION", "Weight (kg)", "Count")

    fig.patch.set_facecolor(COLORS['bg'])
    plt.tight_layout(pad=2)
    st.pyplot(fig)
    plt.close()

# ─────────────────────────────────────────────
#  PAGE 6: CORRELATION
# ─────────────────────────────────────────────
elif page == "🔗  Correlation":
    section_header("🔗", "Correlation Analysis")

    num_df = df[['Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight', 'Price']]
    corr   = num_df.corr()

    sub_title("🔥", "Correlation Heatmap")
    fig, ax = make_fig(w=10, h=6)

    cols     = corr.columns.tolist()
    n        = len(cols)
    short    = ['Proc\nSpeed', 'RAM', 'Storage', 'Screen', 'Weight', 'Price']
    mat      = corr.values

    im = ax.imshow(mat, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(n)); ax.set_xticklabels(short, color=COLORS['text'], fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, color=COLORS['text'], fontsize=9)
    ax.set_facecolor(COLORS['card'])

    for i in range(n):
        for j in range(n):
            val  = mat[i, j]
            clr  = 'black' if abs(val) > 0.5 else COLORS['text']
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=clr, fontsize=9, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.yaxis.set_tick_params(color=COLORS['text'])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=COLORS['text'])
    cbar.ax.set_facecolor(COLORS['card'])

    ax.set_title("CORRELATION HEATMAP", color=COLORS['cyan'], fontsize=12,
                 fontweight='bold', fontfamily='monospace', pad=12)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS['grid'])

    fig.patch.set_facecolor(COLORS['bg'])
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # Correlation with Price bar chart
    sub_title("📊", "Feature Correlation with Price")
    fig2, ax2 = make_fig(w=10, h=4.5)

    price_corr = corr['Price'].drop('Price').sort_values()
    bar_clrs   = [COLORS['orange'] if v < 0 else COLORS['cyan'] for v in price_corr.values]
    edge_clrs  = [COLORS['orange'] if v < 0 else COLORS['cyan'] for v in price_corr.values]

    bars = ax2.barh(price_corr.index, price_corr.values,
                    color=[c+'99' for c in bar_clrs],
                    edgecolor=edge_clrs, linewidth=1.5, height=0.5)

    for bar, val in zip(bars, price_corr.values):
        xpos = val + 0.01 if val >= 0 else val - 0.01
        ha   = 'left' if val >= 0 else 'right'
        ax2.text(xpos, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', ha=ha,
                 color=COLORS['text'], fontsize=9, fontweight='bold')

    ax2.axvline(0, color=COLORS['muted'], linewidth=1, linestyle='--', alpha=0.6)
    ax2.set_facecolor(COLORS['card'])
    ax2.tick_params(colors=COLORS['text'])
    for spine in ax2.spines.values():
        spine.set_edgecolor(COLORS['grid'])
    ax2.grid(axis='x', color=COLORS['grid'], linewidth=0.6, alpha=0.7)
    ax2.set_title("FEATURE CORRELATION WITH PRICE", color=COLORS['cyan'],
                  fontsize=11, fontweight='bold', fontfamily='monospace', pad=10)
    ax2.set_xlabel("Correlation Coefficient", color=COLORS['muted'], fontsize=9)
    ax2.tick_params(axis='y', colors=COLORS['text'])

    # Key insight
    st.markdown(f"""
    <div style='background:rgba(0,255,204,0.07); border:1px solid rgba(0,255,204,0.3);
                border-radius:10px; padding:14px 18px; margin-bottom:12px;'>
        <span style='color:#00ffcc; font-weight:700; font-size:0.95rem;'>
        🔑 Key Insight:
        </span>
        <span style='color:#c8d8ec; font-size:0.9rem;'>
        Storage Capacity has the highest correlation with Price (0.998) —
        meaning storage is the strongest predictor of laptop price in this dataset.
        </span>
    </div>
    """, unsafe_allow_html=True)

    fig2.patch.set_facecolor(COLORS['bg'])
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
