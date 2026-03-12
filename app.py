"""
GeoSight — Streamlit UI
Run: streamlit run app.py
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
from dotenv import load_dotenv

load_dotenv()

from geosight.agent import run_agent

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GeoSight — UK Land Intelligence",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=DM+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -1px;
        color: #0f172a;
        line-height: 1.1;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748b;
        margin-top: 0.3rem;
        margin-bottom: 2rem;
    }
    .tag {
        display: inline-block;
        background: #f0fdf4;
        color: #166534;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 4px;
        margin-bottom: 4px;
        border: 1px solid #bbf7d0;
    }
    .stButton > button {
        background: #0f172a;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
    }
    .stButton > button:hover { background: #1e293b; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_title, col_badges = st.columns([3, 1])
with col_title:
    st.markdown('<div class="main-title">🗺️ GeoSight</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">UK Land Intelligence Agent — '
        'flood risk · protected designations · land use · planning policy · vision</div>',
        unsafe_allow_html=True
    )
with col_badges:
    st.markdown("""
    <div style="text-align:right; margin-top:0.5rem;">
        <span class="tag">LangGraph</span>
        <span class="tag">RAG</span>
        <span class="tag">LLaVA</span>
        <span class="tag">Free to run</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
col_input, col_upload = st.columns([2, 1])

with col_input:
    postcode = st.text_input(
        "UK Postcode",
        placeholder="e.g. TQ13 8HH, SO41 8DQ, NG1 1AA",
        help="Enter any UK postcode.",
    )

with col_upload:
    uploaded_image = st.file_uploader(
        "Site photo (optional)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a site photo for AI visual analysis via LLaVA.",
    )

run_button = st.button("🔍 Analyse Land", use_container_width=True)

# ---------------------------------------------------------------------------
# Run agent
# ---------------------------------------------------------------------------
if run_button and postcode.strip():
    image_bytes = uploaded_image.read() if uploaded_image else None

    with st.spinner("Running GeoSight agent pipeline... this takes 30-60 seconds"):
        try:
            result = run_agent(postcode.strip(), image_bytes=image_bytes)
        except Exception as e:
            st.error(f"Agent failed: {e}")
            st.stop()

    if result.get("errors"):
        with st.expander("⚠️ Warnings during analysis", expanded=False):
            for err in result["errors"]:
                st.warning(err)

    # ---------------------------------------------------------------------------
    # Map
    # ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
    # Map
    # ---------------------------------------------------------------------------
    loc = result.get("location")
    if loc:
        st.subheader("📍 Location")
        m = folium.Map(
            location=[loc.lat, loc.lon],
            zoom_start=13,
            tiles="CartoDB positron"
        )
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satellite",
            overlay=False,
            control=True,
        ).add_to(m)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Terrain",
            overlay=False,
            control=True,
        ).add_to(m)
        folium.LayerControl(position="topright").add_to(m)
        folium.Marker(
            [loc.lat, loc.lon],
            popup=f"<b>{postcode.upper()}</b><br>{loc.display_name}",
            icon=folium.Icon(color="darkblue", icon="map-marker", prefix="fa"),
        ).add_to(m)
        folium.Circle(
            [loc.lat, loc.lon],
            radius=2000,
            color="#166534",
            fill=True,
            fill_opacity=0.05,
            popup="2km search radius",
        ).add_to(m)
        st_folium(m, width=None, height=380, returned_objects=[], key="main_map")

    # ---------------------------------------------------------------------------
    # Data panels
    # ---------------------------------------------------------------------------
    st.divider()
    st.subheader("📊 Data Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        flood = result.get("flood_risk")
        st.markdown("**🌊 Flood Risk**")
        if flood:
            if flood.severe_warnings > 0:
                st.error(flood.summary)
            elif flood.warnings > 0 or flood.alerts > 0:
                st.warning(flood.summary)
            else:
                st.success(flood.summary)
            if flood.nearby_stations:
                st.caption("Nearby monitoring stations:")
                for s in flood.nearby_stations:
                    st.caption(f"  · {s['name']} ({s['river']})")
        else:
            st.caption("Data unavailable")

    with col2:
        pa = result.get("protected_areas")
        st.markdown("**🌿 Protected Areas**")
        if pa:
            if pa.designations:
                st.warning(f"{len(pa.designations)} designation(s) found")
                for d in pa.designations[:4]:
                    st.markdown(
                        f'<span class="tag">{d.type}</span> {d.name}',
                        unsafe_allow_html=True
                    )
            else:
                st.success("No designations within 2km")
        else:
            st.caption("Data unavailable")

    with col3:
        lu = result.get("land_use")
        st.markdown("**🗺️ Land Use (OSM)**")
        if lu:
            all_features = lu.land_uses + lu.natural_features + lu.waterways
            if all_features:
                for f in all_features[:6]:
                    st.markdown(
                        f'<span class="tag">{f}</span>',
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No detailed land use data found")
        else:
            st.caption("Data unavailable")

    # ---------------------------------------------------------------------------
    # Vision
    # ---------------------------------------------------------------------------
    vis = result.get("vision")
    if vis and vis.image_provided:
        st.divider()
        st.subheader("📷 Site Visual Analysis")
        col_img, col_desc = st.columns([1, 2])
        with col_img:
            if uploaded_image:
                st.image(uploaded_image, use_container_width=True)
        with col_desc:
            st.markdown(vis.description)

    # ---------------------------------------------------------------------------
    # Full report
    # ---------------------------------------------------------------------------
    st.divider()
    st.subheader("📄 Land Intelligence Report")

    if result.get("report"):
        st.markdown(result["report"])
    else:
        st.warning("Report could not be generated.")

    # RAG sources
    rag = result.get("rag")
    if rag and rag.chunks:
        with st.expander("📚 Policy Document Sources", expanded=False):
            for i, chunk in enumerate(rag.chunks, 1):
                page_str = f", p.{chunk.page}" if chunk.page else ""
                st.markdown(
                    f"**[{i}]** {chunk.source}{page_str} "
                    f"*(relevance: {chunk.score:.2f})*"
                )
                st.caption(chunk.text[:300] + "...")
                st.divider()

elif run_button:
    st.warning("Please enter a UK postcode.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:0.85rem;">
GeoSight uses free public APIs (Environment Agency, Natural England, OpenStreetMap)
and runs entirely locally via Ollama. No API keys required.<br>
Built by <a href="https://linkedin.com/in/syed-sabeeth" style="color:#64748b;">
Syed Sabeeth Shoeb</a> ·
<a href="https://github.com/ssabeeth/geosight" style="color:#64748b;">GitHub</a>
</div>
""", unsafe_allow_html=True)