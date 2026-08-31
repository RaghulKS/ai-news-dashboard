"""Financial Event Intelligence Dashboard — event clusters, not article spam."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from finintel.config import load_config
from finintel.market.research import EventReturnResearch
from finintel.models.event import EventType, SentimentLabel
from finintel.pipeline.orchestrator import FinancialEventPipeline
from finintel.storage.database import ArticleStore

logging.basicConfig(level=logging.INFO)
st.set_page_config(
    page_title="Financial Event Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.4rem; font-weight: 700; color: #1a365d; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #4a5568; margin-bottom: 1.5rem; }
    .cluster-card {
        background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 1.25rem; margin-bottom: 1rem;
    }
    .event-badge {
        display: inline-block; padding: 0.2rem 0.6rem; border-radius: 6px;
        font-size: 0.8rem; font-weight: 600; margin-right: 0.4rem;
    }
    .metric-row { margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

EVENT_COLORS = {
    "earnings": "#3182ce", "guidance": "#805ad5", "m_and_a": "#d69e2e",
    "regulatory": "#e53e3e", "product": "#38a169", "layoffs": "#dd6b20",
    "litigation": "#c53030", "macro": "#2b6cb0", "analyst": "#6b46c1",
    "dividend": "#2f855a", "ipo": "#b7791f", "other": "#718096",
}

SENTIMENT_COLORS = {
    "positive": "#38a169", "negative": "#e53e3e", "neutral": "#a0aec0",
}


def init_session():
    if "pipeline_result" not in st.session_state:
        st.session_state.pipeline_result = None
    if "last_query" not in st.session_state:
        st.session_state.last_query = "AAPL"


@st.cache_resource
def get_pipeline():
    return FinancialEventPipeline(load_config())


@st.cache_resource
def get_store():
    return ArticleStore()


def render_cluster_card(cluster):
    event_color = EVENT_COLORS.get(cluster.event_type.value, "#718096")
    sent_color = SENTIMENT_COLORS.get(cluster.sentiment_consensus.value, "#a0aec0")

    st.markdown(f'<div class="cluster-card">', unsafe_allow_html=True)
    st.markdown(f"### {cluster.representative_title}")

    cols = st.columns(6)
    cols[0].markdown(
        f'<span class="event-badge" style="background:{event_color};color:white">'
        f'{cluster.event_type.value.replace("_", " ").upper()}</span>',
        unsafe_allow_html=True,
    )
    cols[1].metric("Articles", cluster.article_count)
    cols[2].metric("Sources", cluster.source_count)
    cols[3].metric("Novelty", f"{cluster.novelty_score:.2f}")
    cols[4].metric("Relevance", f"{cluster.market_relevance:.2f}")
    cols[5].metric("Contradiction", f"{cluster.contradiction_score:.2f}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f"**Sentiment:** {cluster.sentiment_consensus.value.title()} "
            f"(avg score: {cluster.avg_sentiment_score:.2f})"
        )
        if cluster.tickers:
            st.markdown(f"**Tickers:** {', '.join(cluster.tickers)}")
        if cluster.entities:
            names = [e.canonical_name for e in cluster.entities[:6]]
            st.markdown(f"**Entities:** {', '.join(names)}")
    with col_b:
        st.markdown(f"**Timeline:** {cluster.first_seen.strftime('%Y-%m-%d %H:%M')} → "
                    f"{cluster.last_seen.strftime('%Y-%m-%d %H:%M')}")
        st.markdown(f"**Sources:** {', '.join(cluster.sources[:5])}")

    if cluster.summary:
        st.info(f"**Summary (grounded in sources):** {cluster.summary}")

    with st.expander(f"View {len(cluster.article_urls)} source articles"):
        for url in cluster.article_urls:
            st.markdown(f"- [{url[:80]}...]({url})")

    st.markdown("</div>", unsafe_allow_html=True)


def render_timeline(clusters):
    if not clusters:
        return
    rows = []
    for c in clusters:
        rows.append({
            "Event": c.representative_title[:60],
            "Start": c.first_seen,
            "End": c.last_seen,
            "Type": c.event_type.value,
            "Articles": c.article_count,
            "Relevance": c.market_relevance,
        })
    df = pd.DataFrame(rows)
    fig = px.timeline(
        df, x_start="Start", x_end="End", y="Event",
        color="Type", hover_data=["Articles", "Relevance"],
        title="Event Timeline",
        color_discrete_map=EVENT_COLORS,
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=max(300, len(clusters) * 40))
    st.plotly_chart(fig, use_container_width=True)


def render_research_tab(ticker: str, clusters):
    st.markdown("### Leakage-Safe Market Impact Research")
    st.caption(
        "Event returns computed using prices available only AFTER event timestamps. "
        "We do not claim predictability unless results are statistically significant (n≥30, |t|>1.96)."
    )

    events = [
        {"event_time": c.first_seen, "event_type": c.event_type.value}
        for c in clusters
    ]
    if not events:
        st.warning("No events to analyze. Run a pipeline query first.")
        return

    research = EventReturnResearch()
    results = research.analyze_events(events, ticker)

    rows = []
    for r in results:
        rows.append({
            "Horizon (days)": r.horizon_days,
            "N Events": r.n_events,
            "Mean Return %": round(r.mean_return, 3),
            "Median Return %": round(r.median_return, 3),
            "Hit Rate": round(r.hit_rate, 3),
            "T-Stat": round(r.t_stat, 3),
            "Significant": r.is_significant,
            "Note": r.disclaimer,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    for r in results:
        if not r.is_significant:
            st.warning(f"Horizon {r.horizon_days}d: {r.disclaimer}")


def main():
    init_session()
    st.markdown('<p class="main-header">Financial Event Intelligence Engine</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Transformer NLP · entity linking · semantic clustering · '
        'novelty detection · leakage-safe market research</p>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Pipeline Controls")
        query = st.text_input("Ticker / Company / Topic", value=st.session_state.last_query)
        count = st.slider("Max articles", 5, 50, 20)
        sentiment_backend = st.selectbox(
            "Sentiment backend",
            ["lexicon", "finbert", "azure"],
            help="FinBERT requires transformers download on first run",
        )

        if st.button("Run Intelligence Pipeline", type="primary"):
            st.session_state.last_query = query
            cfg = load_config()
            cfg.nlp.sentiment_backend = sentiment_backend
            with st.spinner("Ingesting → NLP → clustering → storage..."):
                pipeline = FinancialEventPipeline(cfg)
                result = pipeline.run(query, count)
                st.session_state.pipeline_result = result

        st.divider()
        st.markdown("**Architecture**")
        st.markdown("""
        - Multi-source ingestion (NewsAPI + RSS)
        - FinBERT / lexicon sentiment
        - Financial entity linking
        - Event extraction (11 classes)
        - Sentence-transformer embeddings
        - Semantic story clustering
        - Novelty & contradiction signals
        - Azure optional (not required)
        """)

    result = st.session_state.pipeline_result

    if result:
        st.success(
            f"Fetched {result.articles_fetched} articles · "
            f"Stored {result.articles_stored} · "
            f"Skipped {result.duplicates_skipped} duplicates · "
            f"Formed {len(result.clusters)} event clusters"
        )

        if result.clusters:
            tab1, tab2, tab3, tab4 = st.tabs([
                "Event Clusters", "Timeline", "Analytics", "Market Research",
            ])

            with tab1:
                for cluster in result.clusters:
                    render_cluster_card(cluster)

            with tab2:
                render_timeline(result.clusters)

            with tab3:
                c1, c2 = st.columns(2)
                event_counts = {}
                for c in result.clusters:
                    event_counts[c.event_type.value] = event_counts.get(c.event_type.value, 0) + 1
                with c1:
                    fig = px.pie(
                        names=list(event_counts.keys()),
                        values=list(event_counts.values()),
                        title="Event Type Distribution",
                        color_discrete_map=EVENT_COLORS,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    sent_data = [c.sentiment_consensus.value for c in result.clusters]
                    fig2 = px.histogram(
                        x=sent_data, title="Cluster Sentiment Consensus",
                        color_dispos_sequence=["#38a169", "#e53e3e", "#a0aec0"],
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                relevance_df = pd.DataFrame([
                    {
                        "Cluster": c.representative_title[:40],
                        "Relevance": c.market_relevance,
                        "Novelty": c.novelty_score,
                        "Contradiction": c.contradiction_score,
                    }
                    for c in result.clusters
                ])
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=relevance_df["Relevance"],
                    y=relevance_df["Novelty"],
                    mode="markers+text",
                    text=relevance_df["Cluster"],
                    textposition="top center",
                    marker=dict(
                        size=relevance_df["Contradiction"] * 30 + 10,
                        color=relevance_df["Contradiction"],
                        colorscale="Reds",
                        showscale=True,
                        colorbar_title="Contradiction",
                    ),
                ))
                fig3.update_layout(
                    title="Market Relevance vs Novelty (size = contradiction)",
                    xaxis_title="Market Relevance",
                    yaxis_title="Novelty Score",
                    height=500,
                )
                st.plotly_chart(fig3, use_container_width=True)

            with tab4:
                ticker = query.upper() if len(query) <= 5 else query
                for c in result.clusters:
                    if c.tickers:
                        ticker = c.tickers[0]
                        break
                render_research_tab(ticker, result.clusters)
        else:
            st.warning("No clusters formed. Try a broader query or increase article count.")
    else:
        store = get_store()
        historical = store.get_clusters(20)
        if historical:
            st.info("Showing stored event clusters from previous runs.")
            for row in historical:
                with st.expander(f"{row['representative_title'][:80]} ({row['event_type']})"):
                    st.json({
                        "articles": row["article_count"],
                        "sources": row["source_count"],
                        "relevance": row["market_relevance"],
                        "novelty": row["novelty_score"],
                        "summary": row.get("summary"),
                    })
        else:
            st.info("Enter a ticker or company in the sidebar and run the intelligence pipeline.")


if __name__ == "__main__":
    main()
