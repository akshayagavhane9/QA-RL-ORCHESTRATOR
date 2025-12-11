import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path
from typing import Optional

# ---------- CONFIG ----------
RESULTS_DIR = Path("results")
RL_FILE = "episode_metrics.csv"
BASELINE_FILE = "baseline_fixed_order_metrics.csv"


# ---------- HELPERS ----------
def load_csv(name: str) -> Optional[pd.DataFrame]:
    path = RESULTS_DIR / name
    if not path.exists():
        st.warning(f"File not found: {path}")
        return None
    df = pd.read_csv(path)
    if df.empty:
        st.warning(f"File is empty: {path}")
        return None
    return df


def get_reward_col(df: pd.DataFrame) -> str:
    """Try common reward column names."""
    for col in ["episode_reward", "reward", "total_reward"]:
        if col in df.columns:
            return col
    raise KeyError(f"No reward column found. Available columns: {list(df.columns)}")


def get_episode_col(df: pd.DataFrame) -> str:
    """Try common episode index column names."""
    for col in ["episode", "episode_index", "ep", "episode_idx"]:
        if col in df.columns:
            return col
    raise KeyError(f"No episode column found. Available columns: {list(df.columns)}")


def get_bugs_col(df: pd.DataFrame) -> str:
    for col in ["bugs_found", "num_bugs", "bugs"]:
        if col in df.columns:
            return col
    raise KeyError(f"No bugs column found. Available columns: {list(df.columns)}")


def get_strategy_col(df: pd.DataFrame) -> Optional[str]:
    for col in ["strategy_name", "strategy", "selected_strategy"]:
        if col in df.columns:
            return col
    return None


def pct_improvement(rl_value: float, base_value: Optional[float]) -> Optional[float]:
    if base_value is None:
        return None
    if base_value == 0:
        return None
    return (rl_value - base_value) / abs(base_value) * 100.0


# ---------- MAIN DASHBOARD ----------
def main():
    st.set_page_config(
        page_title="QA-RL Orchestrator Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ----- TITLE -----
    st.title("🧠 QA-RL Orchestrator Dashboard")
    st.caption("Reinforcement Learning for CI Test Prioritization")

    # Sidebar controls
    st.sidebar.header("⚙️ Display Settings")
    window = st.sidebar.slider(
        "Rolling window (episodes) for smoothing",
        min_value=1,
        max_value=50,
        value=20,
        step=1,
    )

    # ----- LOAD DATA -----
    rl_df = load_csv(RL_FILE)
    baseline_df = load_csv(BASELINE_FILE)

    if rl_df is None:
        st.error(
            "RL metrics not found. Run `python -m src.main` to generate "
            "`results/episode_metrics.csv`."
        )
        return

    # Detect columns
    episode_col_rl = get_episode_col(rl_df)
    reward_col_rl = get_reward_col(rl_df)
    bugs_col_rl = get_bugs_col(rl_df)
    strategy_col_rl = get_strategy_col(rl_df)

    if baseline_df is not None:
        episode_col_base = get_episode_col(baseline_df)
        reward_col_base = get_reward_col(baseline_df)
        bugs_col_base = get_bugs_col(baseline_df)
    else:
        episode_col_base = reward_col_base = bugs_col_base = None

    # ===== SUMMARY METRICS =====
    st.header("📊 Summary Metrics")

    avg_reward_rl = rl_df[reward_col_rl].mean()
    avg_bugs_rl = rl_df[bugs_col_rl].mean()

    if baseline_df is not None:
        avg_reward_base = baseline_df[reward_col_base].mean()
        avg_bugs_base = baseline_df[bugs_col_base].mean()
    else:
        avg_reward_base = avg_bugs_base = None

    reward_improvement = pct_improvement(avg_reward_rl, avg_reward_base)
    bugs_improvement = pct_improvement(avg_bugs_rl, avg_bugs_base)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("RL Agent")
        st.metric("Avg Reward (RL)", f"{avg_reward_rl:.2f}")
        st.metric("Avg Bugs Found (RL)", f"{avg_bugs_rl:.2f}")

    with col2:
        st.subheader("Baseline")
        if baseline_df is not None:
            st.metric("Avg Reward (Baseline)", f"{avg_reward_base:.2f}")
            st.metric("Avg Bugs Found (Baseline)", f"{avg_bugs_base:.2f}")
        else:
            st.info(
                "Baseline metrics not available.\n"
                "Run `python -m scripts.run_baseline` to generate baseline results."
            )

    with col3:
        st.subheader("Improvement")
        if baseline_df is not None and reward_improvement is not None:
            st.metric(
                "Reward Improvement",
                f"{reward_improvement:.1f} %",
                help="(AvgReward_RL - AvgReward_Baseline) / |Baseline| * 100",
            )
        if baseline_df is not None and bugs_improvement is not None:
            st.metric(
                "Bugs Improvement",
                f"{bugs_improvement:.1f} %",
                help="(AvgBugs_RL - AvgBugs_Baseline) / |Baseline| * 100",
            )
        if baseline_df is None:
            st.info("Improvement metrics require baseline results.")

    st.markdown("---")

    # ======================================================
    # 📈 RL LEARNING DYNAMICS – AREA + LINE, SIDE BY SIDE
    # ======================================================
    st.header("📈 RL Learning Dynamics")

    rl_df = rl_df.sort_values(by=episode_col_rl)
    rl_df["Episode"] = rl_df[episode_col_rl]

    reward_series = rl_df[reward_col_rl]
    reward_rolling = reward_series.rolling(window=window).mean()
    rl_df["Reward"] = reward_series
    rl_df[f"RewardRolling{window}"] = reward_rolling

    bugs_series = rl_df[bugs_col_rl]
    bugs_rolling = bugs_series.rolling(window=window).mean()
    rl_df["Bugs"] = bugs_series
    rl_df[f"BugsRolling{window}"] = bugs_rolling

    c1, c2 = st.columns(2)

    # Reward chart: area (raw) + line (rolling)
    with c1:
        st.subheader("Reward vs Episodes")
        base_reward = alt.Chart(rl_df).encode(
            x=alt.X("Episode:Q", title="Episode"),
        )

        reward_area = base_reward.mark_area(opacity=0.25).encode(
            y=alt.Y("Reward:Q", title="Reward"),
            tooltip=["Episode", "Reward"],
        )

        reward_line = base_reward.mark_line(strokeWidth=2).encode(
            y=f"RewardRolling{window}:Q",
            color=alt.value("#1f77b4"),
            tooltip=["Episode", alt.Tooltip(f"RewardRolling{window}:Q", title="Rolling Reward")],
        )

        st.altair_chart((reward_area + reward_line).properties(height=280), use_container_width=True)

    # Bugs chart: area (raw) + line (rolling)
    with c2:
        st.subheader("Bugs Found vs Episodes")
        base_bugs = alt.Chart(rl_df).encode(
            x=alt.X("Episode:Q", title="Episode"),
        )

        bugs_area = base_bugs.mark_area(opacity=0.25, color="#ff7f0e").encode(
            y=alt.Y("Bugs:Q", title="Bugs Found"),
            tooltip=["Episode", "Bugs"],
        )

        bugs_line = base_bugs.mark_line(strokeWidth=2, color="#d62728").encode(
            y=f"BugsRolling{window}:Q",
            tooltip=["Episode", alt.Tooltip(f"BugsRolling{window}:Q", title="Rolling Bugs")],
        )

        st.altair_chart((bugs_area + bugs_line).properties(height=280), use_container_width=True)

    # ======================================================
    # 🔥 RL vs BASELINE COMPARISON – BARS + DIFFERENCE AREA
    # ======================================================
    st.markdown("---")
    st.header("🔥 RL vs Baseline Comparison")

    if baseline_df is not None:
        baseline_df = baseline_df.sort_values(by=episode_col_base)
        baseline_df["Episode"] = baseline_df[episode_col_base]

        min_len = min(len(rl_df), len(baseline_df))
        rl_comp = rl_df.head(min_len).reset_index(drop=True)
        base_comp = baseline_df.head(min_len).reset_index(drop=True)

        # ---------- (A) Average metrics bar chart ----------
        st.subheader("Average Performance (Across All Episodes)")
        avg_df = pd.DataFrame(
            {
                "System": ["RL", "Baseline", "RL", "Baseline"],
                "Metric": ["Reward", "Reward", "Bugs", "Bugs"],
                "Value": [
                    avg_reward_rl,
                    avg_reward_base,
                    avg_bugs_rl,
                    avg_bugs_base,
                ],
            }
        )

        avg_chart = (
            alt.Chart(avg_df)
            .mark_bar()
            .encode(
                x=alt.X("Metric:N", title="Metric"),
                y=alt.Y("Value:Q", title="Average Value"),
                color=alt.Color("System:N", title="System"),
                column=alt.Column("Metric:N", header=alt.Header(labelOrient="bottom")),
                tooltip=["System", "Metric", "Value"],
            )
            .resolve_scale(y="independent")
            .properties(height=260)
        )

        st.altair_chart(avg_chart, use_container_width=True)

        # ---------- (B) Per-episode difference area chart ----------
        st.subheader("Per-Episode Advantage of RL (Reward Difference)")

        diff_df = pd.DataFrame(
            {
                "Episode": rl_comp["Episode"],
                "RewardDiff": rl_comp[reward_col_rl] - base_comp[reward_col_base],
            }
        )

        diff_chart = (
            alt.Chart(diff_df)
            .mark_area(opacity=0.5)
            .encode(
                x=alt.X("Episode:Q", title="Episode"),
                y=alt.Y("RewardDiff:Q", title="RL Reward − Baseline Reward"),
                color=alt.condition(
                    "datum.RewardDiff > 0",
                    alt.value("#2ca02c"),
                    alt.value("#d62728"),
                ),
                tooltip=["Episode", "RewardDiff"],
            )
            .properties(height=260)
        )

        st.altair_chart(diff_chart, use_container_width=True)

        if reward_improvement is not None and bugs_improvement is not None:
            st.info(
                f"On average, the RL agent improves **reward by "
                f"{reward_improvement:.1f}%** and **bugs found by "
                f"{bugs_improvement:.1f}%** compared to the baseline."
            )
    else:
        st.warning(
            "Baseline metrics not found. Run `python -m scripts.run_baseline` "
            "to enable RL vs Baseline comparison."
        )

    # ======================================================
    # ⭐ BEST EPISODES HIGHLIGHT
    # ======================================================
    st.markdown("---")
    st.header("⭐ Best Episodes (RL)")

    # Safely find best reward and best bugs episodes
    try:
        best_reward_idx = rl_df[reward_col_rl].idxmax()
        best_reward_row = rl_df.loc[best_reward_idx]
    except Exception:
        best_reward_row = None

    try:
        best_bugs_idx = rl_df[bugs_col_rl].idxmax()
        best_bugs_row = rl_df.loc[best_bugs_idx]
    except Exception:
        best_bugs_row = None

    col_best1, col_best2 = st.columns(2)

    with col_best1:
        st.subheader("🏆 Highest Reward Episode")
        if best_reward_row is not None:
            st.metric(
                "Episode",
                int(best_reward_row[episode_col_rl]),
            )
            st.metric(
                "Reward",
                f"{best_reward_row[reward_col_rl]:.2f}",
            )
            st.metric(
                "Bugs Found",
                int(best_reward_row[bugs_col_rl]),
            )
            # Show a small table with key context columns
            st.caption("Details")
            cols_to_show = [
                episode_col_rl,
                reward_col_rl,
                bugs_col_rl,
                "tests_run",
                "total_time" if "total_time" in rl_df.columns else bugs_col_rl,
            ]
            cols_to_show = [c for c in cols_to_show if c in rl_df.columns]
            st.table(best_reward_row[cols_to_show].to_frame().T)
        else:
            st.info("Could not compute best-reward episode.")

    with col_best2:
        st.subheader("🐛 Most Bugs Found Episode")
        if best_bugs_row is not None:
            st.metric(
                "Episode",
                int(best_bugs_row[episode_col_rl]),
            )
            st.metric(
                "Bugs Found",
                int(best_bugs_row[bugs_col_rl]),
            )
            st.metric(
                "Reward",
                f"{best_bugs_row[reward_col_rl]:.2f}",
            )
            st.caption("Details")
            cols_to_show = [
                episode_col_rl,
                reward_col_rl,
                bugs_col_rl,
                "tests_run",
                "total_time" if "total_time" in rl_df.columns else reward_col_rl,
            ]
            cols_to_show = [c for c in cols_to_show if c in rl_df.columns]
            st.table(best_bugs_row[cols_to_show].to_frame().T)
        else:
            st.info("Could not compute best-bugs episode.")

    # ======================================================
    # 🎯 STRATEGY USAGE – HORIZONTAL BAR + PERCENTAGES
    # ======================================================
    st.markdown("---")
    st.header("🎯 Strategy Usage (RL)")

    if strategy_col_rl is not None:
        strategy_counts = rl_df[strategy_col_rl].value_counts().reset_index()
        strategy_counts.columns = ["Strategy", "Episodes"]
        total_eps = strategy_counts["Episodes"].sum()
        strategy_counts["Percent"] = strategy_counts["Episodes"] / total_eps * 100.0

        strat_chart = (
            alt.Chart(strategy_counts)
            .mark_bar()
            .encode(
                y=alt.Y(
                    "Strategy:N",
                    sort="-x",
                    title="Strategy",
                ),
                x=alt.X("Episodes:Q", title="Number of Episodes"),
                tooltip=[
                    "Strategy",
                    "Episodes",
                    alt.Tooltip("Percent:Q", format=".1f", title="Percent of Episodes"),
                ],
            )
            .properties(height=260)
        )

        # Add text labels with percentages
        text_layer = (
            alt.Chart(strategy_counts)
            .mark_text(align="left", dx=3)
            .encode(
                y="Strategy:N",
                x="Episodes:Q",
                text=alt.Text("Percent:Q", format=".1f"),
            )
        )

        st.altair_chart(strat_chart + text_layer, use_container_width=True)
        st.caption(
            "Horizontal bars show how often each strategy was selected. "
            "Percent labels indicate share of total episodes."
        )
    else:
        st.info("No strategy column found in RL metrics; skipping strategy usage chart.")

    # ======================================================
    # 📁 RAW DATA PREVIEW
    # ======================================================
    st.markdown("---")
    st.header("📁 Raw Metrics Preview")

    if baseline_df is not None:
        tab1, tab2 = st.tabs(["RL Metrics", "Baseline Metrics"])
        with tab1:
            st.subheader("RL Episode Metrics")
            st.dataframe(rl_df.head(50))
        with tab2:
            st.subheader("Baseline Episode Metrics")
            st.dataframe(baseline_df.head(50))
    else:
        st.subheader("RL Episode Metrics")
        st.dataframe(rl_df.head(50))


if __name__ == "__main__":
    main()
