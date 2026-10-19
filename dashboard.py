from streamlit_autorefresh import st_autorefresh
import streamlit as st
import pandas as pd

from database import (
    get_unique_visitor_count,
    get_visitor_events,
)

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Katomaran Visitor Dashboard",
    page_icon="👤",
    layout="wide",
)

st.title("👤 Visitor Face Recognition Dashboard")
st.caption("AI-powered visitor entry and exit monitoring")
st_autorefresh(interval=5000, key="visitor_dashboard_refresh")

# --------------------------------------------------
# 2. REFRESH CONTROL
# --------------------------------------------------

if st.button("🔄 Refresh Data"):
    st.rerun()

# --------------------------------------------------
# 3. LOAD DATABASE RECORDS
# --------------------------------------------------

try:
    total_visitors = get_unique_visitor_count()
    events = get_visitor_events()

    # Count entry and exit events
    total_entries = sum(
        1 for event in events
        if event[2] == "entry"
    )

    total_exits = sum(
        1 for event in events
        if event[2] == "exit"
    )

    # --------------------------------------------------
    # 4. DASHBOARD METRICS
    # --------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Registered Visitors",
        total_visitors,
    )

    col2.metric(
        "Total Events",
        len(events),
    )

    col3.metric(
        "Entry Events",
        total_entries,
    )

    col4.metric(
        "Exit Events",
        total_exits,
    )

    # --------------------------------------------------
    # 5. ENTRY VS EXIT CHART
    # --------------------------------------------------

    st.subheader("📊 Entry vs Exit Overview")

    chart_data = pd.DataFrame({
        "Event Type": ["Entry", "Exit"],
        "Count": [total_entries, total_exits],
    })

    st.bar_chart(
        chart_data,
        x="Event Type",
        y="Count",
    )

    # --------------------------------------------------
    # 6. PREPARE EVENT DATA
    # --------------------------------------------------

    columns = [
        "Event ID",
        "Visitor ID",
        "Event Type",
        "Event Time",
        "Image Path",
    ]

    rows = []

    for event in events:
        event_id, visitor_id, event_type, event_time, image_path = event

        rows.append({
            "Event ID": event_id,
            "Visitor ID": visitor_id,
            "Event Type": event_type.upper(),
            "Event Time": event_time,
            "Image Path": image_path,
        })

    df = pd.DataFrame(rows, columns=columns)

    # --------------------------------------------------
    # 7. VISITOR FILTER
    # --------------------------------------------------

    st.subheader("🔎 Filter Visitor Activity")

    if not df.empty:
        visitor_ids = sorted(
            df["Visitor ID"].dropna().unique().tolist()
        )

        selected_visitor = st.selectbox(
            "Select Visitor",
            ["All Visitors"] + visitor_ids,
        )

        event_types = st.multiselect(
            "Filter Event Type",
            options=["ENTRY", "EXIT"],
            default=["ENTRY", "EXIT"],
        )

        filtered_df = df.copy()

        if selected_visitor != "All Visitors":
            filtered_df = filtered_df[
                filtered_df["Visitor ID"] == selected_visitor
            ]

        filtered_df = filtered_df[
            filtered_df["Event Type"].isin(event_types)
        ]

        # --------------------------------------------------
        # 8. RECENT ACTIVITY TABLE
        # --------------------------------------------------

        st.subheader("📋 Recent Visitor Activity")

        st.caption(
            f"Showing {len(filtered_df)} matching events"
        )

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
        )

        # --------------------------------------------------
        # 9. CSV DOWNLOAD
        # --------------------------------------------------

        csv_data = filtered_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            label="⬇️ Download Filtered CSV Report",
            data=csv_data,
            file_name="visitor_report.csv",
            mime="text/csv",
        )

    else:
        st.info("No visitor events recorded yet.")

except Exception as error:
    st.error("Unable to load visitor data.")
    st.exception(error)

    st.info(
        "Check that MySQL is running and the "
        "KATOMARAN_DB_PASSWORD environment variable "
        "is available in this terminal."
    )
