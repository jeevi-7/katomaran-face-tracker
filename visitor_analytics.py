
import pandas as pd
import streamlit as st

from database import get_unique_visitor_count, get_visitor_events

st.set_page_config(
    page_title="Visitor Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Visitor Analytics")
st.caption("Summary of recorded visitor activity")

if st.button("Refresh Analytics"):
    st.rerun()

try:
    events = get_visitor_events()
    visitor_count = get_unique_visitor_count()

    columns = [
        "Event ID",
        "Visitor ID",
        "Event Type",
        "Event Time",
        "Image Path",
    ]

    df = pd.DataFrame(events, columns=columns)

    if not df.empty:
        df["Event Type"] = df["Event Type"].str.upper()
        df["Event Time"] = pd.to_datetime(df["Event Time"])

    # Summary metrics
    entries = int((df["Event Type"] == "ENTRY").sum()) if not df.empty else 0
    exits = int((df["Event Type"] == "EXIT").sum()) if not df.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registered Visitors", visitor_count)
    col2.metric("Total Events", len(df))
    col3.metric("Entries", entries)
    col4.metric("Exits", exits)

    if not df.empty:
        st.subheader("Entry and Exit Comparison")

        chart = pd.DataFrame({
            "Event Type": ["Entry", "Exit"],
            "Count": [entries, exits],
        }).set_index("Event Type")

        st.bar_chart(chart)

        st.subheader("Activity by Visitor")

        summary = (
            df.groupby(["Visitor ID", "Event Type"])
            .size()
            .unstack(fill_value=0)
        )

        for event_type in ["ENTRY", "EXIT"]:
            if event_type not in summary.columns:
                summary[event_type] = 0

        summary = summary[["ENTRY", "EXIT"]]
        summary["Total Events"] = summary["ENTRY"] + summary["EXIT"]

        st.dataframe(
            summary.reset_index(),
            use_container_width=True,
            hide_index=True,
        )

        latest = df.sort_values("Event Time", ascending=False).iloc[0]

        st.subheader("Latest Recorded Event")
        st.write(f"**Visitor:** {latest['Visitor ID']}")
        st.write(f"**Event:** {latest['Event Type']}")
        st.write(f"**Time:** {latest['Event Time']}")

        st.subheader("All Recorded Events")
        st.dataframe(
            df.sort_values("Event Time", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info("No visitor events have been recorded yet.")

except Exception as error:
    st.error(f"Unable to load analytics: {error}")
