
import csv
from pathlib import Path
from database import get_visitor_events

OUTPUT_FILE = Path(__file__).resolve().parent / "visitor_report.csv"

def main():
    events = get_visitor_events()

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Event ID",
            "Visitor ID",
            "Event Type",
            "Event Time",
            "Image Path",
        ])

        for event in events:
            writer.writerow(event)

    print(f"CSV report exported successfully: {OUTPUT_FILE}")
    print(f"Total records exported: {len(events)}")

if __name__ == "__main__":
    main()
