
from database import (
    get_unique_visitor_count,
    get_visitor_events,
)

def main():
    print("\n========== VISITOR REPORT ==========")

    total_visitors = get_unique_visitor_count()
    events = get_visitor_events()

    print(f"Total registered visitors: {total_visitors}")
    print(f"Total recorded events: {len(events)}")

    print("\n--- Recent Visitor Activity ---")

    if not events:
        print("No events recorded yet.")
        return

    for event in events:
        event_id, visitor_id, event_type, event_time, image_path = event

        print(f"\nEvent ID   : {event_id}")
        print(f"Visitor ID : {visitor_id}")
        print(f"Event Type : {event_type.upper()}")
        print(f"Time       : {event_time}")
        print(f"Image Path : {image_path}")

    print("\n====================================")

if __name__ == "__main__":
    main()
