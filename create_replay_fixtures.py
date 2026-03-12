#!/usr/bin/env python3
"""
Create conversation fixtures from travel agent test data for use with replay_session_script.py
"""
import json
from pathlib import Path


def load_travel_data():
    """Load the travel agent test data."""
    with open("tests/system/test_data_travel_agent.json") as f:
        return json.load(f)


def create_fixture(messages, session_id, namespace="travel-agent", user_id=None):
    """Create a fixture in the format expected by replay_session_script.py"""
    return {
        "data": {"dataset_id": session_id},
        "namespace": namespace,
        "user_id": user_id,
        "messages": messages,
    }


def main():
    print("Loading travel agent test data...")
    data = load_travel_data()
    
    output_dir = Path("temp_fixtures")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Short conversation (10 messages)
    print("Creating short_weekend_trip.json...")
    short_fixture = create_fixture(
        messages=data["short_conversation"]["messages"],
        session_id="weekend-paris-replay",
        user_id="test-user-001",
    )
    with open(output_dir / "short_weekend_trip.json", "w") as f:
        json.dump(short_fixture, f, indent=2)
    
    # 2. Greece trip (for summarization)
    print("Creating greece_trip.json...")
    greece_fixture = create_fixture(
        messages=data["greece_trip"]["messages"],
        session_id="greece-anniversary-replay",
        user_id="test-user-002",
    )
    with open(output_dir / "greece_trip.json", "w") as f:
        json.dump(greece_fixture, f, indent=2)
    
    # 3. Returning client trips
    print("Creating returning client trip fixtures...")
    trips = data["returning_client_scenario"]["trips"]
    
    for trip in trips:
        trip_num = trip["trip_number"]
        session_id = trip["session_id"]
        messages = trip["sample_messages"]
        
        fixture = create_fixture(
            messages=messages,
            session_id=session_id,
            user_id="sarah-johnson-001",
            namespace="travel-agent",
        )
        
        filename = f"trip_{trip_num}_{session_id.split('-')[2]}.json"
        print(f"  - {filename}")
        with open(output_dir / filename, "w") as f:
            json.dump(fixture, f, indent=2)
    
    print("\n✅ Fixtures created in temp_fixtures/")
    print("\nAvailable fixtures:")
    for fixture_file in sorted(output_dir.glob("*.json")):
        print(f"  - {fixture_file.name}")
    
    print("\nExample usage:")
    print("  python3 replay_session_script.py \\")
    print("    temp_fixtures/short_weekend_trip.json \\")
    print("    --base-url http://localhost:8001 \\")
    print("    --reset-session \\")
    print("    --snapshot-file metrics/short_conversation.jsonl")


if __name__ == "__main__":
    main()

