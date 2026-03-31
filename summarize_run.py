import json
from pathlib import Path
from statistics import mean

# Change this if needed
JSON_PATH = Path("C:/Users/User/Desktop/wildlife_captures/SCP/freq_600/run_logs/99f946b37f6c_timed.json")


def safe_mean(values):
    return mean(values) if values else None


def fmt_ms(value):
    return f"{value:.2f} ms" if value is not None else "N/A"


def fmt_pct(value):
    return f"{value:.2f}%" if value is not None else "N/A"


def fmt_mb(value_bytes):
    return f"{value_bytes / (1024 * 1024):.2f} MB" if value_bytes is not None else "N/A"


def summarize_numeric(values):
    if not values:
        return {"avg": None, "min": None, "max": None}
    return {
        "avg": mean(values),
        "min": min(values),
        "max": max(values),
    }


def collect_snapshot_values(events, snapshot_key, cpu_key="cpu_percent_total"):
    cpu_vals = []
    ram_percent_vals = []
    ram_used_vals = []

    for event in events:
        system = event.get("system", {})
        snap = system.get(snapshot_key, {})
        memory = snap.get("memory", {})

        cpu = snap.get(cpu_key)
        ram_percent = memory.get("percent")
        ram_used = memory.get("used_bytes")

        if cpu is not None:
            cpu_vals.append(cpu)
        if ram_percent is not None:
            ram_percent_vals.append(ram_percent)
        if ram_used is not None:
            ram_used_vals.append(ram_used)

    return {
        "cpu_percent_total": summarize_numeric(cpu_vals),
        "ram_percent": summarize_numeric(ram_percent_vals),
        "ram_used_bytes": summarize_numeric(ram_used_vals),
    }


def collect_combined_snapshot_values(events, cpu_key="cpu_percent_total"):
    cpu_vals = []
    ram_percent_vals = []
    ram_used_vals = []

    for event in events:
        system = event.get("system", {})
        for snapshot_key in ("before", "after"):
            snap = system.get(snapshot_key, {})
            memory = snap.get("memory", {})

            cpu = snap.get(cpu_key)
            ram_percent = memory.get("percent")
            ram_used = memory.get("used_bytes")

            if cpu is not None:
                cpu_vals.append(cpu)
            if ram_percent is not None:
                ram_percent_vals.append(ram_percent)
            if ram_used is not None:
                ram_used_vals.append(ram_used)

    return {
        "cpu_percent_total": summarize_numeric(cpu_vals),
        "ram_percent": summarize_numeric(ram_percent_vals),
        "ram_used_bytes": summarize_numeric(ram_used_vals),
    }


def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = data.get("events", [])
    if not events:
        print("No events found in JSON.")
        return

    # 1 & 2. First run / Last run
    # Using event timestamps for the first/last pipeline run
    first_event = min(events, key=lambda e: e.get("timestamp", ""))
    last_event = max(events, key=lambda e: e.get("timestamp", ""))

    print("=" * 70)
    print("RUN SUMMARY")
    print("=" * 70)
    print(f"Run ID           : {data.get('run_id', 'N/A')}")
    print(f"Mode             : {data.get('mode', 'N/A')}")
    print(f"Started At       : {data.get('started_at', 'N/A')}")
    print(f"Completed At     : {data.get('completed_at', 'N/A')}")
    print(f"Event Count      : {data.get('event_count', len(events))}")
    print()

    print("1. First run")
    print(f"   Event ID      : {first_event.get('event_id', 'N/A')}")
    print(f"   Timestamp     : {first_event.get('timestamp', 'N/A')}")
    print()

    print("2. Last run")
    print(f"   Event ID      : {last_event.get('event_id', 'N/A')}")
    print(f"   Timestamp     : {last_event.get('timestamp', 'N/A')}")
    print()

    # 3. Average latency of each pipeline run/stage
    latency_keys = set()
    for event in events:
        latency_keys.update(event.get("latency_ms", {}).keys())

    latency_summary = {}
    for key in sorted(latency_keys):
        vals = [
            event.get("latency_ms", {}).get(key)
            for event in events
            if event.get("latency_ms", {}).get(key) is not None
        ]
        latency_summary[key] = summarize_numeric(vals)

    print("3. Average latency of each pipeline stage")
    for key, stats in latency_summary.items():
        print(
            f"   {key:15} avg={fmt_ms(stats['avg'])} | "
            f"min={fmt_ms(stats['min'])} | max={fmt_ms(stats['max'])}"
        )
    print()

    # 4 & 5. CPU / RAM resource summary
    before_stats = collect_snapshot_values(events, "before")
    after_stats = collect_snapshot_values(events, "after")
    combined_stats = collect_combined_snapshot_values(events)

    print("4. Average CPU, RAM resource")
    print("   Combined (before + after snapshots)")
    print(f"   CPU total      : {fmt_pct(combined_stats['cpu_percent_total']['avg'])}")
    print(f"   RAM percent    : {fmt_pct(combined_stats['ram_percent']['avg'])}")
    print(f"   RAM used       : {fmt_mb(combined_stats['ram_used_bytes']['avg'])}")
    print()

    print("5. Min and max CPU, RAM resource")
    print("   Combined (before + after snapshots)")
    print(
        f"   CPU total      : min={fmt_pct(combined_stats['cpu_percent_total']['min'])} | "
        f"max={fmt_pct(combined_stats['cpu_percent_total']['max'])}"
    )
    print(
        f"   RAM percent    : min={fmt_pct(combined_stats['ram_percent']['min'])} | "
        f"max={fmt_pct(combined_stats['ram_percent']['max'])}"
    )
    print(
        f"   RAM used       : min={fmt_mb(combined_stats['ram_used_bytes']['min'])} | "
        f"max={fmt_mb(combined_stats['ram_used_bytes']['max'])}"
    )
    print()

    # Optional: show before vs after separately
    print("=" * 70)
    print("OPTIONAL BREAKDOWN: BEFORE SNAPSHOT")
    print("=" * 70)
    print(
        f"CPU total        : avg={fmt_pct(before_stats['cpu_percent_total']['avg'])} | "
        f"min={fmt_pct(before_stats['cpu_percent_total']['min'])} | "
        f"max={fmt_pct(before_stats['cpu_percent_total']['max'])}"
    )
    print(
        f"RAM percent      : avg={fmt_pct(before_stats['ram_percent']['avg'])} | "
        f"min={fmt_pct(before_stats['ram_percent']['min'])} | "
        f"max={fmt_pct(before_stats['ram_percent']['max'])}"
    )
    print(
        f"RAM used         : avg={fmt_mb(before_stats['ram_used_bytes']['avg'])} | "
        f"min={fmt_mb(before_stats['ram_used_bytes']['min'])} | "
        f"max={fmt_mb(before_stats['ram_used_bytes']['max'])}"
    )
    print()

    print("=" * 70)
    print("OPTIONAL BREAKDOWN: AFTER SNAPSHOT")
    print("=" * 70)
    print(
        f"CPU total        : avg={fmt_pct(after_stats['cpu_percent_total']['avg'])} | "
        f"min={fmt_pct(after_stats['cpu_percent_total']['min'])} | "
        f"max={fmt_pct(after_stats['cpu_percent_total']['max'])}"
    )
    print(
        f"RAM percent      : avg={fmt_pct(after_stats['ram_percent']['avg'])} | "
        f"min={fmt_pct(after_stats['ram_percent']['min'])} | "
        f"max={fmt_pct(after_stats['ram_percent']['max'])}"
    )
    print(
        f"RAM used         : avg={fmt_mb(after_stats['ram_used_bytes']['avg'])} | "
        f"min={fmt_mb(after_stats['ram_used_bytes']['min'])} | "
        f"max={fmt_mb(after_stats['ram_used_bytes']['max'])}"
    )


if __name__ == "__main__":
    main()