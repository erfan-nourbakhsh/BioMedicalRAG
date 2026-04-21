def build_context_string(results, max_chars=1500):
    parts = []
    total_chars = 0

    for r in results:
        entry = f"[Passage {r['rank']} (source: {r['source']})]: {r['text']}"
        if total_chars + len(entry) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 50:
                parts.append(entry[:remaining] + "...")
            break
        parts.append(entry)
        total_chars += len(entry) + 1

    return "\n".join(parts)
