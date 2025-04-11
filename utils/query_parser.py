def parse_query(query: str) -> str:
    query = query.lower()
    if "understock" in query:
        return "understocked"
    elif "reorder" in query:
        return "reorder"
    elif "forecast" in query or "demand" in query:
        return "forecast"
    else:
        return "unknown"
