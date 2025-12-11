import re
import pandas as pd
from datetime import datetime
import dateparser

# Aggregation keywords mapped to pandas functions
AGGREGATIONS = {
    "average": "mean",
    "avg": "mean",
    "mean": "mean",
    "maximum": "max",
    "max": "max",
    "minimum": "min",
    "min": "min",
    "count": "count",
    "median": "median",
    "sum": "sum",
    "std": "std"
}

# Synonyms for bike queries
BIKE_KEYWORDS = {
    "available": ["available bikes", "bike availability", "free bikes", "top stations"],
    "empty": ["empty station", "no bikes", "stations with zero bikes"],
    "full": ["full station", "no docks", "full stations"],
    "status": ["status of", "station status"],
    "system": ["system status", "overall system", "status of system"]
}

def detect_query_type(query: str):
    """Determine the type of query based on keywords"""
    q = query.lower()
    for key, synonyms in BIKE_KEYWORDS.items():
        if any(kw in q for kw in synonyms):
            return key
    return "unknown"

def extract_station_id(query: str):
    """Extract numeric station ID from query"""
    station_id = ''.join(filter(str.isdigit, query))
    return int(station_id) if station_id else None

def extract_aggregation(query: str):
    """Detect aggregation function from query"""
    for word, func in AGGREGATIONS.items():
        if re.search(rf'\b{word}\b', query.lower()):
            return func
    return None

def extract_dates(query: str):
    """Parse dates from query using dateparser"""
    parsed = dateparser.parse(query, settings={'RELATIVE_BASE': datetime.now()})
    return parsed.date() if parsed else None

def process_query(query: str, df: pd.DataFrame):
    """Return structured info for filtering or aggregation"""
    query_type = detect_query_type(query)
    station_id = extract_station_id(query)
    aggregation = extract_aggregation(query)
    date_filter = extract_dates(query)

    return {
        "query_type": query_type,
        "station_id": station_id,
        "aggregation": aggregation,
        "date": date_filter
    }
