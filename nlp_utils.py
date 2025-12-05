import re
from datetime import datetime, timedelta
import spacy
import dateparser
import pandas as pd

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

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

def extract_aggregation(query: str):
    """Detect aggregation function from query"""
    for word, func in AGGREGATIONS.items():
        if re.search(rf'\b{word}\b', query.lower()):
            return func
    return None

def extract_dates(query: str):
    """Parse dates from query using dateparser"""
    parsed = dateparser.parse(query, settings={'RELATIVE_BASE': datetime.now()})
    if parsed:
        return parsed.date()
    return None

def extract_entities(query: str, possible_values):
    """Return list of entities mentioned in query"""
    found = [val for val in possible_values if re.search(rf'\b{re.escape(str(val).lower())}\b', query.lower())]
    return found

def extract_numeric_conditions(query: str):
    """Extract conditions like '>80' or '<50'"""
    matches = re.findall(r'(>=|<=|>|<|=)\s*(\d+(\.\d+)?)', query)
    return [(op, float(val)) for op, val, _ in matches]

def process_query(query: str, df: pd.DataFrame):
    """Process query and return structured filter/aggregation info"""
    aggregation = extract_aggregation(query)
    date_filter = extract_dates(query)
    
    # Detect sensor types, buildings, floors
    sensor_filter = extract_entities(query, df['sensor_type'].unique())
    building_filter = extract_entities(query, df['building'].unique())
    floor_filter = extract_entities(query, df['floor'].astype(str).unique())
    
    numeric_conditions = extract_numeric_conditions(query)

    return {
        "aggregation": aggregation,
        "date": date_filter,
        "sensor_types": sensor_filter,
        "buildings": building_filter,
        "floors": floor_filter,
        "numeric_conditions": numeric_conditions
    }
