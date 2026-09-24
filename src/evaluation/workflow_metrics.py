import math
from statistics import mean


def average(values):
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def percentile(values, quantile):
    """Nearest-rank percentile; intended for reporting, not tiny-sample SLO claims."""
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(quantile * len(values)) - 1)]


def abstention_metrics(rows):
    negative = [row for row in rows if not row["expected_answerable"]]
    abstained = [row for row in rows if row.get("status") == "abstained"]
    correct = sum(not row["expected_answerable"] for row in abstained)
    return {"abstention_precision": correct / len(abstained) if abstained else None,
            "abstention_recall": correct / len(negative) if negative else None}
