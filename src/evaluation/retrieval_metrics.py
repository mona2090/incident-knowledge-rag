import math


def unique_source_ids(hits):
    return list(dict.fromkeys(hit.document.metadata["source_id"] for hit in hits))


def ranking_metrics(predicted, relevance, k=2):
    """Document-level metrics, one-based ranks; duplicate source IDs count once."""
    if k < 1:
        raise ValueError("k must be positive")
    ranked = list(dict.fromkeys(predicted))[:k]
    relevant = {key for key, grade in relevance.items() if grade > 0}
    if not relevant:
        return {name: None for name in ["precision", "recall", "mrr", "ndcg"]}
    found = sum(item in relevant for item in ranked)
    dcg = sum((2 ** relevance.get(item, 0) - 1) / math.log2(rank + 1)
              for rank, item in enumerate(ranked, 1))
    ideal = sum((2 ** grade - 1) / math.log2(rank + 1)
                for rank, grade in enumerate(sorted(relevance.values(), reverse=True)[:k], 1))
    return {"precision": found / k, "recall": found / len(relevant),
            "mrr": next((1 / rank for rank, item in enumerate(ranked, 1) if item in relevant), 0.0),
            "ndcg": dcg / ideal if ideal else 0.0}


def filter_violations(hits, scope):
    return sum(not scope.allows(hit.document.metadata) for hit in hits)
