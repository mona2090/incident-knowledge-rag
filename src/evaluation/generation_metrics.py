def structural_metrics(result, expected_answerable, allowed_chunk_ids):
    claims = result.answer.claims
    references = [citation for claim in claims for citation in claim.citations]
    valid = sum(citation in result.sources and
                result.sources[citation].get("chunk_id") in allowed_chunk_ids
                for citation in references)
    # An error is not an abstention and cannot receive answerability credit.
    decision_correct = result.status != "error" and result.answer.answerable == expected_answerable
    return {"answerability_correct": int(decision_correct),
            "output_valid": int(result.status != "error"),
            "claim_citation_coverage": sum(bool(claim.citations) for claim in claims) / len(claims) if claims else None,
            "citation_id_validity": valid / len(references) if references else None}
