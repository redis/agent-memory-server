from __future__ import annotations

from typing import Any

from redisvl.query import AggregationQuery, RangeQuery, VectorQuery


class RecencyAggregationQuery(AggregationQuery):
    """AggregationQuery helper for KNN + recency boosting with APPLY/SORTBY and paging.

    Usage:
      - Build a VectorQuery or RangeQuery (hybrid filter expression allowed)
      - Call RecencyAggregationQuery.from_vector_query(...)
      - Chain .load_default_fields().apply_recency(params).sort_by_boosted_desc().paginate(offset, limit)
    """

    DEFAULT_RETURN_FIELDS = [
        "id_",
        "session_id",
        "user_id",
        "namespace",
        "created_at",
        "last_accessed",
        "updated_at",
        "pinned",
        "access_count",
        "topics",
        "entities",
        "memory_hash",
        "discrete_memory_extracted",
        "memory_type",
        "persisted_at",
        "extracted_from",
        "event_date",
        "text",
        "__vector_score",
    ]

    @classmethod
    def from_vector_query(
        cls,
        vq: VectorQuery | RangeQuery,
        *,
        filter_expression: Any | None = None,
    ) -> RecencyAggregationQuery:
        return cls(vq.query, filter_expression=filter_expression)

    def load_default_fields(self) -> RecencyAggregationQuery:
        self.load(self.DEFAULT_RETURN_FIELDS)
        return self

    def apply_recency(
        self, *, now_ts: int, params: dict[str, Any] | None = None
    ) -> RecencyAggregationQuery:
        params = params or {}
        w_sem = float(params.get("w_sem", 0.8))
        w_rec = float(params.get("w_recency", 0.2))
        wf = float(params.get("wf", 0.6))
        wa = float(params.get("wa", 0.4))
        hl_la = float(params.get("half_life_last_access_days", 7.0))
        hl_cr = float(params.get("half_life_created_days", 30.0))

        self.apply(
            f"max(0, ({now_ts} - @last_accessed)/86400.0)", AS="days_since_access"
        ).apply(
            f"max(0, ({now_ts} - @created_at)/86400.0)", AS="days_since_created"
        ).apply(f"pow(2, -@days_since_access/{hl_la})", AS="freshness").apply(
            f"pow(2, -@days_since_created/{hl_cr})", AS="novelty"
        ).apply(f"{wf}*@freshness+{wa}*@novelty", AS="recency").apply(
            "1-(@__vector_score/2)", AS="sim"
        ).apply(f"{w_sem}*@sim+{w_rec}*@recency", AS="boosted_score")

        return self

    def sort_by_boosted_desc(self) -> RecencyAggregationQuery:
        self.sort_by([("boosted_score", "DESC")])
        return self

    def paginate(self, offset: int, limit: int) -> RecencyAggregationQuery:
        self.limit(offset, limit)
        return self
