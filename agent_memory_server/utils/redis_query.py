from __future__ import annotations

from typing import Any

from redisvl.query import AggregationQuery, RangeQuery, VectorQuery

# Import constants from utils.recency module
from agent_memory_server.utils.recency import SECONDS_PER_DAY


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
        agg = cls(vq.query)
        if filter_expression is not None:
            agg.filter(filter_expression)
        return agg

    def load_default_fields(self) -> RecencyAggregationQuery:
        self.load(self.DEFAULT_RETURN_FIELDS)
        return self

    def apply_recency(
        self, *, now_ts: int, params: dict[str, Any] | None = None
    ) -> RecencyAggregationQuery:
        params = params or {}

        semantic_weight = float(params.get("semantic_weight", 0.8))
        recency_weight = float(params.get("recency_weight", 0.2))
        freshness_weight = float(params.get("freshness_weight", 0.6))
        novelty_weight = float(params.get("novelty_weight", 0.4))
        half_life_access = float(params.get("half_life_last_access_days", 7.0))
        half_life_created = float(params.get("half_life_created_days", 30.0))

        self.apply(
            days_since_access=f"max(0, ({now_ts} - @last_accessed)/{SECONDS_PER_DAY})"
        )
        self.apply(
            days_since_created=f"max(0, ({now_ts} - @created_at)/{SECONDS_PER_DAY})"
        )
        self.apply(freshness=f"pow(2, -@days_since_access/{half_life_access})")
        self.apply(novelty=f"pow(2, -@days_since_created/{half_life_created})")
        self.apply(recency=f"{freshness_weight}*@freshness+{novelty_weight}*@novelty")
        self.apply(sim="1-(@__vector_score/2)")
        self.apply(boosted_score=f"{semantic_weight}*@sim+{recency_weight}*@recency")

        return self

    def sort_by_boosted_desc(self) -> RecencyAggregationQuery:
        self.sort_by([("boosted_score", "DESC")])
        return self

    def paginate(self, offset: int, limit: int) -> RecencyAggregationQuery:
        self.limit(offset, limit)
        return self

    def build_args(self) -> list:
        """Build the query arguments for Redis search."""
        return super().build_args()
