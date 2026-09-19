"""Import every model so Base.metadata is complete for Alembic autogenerate."""

from app.models.document import Chunk, Document
from app.models.eval_test_suite import EvalTestSuite
from app.models.evaluation import Evaluation
from app.models.knowledge_base import KnowledgeBase
from app.models.query import Query
from app.models.retrieval_config import RetrievalConfig
from app.models.tenant import Tenant

__all__ = [
    "Chunk",
    "Document",
    "EvalTestSuite",
    "Evaluation",
    "KnowledgeBase",
    "Query",
    "RetrievalConfig",
    "Tenant",
]
