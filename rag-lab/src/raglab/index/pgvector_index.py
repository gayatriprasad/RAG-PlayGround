"""
pgvector Index — PostgreSQL with vector similarity search.

DSN from env PGVECTOR_DSN (postgresql://user:pass@host/db).
Uses cosine similarity via <=> operator.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from raglab.index.base import BaseIndex
from raglab.types import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


class PgVectorIndex(BaseIndex):
    """
    PostgreSQL + pgvector index.

    Requires:
      - PGVECTOR_DSN env var (e.g. postgresql://user:pass@localhost:5432/raglab)
      - pgvector extension installed on the database
    """

    def __init__(self, cfg, embed_cfg):
        self.cfg = cfg
        self.embed_cfg = embed_cfg
        self._embedder = None
        self._table = getattr(cfg, "pgvector_table", "chunks")

    def _get_embedder(self):
        if self._embedder is None:
            from raglab.utils.embedder import get_embedder
            self._embedder = get_embedder(self.embed_cfg)
        return self._embedder

    def _get_dsn(self) -> str:
        dsn = os.environ.get("PGVECTOR_DSN") or os.environ.get("DATABASE_URL")
        if not dsn:
            raise ValueError(
                "PGVECTOR_DSN or DATABASE_URL environment variable required"
            )
        return dsn

    def _get_connection(self):
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2-binary package required. Install with: pip install psycopg2-binary"
            )

        return psycopg2.connect(self._get_dsn())

    def build(self, chunks: List[Chunk]) -> None:
        """Create table with vector column and insert embeddings."""
        import psycopg2.extras

        embedder = self._get_embedder()

        # Get dimension
        sample_emb = embedder.encode(["test"])
        dim = len(sample_emb[0])

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

                # Create table
                table = self._table
                cur.execute(f"DROP TABLE IF EXISTS {table}")
                cur.execute(f"""
                    CREATE TABLE {table} (
                        id TEXT PRIMARY KEY,
                        content TEXT,
                        source_type TEXT,
                        doc_id TEXT,
                        chunk_index INTEGER,
                        metadata JSONB DEFAULT '{{}}',
                        embedding vector({dim})
                    )
                """)

                # Insert in batches
                batch_size = 200
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i : i + batch_size]
                    texts = [c.content for c in batch]
                    embeddings = embedder.encode(texts, show_progress_bar=False)

                    values = []
                    for chunk, emb in zip(batch, embeddings):
                        vec = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                        vec_str = "[" + ",".join(str(v) for v in vec) + "]"
                        values.append((
                            chunk.id,
                            chunk.content,
                            chunk.source_type,
                            chunk.doc_id,
                            chunk.chunk_index,
                            "{}",
                            vec_str,
                        ))

                    psycopg2.extras.execute_values(
                        cur,
                        f"""
                        INSERT INTO {table}
                            (id, content, source_type, doc_id, chunk_index, metadata, embedding)
                        VALUES %s
                        """,
                        values,
                        template="(%s, %s, %s, %s, %s, %s::jsonb, %s::vector)",
                    )
                    logger.info(f"Inserted batch {i // batch_size + 1} ({len(batch)} rows)")

                # Create IVFFlat index for faster retrieval
                cur.execute(f"""
                    CREATE INDEX ON {table}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)

            conn.commit()
            logger.info(f"pgvector table '{table}' built with {len(chunks)} rows (dim={dim})")
        finally:
            conn.close()

    def retrieve(
        self,
        query: str,
        top_k: int,
        experiment_name: str = "default",
        source_type: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Cosine similarity search via <=> operator."""
        embedder = self._get_embedder()
        query_emb = embedder.encode([query])[0]
        vec = query_emb.tolist() if hasattr(query_emb, "tolist") else list(query_emb)
        vec_str = "[" + ",".join(str(v) for v in vec) + "]"

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                table = self._table

                if source_type:
                    cur.execute(
                        f"""
                        SELECT id, content, source_type, doc_id, chunk_index,
                               1 - (embedding <=> %s::vector) AS score
                        FROM {table}
                        WHERE source_type = %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (vec_str, source_type, vec_str, top_k),
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT id, content, source_type, doc_id, chunk_index,
                               1 - (embedding <=> %s::vector) AS score
                        FROM {table}
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (vec_str, vec_str, top_k),
                    )

                rows = cur.fetchall()

            retrieved = []
            for row in rows:
                chunk = Chunk(
                    id=row[0],
                    content=row[1],
                    source_type=row[2],
                    doc_id=row[3],
                    chunk_index=row[4],
                )
                retrieved.append(RetrievedChunk(chunk=chunk, score=float(row[5])))

            return retrieved
        finally:
            conn.close()

    def is_built(self, experiment_name: str) -> bool:
        """Check if pgvector table exists and has rows."""
        try:
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                        (self._table,),
                    )
                    return cur.fetchone()[0]
            finally:
                conn.close()
        except Exception:
            return False
