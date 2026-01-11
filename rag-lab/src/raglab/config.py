from __future__ import annotations
from pydantic import BaseModel
from typing import List, Dict, Any

class ChunkCfg(BaseModel):
    strategy: str = "fixed"
    chunk_tokens: int = 512
    overlap: int = 50

class RetrieveCfg(BaseModel):
    top_k: int = 5

class GoldenCfg(BaseModel):
    path: str

class ExperimentCfg(BaseModel):
    name: str
    corpus_glob: List[str]
    representations: List[str]

class Config(BaseModel):
    experiment: ExperimentCfg
    chunk: ChunkCfg
    retrieve: RetrieveCfg
    golden: GoldenCfg
