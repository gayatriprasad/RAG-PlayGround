"""LangGraph orchestration for multi-agent RAG pipeline."""

import logging
from typing import Dict, Any, Literal, List
from .state import RAGState
from .planner import QueryPlannerAgent
from .retriever import RetrievalAgent
from .synthesizer import SynthesisAgent
from .critic import CriticAgent
from ..config import Config
from ..classifiers import get_classifier

logger = logging.getLogger(__name__)

# Try to import LangGraph - graceful fallback if not installed
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("langgraph not installed - multi-agent mode unavailable")


def validate_rag_state(state: RAGState) -> Dict[str, Any]:
    """
    Validate RAGState invariants before critique — Skill 52(C).

    A malformed state (e.g. an empty draft answer, chunks missing content,
    an iteration counter that has run away) must not silently propagate
    through the graph and produce a nonsensical final answer. Detected
    violations are recorded in trace["validation_errors"] rather than
    raising — this is a diagnostic guard, not a hard failure, since the
    graph should still finalize with SOME answer rather than crash.

    Returns the partial state update (a `trace` dict merge) that a LangGraph
    node function is expected to return.
    """
    errors: List[str] = []

    draft = state.get("draft_answer")
    if not draft or not draft.strip():
        errors.append("draft_answer is empty or missing")

    chunks = state.get("retrieved_chunks") or []
    for i, rc in enumerate(chunks):
        if not getattr(rc.chunk, "content", "").strip():
            errors.append(f"retrieved_chunks[{i}] has empty content")
            break

    iteration = state.get("iteration", 0)
    if iteration > 10:
        errors.append(f"iteration counter runaway: {iteration}")

    if errors:
        logger.warning(f"⚠️  [Node: validate] State validation issues: {errors}")

    return {
        "trace": {
            **state.get("trace", {}),
            "validation_errors": errors,
            "validation_passed": len(errors) == 0,
        }
    }


def create_rag_graph(cfg: Config, index):
    """
    Create and compile the multi-agent RAG graph.
    
    Flow:
    1. classify → intent classification
    2. plan → query decomposition
    3. retrieve → execute retrieval plan
    4. synthesize → generate draft answer with citations
    5. critique → evaluate answer quality
    6. [conditional] revise (if confidence < 0.6) OR finalize
    
    Args:
        cfg: Configuration
        index: Index instance for retrieval
        
    Returns:
        Compiled LangGraph app
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError(
            "langgraph is required for multi-agent mode. "
            "Install with: pip install langgraph langchain-core"
        )
    
    # Initialize agents
    classifier = get_classifier(cfg.intent, cfg.llm)
    planner = QueryPlannerAgent(cfg)
    retriever = RetrievalAgent(cfg, index)
    synthesizer = SynthesisAgent(cfg)
    critic = CriticAgent(cfg)
    
    # Define node functions
    def classify_intent(state: RAGState) -> Dict[str, Any]:
        """Node: Classify query intent."""
        question = state["question"]
        logger.info(f"🎯 [Node: classify] Question: {question.text[:60]}...")
        
        intent_result = classifier.classify(question.text)
        
        return {
            "intent": intent_result,
            "trace": {
                **state.get("trace", {}),
                "intent_label": intent_result.label,
                "intent_confidence": intent_result.confidence,
                "intent_method": intent_result.method
            }
        }
    
    def plan_queries(state: RAGState) -> Dict[str, Any]:
        """Node: Generate retrieval plan."""
        logger.info("📋 [Node: plan] Generating retrieval plan...")
        return planner.plan(state)
    
    def retrieve_chunks(state: RAGState) -> Dict[str, Any]:
        """Node: Execute retrieval."""
        logger.info("🔍 [Node: retrieve] Executing retrieval...")
        return retriever.retrieve(state)
    
    def synthesize_answer(state: RAGState) -> Dict[str, Any]:
        """Node: Generate draft answer."""
        logger.info("✍️  [Node: synthesize] Generating answer...")
        return synthesizer.synthesize(state)
    
    def validate_state(state: RAGState) -> Dict[str, Any]:
        """Node: Validate RAGState invariants before critique — Skill 52(C)."""
        logger.info("🛡️  [Node: validate] Checking state invariants...")
        return validate_rag_state(state)
    
    def critique_answer(state: RAGState) -> Dict[str, Any]:
        """Node: Critique draft answer."""
        logger.info("🔍 [Node: critique] Evaluating answer...")
        return critic.critique(state)
    
    def finalize_answer(state: RAGState) -> Dict[str, Any]:
        """Node: Finalize the answer."""
        logger.info("✅ [Node: finalize] Finalizing answer...")
        
        draft = state.get("draft_answer", "")
        critique = state.get("critique", {})
        
        # If critique found issues, add a disclaimer
        if critique.get("unsupported_claims") or critique.get("errors"):
            disclaimer = "\n\n⚠️ Note: Some claims may need verification."
            final = draft + disclaimer
        else:
            final = draft
        
        return {
            "final_answer": final,
            "trace": {
                **state.get("trace", {}),
                "finalized": True,
                "total_iterations": state.get("iteration", 0)
            }
        }
    
    # Define conditional routing
    def should_revise(state: RAGState) -> Literal["retrieve", "finalize"]:
        """
        Decide whether to revise (re-retrieve) or finalize.
        
        Rules:
        - Max 2 iterations (revision loops)
        - If confidence < 0.6 and iterations < 2, revise
        - Otherwise, finalize
        """
        iteration = state.get("iteration", 0)
        critique = state.get("critique", {})
        confidence = critique.get("confidence", 1.0)
        
        if iteration >= 2:
            logger.info(f"📊 [Routing] Max iterations reached ({iteration}) → finalize")
            return "finalize"
        
        if confidence < 0.6:
            logger.info(f"📊 [Routing] Low confidence ({confidence:.2f}) → revise (iteration {iteration + 1})")
            return "retrieve"
        
        logger.info(f"📊 [Routing] Sufficient confidence ({confidence:.2f}) → finalize")
        return "finalize"
    
    # Build graph
    logger.info("🔨 Building LangGraph StateGraph...")
    graph = StateGraph(RAGState)
    
    # Add nodes
    graph.add_node("classify", classify_intent)
    graph.add_node("plan", plan_queries)
    graph.add_node("retrieve", retrieve_chunks)
    graph.add_node("synthesize", synthesize_answer)
    graph.add_node("validate", validate_state)
    graph.add_node("critique", critique_answer)
    graph.add_node("finalize", finalize_answer)
    
    # Define edges
    graph.set_entry_point("classify")
    graph.add_edge("classify", "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "validate")
    graph.add_edge("validate", "critique")
    
    # Conditional edge: critique → revise OR finalize
    graph.add_conditional_edges(
        "critique",
        should_revise,
        {
            "retrieve": "retrieve",  # Re-retrieve with refined query
            "finalize": "finalize"
        }
    )
    
    graph.add_edge("finalize", END)
    
    # Compile graph
    logger.info("✅ LangGraph compiled successfully")
    compiled_graph = graph.compile()
    
    return compiled_graph


def run_agentic_graph(cfg: Config, index, question) -> Dict[str, Any]:
    """
    Convenience wrapper to run the agentic graph on a single question.
    
    Args:
        cfg: Configuration
        index: Index instance
        question: Question object
        
    Returns:
        Final state after graph execution
    """
    graph = create_rag_graph(cfg, index)
    
    # Initialize state
    initial_state: RAGState = {
        "question": question,
        "intent": None,
        "retrieval_plan": [],
        "retrieved_chunks": [],
        "draft_answer": None,
        "citations": {},
        "critique": None,
        "final_answer": None,
        "trace": {},
        "iteration": 0
    }
    
    # Execute graph
    logger.info(f"🚀 Starting agentic graph execution for question: {question.id}")
    final_state = graph.invoke(initial_state)
    
    logger.info(f"✅ Graph execution complete. Final answer: {final_state.get('final_answer', 'N/A')[:100]}...")
    
    return final_state
