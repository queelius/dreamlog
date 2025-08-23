"""
DreamLog Agent-Based API Server

A persistent, learning agent that provides RESTful API access to DreamLog.
Features:
- Persistent agent with state management
- Dream cycles for learning and optimization
- RAG-based example retrieval
- User feedback collection
- Active learning
- WebSocket support for real-time interaction

Usage:
    python dreamlog_agent_server.py --config config.yaml --port 8000
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import uuid
import time
from pathlib import Path
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dreamlog import DreamLogEngine
from dreamlog.extended_config import ExtendedDreamLogConfig, get_extended_config
from dreamlog.rag_framework import RAGDatabase, MetaLearningTracker
from dreamlog.embedding_providers import create_embedding_provider


# ===== Agent State Management =====

class AgentState(str, Enum):
    """Agent lifecycle states"""
    AWAKE = "awake"          # Active, processing queries
    DREAMING = "dreaming"    # Running dream cycle
    IDLE = "idle"           # Waiting for activity
    SLEEPING = "sleeping"    # Deep sleep, minimal resource usage
    LEARNING = "learning"    # Active learning mode


class DreamLogAgent:
    """
    Persistent DreamLog agent with learning capabilities.
    This is the core agent that maintains state across requests.
    """
    
    def __init__(self, config: ExtendedDreamLogConfig):
        self.config = config
        self.state = AgentState.AWAKE
        self.engine = DreamLogEngine()
        
        # Initialize RAG systems if enabled
        self.example_rag = None
        self.template_rag = None
        self.negative_rag = None
        self.meta_tracker = None
        
        if config.rag.examples.enabled:
            embed_provider = create_embedding_provider(
                config.rag.embedding.provider,
                **config.get_embedding_config()
            )
            self.example_rag = RAGDatabase(
                embed_provider,
                Path(config.rag.examples.db_path).expanduser()
            )
            
        if config.rag.templates.enabled:
            self.template_rag = RAGDatabase(
                embed_provider,
                Path(config.rag.templates.db_path).expanduser()
            )
            
        if config.rag.negative_examples.enabled:
            self.negative_rag = RAGDatabase(
                embed_provider,
                Path(config.rag.negative_examples.db_path).expanduser()
            )
        
        # Meta-learning tracker
        self.meta_tracker = MetaLearningTracker(
            Path(config.rag.meta_tracking_path).expanduser()
        )
        
        # Activity tracking
        self.activity_log = []
        self.last_activity = datetime.now()
        self.session_id = str(uuid.uuid4())
        self.stats = {
            "queries_processed": 0,
            "dreams_completed": 0,
            "examples_learned": 0,
            "user_feedback_received": 0
        }
        
        # Dream cycle task
        self.dream_task = None
        self.dream_lock = asyncio.Lock()
        
        # User feedback queue
        self.feedback_queue = []
        
    async def process_query(self, query: str, collect_feedback: bool = False) -> Dict[str, Any]:
        """
        Process a query with optional feedback collection.
        """
        self.last_activity = datetime.now()
        self.state = AgentState.AWAKE
        
        # Log activity for later analysis
        activity_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "session_id": self.session_id
        }
        
        # Retrieve relevant examples if RAG enabled
        examples = []
        if self.example_rag:
            examples = self.example_rag.retrieve_weighted(
                query,
                k=self.config.rag.examples.retrieval_k,
                temperature=self.config.rag.examples.retrieval_temperature
            )
            activity_entry["examples_used"] = [ex.id for ex in examples]
        
        # Execute query
        start_time = time.time()
        solutions = list(self.engine.query_from_string(query))
        query_time = time.time() - start_time
        
        activity_entry["solutions"] = solutions
        activity_entry["query_time"] = query_time
        activity_entry["success"] = len(solutions) > 0
        
        # Add to activity log
        self.activity_log.append(activity_entry)
        self.stats["queries_processed"] += 1
        
        # Update RAG statistics if examples were used
        if self.example_rag and examples:
            # This will be evaluated during dream cycle
            activity_entry["pending_evaluation"] = True
        
        result = {
            "query": query,
            "solutions": solutions,
            "count": len(solutions),
            "time_ms": query_time * 1000,
            "examples_used": len(examples),
            "collect_feedback": collect_feedback
        }
        
        if collect_feedback:
            result["feedback_id"] = activity_entry.get("feedback_id", str(uuid.uuid4()))
            activity_entry["feedback_id"] = result["feedback_id"]
        
        return result
    
    async def submit_feedback(self, feedback_id: str, rating: str, explanation: Optional[str] = None):
        """
        Submit user feedback for a query.
        
        Args:
            feedback_id: ID from the query result
            rating: "good", "bad", or "unsure"
            explanation: Optional explanation
        """
        # Find the corresponding activity entry
        for entry in self.activity_log:
            if entry.get("feedback_id") == feedback_id:
                entry["user_feedback"] = rating
                entry["user_explanation"] = explanation
                self.stats["user_feedback_received"] += 1
                
                # Immediately update RAG if feedback is definitive
                if rating in ["good", "bad"] and self.example_rag:
                    success = rating == "good"
                    for example_id in entry.get("examples_used", []):
                        self.example_rag.update_item_stats(example_id, success)
                
                return {"status": "feedback_recorded", "rating": rating}
        
        raise ValueError(f"Feedback ID {feedback_id} not found")
    
    async def start_dream_cycle(self):
        """
        Start a dream cycle for learning and optimization.
        """
        async with self.dream_lock:
            if self.state == AgentState.DREAMING:
                return {"status": "already_dreaming"}
            
            self.state = AgentState.DREAMING
            dream_start = datetime.now()
            
            try:
                # Evaluate recent queries
                await self._evaluate_activities()
                
                # Update RAG databases
                await self._update_rag_systems()
                
                # Compress knowledge if enabled
                if self.config.dream_cycle.compression_enabled:
                    await self._compress_knowledge()
                
                # Meta-learning analysis
                await self._analyze_meta_patterns()
                
                self.stats["dreams_completed"] += 1
                
                return {
                    "status": "dream_complete",
                    "duration_seconds": (datetime.now() - dream_start).total_seconds(),
                    "activities_evaluated": len([a for a in self.activity_log if a.get("evaluated")])
                }
                
            finally:
                self.state = AgentState.AWAKE
    
    async def _evaluate_activities(self):
        """Evaluate activities using configured methods."""
        for activity in self.activity_log:
            if activity.get("evaluated") or not activity.get("pending_evaluation"):
                continue
            
            # Use LLM judge if configured
            if self.config.dream_cycle.llm_judge_enabled:
                # TODO: Implement LLM evaluation
                pass
            
            # Use user feedback if available
            if activity.get("user_feedback"):
                activity["success_score"] = 1.0 if activity["user_feedback"] == "good" else 0.0
            else:
                # Automatic evaluation
                activity["success_score"] = 1.0 if activity.get("success") else 0.5
            
            activity["evaluated"] = True
    
    async def _update_rag_systems(self):
        """Update RAG databases based on evaluations."""
        for activity in self.activity_log:
            if not activity.get("evaluated"):
                continue
            
            success = activity.get("success_score", 0.5) > self.config.dream_cycle.success_threshold
            
            # Update example statistics
            if self.example_rag and "examples_used" in activity:
                for example_id in activity["examples_used"]:
                    self.example_rag.update_item_stats(example_id, success)
            
            # Add successful patterns as new examples
            if success and activity["success_score"] > self.config.dream_cycle.high_confidence_threshold:
                if self.example_rag:
                    self.example_rag.add_item(
                        content={"query": activity["query"], "solutions": activity["solutions"]},
                        text_for_embedding=activity["query"],
                        metadata={"learned_at": datetime.now().isoformat()},
                        source="learned"
                    )
                    self.stats["examples_learned"] += 1
    
    async def _compress_knowledge(self):
        """Compress and optimize knowledge base."""
        # TODO: Implement knowledge compression
        pass
    
    async def _analyze_meta_patterns(self):
        """Analyze patterns for meta-learning."""
        for activity in self.activity_log:
            if activity.get("evaluated"):
                self.meta_tracker.record_usage(
                    query=activity["query"],
                    query_type=self._classify_query(activity["query"]),
                    selected_items=activity.get("examples_used", []),
                    success=activity.get("success_score", 0.5) > 0.7,
                    context={"session_id": self.session_id}
                )
    
    def _classify_query(self, query: str) -> str:
        """Classify query type for meta-learning."""
        # Simple classification based on keywords
        if "grandparent" in query or "ancestor" in query:
            return "transitive"
        elif "sibling" in query or "cousin" in query:
            return "relational"
        elif any(kw in query for kw in ["recursive", "descendant"]):
            return "recursive"
        else:
            return "general"
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        idle_seconds = (datetime.now() - self.last_activity).total_seconds()
        
        return {
            "state": self.state.value,
            "session_id": self.session_id,
            "idle_seconds": idle_seconds,
            "stats": self.stats,
            "kb_size": {
                "facts": len(self.engine.kb.facts),
                "rules": len(self.engine.kb.rules)
            },
            "rag_status": {
                "examples": self.example_rag.get_statistics() if self.example_rag else None,
                "templates": self.template_rag.get_statistics() if self.template_rag else None,
            },
            "config": {
                "dream_cycle_enabled": self.config.dream_cycle.enabled,
                "user_feedback_enabled": self.config.user_feedback.collection_enabled,
                "active_learning_enabled": self.config.user_feedback.active_learning_enabled
            }
        }
    
    async def sleep(self):
        """Put agent into sleep mode."""
        self.state = AgentState.SLEEPING
        # Could persist state to disk here
        return {"status": "sleeping"}
    
    async def wake(self):
        """Wake agent from sleep."""
        self.state = AgentState.AWAKE
        self.last_activity = datetime.now()
        return {"status": "awake"}


# ===== API Models =====

class QueryRequest(BaseModel):
    query: str = Field(..., description="Query in S-expression format")
    collect_feedback: bool = Field(False, description="Request feedback collection")
    timeout: int = Field(30, description="Query timeout in seconds")


class QueryResponse(BaseModel):
    query: str
    solutions: List[Dict[str, Any]]
    count: int
    time_ms: float
    examples_used: int
    feedback_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    feedback_id: str
    rating: Literal["good", "bad", "unsure"]
    explanation: Optional[str] = None


class DreamRequest(BaseModel):
    max_duration: int = Field(60, description="Maximum dream duration in seconds")
    force: bool = Field(False, description="Force dream even if recently completed")


# ===== FastAPI Application =====

app = FastAPI(
    title="DreamLog Agent API",
    description="Agent-based API for DreamLog with learning capabilities",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent: Optional[DreamLogAgent] = None
background_scheduler = None


# ===== Dependency Injection =====

async def get_agent() -> DreamLogAgent:
    """Dependency to get the agent instance."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return agent


# ===== API Endpoints =====

@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup."""
    global agent, background_scheduler
    
    # Load configuration
    config_path = os.getenv("DREAMLOG_CONFIG")
    config = ExtendedDreamLogConfig.load(config_path)
    
    # Validate configuration
    issues = config.validate()
    if issues:
        print("Configuration warnings:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Create agent
    agent = DreamLogAgent(config)
    print(f"Agent initialized with session ID: {agent.session_id}")
    
    # Start background tasks
    if config.dream_cycle.enabled:
        background_scheduler = asyncio.create_task(dream_scheduler(agent, config))


@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown of the agent."""
    if agent:
        await agent.sleep()
    
    if background_scheduler:
        background_scheduler.cancel()


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "DreamLog Agent API",
        "version": "2.0.0",
        "endpoints": {
            "agent": "/agent/status",
            "query": "/query",
            "feedback": "/feedback",
            "dream": "/agent/dream",
            "knowledge": "/kb",
            "learning": "/learning"
        }
    }


@app.get("/agent/status")
async def agent_status(agent: DreamLogAgent = Depends(get_agent)):
    """Get agent status and statistics."""
    return await agent.get_status()


@app.post("/agent/sleep")
async def agent_sleep(agent: DreamLogAgent = Depends(get_agent)):
    """Put agent to sleep."""
    return await agent.sleep()


@app.post("/agent/wake")
async def agent_wake(agent: DreamLogAgent = Depends(get_agent)):
    """Wake agent from sleep."""
    return await agent.wake()


@app.post("/agent/dream")
async def trigger_dream(
    request: DreamRequest,
    background_tasks: BackgroundTasks,
    agent: DreamLogAgent = Depends(get_agent)
):
    """Trigger a dream cycle."""
    if agent.state == AgentState.DREAMING:
        return {"status": "already_dreaming"}
    
    # Run dream in background
    background_tasks.add_task(agent.start_dream_cycle)
    return {"status": "dream_started"}


@app.post("/query", response_model=QueryResponse)
async def execute_query(
    request: QueryRequest,
    agent: DreamLogAgent = Depends(get_agent)
):
    """Execute a query."""
    try:
        result = await agent.process_query(
            request.query,
            request.collect_feedback
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    agent: DreamLogAgent = Depends(get_agent)
):
    """Submit feedback for a query."""
    try:
        return await agent.submit_feedback(
            request.feedback_id,
            request.rating,
            request.explanation
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/learning/stats")
async def learning_statistics(agent: DreamLogAgent = Depends(get_agent)):
    """Get learning statistics."""
    meta_insights = agent.meta_tracker.analyze_patterns() if agent.meta_tracker else {}
    
    return {
        "total_queries": agent.stats["queries_processed"],
        "dreams_completed": agent.stats["dreams_completed"],
        "examples_learned": agent.stats["examples_learned"],
        "user_feedback_received": agent.stats["user_feedback_received"],
        "meta_insights": meta_insights
    }


@app.get("/learning/recommendations/{query_type}")
async def get_recommendations(
    query_type: str,
    agent: DreamLogAgent = Depends(get_agent)
):
    """Get recommendations for a query type."""
    if not agent.meta_tracker:
        return {"recommendations": []}
    
    recommendations = agent.meta_tracker.get_recommendations(query_type)
    return {"query_type": query_type, "recommendations": recommendations}


@app.websocket("/ws/interactive")
async def interactive_session(websocket: WebSocket):
    """WebSocket endpoint for interactive sessions with active learning."""
    await websocket.accept()
    
    if not agent:
        await websocket.send_json({"error": "Agent not initialized"})
        await websocket.close()
        return
    
    try:
        await websocket.send_json({
            "type": "welcome",
            "message": "DreamLog Interactive Session",
            "session_id": agent.session_id,
            "active_learning": agent.config.user_feedback.active_learning_enabled
        })
        
        while True:
            data = await websocket.receive_json()
            
            if data["type"] == "query":
                result = await agent.process_query(
                    data["query"],
                    collect_feedback=True
                )
                await websocket.send_json({
                    "type": "result",
                    **result
                })
                
                # Active learning - maybe ask for clarification
                if agent.config.user_feedback.active_learning_enabled:
                    # TODO: Implement active learning questions
                    pass
            
            elif data["type"] == "feedback":
                feedback_result = await agent.submit_feedback(
                    data["feedback_id"],
                    data["rating"],
                    data.get("explanation")
                )
                await websocket.send_json({
                    "type": "feedback_received",
                    **feedback_result
                })
            
            elif data["type"] == "status":
                status = await agent.get_status()
                await websocket.send_json({
                    "type": "status",
                    **status
                })
            
            elif data["type"] == "close":
                break
    
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()


# ===== Background Tasks =====

async def dream_scheduler(agent: DreamLogAgent, config: ExtendedDreamLogConfig):
    """Background task to schedule dream cycles."""
    while True:
        try:
            # Check if it's time to dream
            idle_time = (datetime.now() - agent.last_activity).total_seconds()
            
            should_dream = False
            
            if config.dream_cycle.schedule_mode == "periodic":
                # Dream every N minutes
                await asyncio.sleep(config.dream_cycle.period_minutes * 60)
                should_dream = True
                
            elif config.dream_cycle.schedule_mode == "idle":
                # Dream when idle
                if idle_time > config.dream_cycle.idle_threshold_seconds:
                    should_dream = True
                else:
                    await asyncio.sleep(30)  # Check every 30 seconds
            
            if should_dream and agent.state != AgentState.DREAMING:
                print(f"Starting scheduled dream cycle at {datetime.now()}")
                await agent.start_dream_cycle()
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in dream scheduler: {e}")
            await asyncio.sleep(60)  # Wait before retrying


# ===== CLI Entry Point =====

def main():
    parser = argparse.ArgumentParser(description="DreamLog Agent API Server")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    
    args = parser.parse_args()
    
    if args.config:
        os.environ["DREAMLOG_CONFIG"] = args.config
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()