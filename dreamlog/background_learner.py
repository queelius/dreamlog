"""
Background Learning Service for DreamLog Persistent Learning

This module implements a long-running background service that manages persistent
learning, sleep cycles, and inter-process communication. It provides a daemon
process that continuously learns and improves the knowledge base.

Key Components:
- BackgroundLearner: Main service daemon
- IPCServer: Inter-process communication server
- LearningSession: Individual learning sessions
- QueryProcessor: Handles incoming queries with learning
- PerformanceMonitor: Tracks system performance and metrics
"""

import json
import logging
import multiprocessing
import socket
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any, Callable, Union
import signal
import sys

from .persistent_learning import PersistentKnowledgeBase, ConflictResolutionStrategy, UserTrustStrategy
from .knowledge_validator import KnowledgeValidator, ValidationReport
from .sleep_cycle import SleepCycleManager, SleepPhase, SleepCycleConfig
from .knowledge import KnowledgeBase, Fact, Rule
from .terms import Term
from .evaluator import PrologEvaluator, Solution
from .engine import DreamLogEngine
from .llm_hook import LLMHook
from .llm_providers import LLMProvider
from .factories import term_from_prefix

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Status of the background learning service"""
    STARTING = "starting"
    RUNNING = "running"
    SLEEPING = "sleeping"
    LEARNING = "learning"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class IPCMessageType(Enum):
    """Types of IPC messages"""
    QUERY = "query"
    ADD_FACTS = "add_facts"
    ADD_RULES = "add_rules"
    ADD_USER_KNOWLEDGE = "add_user_knowledge"
    GET_STATUS = "get_status"
    GET_METRICS = "get_metrics"
    FORCE_SLEEP = "force_sleep"
    SHUTDOWN = "shutdown"
    RESPONSE = "response"
    ERROR = "error"


@dataclass
class IPCMessage:
    """Inter-process communication message"""
    message_id: str
    message_type: IPCMessageType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps({
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'IPCMessage':
        """Deserialize from JSON"""
        data = json.loads(json_str)
        return cls(
            message_id=data["message_id"],
            message_type=IPCMessageType(data["message_type"]),
            data=data["data"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


@dataclass
class LearningSession:
    """Individual learning session tracking"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    queries_processed: int = 0
    facts_learned: int = 0
    rules_learned: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    validation_errors: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_active(self) -> bool:
        return self.end_time is None


class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.query_times: List[float] = []
        self.learning_events: List[Tuple[datetime, str, Dict[str, Any]]] = []
        self.error_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
    
    def record_query_time(self, time_ms: float) -> None:
        """Record query execution time"""
        with self._lock:
            self.query_times.append(time_ms)
            if len(self.query_times) > self.window_size:
                self.query_times.pop(0)
    
    def record_learning_event(self, event_type: str, metadata: Dict[str, Any]) -> None:
        """Record a learning event"""
        with self._lock:
            self.learning_events.append((datetime.now(), event_type, metadata))
            if len(self.learning_events) > self.window_size:
                self.learning_events.pop(0)
    
    def record_error(self, error_type: str) -> None:
        """Record an error occurrence"""
        with self._lock:
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        with self._lock:
            if not self.query_times:
                avg_query_time = 0
                max_query_time = 0
                min_query_time = 0
            else:
                avg_query_time = sum(self.query_times) / len(self.query_times)
                max_query_time = max(self.query_times)
                min_query_time = min(self.query_times)
            
            recent_events = len([e for e in self.learning_events 
                               if e[0] > datetime.now() - timedelta(hours=1)])
            
            return {
                "avg_query_time_ms": avg_query_time,
                "max_query_time_ms": max_query_time,
                "min_query_time_ms": min_query_time,
                "total_queries": len(self.query_times),
                "recent_events_1h": recent_events,
                "error_counts": self.error_counts.copy(),
                "total_errors": sum(self.error_counts.values())
            }


class IPCServer:
    """IPC server for communication with background learner"""
    
    def __init__(self, port: int, message_handler: Callable[[IPCMessage], IPCMessage]):
        self.port = port
        self.message_handler = message_handler
        self.socket: Optional[socket.socket] = None
        self.running = False
        self.server_thread: Optional[threading.Thread] = None
        self.client_handlers: List[threading.Thread] = []
    
    def start(self) -> None:
        """Start the IPC server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(('localhost', self.port))
            self.socket.listen(5)
            
            self.running = True
            self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
            self.server_thread.start()
            
            logger.info(f"IPC server started on port {self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start IPC server: {e}")
            raise
    
    def stop(self) -> None:
        """Stop the IPC server"""
        self.running = False
        
        if self.socket:
            self.socket.close()
        
        # Wait for client handlers to finish
        for handler in self.client_handlers:
            handler.join(timeout=2.0)
        
        if self.server_thread:
            self.server_thread.join(timeout=5.0)
        
        logger.info("IPC server stopped")
    
    def _server_loop(self) -> None:
        """Main server loop"""
        while self.running:
            try:
                if not self.socket:
                    break
                
                client_socket, address = self.socket.accept()
                logger.debug(f"Client connected from {address}")
                
                # Handle client in separate thread
                handler = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,),
                    daemon=True
                )
                handler.start()
                self.client_handlers.append(handler)
                
                # Clean up finished handlers
                self.client_handlers = [h for h in self.client_handlers if h.is_alive()]
                
            except OSError:
                if self.running:
                    logger.error("Socket error in server loop")
                break
            except Exception as e:
                logger.error(f"Error in server loop: {e}")
                if self.running:
                    time.sleep(1)
    
    def _handle_client(self, client_socket: socket.socket) -> None:
        """Handle individual client connection"""
        try:
            # Read message
            data = client_socket.recv(8192)
            if not data:
                return
            
            # Parse message
            message_json = data.decode('utf-8')
            request = IPCMessage.from_json(message_json)
            
            logger.debug(f"Received {request.message_type.value} message")
            
            # Process message
            response = self.message_handler(request)
            
            # Send response
            response_json = response.to_json()
            client_socket.sendall(response_json.encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Error handling client: {e}")
            
            # Send error response
            error_response = IPCMessage(
                message_id=str(uuid.uuid4()),
                message_type=IPCMessageType.ERROR,
                data={"error": str(e)}
            )
            
            try:
                client_socket.sendall(error_response.to_json().encode('utf-8'))
            except:
                pass
        
        finally:
            client_socket.close()


class BackgroundLearner:
    """
    Main background learning service
    
    Long-running daemon that manages persistent learning, sleep cycles,
    and provides IPC interface for external queries and knowledge injection.
    """
    
    def __init__(self, 
                 storage_path: Path,
                 llm_provider: Optional[LLMProvider] = None,
                 ipc_port: int = 7777,
                 config: Optional[Dict[str, Any]] = None):
        
        self.storage_path = Path(storage_path)
        self.llm_provider = llm_provider
        self.ipc_port = ipc_port
        self.config = config or {}
        
        # Core components
        self.persistent_kb: Optional[PersistentKnowledgeBase] = None
        self.sleep_manager: Optional[SleepCycleManager] = None
        self.validator = KnowledgeValidator()
        self.performance_monitor = PerformanceMonitor()
        
        # Service state
        self.status = ServiceStatus.STOPPED
        self.start_time: Optional[datetime] = None
        self.current_session: Optional[LearningSession] = None
        self.sessions: List[LearningSession] = []
        
        # IPC and threading
        self.ipc_server: Optional[IPCServer] = None
        self.main_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Signal handling
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def start(self) -> None:
        """Start the background learning service"""
        if self.running:
            logger.warning("Background learner already running")
            return
        
        try:
            self.status = ServiceStatus.STARTING
            self.start_time = datetime.now()
            
            logger.info("Starting background learning service...")
            
            # Initialize persistent knowledge base
            self.persistent_kb = PersistentKnowledgeBase(self.storage_path)
            logger.info(f"Loaded persistent KB: {self.persistent_kb.get_performance_metrics()}")
            
            # Initialize sleep cycle manager
            sleep_config = SleepCycleConfig()
            if "sleep_config" in self.config:
                # Update config with provided values
                for key, value in self.config["sleep_config"].items():
                    if hasattr(sleep_config, key):
                        setattr(sleep_config, key, value)
            
            self.sleep_manager = SleepCycleManager(self.persistent_kb, sleep_config)
            
            # Start IPC server
            self.ipc_server = IPCServer(self.ipc_port, self._handle_ipc_message)
            self.ipc_server.start()
            
            # Start sleep cycle manager
            self.sleep_manager.start_background_sleep()
            
            # Start main service loop
            self.running = True
            self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
            self.main_thread.start()
            
            self.status = ServiceStatus.RUNNING
            logger.info(f"Background learning service started on port {self.ipc_port}")
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            logger.error(f"Failed to start background learner: {e}")
            raise
    
    def stop(self) -> None:
        """Stop the background learning service"""
        logger.info("Stopping background learning service...")
        
        self.status = ServiceStatus.STOPPING
        self.running = False
        
        # Stop sleep manager
        if self.sleep_manager:
            self.sleep_manager.stop_background_sleep()
        
        # Stop IPC server
        if self.ipc_server:
            self.ipc_server.stop()
        
        # Finish current session
        if self.current_session and self.current_session.is_active:
            self.current_session.end_time = datetime.now()
        
        # Save final state
        if self.persistent_kb:
            self.persistent_kb.save()
        
        # Wait for main thread
        if self.main_thread:
            self.main_thread.join(timeout=5.0)
        
        self.status = ServiceStatus.STOPPED
        logger.info("Background learning service stopped")
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def _main_loop(self) -> None:
        """Main service loop"""
        while self.running:
            try:
                # Start new learning session if needed
                if not self.current_session or not self.current_session.is_active:
                    self._start_new_session()
                
                # Trigger activity in sleep manager
                if self.sleep_manager:
                    self.sleep_manager.trigger_activity()
                
                # Periodic maintenance
                if datetime.now().minute % 10 == 0:  # Every 10 minutes
                    self._perform_maintenance()
                
                # Sleep briefly
                time.sleep(60)  # Main loop runs every minute
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.performance_monitor.record_error("main_loop_error")
                time.sleep(60)
    
    def _start_new_session(self) -> None:
        """Start a new learning session"""
        if self.current_session and self.current_session.is_active:
            self.current_session.end_time = datetime.now()
            self.sessions.append(self.current_session)
        
        self.current_session = LearningSession(
            session_id=str(uuid.uuid4()),
            start_time=datetime.now()
        )
        
        logger.info(f"Started new learning session: {self.current_session.session_id}")
    
    def _perform_maintenance(self) -> None:
        """Perform periodic maintenance tasks"""
        try:
            # Clean up old sessions
            cutoff = datetime.now() - timedelta(days=7)
            self.sessions = [s for s in self.sessions if s.start_time > cutoff]
            
            # Save state
            if self.persistent_kb:
                self.persistent_kb.save()
            
            # Log metrics
            metrics = self.get_service_metrics()
            logger.info(f"Service metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Error in maintenance: {e}")
    
    def _handle_ipc_message(self, request: IPCMessage) -> IPCMessage:
        """Handle incoming IPC messages"""
        try:
            if request.message_type == IPCMessageType.QUERY:
                return self._handle_query(request)
            elif request.message_type == IPCMessageType.ADD_FACTS:
                return self._handle_add_facts(request)
            elif request.message_type == IPCMessageType.ADD_RULES:
                return self._handle_add_rules(request)
            elif request.message_type == IPCMessageType.ADD_USER_KNOWLEDGE:
                return self._handle_add_user_knowledge(request)
            elif request.message_type == IPCMessageType.GET_STATUS:
                return self._handle_get_status(request)
            elif request.message_type == IPCMessageType.GET_METRICS:
                return self._handle_get_metrics(request)
            elif request.message_type == IPCMessageType.FORCE_SLEEP:
                return self._handle_force_sleep(request)
            elif request.message_type == IPCMessageType.SHUTDOWN:
                return self._handle_shutdown(request)
            else:
                raise ValueError(f"Unknown message type: {request.message_type}")
                
        except Exception as e:
            logger.error(f"Error handling IPC message: {e}")
            return IPCMessage(
                message_id=str(uuid.uuid4()),
                message_type=IPCMessageType.ERROR,
                data={"error": str(e), "request_id": request.message_id}
            )
    
    def _handle_query(self, request: IPCMessage) -> IPCMessage:
        """Handle query requests"""
        start_time = time.time()
        
        try:
            # Parse query
            query_data = request.data.get("query", [])
            goals = [term_from_prefix(goal_data) for goal_data in query_data]
            
            # Execute query with learning
            solutions = self.persistent_kb.query_with_tracking(goals)
            
            # Convert solutions to serializable format
            solution_data = []
            for solution in solutions:
                bindings = {}
                for var_name, term in solution.get_ground_bindings().items():
                    bindings[var_name] = term.to_prefix()
                solution_data.append(bindings)
            
            # Update session metrics
            if self.current_session:
                self.current_session.queries_processed += 1
            
            # Record performance
            query_time = (time.time() - start_time) * 1000
            self.performance_monitor.record_query_time(query_time)
            
            return IPCMessage(
                message_id=str(uuid.uuid4()),
                message_type=IPCMessageType.RESPONSE,
                data={
                    "solutions": solution_data,
                    "solution_count": len(solutions),
                    "query_time_ms": query_time,
                    "request_id": request.message_id
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    def _handle_add_facts(self, request: IPCMessage) -> IPCMessage:
        """Handle add facts requests"""
        try:
            facts_data = request.data.get("facts", [])
            
            facts = []
            for fact_data in facts_data:
                fact = Fact.from_prefix(fact_data)
                facts.append(fact)
            
            # Add to learned knowledge
            self.persistent_kb.add_learned_knowledge(facts, [])
            
            # Update session metrics
            if self.current_session:
                self.current_session.facts_learned += len(facts)
            
            return IPCMessage(
                message_id=str(uuid.uuid4()),
                message_type=IPCMessageType.RESPONSE,
                data={
                    "facts_added": len(facts),
                    "request_id": request.message_id
                }
            )
            
        except Exception as e:
            logger.error(f"Error adding facts: {e}")
            raise
    
    def _handle_add_rules(self, request: IPCMessage) -> IPCMessage:
        """Handle add rules requests"""
        try:
            rules_data = request.data.get("rules", [])
            
            rules = []
            for rule_data in rules_data:
                rule = Rule.from_prefix(rule_data)
                rules.append(rule)
            
            # Add to learned knowledge
            self.persistent_kb.add_learned_knowledge([], rules)
            
            # Update session metrics
            if self.current_session:
                self.current_session.rules_learned += len(rules)
            
            return IPCMessage(
                message_id=str(uuid.uuid4()),
                message_type=IPCMessageType.RESPONSE,
                data={
                    "rules_added": len(rules),
                    "request_id": request.message_id
                }
            )
            
        except Exception as e:
            logger.error(f"Error adding rules: {e}")
            raise
    
    def _handle_add_user_knowledge(self, request: IPCMessage) -> IPCMessage:
        """Handle add user knowledge requests"""
        try:
            facts_data = request.data.get("facts", [])
            rules_data = request.data.get("rules", [])
            
            facts = [Fact.from_prefix(fact_data) for fact_data in facts_data]
            rules = [Rule.from_prefix(rule_data) for rule_data in rules_data]
            
            # Add to user knowledge and detect conflicts
            conflicts = self.persistent_kb.add_user_knowledge(facts, rules)
            
            # Resolve conflicts if any
            if conflicts:
                resolution_results = self.persistent_kb.resolve_conflicts()
                
                # Update session metrics
                if self.current_session:
                    self.current_session.conflicts_detected += len(conflicts)
                    self.current_session.conflicts_resolved += resolution_results["conflicts_resolved"]
            
            return IPCMessage(
                message_id=str(uuid.uuid4()),
                message_type=IPCMessageType.RESPONSE,
                data={
                    "facts_added": len(facts),
                    "rules_added": len(rules),
                    "conflicts_detected": len(conflicts),
                    "conflicts_resolved": len(conflicts),
                    "request_id": request.message_id
                }
            )
            
        except Exception as e:
            logger.error(f"Error adding user knowledge: {e}")
            raise
    
    def _handle_get_status(self, request: IPCMessage) -> IPCMessage:
        """Handle get status requests"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        status_data = {
            "status": self.status.value,
            "uptime_seconds": uptime,
            "current_session": asdict(self.current_session) if self.current_session else None,
            "total_sessions": len(self.sessions),
            "kb_metrics": self.persistent_kb.get_performance_metrics() if self.persistent_kb else {},
            "sleep_metrics": self.sleep_manager.get_sleep_metrics() if self.sleep_manager else {}
        }
        
        return IPCMessage(
            message_id=str(uuid.uuid4()),
            message_type=IPCMessageType.RESPONSE,
            data=status_data
        )
    
    def _handle_get_metrics(self, request: IPCMessage) -> IPCMessage:
        """Handle get metrics requests"""
        metrics = self.get_service_metrics()
        
        return IPCMessage(
            message_id=str(uuid.uuid4()),
            message_type=IPCMessageType.RESPONSE,
            data=metrics
        )
    
    def _handle_force_sleep(self, request: IPCMessage) -> IPCMessage:
        """Handle force sleep requests"""
        try:
            phase_name = request.data.get("phase", "LIGHT_SLEEP")
            phase = SleepPhase(phase_name.lower())
            
            if not self.sleep_manager:
                raise ValueError("Sleep manager not available")
            
            # Force sleep cycle
            report = self.sleep_manager.force_sleep_cycle(phase)
            
            return IPCMessage(
                message_id=str(uuid.uuid4()),
                message_type=IPCMessageType.RESPONSE,
                data={
                    "cycle_id": report.cycle_id,
                    "phase": report.phase.value,
                    "operations": report.operations,
                    "compression_ratio": report.compression_ratio,
                    "duration_ms": report.duration.total_seconds() * 1000,
                    "request_id": request.message_id
                }
            )
            
        except Exception as e:
            logger.error(f"Error forcing sleep cycle: {e}")
            raise
    
    def _handle_shutdown(self, request: IPCMessage) -> IPCMessage:
        """Handle shutdown requests"""
        # Schedule shutdown in separate thread to allow response
        def delayed_shutdown():
            time.sleep(1)
            self.stop()
        
        shutdown_thread = threading.Thread(target=delayed_shutdown, daemon=True)
        shutdown_thread.start()
        
        return IPCMessage(
            message_id=str(uuid.uuid4()),
            message_type=IPCMessageType.RESPONSE,
            data={
                "message": "Shutdown initiated",
                "request_id": request.message_id
            }
        )
    
    def get_service_metrics(self) -> Dict[str, Any]:
        """Get comprehensive service metrics"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        metrics = {
            "service": {
                "status": self.status.value,
                "uptime_seconds": uptime,
                "start_time": self.start_time.isoformat() if self.start_time else None
            },
            "sessions": {
                "current_session_id": self.current_session.session_id if self.current_session else None,
                "total_sessions": len(self.sessions),
                "active_session_duration": self.current_session.duration.total_seconds() if self.current_session and self.current_session.duration else 0
            },
            "performance": self.performance_monitor.get_metrics()
        }
        
        # Add KB metrics if available
        if self.persistent_kb:
            metrics["knowledge_base"] = self.persistent_kb.get_performance_metrics()
        
        # Add sleep metrics if available
        if self.sleep_manager:
            metrics["sleep_cycles"] = self.sleep_manager.get_sleep_metrics()
        
        # Add current session metrics
        if self.current_session:
            metrics["current_session"] = {
                "queries_processed": self.current_session.queries_processed,
                "facts_learned": self.current_session.facts_learned,
                "rules_learned": self.current_session.rules_learned,
                "conflicts_detected": self.current_session.conflicts_detected,
                "conflicts_resolved": self.current_session.conflicts_resolved
            }
        
        return metrics


# Client utility for communicating with background learner
class BackgroundLearnerClient:
    """Client for communicating with BackgroundLearner service"""
    
    def __init__(self, port: int = 7777):
        self.port = port
    
    def _send_message(self, message: IPCMessage) -> IPCMessage:
        """Send message to background learner"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect(('localhost', self.port))
                
                # Send message
                message_json = message.to_json()
                sock.sendall(message_json.encode('utf-8'))
                
                # Receive response
                response_data = sock.recv(8192)
                response_json = response_data.decode('utf-8')
                response = IPCMessage.from_json(response_json)
                
                return response
                
        except Exception as e:
            logger.error(f"Error communicating with background learner: {e}")
            raise
    
    def query(self, goals: List[Term]) -> List[Dict[str, Any]]:
        """Query the background learner"""
        message = IPCMessage(
            message_id=str(uuid.uuid4()),
            message_type=IPCMessageType.QUERY,
            data={"query": [goal.to_prefix() for goal in goals]}
        )
        
        response = self._send_message(message)
        
        if response.message_type == IPCMessageType.ERROR:
            raise RuntimeError(f"Query failed: {response.data.get('error')}")
        
        return response.data.get("solutions", [])
    
    def add_user_knowledge(self, facts: List[Fact], rules: List[Rule]) -> Dict[str, Any]:
        """Add user knowledge to the background learner"""
        message = IPCMessage(
            message_id=str(uuid.uuid4()),
            message_type=IPCMessageType.ADD_USER_KNOWLEDGE,
            data={
                "facts": [fact.to_prefix() for fact in facts],
                "rules": [rule.to_prefix() for rule in rules]
            }
        )
        
        response = self._send_message(message)
        
        if response.message_type == IPCMessageType.ERROR:
            raise RuntimeError(f"Add user knowledge failed: {response.data.get('error')}")
        
        return response.data
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        message = IPCMessage(
            message_id=str(uuid.uuid4()),
            message_type=IPCMessageType.GET_STATUS,
            data={}
        )
        
        response = self._send_message(message)
        
        if response.message_type == IPCMessageType.ERROR:
            raise RuntimeError(f"Get status failed: {response.data.get('error')}")
        
        return response.data
    
    def force_sleep_cycle(self, phase: SleepPhase = SleepPhase.LIGHT_SLEEP) -> Dict[str, Any]:
        """Force a sleep cycle"""
        message = IPCMessage(
            message_id=str(uuid.uuid4()),
            message_type=IPCMessageType.FORCE_SLEEP,
            data={"phase": phase.value.upper()}
        )
        
        response = self._send_message(message)
        
        if response.message_type == IPCMessageType.ERROR:
            raise RuntimeError(f"Force sleep failed: {response.data.get('error')}")
        
        return response.data
    
    def shutdown(self) -> None:
        """Shutdown the background learner"""
        message = IPCMessage(
            message_id=str(uuid.uuid4()),
            message_type=IPCMessageType.SHUTDOWN,
            data={}
        )
        
        response = self._send_message(message)
        
        if response.message_type == IPCMessageType.ERROR:
            raise RuntimeError(f"Shutdown failed: {response.data.get('error')}")