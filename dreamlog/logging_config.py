"""
Logging Configuration for DreamLog Persistent Learning

This module provides comprehensive logging setup for the persistent learning system,
including structured logging, performance monitoring, and error tracking.
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


class LearningSystemLogger:
    """Specialized logger for the learning system"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """Setup logger configuration"""
        # Don't add handlers if already configured
        if self.logger.handlers:
            return
            
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler for development
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def log_knowledge_injection(self, 
                               facts_count: int, 
                               rules_count: int,
                               conflicts: int,
                               execution_time: float) -> None:
        """Log knowledge injection event"""
        self.logger.info(
            "Knowledge injection completed",
            extra={
                'extra_fields': {
                    'event_type': 'knowledge_injection',
                    'facts_added': facts_count,
                    'rules_added': rules_count,
                    'conflicts_detected': conflicts,
                    'execution_time_ms': execution_time
                }
            }
        )
    
    def log_sleep_cycle(self, 
                       cycle_id: str,
                       phase: str,
                       operations: list,
                       compression_ratio: float) -> None:
        """Log sleep cycle completion"""
        self.logger.info(
            f"Sleep cycle {phase} completed",
            extra={
                'extra_fields': {
                    'event_type': 'sleep_cycle',
                    'cycle_id': cycle_id,
                    'phase': phase,
                    'operations': operations,
                    'compression_ratio': compression_ratio
                }
            }
        )
    
    def log_query_performance(self, 
                            query: str,
                            solution_count: int,
                            execution_time: float) -> None:
        """Log query performance metrics"""
        self.logger.debug(
            "Query executed",
            extra={
                'extra_fields': {
                    'event_type': 'query_performance',
                    'query': query,
                    'solution_count': solution_count,
                    'execution_time_ms': execution_time
                }
            }
        )
    
    def log_conflict_detection(self, 
                              conflict_type: str,
                              severity: float,
                              description: str) -> None:
        """Log conflict detection"""
        self.logger.warning(
            f"Conflict detected: {conflict_type}",
            extra={
                'extra_fields': {
                    'event_type': 'conflict_detection',
                    'conflict_type': conflict_type,
                    'severity': severity,
                    'description': description
                }
            }
        )
    
    def log_validation_failure(self, 
                              test_name: str,
                              error_message: str,
                              severity: str) -> None:
        """Log validation failure"""
        self.logger.error(
            f"Validation failed: {test_name}",
            extra={
                'extra_fields': {
                    'event_type': 'validation_failure',
                    'test_name': test_name,
                    'error_message': error_message,
                    'severity': severity
                }
            }
        )


def setup_logging(log_dir: Optional[Path] = None, 
                 log_level: str = "INFO",
                 enable_file_logging: bool = True,
                 enable_structured_logging: bool = False) -> None:
    """
    Setup comprehensive logging for the persistent learning system
    
    Args:
        log_dir: Directory for log files (optional)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_file_logging: Whether to enable file logging
        enable_structured_logging: Whether to use structured JSON logging
    """
    
    # Create log directory if needed
    if log_dir and enable_file_logging:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if enable_structured_logging:
        console_formatter = StructuredFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handlers if enabled
    if enable_file_logging and log_dir:
        
        # Main log file (rotating)
        main_log_handler = logging.handlers.RotatingFileHandler(
            log_dir / "dreamlog.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        main_log_handler.setLevel(logging.INFO)
        
        if enable_structured_logging:
            main_log_formatter = StructuredFormatter()
        else:
            main_log_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        main_log_handler.setFormatter(main_log_formatter)
        root_logger.addHandler(main_log_handler)
        
        # Error log file
        error_log_handler = logging.handlers.RotatingFileHandler(
            log_dir / "dreamlog_errors.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_log_handler.setLevel(logging.ERROR)
        error_log_handler.setFormatter(main_log_formatter)
        root_logger.addHandler(error_log_handler)
        
        # Performance log file (for detailed performance metrics)
        perf_log_handler = logging.handlers.RotatingFileHandler(
            log_dir / "dreamlog_performance.log",
            maxBytes=20*1024*1024,  # 20MB
            backupCount=3
        )
        perf_log_handler.setLevel(logging.DEBUG)
        
        # Filter for performance events only
        class PerformanceFilter(logging.Filter):
            def filter(self, record):
                return hasattr(record, 'extra_fields') and \
                       record.extra_fields.get('event_type') in [
                           'query_performance', 'sleep_cycle', 'knowledge_injection'
                       ]
        
        perf_log_handler.addFilter(PerformanceFilter())
        perf_log_handler.setFormatter(StructuredFormatter())  # Always use structured format for perf logs
        root_logger.addHandler(perf_log_handler)
    
    # Setup specific loggers
    setup_component_loggers()


def setup_component_loggers() -> None:
    """Setup specialized loggers for different components"""
    
    # Persistent learning components
    logging.getLogger('dreamlog.persistent_learning').setLevel(logging.INFO)
    logging.getLogger('dreamlog.knowledge_validator').setLevel(logging.INFO)
    logging.getLogger('dreamlog.sleep_cycle').setLevel(logging.INFO)
    logging.getLogger('dreamlog.background_learner').setLevel(logging.INFO)
    logging.getLogger('dreamlog.learning_api').setLevel(logging.INFO)
    
    # Core DreamLog components
    logging.getLogger('dreamlog.engine').setLevel(logging.WARNING)
    logging.getLogger('dreamlog.evaluator').setLevel(logging.WARNING)
    logging.getLogger('dreamlog.unification').setLevel(logging.WARNING)
    logging.getLogger('dreamlog.llm_hook').setLevel(logging.INFO)
    
    # External libraries
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


# Convenience function to get logger with proper setup
def get_logger(name: str) -> LearningSystemLogger:
    """Get a properly configured logger for the learning system"""
    return LearningSystemLogger(name)


# Error tracking utilities
class ErrorTracker:
    """Track and analyze errors across the system"""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.recent_errors: list = []
        self.logger = get_logger(__name__)
    
    def record_error(self, 
                    error_type: str, 
                    error_message: str,
                    component: str,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record an error occurrence"""
        
        # Update error counts
        key = f"{component}_{error_type}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        # Track recent errors
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_message,
            'component': component,
            'metadata': metadata or {}
        }
        
        self.recent_errors.append(error_record)
        
        # Keep only recent errors (last 100)
        if len(self.recent_errors) > 100:
            self.recent_errors.pop(0)
        
        # Log the error
        self.logger.logger.error(
            f"Error in {component}: {error_message}",
            extra={
                'extra_fields': {
                    'event_type': 'error_tracking',
                    'error_type': error_type,
                    'component': component,
                    'metadata': metadata or {}
                }
            }
        )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of tracked errors"""
        return {
            'total_error_types': len(self.error_counts),
            'error_counts': self.error_counts.copy(),
            'recent_errors_count': len(self.recent_errors),
            'recent_errors': self.recent_errors[-10:] if self.recent_errors else []
        }


# Global error tracker instance
error_tracker = ErrorTracker()


def log_and_track_error(component: str,
                       error_type: str,
                       error_message: str,
                       metadata: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to log and track errors"""
    error_tracker.record_error(error_type, error_message, component, metadata)