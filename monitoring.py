"""
Lightweight monitoring setup - minimal startup overhead
NO IMPORTS FROM bot.py TO AVOID CIRCULAR DEPENDENCY
"""
import os
import time
import functools
from typing import Optional, Callable, Any
from fastapi import FastAPI

# Prometheus (optional)
try:
    from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Structured logging - INDEPENDENT setup (no bot.py import)
import structlog
from structlog.processors import JSONRenderer

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()

# Lazy metrics - only create if Prometheus is available
if PROMETHEUS_AVAILABLE:
    http_requests_total = Counter(
        'http_requests_total',
        'Total HTTP requests',
        ['method', 'endpoint', 'status']
    )
    
    http_request_duration_seconds = Histogram(
        'http_request_duration_seconds',
        'HTTP request duration',
        ['method', 'endpoint']
    )
    
    active_connections = Gauge(
        'active_connections',
        'Number of active connections'
    )
    
    chat_requests_total = Counter(
        'chat_requests_total',
        'Total chat requests',
        ['business_id']
    )
    
    document_uploads_total = Counter(
        'document_uploads_total',
        'Total document uploads',
        ['business_id']
    )
    
    # Operation metrics
    operation_duration = Histogram(
        'operation_duration_seconds',
        'Operation duration in seconds',
        ['operation']
    )
    
    operation_total = Counter(
        'operation_total',
        'Total operations',
        ['operation', 'status']
    )
    
    # LLM metrics
    llm_tokens_total = Counter(
        'llm_tokens_total',
        'Total LLM tokens used',
        ['model', 'type']
    )


def setup_monitoring(app: FastAPI):
    """
    Lightweight monitoring setup - no heavy instrumentation
    Only adds metrics endpoint, no tracing or complex instrumentation
    """
    logger.info("monitoring_setup_started", mode="lightweight")
    
    # Only add Prometheus if available
    if PROMETHEUS_AVAILABLE:
        try:
            # Mount metrics endpoint
            metrics_app = make_asgi_app()
            app.mount("/metrics", metrics_app)
            logger.info("prometheus_metrics_enabled", endpoint="/metrics")
        except Exception as e:
            logger.warning("prometheus_setup_failed", error=str(e))
    else:
        logger.info("prometheus_not_available", 
                   message="Install prometheus-client for metrics")
    
    # Add simple middleware for request counting
    @app.middleware("http")
    async def metrics_middleware(request, call_next):
        if PROMETHEUS_AVAILABLE:
            active_connections.inc()
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            if PROMETHEUS_AVAILABLE:
                duration = time.time() - start_time
                
                http_requests_total.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status=response.status_code
                ).inc()
                
                http_request_duration_seconds.labels(
                    method=request.method,
                    endpoint=request.url.path
                ).observe(duration)
            
            return response
        finally:
            if PROMETHEUS_AVAILABLE:
                active_connections.dec()
    
    logger.info("monitoring_setup_completed", mode="lightweight")


def track_chat_request(business_id: str):
    """Track a chat request"""
    if PROMETHEUS_AVAILABLE:
        chat_requests_total.labels(business_id=business_id).inc()


def track_document_upload(business_id: str):
    """Track a document upload"""
    if PROMETHEUS_AVAILABLE:
        document_uploads_total.labels(business_id=business_id).inc()


def monitor(
    operation: Optional[str] = None,
    model: Optional[str] = None,
    track_tokens: bool = False
) -> Callable:
    """
    Decorator to monitor operations with metrics and logging
    
    Usage:
        @monitor(operation="chat")
        async def chat_function():
            ...
        
        @monitor(model="llama-3.3-70b", track_tokens=True)
        async def llm_call():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            op_name = operation or func.__name__
            start_time = time.time()
            
            logger.info(f"{op_name}_started")
            
            try:
                result = await func(*args, **kwargs)
                
                duration = time.time() - start_time
                
                if PROMETHEUS_AVAILABLE:
                    operation_duration.labels(operation=op_name).observe(duration)
                    operation_total.labels(operation=op_name, status="success").inc()
                
                logger.info(
                    f"{op_name}_completed",
                    duration_ms=round(duration * 1000, 2)
                )
                
                # Track tokens if enabled
                if track_tokens and PROMETHEUS_AVAILABLE and hasattr(result, 'usage_metadata'):
                    usage = result.usage_metadata
                    if hasattr(usage, 'input_tokens'):
                        llm_tokens_total.labels(
                            model=model or "unknown",
                            type="input"
                        ).inc(usage.input_tokens)
                    if hasattr(usage, 'output_tokens'):
                        llm_tokens_total.labels(
                            model=model or "unknown",
                            type="output"
                        ).inc(usage.output_tokens)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                if PROMETHEUS_AVAILABLE:
                    operation_total.labels(operation=op_name, status="error").inc()
                
                logger.error(
                    f"{op_name}_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    duration_ms=round(duration * 1000, 2)
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            op_name = operation or func.__name__
            start_time = time.time()
            
            logger.info(f"{op_name}_started")
            
            try:
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                
                if PROMETHEUS_AVAILABLE:
                    operation_duration.labels(operation=op_name).observe(duration)
                    operation_total.labels(operation=op_name, status="success").inc()
                
                logger.info(
                    f"{op_name}_completed",
                    duration_ms=round(duration * 1000, 2)
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                if PROMETHEUS_AVAILABLE:
                    operation_total.labels(operation=op_name, status="error").inc()
                
                logger.error(
                    f"{op_name}_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    duration_ms=round(duration * 1000, 2)
                )
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator