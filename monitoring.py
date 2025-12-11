"""
Comprehensive monitoring setup with Prometheus and OpenTelemetry
"""

# ============================================
# CRITICAL: Windows Event Loop Fix - MUST BE FIRST
# ============================================
import sys
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ============================================
# NOW SAFE TO IMPORT OTHER MODULES
# ============================================

import os
import time
import functools
from typing import Optional, Callable, Any
from contextlib import contextmanager

# Prometheus
from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    start_http_server, REGISTRY
)

# OpenTelemetry - Tracing
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION

# OpenTelemetry - Metrics
from opentelemetry import metrics as otel_metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

# OpenTelemetry - Auto-instrumentation
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    FastAPIInstrumentor = None
    import warnings
    warnings.warn("OpenTelemetry not available - monitoring disabled")
    
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor

# Logging
import structlog

logger = structlog.get_logger()

# ============================================
# GLOBAL STATE
# ============================================

_monitoring_initialized = False
_tracer: Optional[trace.Tracer] = None
_meter: Optional[otel_metrics.Meter] = None

# ============================================
# PROMETHEUS METRICS
# ============================================

# Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

# LLM metrics
llm_requests_total = Counter(
    'llm_requests_total',
    'Total LLM API requests',
    ['model', 'status']
)

llm_request_duration = Histogram(
    'llm_request_duration_seconds',
    'LLM request duration',
    ['model']
)

llm_tokens_total = Counter(
    'llm_tokens_total',
    'Total tokens processed',
    ['model', 'type']  # type: prompt, completion
)

llm_cost_total = Counter(
    'llm_cost_total',
    'Total estimated LLM cost in USD',
    ['model']
)

# Database metrics
db_operations_total = Counter(
    'db_operations_total',
    'Total database operations',
    ['operation', 'status']
)

db_operation_duration = Histogram(
    'db_operation_duration_seconds',
    'Database operation duration',
    ['operation']
)

db_pool_connections = Gauge(
    'db_pool_connections',
    'Current database pool connections',
    ['state']  # state: idle, active, total
)

# RAG metrics
rag_queries_total = Counter(
    'rag_queries_total',
    'Total RAG queries',
    ['business_id', 'status']
)

rag_query_duration = Histogram(
    'rag_query_duration_seconds',
    'RAG query duration',
    ['business_id']
)

rag_cache_hits = Counter(
    'rag_cache_hits_total',
    'RAG cache hits',
    ['business_id']
)

# Cache metrics
cache_operations_total = Counter(
    'cache_operations_total',
    'Total cache operations',
    ['operation', 'status']  # operation: get, set, delete
)

cache_hit_ratio = Gauge(
    'cache_hit_ratio',
    'Cache hit ratio'
)

# Circuit breaker metrics
circuit_breaker_state = Gauge(
    'circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open)',
    ['service']
)

circuit_breaker_failures = Counter(
    'circuit_breaker_failures_total',
    'Circuit breaker failures',
    ['service']
)

# Application info
app_info = Info(
    'chatbot_application',
    'Chatbot application information'
)

# ============================================
# OPENTELEMETRY SETUP
# ============================================

def setup_opentelemetry(
    service_name: str = "whatsapp-chatbot",
    service_version: str = "1.0.0",
    otlp_endpoint: Optional[str] = None,
    environment: str = "production"
):
    """
    Setup OpenTelemetry with tracing and metrics
    
    Args:
        service_name: Name of the service
        service_version: Version of the service
        otlp_endpoint: OTLP endpoint (e.g., "http://localhost:4317" for local Jaeger)
        environment: Environment (production, staging, development)
    """
    global _tracer, _meter
    
    # Create resource with service information
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
        "environment": environment,
        "deployment.environment": environment,
    })
    
    # ============================================
    # TRACING SETUP
    # ============================================
    
    tracer_provider = TracerProvider(resource=resource)
    
    if otlp_endpoint:
        # OTLP exporter (for Jaeger, Tempo, etc.)
        otlp_trace_exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True  # Use secure=False for production with TLS
        )
        tracer_provider.add_span_processor(
            BatchSpanProcessor(otlp_trace_exporter)
        )
        logger.info("otlp_trace_exporter_configured", endpoint=otlp_endpoint)
    else:
        logger.warning("otlp_endpoint_not_set", 
                      message="Traces will not be exported. Set OTLP_ENDPOINT env var.")
    
    # Set global tracer provider
    trace.set_tracer_provider(tracer_provider)
    _tracer = trace.get_tracer(__name__)
    
    logger.info("opentelemetry_tracing_initialized", 
                service=service_name, 
                version=service_version)
    
    # ============================================
    # METRICS SETUP
    # ============================================
    
    if otlp_endpoint:
        # OTLP metric exporter
        otlp_metric_exporter = OTLPMetricExporter(
            endpoint=otlp_endpoint,
            insecure=True
        )
        
        # Metric reader with periodic export
        metric_reader = PeriodicExportingMetricReader(
            otlp_metric_exporter,
            export_interval_millis=60000  # Export every 60 seconds
        )
        
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader]
        )
        
        otel_metrics.set_meter_provider(meter_provider)
        _meter = otel_metrics.get_meter(__name__)
        
        logger.info("opentelemetry_metrics_initialized")
    else:
        logger.warning("otlp_metrics_not_configured")
    
    return _tracer, _meter

# ============================================
# AUTO-INSTRUMENTATION
# ============================================

def setup_auto_instrumentation(
    instrument_fastapi: bool = True,
    instrument_sqlalchemy: bool = True,
    instrument_psycopg: bool = True,
    instrument_asyncpg: bool = True
):
    """
    Setup automatic instrumentation for common libraries
    """
    
    if instrument_sqlalchemy:
        try:
            SQLAlchemyInstrumentor().instrument()
            logger.info("sqlalchemy_instrumentation_enabled")
        except Exception as e:
            logger.warning("sqlalchemy_instrumentation_failed", error=str(e))
    
    if instrument_psycopg:
        try:
            PsycopgInstrumentor().instrument()
            logger.info("psycopg_instrumentation_enabled")
        except Exception as e:
            logger.warning("psycopg_instrumentation_failed", error=str(e))
    
    if instrument_asyncpg:
        try:
            AsyncPGInstrumentor().instrument()
            logger.info("asyncpg_instrumentation_enabled")
        except Exception as e:
            logger.warning("asyncpg_instrumentation_failed", error=str(e))
    
    # FastAPI instrumentation is done separately in the app

# ============================================
# MAIN SETUP FUNCTION
# ============================================

def setup_monitoring(
    app=None,
    prometheus_port: int = 9090,
    otlp_endpoint: Optional[str] = None,
    service_name: str = "whatsapp-chatbot",
    service_version: str = "1.0.0",
    environment: str = None,
    auto_instrument_db: bool = True
):
    """
    Setup complete monitoring with Prometheus and OpenTelemetry
    
    Args:
        app: FastAPI app instance (for instrumentation)
        prometheus_port: Port for Prometheus metrics server (or FastAPI app object)
        otlp_endpoint: OTLP endpoint for traces/metrics (e.g., "http://localhost:4317")
        service_name: Service name for tracing
        service_version: Service version
        environment: Environment name
        auto_instrument_db: Enable auto-instrumentation for database libraries
    
    Example:
        setup_monitoring(
            app=app,
            prometheus_port=9090,
            otlp_endpoint=os.getenv("OTLP_ENDPOINT", "http://localhost:4317"),
            environment="production"
        )
    """
    global _monitoring_initialized
    
    if _monitoring_initialized:
        logger.warning("monitoring_already_initialized")
        return
    
    # Detect environment
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "production")
    
    logger.info("monitoring_setup_started", 
                prometheus_port=prometheus_port,
                otlp_endpoint=otlp_endpoint,
                environment=environment)
    
    # ============================================
    # PROMETHEUS
    # ============================================
    
    try:
        # Check if prometheus_port is actually an int
        if isinstance(prometheus_port, int):
            start_http_server(prometheus_port)
            logger.info("prometheus_metrics_server_started", port=prometheus_port)
        else:
            logger.warning("prometheus_port_invalid", 
                          prometheus_port=str(type(prometheus_port)))
        
        # Set application info
        app_info.info({
            'version': service_version,
            'environment': environment,
            'service': service_name
        })
        
    except Exception as e:
        logger.error("prometheus_setup_failed", error=str(e))
    
    # ============================================
    # OPENTELEMETRY
    # ============================================
    
    try:
        setup_opentelemetry(
            service_name=service_name,
            service_version=service_version,
            otlp_endpoint=otlp_endpoint,
            environment=environment
        )
    except Exception as e:
        logger.error("opentelemetry_setup_failed", error=str(e))
    
    # ============================================
    # AUTO-INSTRUMENTATION
    # ============================================
    
    if auto_instrument_db:
        try:
            setup_auto_instrumentation()
        except Exception as e:
            logger.error("auto_instrumentation_failed", error=str(e))
    
    _monitoring_initialized = True
    logger.info("monitoring_setup_completed")

# ============================================
# DECORATOR FOR MONITORING
# ============================================

def monitor(
    operation: str = None,
    model: str = None,
    track_tokens: bool = False
):
    """
    Decorator to add monitoring to functions
    
    Usage:
        @monitor(operation="rag_search")
        async def search_documents(...):
            ...
        
        @monitor(model="gpt-4", track_tokens=True)
        async def call_llm(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            # Start OpenTelemetry span
            span_name = f"{op_name}"
            with _tracer.start_as_current_span(span_name) if _tracer else contextmanager(lambda: (yield))():
                try:
                    # Add span attributes
                    if _tracer:
                        span = trace.get_current_span()
                        span.set_attribute("operation", op_name)
                        if model:
                            span.set_attribute("model", model)
                    
                    # Execute function
                    result = await func(*args, **kwargs)
                    
                    # Track tokens if applicable
                    if track_tokens and model and hasattr(result, 'usage_metadata'):
                        usage = result.usage_metadata
                        prompt_tokens = getattr(usage, 'input_tokens', 0)
                        completion_tokens = getattr(usage, 'output_tokens', 0)
                        
                        llm_tokens_total.labels(model=model, type='prompt').inc(prompt_tokens)
                        llm_tokens_total.labels(model=model, type='completion').inc(completion_tokens)
                        
                        if _tracer:
                            span = trace.get_current_span()
                            span.set_attribute("tokens.prompt", prompt_tokens)
                            span.set_attribute("tokens.completion", completion_tokens)
                    
                    return result
                    
                except Exception as e:
                    status = "error"
                    
                    # Record error in span
                    if _tracer:
                        span = trace.get_current_span()
                        span.set_attribute("error", True)
                        span.set_attribute("error.type", type(e).__name__)
                        span.set_attribute("error.message", str(e))
                    
                    # Update metrics
                    if model:
                        llm_requests_total.labels(model=model, status='error').inc()
                    
                    raise
                    
                finally:
                    # Record duration
                    duration = time.time() - start_time
                    
                    if model:
                        llm_request_duration.labels(model=model).observe(duration)
                        llm_requests_total.labels(model=model, status=status).inc()
                    else:
                        db_operation_duration.labels(operation=op_name).observe(duration)
                        db_operations_total.labels(operation=op_name, status=status).inc()
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            span_name = f"{op_name}"
            with _tracer.start_as_current_span(span_name) if _tracer else contextmanager(lambda: (yield))():
                try:
                    if _tracer:
                        span = trace.get_current_span()
                        span.set_attribute("operation", op_name)
                        if model:
                            span.set_attribute("model", model)
                    
                    result = func(*args, **kwargs)
                    return result
                    
                except Exception as e:
                    status = "error"
                    
                    if _tracer:
                        span = trace.get_current_span()
                        span.set_attribute("error", True)
                        span.set_attribute("error.type", type(e).__name__)
                        span.set_attribute("error.message", str(e))
                    
                    raise
                    
                finally:
                    duration = time.time() - start_time
                    
                    if model:
                        llm_request_duration.labels(model=model).observe(duration)
                        llm_requests_total.labels(model=model, status=status).inc()
                    else:
                        db_operation_duration.labels(operation=op_name).observe(duration)
                        db_operations_total.labels(operation=op_name, status=status).inc()
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# ============================================
# HELPER FUNCTIONS
# ============================================

def record_cache_operation(operation: str, status: str):
    """Record cache operation"""
    cache_operations_total.labels(operation=operation, status=status).inc()

def record_circuit_breaker_event(service: str, state: str):
    """Record circuit breaker state change"""
    state_value = 1 if state == "open" else 0
    circuit_breaker_state.labels(service=service).set(state_value)
    if state == "open":
        circuit_breaker_failures.labels(service=service).inc()

def get_tracer() -> Optional[trace.Tracer]:
    """Get the global tracer instance"""
    return _tracer

def get_meter() -> Optional[otel_metrics.Meter]:
    """Get the global meter instance"""
    return _meter