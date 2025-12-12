"""
Lightweight monitoring setup - minimal startup overhead
"""
import os
from fastapi import FastAPI

try:
    from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from bot import logger

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
        
        try:
            response = await call_next(request)
            
            if PROMETHEUS_AVAILABLE:
                http_requests_total.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status=response.status_code
                ).inc()
            
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