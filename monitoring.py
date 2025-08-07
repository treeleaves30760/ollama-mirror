"""
Monitoring and metrics module for Ollama Mirror Server

Provides metrics collection, health checks, and monitoring capabilities.
"""

import time
import asyncio
import logging
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Individual request metrics"""
    timestamp: datetime
    endpoint: str
    method: str
    status_code: int
    duration_ms: float
    model_name: Optional[str] = None
    bytes_transferred: int = 0


@dataclass
class ModelMetrics:
    """Metrics for a specific model"""
    name: str
    pull_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_bytes_served: int = 0
    last_accessed: Optional[datetime] = None
    first_cached: Optional[datetime] = None


@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_usage_mb: float
    memory_percent: float
    disk_usage_gb: float
    disk_usage_percent: float
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0


class MetricsCollector:
    """Collects and stores various metrics"""

    def __init__(self, max_request_history: int = 10000):
        self.max_request_history = max_request_history

        # Request tracking
        self.request_history: deque = deque(maxlen=max_request_history)
        self.endpoint_stats: Dict[str, Dict] = defaultdict(lambda: {
            'count': 0,
            'total_duration': 0.0,
            'error_count': 0,
            'avg_duration': 0.0
        })

        # Model tracking
        self.model_metrics: Dict[str, ModelMetrics] = {}

        # System metrics
        # 24 hours at 1-minute intervals
        self.system_history: deque = deque(maxlen=1440)

        # Performance counters
        self.start_time = datetime.now()
        self.total_requests = 0
        self.total_errors = 0
        self.total_bytes_served = 0

        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info("Metrics collector initialized")

    def record_request(self,
                       endpoint: str,
                       method: str,
                       status_code: int,
                       duration_ms: float,
                       model_name: Optional[str] = None,
                       bytes_transferred: int = 0):
        """Record a request"""

        # Create request metric
        request = RequestMetrics(
            timestamp=datetime.now(),
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            duration_ms=duration_ms,
            model_name=model_name,
            bytes_transferred=bytes_transferred
        )

        # Add to history
        self.request_history.append(request)

        # Update endpoint stats
        self.endpoint_stats[endpoint]['count'] += 1
        self.endpoint_stats[endpoint]['total_duration'] += duration_ms

        if status_code >= 400:
            self.endpoint_stats[endpoint]['error_count'] += 1
            self.total_errors += 1

        # Calculate average duration
        stats = self.endpoint_stats[endpoint]
        stats['avg_duration'] = stats['total_duration'] / stats['count']

        # Update totals
        self.total_requests += 1
        self.total_bytes_served += bytes_transferred

        # Update model metrics
        if model_name:
            self._update_model_metrics(
                model_name, status_code, bytes_transferred)

    def _update_model_metrics(self, model_name: str, status_code: int, bytes_transferred: int):
        """Update metrics for a specific model"""
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = ModelMetrics(name=model_name)

        metrics = self.model_metrics[model_name]
        metrics.last_accessed = datetime.now()
        metrics.total_bytes_served += bytes_transferred

        # Determine if this was a pull request
        if status_code < 400:
            metrics.pull_count += 1

    def record_cache_hit(self, model_name: str):
        """Record a cache hit"""
        self.cache_hits += 1

        if model_name in self.model_metrics:
            self.model_metrics[model_name].cache_hits += 1

    def record_cache_miss(self, model_name: str):
        """Record a cache miss"""
        self.cache_misses += 1

        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = ModelMetrics(name=model_name)

        self.model_metrics[model_name].cache_misses += 1

    def record_model_cached(self, model_name: str):
        """Record when a model is first cached"""
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = ModelMetrics(name=model_name)

        if self.model_metrics[model_name].first_cached is None:
            self.model_metrics[model_name].first_cached = datetime.now()

    def collect_system_metrics(self):
        """Collect current system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage_mb = memory.used / (1024 * 1024)
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_gb = disk.used / (1024 * 1024 * 1024)
            disk_usage_percent = (disk.used / disk.total) * 100

            # Network I/O
            network = psutil.net_io_counters()

            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_usage_mb=memory_usage_mb,
                memory_percent=memory_percent,
                disk_usage_gb=disk_usage_gb,
                disk_usage_percent=disk_usage_percent,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv
            )

            self.system_history.append(metrics)

        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")

    def get_request_stats(self, minutes: int = 60) -> Dict[str, Any]:
        """Get request statistics for the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_requests = [
            r for r in self.request_history if r.timestamp >= cutoff_time]

        if not recent_requests:
            return {
                'total_requests': 0,
                'error_rate': 0.0,
                'avg_duration_ms': 0.0,
                'requests_per_minute': 0.0
            }

        total_requests = len(recent_requests)
        error_count = sum(1 for r in recent_requests if r.status_code >= 400)
        total_duration = sum(r.duration_ms for r in recent_requests)

        return {
            'total_requests': total_requests,
            'error_rate': (error_count / total_requests) * 100,
            'avg_duration_ms': total_duration / total_requests,
            'requests_per_minute': total_requests / minutes
        }

    def get_model_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all models"""
        stats = []

        for model_name, metrics in self.model_metrics.items():
            cache_total = metrics.cache_hits + metrics.cache_misses
            cache_hit_rate = (metrics.cache_hits /
                              cache_total * 100) if cache_total > 0 else 0

            stats.append({
                'name': model_name,
                'pull_count': metrics.pull_count,
                'cache_hits': metrics.cache_hits,
                'cache_misses': metrics.cache_misses,
                'cache_hit_rate': cache_hit_rate,
                'total_bytes_served': metrics.total_bytes_served,
                'last_accessed': metrics.last_accessed.isoformat() if metrics.last_accessed else None,
                'first_cached': metrics.first_cached.isoformat() if metrics.first_cached else None
            })

        return sorted(stats, key=lambda x: x['pull_count'], reverse=True)

    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        if not self.system_history:
            return {}

        latest = self.system_history[-1]

        # Calculate averages over last hour
        hour_ago = datetime.now() - timedelta(hours=1)
        recent_metrics = [
            m for m in self.system_history if m.timestamp >= hour_ago]

        if recent_metrics:
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / \
                len(recent_metrics)
            avg_memory = sum(
                m.memory_percent for m in recent_metrics) / len(recent_metrics)
        else:
            avg_cpu = latest.cpu_percent
            avg_memory = latest.memory_percent

        return {
            'current': {
                'cpu_percent': latest.cpu_percent,
                'memory_usage_mb': latest.memory_usage_mb,
                'memory_percent': latest.memory_percent,
                'disk_usage_gb': latest.disk_usage_gb,
                'disk_usage_percent': latest.disk_usage_percent
            },
            'averages_1h': {
                'cpu_percent': avg_cpu,
                'memory_percent': avg_memory
            },
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_cache_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_cache_requests *
                    100) if total_cache_requests > 0 else 0

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_cache_requests
        }

    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for monitoring systems"""
        return {
            'overview': {
                'total_requests': self.total_requests,
                'total_errors': self.total_errors,
                'error_rate': (self.total_errors / self.total_requests * 100) if self.total_requests > 0 else 0,
                'total_bytes_served': self.total_bytes_served,
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
            },
            'requests': self.get_request_stats(),
            'models': self.get_model_stats(),
            'system': self.get_system_stats(),
            'cache': self.get_cache_stats(),
            'endpoints': dict(self.endpoint_stats)
        }


class HealthChecker:
    """Health check functionality"""

    def __init__(self, cache_dir: Path, upstream_url: str):
        self.cache_dir = cache_dir
        self.upstream_url = upstream_url
        self.last_check = None
        self.status = "unknown"
        self.issues = []

    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        self.last_check = datetime.now()
        self.issues = []

        # Check cache directory
        cache_ok = self._check_cache_directory()

        # Check upstream connectivity
        upstream_ok = await self._check_upstream()

        # Check system resources
        system_ok = self._check_system_resources()

        # Determine overall status
        if cache_ok and upstream_ok and system_ok:
            self.status = "healthy"
        elif cache_ok and system_ok:
            self.status = "degraded"  # Can serve cached models
        else:
            self.status = "unhealthy"

        return {
            'status': self.status,
            'timestamp': self.last_check.isoformat(),
            'checks': {
                'cache_directory': cache_ok,
                'upstream_connectivity': upstream_ok,
                'system_resources': system_ok
            },
            'issues': self.issues
        }

    def _check_cache_directory(self) -> bool:
        """Check cache directory accessibility"""
        try:
            if not self.cache_dir.exists():
                self.issues.append("Cache directory does not exist")
                return False

            if not self.cache_dir.is_dir():
                self.issues.append("Cache path is not a directory")
                return False

            # Test write access
            test_file = self.cache_dir / ".health_check"
            test_file.write_text("test")
            test_file.unlink()

            return True

        except Exception as e:
            self.issues.append(f"Cache directory check failed: {e}")
            return False

    async def _check_upstream(self) -> bool:
        """Check upstream connectivity"""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.upstream_url}/v2/")

                if response.status_code == 200:
                    return True
                else:
                    self.issues.append(
                        f"Upstream returned status {response.status_code}")
                    return False

        except Exception as e:
            self.issues.append(f"Upstream connectivity check failed: {e}")
            return False

    def _check_system_resources(self) -> bool:
        """Check system resource availability"""
        try:
            # Check available disk space
            disk = psutil.disk_usage(str(self.cache_dir))
            free_gb = disk.free / (1024 * 1024 * 1024)

            if free_gb < 1.0:  # Less than 1GB free
                self.issues.append(
                    f"Low disk space: {free_gb:.1f}GB available")
                return False

            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 95:
                self.issues.append(f"High memory usage: {memory.percent:.1f}%")
                return False

            return True

        except Exception as e:
            self.issues.append(f"System resource check failed: {e}")
            return False


# Global metrics collector instance
metrics_collector: Optional[MetricsCollector] = None
health_checker: Optional[HealthChecker] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance"""
    global metrics_collector
    if metrics_collector is None:
        metrics_collector = MetricsCollector()
    return metrics_collector


def get_health_checker(cache_dir: Path, upstream_url: str) -> HealthChecker:
    """Get the global health checker instance"""
    global health_checker
    if health_checker is None:
        health_checker = HealthChecker(cache_dir, upstream_url)
    return health_checker


async def periodic_metrics_collection():
    """Background task for periodic metrics collection"""
    collector = get_metrics_collector()

    while True:
        try:
            collector.collect_system_metrics()
            await asyncio.sleep(60)  # Collect every minute
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            await asyncio.sleep(60)
