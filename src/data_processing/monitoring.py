"""
Monitoring and metrics collection for PSX data download operations.
"""

import json
import time
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class DownloadMetrics:
    """Metrics for tracking download performance"""
    total_symbols: int = 0
    successful_downloads: int = 0
    failed_downloads: int = 0
    total_response_time: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    failed_symbols: List[str] = None
    response_times: List[float] = None
    
    def __post_init__(self):
        if self.failed_symbols is None:
            self.failed_symbols = []
        if self.response_times is None:
            self.response_times = []
        if self.start_time is None:
            self.start_time = datetime.now()
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time"""
        if self.response_times:
            return sum(self.response_times) / len(self.response_times)
        return 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_symbols > 0:
            return (self.successful_downloads / self.total_symbols) * 100
        return 0.0
    
    @property
    def duration(self) -> timedelta:
        """Calculate total duration"""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return timedelta()
    
    def add_response_time(self, response_time: float):
        """Add a response time measurement"""
        self.response_times.append(response_time)
        self.total_response_time += response_time
    
    def mark_success(self, symbol: str):
        """Mark a successful download"""
        self.successful_downloads += 1
    
    def mark_failure(self, symbol: str):
        """Mark a failed download"""
        self.failed_downloads += 1
        self.failed_symbols.append(symbol)
    
    def finalize(self):
        """Mark the end of the download session"""
        self.end_time = datetime.now()

class MetricsCollector:
    """Collects and manages download metrics"""
    
    def __init__(self, metrics_file: str = "download_metrics.json"):
        self.metrics_file = Path(metrics_file)
        self.current_metrics = DownloadMetrics()
        self.logger = logging.getLogger(__name__)
        
        # Ensure metrics directory exists
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    def start_session(self, total_symbols: int):
        """Start a new metrics collection session"""
        self.current_metrics = DownloadMetrics(
            total_symbols=total_symbols,
            start_time=datetime.now()
        )
        self.logger.info(f"Started metrics collection for {total_symbols} symbols")
    
    def record_download_attempt(self, symbol: str, success: bool, response_time: float = 0.0):
        """Record a download attempt"""
        self.current_metrics.add_response_time(response_time)
        
        if success:
            self.current_metrics.mark_success(symbol)
            self.logger.debug(f"Successful download for {symbol} in {response_time:.2f}s")
        else:
            self.current_metrics.mark_failure(symbol)
            self.logger.warning(f"Failed download for {symbol}")
    
    def get_current_metrics(self) -> DownloadMetrics:
        """Get current metrics"""
        return self.current_metrics
    
    def save_metrics(self):
        """Save current metrics to file"""
        self.current_metrics.finalize()
        
        try:
            # Load existing metrics or create new list
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    historical_metrics = json.load(f)
            else:
                historical_metrics = []
            
            # Convert current metrics to dict (handling datetime serialization)
            metrics_dict = asdict(self.current_metrics)
            metrics_dict['start_time'] = self.current_metrics.start_time.isoformat() if self.current_metrics.start_time else None
            metrics_dict['end_time'] = self.current_metrics.end_time.isoformat() if self.current_metrics.end_time else None
            metrics_dict['duration_seconds'] = self.current_metrics.duration.total_seconds()
            
            # Add to historical metrics
            historical_metrics.append(metrics_dict)
            
            # Save back to file
            with open(self.metrics_file, 'w') as f:
                json.dump(historical_metrics, f, indent=2)
            
            self.logger.info(f"Metrics saved to {self.metrics_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    def get_summary_report(self) -> str:
        """Generate a summary report of current metrics"""
        metrics = self.current_metrics
        
        report = f"""
Download Session Summary:
========================
Total Symbols: {metrics.total_symbols}
Successful Downloads: {metrics.successful_downloads}
Failed Downloads: {metrics.failed_downloads}
Success Rate: {metrics.success_rate:.2f}%
Average Response Time: {metrics.average_response_time:.2f}s
Total Duration: {metrics.duration}
Failed Symbols: {', '.join(metrics.failed_symbols[:10])}{'...' if len(metrics.failed_symbols) > 10 else ''}
"""
        return report.strip()

class HealthChecker:
    """Monitors system health during data download operations"""
    
    def __init__(self, check_interval: int = 300):
        self.check_interval = check_interval
        self.last_check = datetime.now()
        self.logger = logging.getLogger(__name__)
        self.health_status = {}
    
    def check_database_health(self, engine) -> bool:
        """Check database connection health"""
        try:
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            self.health_status['database'] = 'healthy'
            return True
        except Exception as e:
            self.health_status['database'] = f'unhealthy: {str(e)}'
            self.logger.error(f"Database health check failed: {e}")
            return False
    
    def check_memory_usage(self) -> bool:
        """Check memory usage"""
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 90:
                self.health_status['memory'] = f'high usage: {memory_percent}%'
                self.logger.warning(f"High memory usage: {memory_percent}%")
                return False
            else:
                self.health_status['memory'] = f'normal: {memory_percent}%'
                return True
        except ImportError:
            self.health_status['memory'] = 'psutil not available'
            return True
        except Exception as e:
            self.health_status['memory'] = f'error: {str(e)}'
            return True
    
    def check_disk_space(self, db_path: str) -> bool:
        """Check available disk space"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(Path(db_path).parent)
            free_percent = (free / total) * 100
            
            if free_percent < 10:
                self.health_status['disk'] = f'low space: {free_percent:.1f}% free'
                self.logger.warning(f"Low disk space: {free_percent:.1f}% free")
                return False
            else:
                self.health_status['disk'] = f'sufficient: {free_percent:.1f}% free'
                return True
        except Exception as e:
            self.health_status['disk'] = f'error: {str(e)}'
            return True
    
    def should_check(self) -> bool:
        """Determine if it's time for a health check"""
        return (datetime.now() - self.last_check).seconds >= self.check_interval
    
    def perform_health_check(self, engine, db_path: str) -> Dict[str, bool]:
        """Perform comprehensive health check"""
        if not self.should_check():
            return self.health_status
        
        self.last_check = datetime.now()
        
        results = {
            'database': self.check_database_health(engine),
            'memory': self.check_memory_usage(),
            'disk': self.check_disk_space(db_path)
        }
        
        overall_health = all(results.values())
        self.health_status['overall'] = 'healthy' if overall_health else 'degraded'
        
        if not overall_health:
            self.logger.warning("System health check detected issues")
        
        return results
