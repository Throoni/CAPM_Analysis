"""
monitoring.py

Phase 12.3: Real-Time Monitoring

Provides real-time monitoring capabilities:
- Health checks
- Performance monitoring
- Error tracking
- Alerting system
"""

import os
import sys
import logging
import time
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HealthChecker:
    """System health checker."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize health checker.
        
        Parameters
        ----------
        project_root : Path, optional
            Project root directory
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent
        self.project_root = Path(project_root)
        
    def check_data_files(self) -> Dict:
        """
        Check if required data files exist and are fresh.
        
        Returns
        -------
        Dict
            Health check results for data files
        """
        logger.info("Checking data files...")
        
        issues = []
        checks = []
        
        # Check processed data
        processed_file = self.project_root / "data" / "processed" / "returns_panel.csv"
        if processed_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(processed_file.stat().st_mtime)
            age_days = file_age.days
            checks.append({
                'file': 'returns_panel.csv',
                'exists': True,
                'age_days': age_days,
                'status': 'ok' if age_days < 30 else 'stale'
            })
            if age_days > 30:
                issues.append(f"returns_panel.csv is {age_days} days old")
        else:
            checks.append({
                'file': 'returns_panel.csv',
                'exists': False,
                'status': 'missing'
            })
            issues.append("returns_panel.csv missing")
        
        # Check results
        results_file = self.project_root / "results" / "data" / "capm_results.csv"
        if results_file.exists():
            file_age = datetime.now() - datetime.fromtimestamp(results_file.stat().st_mtime)
            age_days = file_age.days
            checks.append({
                'file': 'capm_results.csv',
                'exists': True,
                'age_days': age_days,
                'status': 'ok' if age_days < 30 else 'stale'
            })
            if age_days > 30:
                issues.append(f"capm_results.csv is {age_days} days old")
        else:
            checks.append({
                'file': 'capm_results.csv',
                'exists': False,
                'status': 'missing'
            })
            issues.append("capm_results.csv missing")
        
        return {
            'healthy': len(issues) == 0,
            'checks': checks,
            'issues': issues
        }
    
    def check_directory_structure(self) -> Dict:
        """
        Check if directory structure is correct.
        
        Returns
        -------
        Dict
            Directory structure health check
        """
        logger.info("Checking directory structure...")
        
        required_dirs = [
            "data/raw",
            "data/processed",
            "results/data",
            "results/figures",
            "results/reports",
            "analysis/core",
            "audit"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
        
        return {
            'healthy': len(missing_dirs) == 0,
            'missing_directories': missing_dirs,
            'total_checked': len(required_dirs)
        }
    
    def get_health_status(self) -> Dict:
        """
        Get overall system health status.
        
        Returns
        -------
        Dict
            Overall health status
        """
        data_health = self.check_data_files()
        dir_health = self.check_directory_structure()
        
        all_healthy = data_health['healthy'] and dir_health['healthy']
        
        return {
            'overall_health': 'healthy' if all_healthy else 'unhealthy',
            'data_files': data_health,
            'directory_structure': dir_health,
            'timestamp': datetime.now().isoformat()
        }


class PerformanceMonitor:
    """Performance monitoring."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = []
        
    def record_execution_time(self, operation: str, duration: float):
        """
        Record execution time for an operation.
        
        Parameters
        ----------
        operation : str
            Operation name
        duration : float
            Duration in seconds
        """
        self.metrics.append({
            'operation': operation,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.debug(f"Operation {operation} took {duration:.2f}s")
    
    def get_performance_summary(self) -> Dict:
        """
        Get performance summary.
        
        Returns
        -------
        Dict
            Performance summary statistics
        """
        if len(self.metrics) == 0:
            return {
                'total_operations': 0,
                'average_duration': 0,
                'max_duration': 0,
                'min_duration': 0
            }
        
        durations = [m['duration'] for m in self.metrics]
        
        return {
            'total_operations': len(self.metrics),
            'average_duration': sum(durations) / len(durations),
            'max_duration': max(durations),
            'min_duration': min(durations),
            'recent_operations': self.metrics[-10:]  # Last 10
        }


class ErrorTracker:
    """Error tracking and logging."""
    
    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize error tracker.
        
        Parameters
        ----------
        log_dir : Path, optional
            Directory for error logs
        """
        if log_dir is None:
            project_root = Path(__file__).parent.parent
            log_dir = project_root / "logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.errors = []
        
    def track_error(self, error_type: str, message: str, details: Dict = None):
        """
        Track an error.
        
        Parameters
        ----------
        error_type : str
            Type of error
        message : str
            Error message
        details : Dict, optional
            Additional error details
        """
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': message,
            'details': details or {}
        }
        
        self.errors.append(error_entry)
        logger.error(f"[{error_type}] {message}")
        
        # Write to error log file
        error_log = self.log_dir / "errors.log"
        with open(error_log, 'a') as f:
            f.write(f"{error_entry['timestamp']} [{error_type}] {message}\n")
    
    def get_error_summary(self, hours: int = 24) -> Dict:
        """
        Get error summary for recent period.
        
        Parameters
        ----------
        hours : int
            Number of hours to look back
        
        Returns
        -------
        Dict
            Error summary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_errors = [
            e for e in self.errors
            if datetime.fromisoformat(e['timestamp']) > cutoff_time
        ]
        
        error_types = {}
        for error in recent_errors:
            error_type = error['type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(recent_errors),
            'error_types': error_types,
            'recent_errors': recent_errors[-10:]  # Last 10
        }


class AlertSystem:
    """Alerting system for critical issues."""
    
    def __init__(self):
        """Initialize alert system."""
        self.alerts = []
        
    def send_alert(self, severity: str, message: str, details: Dict = None):
        """
        Send an alert.
        
        Parameters
        ----------
        severity : str
            Alert severity ('critical', 'warning', 'info')
        message : str
            Alert message
        details : Dict, optional
            Additional details
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'message': message,
            'details': details or {}
        }
        
        self.alerts.append(alert)
        
        if severity == 'critical':
            logger.critical(f"ALERT: {message}")
        elif severity == 'warning':
            logger.warning(f"ALERT: {message}")
        else:
            logger.info(f"ALERT: {message}")
    
    def get_critical_alerts(self) -> List[Dict]:
        """
        Get all critical alerts.
        
        Returns
        -------
        List[Dict]
            List of critical alerts
        """
        return [a for a in self.alerts if a['severity'] == 'critical']


class SystemMonitor:
    """Main system monitoring class."""
    
    def __init__(self):
        """Initialize system monitor."""
        project_root = Path(__file__).parent.parent
        self.health_checker = HealthChecker(project_root)
        self.performance_monitor = PerformanceMonitor()
        self.error_tracker = ErrorTracker()
        self.alert_system = AlertSystem()
        
    def run_health_check(self) -> Dict:
        """
        Run comprehensive health check.
        
        Returns
        -------
        Dict
            Health check results
        """
        logger.info("Running system health check...")
        
        health_status = self.health_checker.get_health_status()
        
        # Send alerts if unhealthy
        if not health_status['data_files']['healthy']:
            self.alert_system.send_alert(
                'warning',
                "Data files health check failed",
                health_status['data_files']
            )
        
        if not health_status['directory_structure']['healthy']:
            self.alert_system.send_alert(
                'critical',
                "Directory structure issues detected",
                health_status['directory_structure']
            )
        
        return health_status
    
    def get_monitoring_summary(self) -> Dict:
        """
        Get comprehensive monitoring summary.
        
        Returns
        -------
        Dict
            Monitoring summary
        """
        return {
            'health': self.run_health_check(),
            'performance': self.performance_monitor.get_performance_summary(),
            'errors': self.error_tracker.get_error_summary(),
            'alerts': {
                'critical': self.alert_system.get_critical_alerts(),
                'total': len(self.alert_system.alerts)
            },
            'timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    monitor = SystemMonitor()
    summary = monitor.get_monitoring_summary()
    
    print("\n" + "="*70)
    print("SYSTEM MONITORING SUMMARY")
    print("="*70)
    print(f"Overall Health: {summary['health']['overall_health']}")
    print(f"Total Operations: {summary['performance']['total_operations']}")
    print(f"Recent Errors: {summary['errors']['total_errors']}")
    print(f"Critical Alerts: {len(summary['alerts']['critical'])}")
    print("="*70)

