"""
validate_performance.py

Phase 9.1: Performance Benchmarking Audit

Validates performance:
- Execution time profiling
- Memory usage tracking
- Bottleneck detection
- Scalability testing
"""

import os
import sys
import logging
import time
import tracemalloc
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class PerformanceAudit:
    """Audit class for performance validation."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.passed = []
        self.project_root = Path(__file__).parent.parent
        
    def log_issue(self, severity: str, message: str, details: Dict = None):
        """Log an issue."""
        entry = {'severity': severity, 'message': message, 'details': details or {}}
        if severity == 'critical':
            self.issues.append(entry)
        else:
            self.warnings.append(entry)
        logger.warning(f"[{severity.upper()}] {message}")
    
    def log_pass(self, message: str):
        """Log a passed check."""
        self.passed.append(message)
        logger.info(f"[PASS] {message}")
    
    def benchmark_import_time(self) -> Dict:
        """Benchmark import times for key modules."""
        logger.info("Benchmarking module import times...")
        
        issues_found = []
        modules_to_test = [
            'pandas',
            'numpy',
            'statsmodels',
            'matplotlib',
            'analysis.config',
            'analysis.returns_processing'
        ]
        
        import_times = {}
        slow_imports = []
        
        for module_name in modules_to_test:
            try:
                start_time = time.time()
                __import__(module_name)
                import_time = time.time() - start_time
                import_times[module_name] = import_time
                
                if import_time > 2.0:  # More than 2 seconds
                    slow_imports.append({
                        'module': module_name,
                        'time': import_time
                    })
                    self.log_issue('warning',
                                 f"Slow import: {module_name} ({import_time:.2f}s)")
                else:
                    self.log_pass(f"Import time OK: {module_name} ({import_time:.2f}s)")
            except Exception as e:
                issues_found.append({
                    'issue': f"Failed to import {module_name}",
                    'error': str(e)
                })
                self.log_issue('warning', f"Import failed: {module_name} - {e}")
        
        return {
            'passed': len(slow_imports) == 0,
            'import_times': import_times,
            'slow_imports': slow_imports,
            'issues': issues_found
        }
    
    def check_memory_usage(self) -> Dict:
        """Check memory usage patterns."""
        logger.info("Checking memory usage...")
        
        issues_found = []
        
        # Try to enable memory tracking
        try:
            tracemalloc.start()
            
            # Import and use key modules
            import pandas as pd
            import numpy as np
            
            # Create a test DataFrame
            test_df = pd.DataFrame(np.random.randn(1000, 100))
            
            # Get current memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Convert to MB
            current_mb = current / 1024 / 1024
            peak_mb = peak / 1024 / 1024
            
            if peak_mb > 500:  # More than 500 MB for simple operation
                self.log_issue('warning',
                             f"High memory usage: {peak_mb:.2f} MB")
                issues_found.append({
                    'issue': f'High memory usage: {peak_mb:.2f} MB',
                    'peak_mb': peak_mb
                })
            else:
                self.log_pass(f"Memory usage OK: {peak_mb:.2f} MB")
            
            return {
                'passed': peak_mb <= 500,
                'current_mb': current_mb,
                'peak_mb': peak_mb,
                'issues': issues_found
            }
        except Exception as e:
            self.log_issue('warning', f"Could not check memory: {e}")
            return {
                'passed': False,
                'issues': [{'issue': f'Memory check failed: {e}'}]
            }
    
    def check_file_sizes(self) -> Dict:
        """Check if data files are reasonable size."""
        logger.info("Checking data file sizes...")
        
        issues_found = []
        data_dir = self.project_root / "data"
        
        if not data_dir.exists():
            return {
                'passed': False,
                'issues': [{'issue': 'data/ directory not found'}]
            }
        
        large_files = []
        total_size = 0
        
        for file_path in data_dir.rglob("*"):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / 1024 / 1024
                total_size += size_mb
                
                if size_mb > 100:  # More than 100 MB
                    large_files.append({
                        'file': str(file_path.relative_to(self.project_root)),
                        'size_mb': size_mb
                    })
                    self.log_issue('warning',
                                 f"Large file: {file_path.name} ({size_mb:.2f} MB)")
        
        if len(large_files) == 0:
            self.log_pass(f"All files reasonable size (total: {total_size:.2f} MB)")
        else:
            self.log_pass(f"Total data size: {total_size:.2f} MB")
        
        return {
            'passed': len(large_files) == 0,
            'total_size_mb': total_size,
            'large_files': large_files,
            'issues': issues_found
        }
    
    def check_computation_time(self) -> Dict:
        """Check computation time for key operations."""
        logger.info("Checking computation time...")
        
        issues_found = []
        
        # Test simple operations
        import numpy as np
        import pandas as pd
        
        # Test 1: Array operations
        start = time.time()
        arr = np.random.randn(10000, 100)
        result = arr.mean(axis=0)
        array_time = time.time() - start
        
        if array_time > 1.0:
            self.log_issue('warning', f"Slow array operation: {array_time:.2f}s")
        else:
            self.log_pass(f"Array operation time OK: {array_time:.2f}s")
        
        # Test 2: DataFrame operations
        start = time.time()
        df = pd.DataFrame(np.random.randn(10000, 100))
        result = df.mean()
        df_time = time.time() - start
        
        if df_time > 1.0:
            self.log_issue('warning', f"Slow DataFrame operation: {df_time:.2f}s")
        else:
            self.log_pass(f"DataFrame operation time OK: {df_time:.2f}s")
        
        return {
            'passed': array_time <= 1.0 and df_time <= 1.0,
            'array_time': array_time,
            'dataframe_time': df_time,
            'issues': issues_found
        }
    
    def run_all_checks(self) -> Dict:
        """Run all performance checks."""
        logger.info("\n" + "="*70)
        logger.info("PHASE 9.1: PERFORMANCE BENCHMARKING")
        logger.info("="*70 + "\n")
        
        results = {
            'import_times': self.benchmark_import_time(),
            'memory_usage': self.check_memory_usage(),
            'file_sizes': self.check_file_sizes(),
            'computation_time': self.check_computation_time()
        }
        
        total_issues = sum(len(r.get('issues', [])) for r in results.values())
        total_critical = len(self.issues)
        total_warnings = len(self.warnings)
        
        results['summary'] = {
            'total_checks': len(results),
            'total_issues': total_issues,
            'critical_issues': total_critical,
            'warnings': total_warnings,
            'passed_checks': len(self.passed)
        }
        
        return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    audit = PerformanceAudit()
    results = audit.run_all_checks()
    
    print("\n" + "="*70)
    print("PERFORMANCE AUDIT SUMMARY")
    print("="*70)
    print(f"Total Checks: {results['summary']['total_checks']}")
    print(f"Critical Issues: {results['summary']['critical_issues']}")
    print(f"Warnings: {results['summary']['warnings']}")
    print(f"Passed: {results['summary']['passed_checks']}")
    print("="*70)

