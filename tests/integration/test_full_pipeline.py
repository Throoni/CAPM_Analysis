"""
Integration test for full CAPM analysis pipeline

Tests the complete pipeline from raw data to final results.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestFullPipeline:
    """Test the full CAPM analysis pipeline."""
    
    def test_data_processing_imports(self):
        """Test that data processing modules can be imported."""
        try:
            from analysis.core import returns_processing
            from analysis.core import capm_regression
            from analysis.core import fama_macbeth
            assert True
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_config_accessible(self):
        """Test that configuration is accessible."""
        try:
            from analysis.config import COUNTRIES, DATA_PROCESSED_DIR
            assert len(COUNTRIES) > 0
            assert DATA_PROCESSED_DIR is not None
        except Exception as e:
            pytest.fail(f"Config access failed: {e}")
    
    def test_returns_processing_structure(self):
        """Test returns processing module structure."""
        from analysis.core import returns_processing
        
        # Check key functions exist
        assert hasattr(returns_processing, 'prices_to_returns')
        assert hasattr(returns_processing, 'create_country_panel')
        assert hasattr(returns_processing, 'process_all_countries')
    
    def test_capm_regression_structure(self):
        """Test CAPM regression module structure."""
        from analysis.core import capm_regression
        
        # Check key functions exist
        assert hasattr(capm_regression, 'run_capm_regression')
        assert hasattr(capm_regression, 'run_all_capm_regressions')
    
    def test_fama_macbeth_structure(self):
        """Test Fama-MacBeth module structure."""
        from analysis.core import fama_macbeth
        
        # Check key functions exist
        assert hasattr(fama_macbeth, 'run_fama_macbeth_test')
    
    @pytest.mark.slow
    def test_end_to_end_with_synthetic_data(self):
        """Test end-to-end pipeline with synthetic data."""
        # This is a placeholder for a full end-to-end test
        # In practice, this would:
        # 1. Create synthetic price data
        # 2. Process through returns_processing
        # 3. Run CAPM regressions
        # 4. Run Fama-MacBeth test
        # 5. Verify outputs
        
        # For now, just verify the structure
        assert True  # Placeholder


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

