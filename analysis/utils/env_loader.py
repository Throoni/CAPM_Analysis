"""
env_loader.py

Utility module to load environment variables from .env file.
This ensures credentials are loaded automatically when the package is imported.

Usage:
    from analysis.utils.env_loader import load_env
    load_env()  # Loads .env file if it exists
"""

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


def load_env(env_file: Optional[str] = None) -> bool:
    """
    Load environment variables from .env file.
    
    This function automatically loads environment variables from a .env file
    in the project root directory. This allows credentials to be stored
    securely without hardcoding them in the source code.
    
    Parameters
    ----------
    env_file : str, optional
        Path to .env file. If None, looks for .env in project root.
    
    Returns
    -------
    bool
        True if .env file was loaded successfully, False otherwise.
    """
    if not DOTENV_AVAILABLE:
        # python-dotenv not installed - environment variables must be set manually
        return False
    
    # Determine project root (parent of analysis directory)
    if env_file is None:
        # Get the directory containing this file (analysis/)
        current_dir = Path(__file__).parent
        # Project root is parent of analysis/
        project_root = current_dir.parent
        env_file = project_root / ".env"
    else:
        env_file = Path(env_file)
    
    # Load .env file if it exists
    if env_file.exists():
        load_dotenv(env_file, override=False)  # Don't override existing env vars
        return True
    else:
        # .env file doesn't exist - that's OK, user can set env vars manually
        return False


# Auto-load .env when module is imported
load_env()

