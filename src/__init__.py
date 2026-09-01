# src/__init__.py
"""
Source package for DevDocs AI
"""

# Try to import main components
try:
    from .simple_search_agent import *
except ImportError as e:
    print(f"Warning: Could not import simple_search_agent: {e}")

# Export commonly used functions and classes
__all__ = ["QAAgent", "QAAgentEvaluator", "Config", "IntentRouterResult"]