"""
LLM roles: every pipeline point that calls a model is identified by one
of these roles.

Backward-compatible re-export. Prefer src.domain.models.LLMRole.

Previous docstring (historical context):
  The backend+model serving each role is configured independently
  (see Settings.role_spec in src/config/settings.py). Each role has
  its own env var, so you can run the final answer on a powerful local
  model and short support tasks on a fast cloud model, without coupling
  them together.
"""

from src.domain.models import LLMRole

__all__ = ["LLMRole"]
