"""
Shared domain types for the RAG system.

Types that multiple layers depend on (CLI, API, graph, LLM, retrieval)
live here instead of in a specific layer's package. This prevents
coupling between layers through a shared dependency on a presentation
package (e.g. cli/).
"""
