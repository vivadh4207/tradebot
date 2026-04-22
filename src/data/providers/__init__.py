"""Multi-source data providers.

Each provider exposes the same interface (quotes, chain, news) but
wraps a different vendor API. The MultiProvider aggregator in
src/data/multi_provider.py fans out to all configured providers,
aggregates responses, and handles fallback.

Each provider is independently enabled via .env keys. A missing key
disables that provider — the others keep working.
"""
