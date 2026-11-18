"""Demand-side services.

Separate from ``agents/`` because this is demand, not an ARTF agent. The
distinction matters: agents run inside the host platform's infrastructure and
propose mutations; this is a buyer's demand endpoint, which sits outside the
seller's cluster and answers bid requests.
"""
