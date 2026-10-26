"""The ``artfhouse`` demand endpoint.

Decides which campaigns may offer on an impression, and at what price. It does
NOT resolve the auction: Prebid does that. This endpoint emits offers and, for
every campaign it considered but did not offer for, the reason why.
"""
