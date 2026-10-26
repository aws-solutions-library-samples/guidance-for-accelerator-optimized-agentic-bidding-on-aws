"""Campaign/deals control plane for the ARTF stack.

An MCP server, run as a Bedrock AgentCore Runtime, that reads and writes the
stack's deals table (``<stack>-deals``): IAB Tech Lab Deals API v1.0 deals and the
artfhouse demand endpoint's campaigns. Deployed only alongside Prebid Server, by
deployment/deploy_control_plane.sh.

Layout, and why it is split this way:

  records.py     validation and status rules for both record types. Imports nothing
                 but the standard library, and is ALSO imported by
                 deployment/scripts/seed_deals_table.py, so the seed file and a tool
                 call are held to one set of rules.
  scope.py       entitlement: which advertisers or sellers a verified caller may act for.
  operations.py  the ten operations against a DynamoDB table. No MCP import.
  server.py      the only module that imports the MCP SDK: tool registration, token
                 verification and transport.

THIS PACKAGE IS NOT CALLED ``mcp``. The SDK's own top-level package has that name, and
a source tree on sys.path with a package called ``mcp`` shadows it.
"""
