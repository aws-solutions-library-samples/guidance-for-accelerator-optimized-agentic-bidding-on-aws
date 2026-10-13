"""The bid simulator wrapper answers without waiting on Nagle."""
import importlib.util
import os
import sys
import types

_PATH = os.path.join(os.path.dirname(__file__), "..", "prebid", "amt-simulator", "server.py")


def test_the_handler_sets_tcp_nodelay(monkeypatch):
    # The release's handler is mounted at deploy time; a stand-in lets the wrapper import.
    monkeypatch.setitem(sys.modules, "handler", types.ModuleType("handler"))
    spec = importlib.util.spec_from_file_location("amt_sim_server", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.SimulatorHandler.disable_nagle_algorithm is True
    assert module.SimulatorHandler.protocol_version == "HTTP/1.1"
