# tests/test_hf_login.py
import importlib
import types


def test_login_retries_then_succeeds(monkeypatch, capsys):
    """login() should retry on failure and eventually succeed (no autologin on import)."""
    calls = {"n": 0}

    def fake_login(_token):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("temporary HF error")
        # success on third attempt

    # Ensure no autologin during import
    monkeypatch.delenv("HF_TOKEN", raising=False)

    # Make sure we import a fresh module
    if "diffusers_helper.hf_login" in importlib.sys.modules:
        del importlib.sys.modules["diffusers_helper.hf_login"]

    # Provide fake huggingface_hub before import
    monkeypatch.setitem(
        importlib.import_module("sys").modules,
        "huggingface_hub",
        types.SimpleNamespace(login=fake_login),
    )

    # Avoid real sleeps
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)

    # Import module (no autologin because HF_TOKEN is unset)
    from diffusers_helper import hf_login

    # Now call the function explicitly (this will do the retries)
    hf_login.login("abc")
    captured = capsys.readouterr().out
    assert calls["n"] == 3
    assert "HF login ok." in captured
