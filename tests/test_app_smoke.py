"""Smoke test do dashboard: executa app.py inteiro via AppTest."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parent.parent


def test_app_roda_sem_excecao():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception, f"app.py levantou exceção: {[str(e.value) for e in at.exception]}"
    assert len(at.metric) > 0
    assert len(at.tabs) == 5
