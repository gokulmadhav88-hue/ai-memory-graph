"""Exercise the actual Streamlit app and its connected demo pipeline."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.runtime import create_runtime


APP = Path(__file__).resolve().parents[1] / 'app' / 'chatbot_app.py'


def test_app_sample_and_cited_chat(tmp_path, monkeypatch):
    def runtime(**kwargs):
        return create_runtime(demo=True, data_path=tmp_path)
    monkeypatch.setattr('src.runtime.create_runtime', runtime)
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    next(b for b in app.button if b.label == 'Load sample document').click().run()
    assert not app.exception
    app.chat_input[0].set_value('Who manages Atlas?').run()
    assert not app.exception and 'Maya' in app.session_state['messages'][-1]['content']
    assert app.session_state['messages'][-1]['response']['sources']


def test_app_production_mode_needs_explicit_connection(tmp_path, monkeypatch):
    monkeypatch.setattr('src.runtime.create_runtime',
                        lambda **kwargs: create_runtime(demo=True, data_path=tmp_path))
    app = AppTest.from_file(str(APP)).run()
    app.radio[0].set_value('Neo4j + Chroma').run()
    assert not app.exception
    assert any(b.label == 'Connect workspace' for b in app.button)
    assert not app.chat_input
