"""Isola os testes do banco real de microestrutura (o TapeReadingAgent padrão fica UNAVAILABLE)."""
import os

import pytest


@pytest.fixture(autouse=True)
def _no_real_microstructure_db(monkeypatch, tmp_path_factory):
    from agents import tape_reading_agent as tra
    monkeypatch.setattr(tra, "MICROSTRUCTURE_DB", os.path.join(str(tmp_path_factory.getbasetemp()), "no_micro.db"))
