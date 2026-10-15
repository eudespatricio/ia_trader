"""Testes (100% mockados) do patch de robustez do healthcheck do supervisor."""
import logging
import os
import sys
from unittest import mock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import system_supervisor as ss  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_supervisor_log():
    """Impede que o teste escreva no datasets/supervisor.log real (remove o FileHandler durante o teste)."""
    removed = [h for h in ss.logger.handlers if isinstance(h, logging.FileHandler)]
    for h in removed:
        ss.logger.removeHandler(h)
    yield
    for h in removed:
        ss.logger.addHandler(h)


def _svc(**kw):
    return ss.ServiceManager("t", ["true"], check_type="http_port", port=5999, **kw)


def test_check_health_logs_pid_when_not_running(caplog):
    s = _svc()
    s.pid = 123
    with mock.patch.object(s, "is_running", return_value=False), \
            caplog.at_level(logging.WARNING, logger="system_supervisor"):
        ss.logger.propagate = True
        assert s.check_health() is False
    assert "PID=123" in caplog.text


def test_check_health_socket_timeout_and_reason(caplog):
    s = _svc()
    s.pid = 77
    with mock.patch.object(s, "is_running", return_value=True), \
            mock.patch("socket.create_connection", side_effect=ConnectionRefusedError("x")) as cc, \
            caplog.at_level(logging.WARNING, logger="system_supervisor"):
        ss.logger.propagate = True
        assert s.check_health() is False
    assert cc.call_args.kwargs["timeout"] == 5.0
    assert "ConnectionRefusedError" in caplog.text


def test_heal_threshold_is_three():
    sup = ss.SystemSupervisor()
    svc = mock.Mock(consecutive_failures=0)
    svc.name = "x"
    svc.check_health.return_value = False
    sup.services = [svc]
    with mock.patch.object(sup, "maybe_checkpoint_wal"), mock.patch.object(sup, "dump_state_file"):
        sup.check_all_services()
        sup.check_all_services()
        svc.auto_heal.assert_not_called()
        sup.check_all_services()
        svc.auto_heal.assert_called_once()


def test_auto_heal_readopts_port_pid():
    s = _svc()
    s.pid = None
    with mock.patch.object(s, "find_port_pid", return_value=4242), \
            mock.patch.object(s, "stop") as stop, mock.patch.object(s, "start") as start:
        s.auto_heal()
    assert s.pid == 4242
    stop.assert_not_called()
    start.assert_not_called()
    assert s.restart_count == 0


def test_auto_heal_restarts_when_no_port_pid():
    s = _svc()
    with mock.patch.object(s, "find_port_pid", return_value=None), \
            mock.patch.object(s, "stop") as stop, mock.patch.object(s, "start") as start, \
            mock.patch("time.sleep"):
        s.auto_heal()
    stop.assert_called_once()
    start.assert_called_once()
    assert s.restart_count == 1


def test_find_port_pid_lsof():
    s = _svc()
    with mock.patch("subprocess.check_output", return_value="999\n"):
        assert s.find_port_pid() == 999


def test_start_redirects_output_to_log(tmp_path):
    log = tmp_path / "logs" / "flask.log"
    s = ss.ServiceManager("t", ["true"], log_file=str(log))
    with mock.patch.object(s, "is_running", return_value=False), \
            mock.patch("subprocess.Popen") as popen:
        popen.return_value.pid = 5
        s.start()
    assert log.exists()
    kw = popen.call_args.kwargs
    assert kw["stdout"] is not subprocess_devnull()
    assert kw["stderr"] == ss.subprocess.STDOUT


def subprocess_devnull():
    return ss.subprocess.DEVNULL


def test_flask_service_uses_flask_log():
    assert ss.SystemSupervisor().flask_service.log_file == ss.FLASK_LOG_FILE
