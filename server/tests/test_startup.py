"""Launcher checks do not start services or install dependencies."""
from importlib.util import spec_from_file_location,module_from_spec
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=spec_from_file_location('handoff_start',ROOT/'start.py')
start=module_from_spec(spec);spec.loader.exec_module(start)

def test_ports_are_distinct_and_bounded():
    assert start.arguments([]).server_port==8000
    assert start.arguments([]).client_port==8080
    for values in (['--server-port','8080'],['--client-port','80'],['--server-port','65536']):
        with pytest.raises(SystemExit):start.arguments(values)

def test_no_install_uses_current_interpreter_without_touching_environment(tmp_path,monkeypatch):
    monkeypatch.setattr(start.subprocess,'run',lambda *a,**k:pytest.fail('No install may run'))
    python=start.environment_python(tmp_path,no_install=True)
    assert python==Path(sys.executable)
    assert list(tmp_path.iterdir())==[]

def test_bootstrap_uses_private_venv_and_exact_lockfile(tmp_path,monkeypatch):
    calls=[]
    class FakeBuilder:
        def __init__(self,**kwargs):assert kwargs=={'with_pip':True,'symlinks':True}
        def create(self,path):
            (path/'bin').mkdir(parents=True);(path/'bin/python').touch()
    monkeypatch.setattr(start.venv,'EnvBuilder',FakeBuilder)
    monkeypatch.setattr(start.subprocess,'run',lambda args,**kwargs:calls.append((args,kwargs)))
    python=start.environment_python(tmp_path,no_install=False)
    assert python==tmp_path/'.venv/bin/python'
    assert calls[0][0][:4]==[str(python),'-m','pip','install']
    assert calls[0][0][-2:]==['-r',str(tmp_path/'server/requirements.lock.txt')]
    assert calls[0][1]['check'] is True

def test_runtime_credentials_are_ephemeral_and_not_exposed_in_logs(tmp_path):
    with start.runtime_credential() as (path,token):
        assert path.is_file() and path.read_bytes()==token
        assert len(token)>=32
        assert path.stat().st_mode & 0o777==0o600
        assert not path.is_relative_to(ROOT)
    assert not path.exists()
