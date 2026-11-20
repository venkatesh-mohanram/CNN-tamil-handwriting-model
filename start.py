#!/usr/bin/env python3
"""Prepare a private CPU environment and supervise the local server and client."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import http.client
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import venv

ROOT=Path(__file__).resolve().parent

def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-port',type=int,default=8000)
    parser.add_argument('--client-port',type=int,default=8080)
    parser.add_argument('--no-install',action='store_true',help='Use the current Python with already installed dependencies')
    parser.add_argument('--setup-only',action='store_true',help='Prepare/check dependencies and exit without starting services')
    options=parser.parse_args(argv)
    if not all(1024<=port<=65535 for port in (options.server_port,options.client_port)):
        parser.error('Use ports between1024 and65535.')
    if options.server_port==options.client_port:
        parser.error('Server and client ports must be different.')
    return options

def environment_python(root:Path,no_install:bool=False)->Path:
    if no_install:return Path(sys.executable)
    directory=root/'.venv'
    python=directory/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.is_file():
        print('[setup] Creating an isolated environment in .venv',flush=True)
        # POSIX standalone Python builds locate libpython through the original
        # executable's path; copying that binary can break its relative rpath.
        venv.EnvBuilder(with_pip=True,symlinks=os.name!='nt').create(directory)
    print('[setup] Installing/checking the pinned CPU dependencies',flush=True)
    subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check','--no-input',
                    '--require-virtualenv','-r',str(root/'server/requirements.lock.txt')],check=True)
    return python

@contextmanager
def runtime_credential():
    with tempfile.TemporaryDirectory(prefix='tamil-handoff-') as folder:
        path=Path(folder)/'service-credential'
        token=secrets.token_urlsafe(48).encode('ascii')
        descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'wb') as stream:stream.write(token)
        yield path,token

def check_ports(ports):
    for port in ports:
        try:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                probe.bind(('127.0.0.1',port))
        except OSError as exc:
            raise RuntimeError(f'Port {port} is unavailable. Choose a different port using the startup options.') from exc

def wait_ready(process,port,path,token,stop,seconds):
    deadline=time.monotonic()+seconds
    while not stop.is_set() and time.monotonic()<deadline:
        if process.poll() is not None:
            raise RuntimeError(f'The service on port {port} exited before it became ready.')
        connection=http.client.HTTPConnection('127.0.0.1',port,timeout=1)
        try:
            headers={'Authorization':'Bearer '+token.decode('ascii')} if token else {}
            connection.request('GET',path,headers=headers)
            response=connection.getresponse()
            if response.status==200:return
        except (OSError,http.client.HTTPException):
            pass  # The child is still binding/loading; retry within this startup deadline.
        finally:connection.close()
        stop.wait(.1)
    if stop.is_set():raise InterruptedError('Startup cancelled.')
    raise RuntimeError(f'Timed out waiting for the service on port {port}. See its console output.')

def stop_children(children):
    for process in reversed(children):
        if process.poll() is None:process.terminate()
    for process in reversed(children):
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill();process.wait(timeout=5)

def run_services(python,options):
    check_ports((options.server_port,options.client_port))
    children=[];stop=threading.Event()
    previous={}
    for name in (signal.SIGINT,signal.SIGTERM):
        previous[name]=signal.signal(name,lambda *_:stop.set())
    try:
        with runtime_credential() as (credential,token):
            environment={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'}
            def launch(script,*args):
                process=subprocess.Popen([str(python),str(ROOT/script),*map(str,args)],cwd=ROOT,
                    env=environment,start_new_session=os.name!='nt')
                children.append(process);return process
            try:
                print('[startup] EXPERIMENTAL model: tamil-symbols-0.1.0. Accuracy/rejection limitations apply.',flush=True)
                api=launch('server/app.py','--token-file',credential,'--port',options.server_port)
                wait_ready(api,options.server_port,'/health/ready',token,stop,45)
                client=launch('client/serve.py','--token-file',credential,'--api-port',options.server_port,'--port',options.client_port)
                wait_ready(client,options.client_port,'/',None,stop,20)
                print(f'\n[ready] Open the client: http://127.0.0.1:{options.client_port}/',flush=True)
                print(f'[ready] Private inference API: http://127.0.0.1:{options.server_port}/v1/tamil-symbols/recognize',flush=True)
                print('[ready] Press Ctrl+C here to stop both services. Drawings are not saved.\n',flush=True)
                while not stop.wait(.25):
                    if any(process.poll() is not None for process in children):
                        raise RuntimeError('A service exited; stopping the other service.')
                return 0
            finally:
                stop_children(children)
                print('[stopped] Client and inference server stopped.',flush=True)
    finally:
        for name,handler in previous.items():signal.signal(name,handler)

def main(argv=None):
    if not (3,11)<=sys.version_info[:2]<(3,14):
        print('Use Python3.11–3.13; Python3.12 is recommended.',file=sys.stderr);return 1
    options=arguments(argv)
    try:
        if not options.setup_only:check_ports((options.server_port,options.client_port))
        python=environment_python(ROOT,options.no_install)
        subprocess.run([str(python),'-c','import fastapi,uvicorn,onnxruntime,numpy,PIL,httpx,python_multipart'],check=True)
        if options.setup_only:
            print('[setup] CPU dependencies are ready. Run ./start.sh to start both services.',flush=True);return 0
        return run_services(python,options)
    except (KeyboardInterrupt,InterruptedError):return 0
    except (OSError,RuntimeError,subprocess.CalledProcessError) as exc:
        print(f'[error] {exc}',file=sys.stderr);return 1

if __name__=='__main__':
    raise SystemExit(main())
