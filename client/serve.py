"""Portable loopback canvas host. Run via ../start.sh or pass --token-file directly."""
from __future__ import annotations
import argparse
import asyncio
from contextlib import asynccontextmanager, suppress
import json
import math
from pathlib import Path
import threading
from uuid import uuid4
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import ClientDisconnect

CLIENT_ROOT=Path(__file__).resolve().parent
PROJECT_ROOT=CLIENT_ROOT.parent
BODY_LIMIT=1024*1024+64*1024
RESPONSE_LIMIT=16*1024
SYMBOLS=frozenset('அஆஇஈஉஊஎஏஐஒஓஔஃ')



class Rejection(Exception):
    def __init__(self,status:int,message:str):
        self.status,self.message=status,message


def error(status:int,message:str,headers:dict|None=None):
    return JSONResponse({'request_id':uuid4().hex,'error':message},status_code=status,headers={'Cache-Control':'no-store',**(headers or {})})


def _credential(path:Path)->str:
    path=Path(path).resolve()
    if path.is_relative_to(PROJECT_ROOT):
        raise ValueError('Keep the demo service credential outside the project directory.')
    with path.open('rb') as stream:
        value=stream.read(4097).strip()
    if not 32<=len(value)<=4096 or any(byte<33 or byte>126 for byte in value):
        raise ValueError('Use a token file containing 32–4096 printable ASCII characters.')
    return value.decode('ascii')


async def _body(request:Request)->bytes:
    declared=request.headers.get('content-length')
    if declared is not None:
        try:
            length=int(declared)
        except ValueError as exc:
            raise Rejection(400,'Invalid request length.') from exc
        if length<0:
            raise Rejection(400,'Invalid request length.')
        if length>BODY_LIMIT:
            raise Rejection(413,'Upload exceeds the size limit.')
    if not request.headers.get('content-type','').lower().startswith('multipart/form-data;'):
        raise Rejection(400,'Send a multipart image upload.')
    result=bytearray()
    try:
        async with asyncio.timeout(10):
            async for chunk in request.stream():
                if len(result)+len(chunk)>BODY_LIMIT:
                    raise Rejection(413,'Upload exceeds the size limit.')
                result.extend(chunk)
    except (TimeoutError,ClientDisconnect) as exc:
        raise Rejection(400,'Upload did not complete.') from exc
    return bytes(result)


def _result(payload:bytes)->dict:
    value=json.loads(payload)
    if not isinstance(value,dict):
        raise ValueError('Invalid result')
    valid=((value.get('status')=='recognized' and value.get('character') in SYMBOLS) or
           (value.get('status')=='unrecognized' and value.get('character') is None))
    if not valid:
        raise ValueError('Invalid result')
    for name in ('request_id','model_version'):
        if not isinstance(value.get(name),str) or not 1<=len(value[name])<=200:
            raise ValueError('Invalid result')
    duration=value.get('processing_ms')
    if isinstance(duration,bool) or not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<0:
        raise ValueError('Invalid result')
    return {key:value[key] for key in ('request_id','status','character','model_version','processing_ms')}


def create_app(token_file:Path,api_port:int=8000,client_port:int=8080,transport=None)->FastAPI:
    """A fixed loopback bridge. Injected HTTP transport exists for local tests only."""
    for port in (api_port,client_port):
        if type(port) is not int or not 1<=port<=65535:
            raise ValueError('Ports must be integers from 1 to 65535.')
    if api_port==client_port:
        raise ValueError('API and client ports must differ.')
    origins={f'http://127.0.0.1:{client_port}',f'http://localhost:{client_port}'}
    hosts={f'127.0.0.1:{client_port}',f'localhost:{client_port}'}
    upstream=f'http://127.0.0.1:{api_port}/v1/tamil-symbols/recognize'
    token=_credential(token_file)
    admission=threading.BoundedSemaphore(20)

    @asynccontextmanager
    async def lifespan(application):
        # No proxy environment or redirect may send backend credentials elsewhere.
        async with httpx.AsyncClient(transport=transport,trust_env=False,follow_redirects=False,
                                     timeout=12,limits=httpx.Limits(max_connections=20)) as client:
            application.state.upstream=client
            yield

    app=FastAPI(lifespan=lifespan,docs_url=None,redoc_url=None,openapi_url=None)

    @app.middleware('http')
    async def local_boundary(request:Request,call_next):
        # Host checks resist DNS rebinding; Origin protects the credential-bearing
        # bridge against a malicious page POSTing to a local TCP listener.
        if (request.client is None or request.client.host not in {'127.0.0.1','::1'} or
                request.headers.get('host') not in hosts):
            return error(403,'This demonstration accepts loopback requests only.')
        if request.method not in {'GET','HEAD'} and request.headers.get('origin') not in origins:
            return error(403,'Use the drawing page served by this local demonstration.')
        response=await call_next(request)
        response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        return response

    @app.get('/')
    async def index():
        return FileResponse(CLIENT_ROOT/'index.html')

    async def forward(body:bytes,content_type:str):
        async with app.state.upstream.stream('POST',upstream,content=body,
                       headers={'Authorization':'Bearer '+token,'Content-Type':content_type,'Accept':'application/json'}) as response:
            if response.status_code!=200:
                status=response.status_code if response.status_code in {400,413,415,422,429,503} else 503
                headers={}
                if status==429:
                    retry=response.headers.get('retry-after','1')
                    headers['Retry-After']=retry if retry.isdigit() and 1<=int(retry)<=60 else '1'
                return error(status,'Recognition unavailable. Please try again.' if status>=429 else 'The image upload could not be recognized.',headers)
            payload=bytearray()
            async for chunk in response.aiter_bytes():
                if len(payload)+len(chunk)>RESPONSE_LIMIT:
                    raise ValueError('Oversized upstream result')
                payload.extend(chunk)
            return JSONResponse(_result(bytes(payload)),headers={'Cache-Control':'no-store'})

    @app.post('/api/tamil-symbols/recognize')
    async def recognize(request:Request):
        if not admission.acquire(blocking=False):
            return error(429,'Recognition capacity is full. Retry shortly.',{'Retry-After':'1'})
        forwarding=watcher=None
        try:
            body=await _body(request)
            forwarding=asyncio.create_task(forward(body,request.headers['content-type']))
            async def disconnected():
                while not await request.is_disconnected():
                    await asyncio.sleep(.05)
            watcher=asyncio.create_task(disconnected())
            done,_=await asyncio.wait({forwarding,watcher},return_when=asyncio.FIRST_COMPLETED)
            if forwarding in done:
                return await forwarding
            return error(400,'The drawing request was cancelled.')
        except Rejection as exc:
            return error(exc.status,exc.message)
        except (httpx.HTTPError,ValueError,KeyError,TypeError):
            # Infrastructure, credential, payload, and model details stay backend-only.
            return error(503,'Recognition unavailable. Please try again.')
        finally:
            for task in (forwarding,watcher):
                if task is not None and not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
            admission.release()

    @app.get('/app.mjs')
    async def script():
        return FileResponse(CLIENT_ROOT/'app.mjs',media_type='text/javascript')

    app.mount('/component',StaticFiles(directory=CLIENT_ROOT/'component',follow_symlink=False),name='component')
    return app


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--token-file',type=Path,required=True)
    parser.add_argument('--api-port','--server-port',dest='api_port',type=int,default=8000)
    parser.add_argument('--port','--client-port',dest='client_port',type=int,default=8080)
    options=parser.parse_args()
    import uvicorn
    # Fixed loopback listener: production hosts must supply their own authenticated
    # backend. Disable access logs; neither uploads nor tokens are logged here.
    uvicorn.run(create_app(options.token_file,options.api_port,options.client_port),host='127.0.0.1',port=options.client_port,access_log=False)


if __name__=='__main__':
    main()
