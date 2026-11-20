"""Local adapter contract; inference responses here are explicit transport fixtures."""
import importlib.util
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

spec=importlib.util.spec_from_file_location('demo_host',Path(__file__).parents[1]/'serve.py')
demo=importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)
ORIGIN='http://127.0.0.1:8080'


def token_file(tmp_path):
    path=tmp_path/'credential'
    path.write_text('a-test-service-secret-with-at-least-32-characters')
    path.chmod(0o600)
    return path


def test_proxy_keeps_credential_backend_only_and_serves_ui(tmp_path,caplog):
    token=token_file(tmp_path)
    calls=[]
    def upstream(request):
        calls.append(request)
        assert request.url==httpx.URL('http://127.0.0.1:8000/v1/tamil-symbols/recognize')
        assert request.headers['authorization']=='Bearer '+token.read_text()
        assert 'cookie' not in request.headers
        return httpx.Response(200,json=dict(request_id='request-1',status='recognized',character='அ',model_version='model-1',processing_ms=2.5,extra_secret='filtered'))
    app=demo.create_app(token,transport=httpx.MockTransport(upstream))
    with TestClient(app,base_url=ORIGIN,client=('127.0.0.1',1234)) as client:
        page=client.get('/')
        assert page.status_code==200
        assert token.read_text() not in page.text
        result=client.post('/api/tamil-symbols/recognize',headers={'Origin':ORIGIN,'Cookie':'private=user'},files={'image':('drawing.png',b'unique-private-image','image/png')})
        assert result.status_code==200
        assert result.json()['character']=='அ'
        assert 'extra_secret' not in result.json()
        assert token.read_text() not in result.text
    assert len(calls)==1
    assert b'unique-private-image' in calls[0].content
    assert token.read_text() not in caplog.text
    assert 'unique-private-image' not in caplog.text


def test_cross_origin_nonloopback_host_and_oversized_body_never_reach_service(tmp_path):
    calls=[]
    app=demo.create_app(token_file(tmp_path),transport=httpx.MockTransport(lambda request:calls.append(request)))
    with TestClient(app,base_url=ORIGIN,client=('127.0.0.1',1234)) as client:
        assert client.post('/api/tamil-symbols/recognize',headers={'Origin':'https://other.example'},content=b'x').status_code==403
        assert client.post('/api/tamil-symbols/recognize',headers={'Origin':ORIGIN,'Host':'rebind.example:8080'},content=b'x').status_code==403
        result=client.post('/api/tamil-symbols/recognize',headers={'Origin':ORIGIN,'Content-Type':'multipart/form-data; boundary=x'},content=b'x'*(demo.BODY_LIMIT+1))
        assert result.status_code==413
        # Streamed uploads have no Content-Length; the same bound still applies.
        result=client.post('/api/tamil-symbols/recognize',headers={'Origin':ORIGIN,'Content-Type':'multipart/form-data; boundary=x'},content=iter([b'x'*600000,b'x'*600000]))
        assert result.status_code==413
    assert not calls


@pytest.mark.parametrize('upstream_status,expected',[(401,503),(429,429),(422,422)])
def test_upstream_errors_are_safe_and_busy_retry_is_preserved(tmp_path,upstream_status,expected):
    def upstream(request):
        return httpx.Response(upstream_status,headers={'Retry-After':'2'},json={'detail':'internal secret path'})
    app=demo.create_app(token_file(tmp_path),transport=httpx.MockTransport(upstream))
    with TestClient(app,base_url=ORIGIN,client=('127.0.0.1',1234)) as client:
        result=client.post('/api/tamil-symbols/recognize',headers={'Origin':ORIGIN},files={'image':('drawing.png',b'x','image/png')})
    assert result.status_code==expected
    assert 'internal secret path' not in result.text
    if expected==429:
        assert result.headers['Retry-After']=='2'


def test_invalid_or_oversized_upstream_result_fails_closed(tmp_path):
    def upstream(request):
        return httpx.Response(200,content=b'x'*(demo.RESPONSE_LIMIT+1))
    app=demo.create_app(token_file(tmp_path),transport=httpx.MockTransport(upstream))
    with TestClient(app,base_url=ORIGIN,client=('127.0.0.1',1234)) as client:
        result=client.post('/api/tamil-symbols/recognize',headers={'Origin':ORIGIN},files={'image':('drawing.png',b'x','image/png')})
    assert result.status_code==503


def test_cancelled_host_request_cancels_upstream_transport(tmp_path):
    import asyncio
    from contextlib import suppress

    async def scenario():
        entered=asyncio.Event()
        cancelled=asyncio.Event()
        async def upstream(request):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        app=demo.create_app(token_file(tmp_path),transport=httpx.MockTransport(upstream))
        async with app.router.lifespan_context(app):
            transport=httpx.ASGITransport(app=app,client=('127.0.0.1',1234))
            async with httpx.AsyncClient(transport=transport,base_url=ORIGIN) as client:
                request=asyncio.create_task(client.post('/api/tamil-symbols/recognize',headers={'Origin':ORIGIN},files={'image':('drawing.png',b'x','image/png')}))
                await asyncio.wait_for(entered.wait(),1)
                request.cancel()
                with suppress(asyncio.CancelledError):
                    await request
                await asyncio.wait_for(cancelled.wait(),1)
    asyncio.run(scenario())


def test_configurable_ports_apply_to_proxy_origin_and_static_assets(tmp_path):
    calls=[]
    def upstream(request):
        calls.append(request)
        return httpx.Response(200,json=dict(request_id='configured',status='unrecognized',character=None,model_version='experimental',processing_ms=3))
    app=demo.create_app(token_file(tmp_path),api_port=9100,client_port=9180,transport=httpx.MockTransport(upstream))
    origin='http://127.0.0.1:9180'
    with TestClient(app,base_url=origin,client=('127.0.0.1',1234)) as client:
        assert client.get('/').status_code==200
        assert client.get('/app.mjs').status_code==200
        assert client.get('/component/index.js').status_code==200
        assert client.get('/serve.py').status_code==404
        assert client.get('/server/model.onnx').status_code==404
        response=client.post('/api/tamil-symbols/recognize',headers={'Origin':origin},files={'image':('drawing.png',b'x','image/png')})
        assert response.status_code==200
        assert client.post('/api/tamil-symbols/recognize',headers={'Origin':ORIGIN},files={'image':('drawing.png',b'x','image/png')}).status_code==403
    assert len(calls)==1
    assert calls[0].url.port==9100


@pytest.mark.parametrize('api_port,client_port',[(0,8080),(8000,65536),(8000,8000),(True,8080)])
def test_invalid_ports_rejected_before_listener_creation(tmp_path,api_port,client_port):
    with pytest.raises(ValueError):
        demo.create_app(token_file(tmp_path),api_port=api_port,client_port=client_port)
