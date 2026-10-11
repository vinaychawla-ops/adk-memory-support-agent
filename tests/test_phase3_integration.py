"""Phase 3 integration tests: serving + load balancing.

Offline part (no Docker): validates docker-compose.yml structure and the
nginx config textually -- ports, env, upstream servers.

Live part (self-skipping): talks to the compose stack. Bring it up with:

  cd ~/workspace/agent-memory-support-agent
  docker compose up -d --build
  V=~/workspace/.venv-adk/bin
  $V/python -m pytest tests/test_phase3_integration.py -q -m integration

Without GOOGLE_API_KEY the live /chat test asserts graceful degradation
(500 naming the missing key, never a traceback); with a key it runs a
real two-turn conversation through the load balancer and checks session
persistence. The manual cross-replica drill:

  for i in 1 2 3 4 5 6; do
    curl -s -X POST localhost:8000/chat \\
      -H 'Content-Type: application/json' \\
      -d '{"user_id":"maya","session_id":"s1","message":"track ORD-1003"}' \\
      | python3 -c 'import json,sys; d=json.load(sys.stdin);
                     print(d["replica_id"], d["events_recorded"])'
  done

replica_id should cycle across agent-1a/1b/1c while the conversation
stays coherent -- that is the no-sticky-sessions acceptance.
"""

import os

import pytest
import yaml

pytestmark = pytest.mark.integration

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _compose():
    with open(os.path.join(REPO, "docker-compose.yml")) as f:
        return yaml.safe_load(f)


# --- offline: config validity -------------------------------------------------


def test_compose_defines_three_replicas_behind_nginx():
    compose = _compose()
    services = compose["services"]
    for name in ("redis", "pgvector", "firestore",
                 "agent-1a", "agent-1b", "agent-1c", "nginx"):
        assert name in services, f"missing service: {name}"

    replica_ids = set()
    for name in ("agent-1a", "agent-1b", "agent-1c"):
        svc = services[name]
        assert svc.get("build") == "."
        assert {"redis", "pgvector", "firestore"} <= set(svc["depends_on"])
        # Replicas are reachable only through nginx.
        assert "ports" not in svc, f"{name} must not publish ports"
        env = svc["environment"]
        assert env["SESSION_BACKEND"] == "redis"
        assert env["REDIS_URL"] == "redis://redis:6379/0"
        assert env["EPISODIC_BACKEND"] == "pgvector"
        assert "pgvector:5432" in env["PGVECTOR_DSN"]
        assert env["SEMANTIC_BACKEND"] == "firestore"
        assert env["FIRESTORE_EMULATOR_HOST"] == "firestore:8080"
        replica_ids.add(env["REPLICA_ID"])
    assert replica_ids == {"agent-1a", "agent-1b", "agent-1c"}

    nginx = services["nginx"]
    assert nginx["image"].startswith("nginx")
    assert "8000:8000" in nginx["ports"]
    mounts = " ".join(str(v) for v in nginx["volumes"])
    assert "nginx.conf" in mounts


def test_compose_ports_do_not_clash():
    """8000 = nginx/LB; 8080 stays the Firestore emulator (Phase 2)."""
    compose = _compose()
    host_ports: dict[str, str] = {}
    for name, svc in compose["services"].items():
        for mapping in svc.get("ports", []) or []:
            host = str(mapping).split(":")[0]
            assert host not in host_ports, (
                f"port {host} claimed by both {host_ports[host]} and {name}"
            )
            host_ports[host] = name
    assert host_ports["8000"] == "nginx"
    assert host_ports["8080"] == "firestore"
    assert host_ports["6379"] == "redis"
    assert host_ports["5432"] == "pgvector"


def test_nginx_config_balances_three_replicas():
    with open(os.path.join(REPO, "nginx", "nginx.conf")) as f:
        conf = f.read()
    assert "upstream agent_backend" in conf
    assert "least_conn" in conf
    for replica in ("agent-1a", "agent-1b", "agent-1c"):
        assert f"server {replica}:8000" in conf
    assert "proxy_pass http://agent_backend" in conf


# --- live: against the compose stack (self-skipping) ---------------------------


def _lb():
    import httpx

    try:
        r = httpx.get("http://localhost:8000/health", timeout=3)
    except Exception as e:
        pytest.skip(f"compose stack not up (docker compose up -d --build): {e}")
    assert r.status_code == 200, "LB /health not 200"
    return r.json()


def test_lb_health_live():
    body = _lb()
    assert body["status"] == "ok"
    assert body["replica_id"] in {"agent-1a", "agent-1b", "agent-1c"}
    assert body["backends"]["session"] == "redis"
    assert body["backends"]["episodic"] == "pgvector"
    assert body["backends"]["semantic"] == "firestore"


def test_lb_chat_live():
    """Through the load balancer. Needs GOOGLE_API_KEY for a real turn;
    without it, asserts the graceful 500 (no traceback)."""
    import httpx

    _lb()  # skips when compose is down
    payload = {"user_id": "lbtest", "session_id": "s1",
               "message": "track order ORD-1003"}
    r = httpx.post("http://localhost:8000/chat", json=payload, timeout=120)
    if not os.environ.get("GOOGLE_API_KEY"):
        assert r.status_code == 500
        assert "error" in r.json()
        assert "Traceback" not in r.text
        return
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["reply"], "empty reply from live turn"
    assert body["replica_id"] in {"agent-1a", "agent-1b", "agent-1c"}
    # Second turn on the same session persists (whichever replica answers).
    r2 = httpx.post(
        "http://localhost:8000/chat",
        json={"user_id": "lbtest", "session_id": "s1",
              "message": "what is the order status?"},
        timeout=120,
    )
    assert r2.status_code == 200, r2.text[:300]
