#!/usr/bin/env bash
# Expose the CoTracker3 MCP server (streamable HTTP on $LOCAL_PORT) through a
# TunnelMate tunnel. Run this AFTER starting cot_mcp_server.py.
#
# Usage: bash cot_mcp/tunnel.sh
# Env overrides: LOCAL_PORT (default 8004), BROKER, BROKER_HOST, BROKER_CTRL_PORT

set -euo pipefail

BROKER="${BROKER:-http://163.61.236.112}"
BROKER_HOST="${BROKER_HOST:-163.61.236.112}"
BROKER_CTRL_PORT="${BROKER_CTRL_PORT:-7000}"
LOCAL_HOST="${LOCAL_HOST:-127.0.0.1}"
LOCAL_PORT="${LOCAL_PORT:-8004}"
ARCH="$(uname -m)"
STATE_DIR="${COT_TUNNEL_STATE_DIR:-/root/cot-mcp-tunnel}"

mkdir -p "$STATE_DIR"

echo "[tunnel] Ensuring tunnelmate binaries..."
if ! command -v tunnelmate-agent >/dev/null 2>&1; then
    curl -fsSL "$BROKER/v1/download/tunnelmate-0.1.0-linux-$ARCH.tar.gz" \
        | tar xz -C /tmp
    install -m 0755 "/tmp/tunnelmate-0.1.0-linux-$ARCH/tunnelmate-agent" \
        "/tmp/tunnelmate-0.1.0-linux-$ARCH/tunnelmate-peer" /usr/local/bin/
fi

echo "[tunnel] Fetching broker certificate..."
CA_PATH=""
if curl -fsS "$BROKER/v1/broker-certificate" -o "$STATE_DIR/broker.crt"; then
    CA_PATH="$STATE_DIR/broker.crt"
fi

if [[ -f "$STATE_DIR/tunnel.json" ]] && [[ -f "$STATE_DIR/agent.conf" ]]; then
    echo "[tunnel] Reusing existing tunnel state"
else
    echo "[tunnel] Creating tunnel on broker..."
    RESPONSE="$(curl -fsS -X POST "$BROKER/v1/tunnels" \
        -H 'Content-Type: application/json' \
        -d '{"scope": "open", "protocol": "tcp"}')"
    echo "$RESPONSE" > "$STATE_DIR/tunnel.json"
    chmod 600 "$STATE_DIR/tunnel.json"
fi

TUNNEL_ID="$(python3 -c "import json;print(json.load(open('$STATE_DIR/tunnel.json'))['tunnel_id'])")"
AGENT_SECRET="$(python3 -c "import json;print(json.load(open('$STATE_DIR/tunnel.json'))['agent_secret'])")"
PUBLIC_PORT="$(python3 -c "import json;print(json.load(open('$STATE_DIR/tunnel.json'))['public_port'])")"

cat > "$STATE_DIR/agent.conf" <<EOF
agent.tunnel_id = $TUNNEL_ID
agent.agent_secret = $AGENT_SECRET
agent.local_host = $LOCAL_HOST
agent.local_port = $LOCAL_PORT
agent.broker_host = $BROKER_HOST
agent.broker_port = $BROKER_CTRL_PORT
agent.protocol = tcp
agent.verify_ca = true
agent.ca_path = $CA_PATH
agent.log_level = info
EOF
chmod 600 "$STATE_DIR/agent.conf"

echo "[tunnel] Starting agent (public: tcp://$BROKER_HOST:$PUBLIC_PORT)"
echo "[tunnel] MCP endpoint: http://$BROKER_HOST:$PUBLIC_PORT/mcp"
pkill -f tunnelmate-agent >/dev/null 2>&1 || true
sleep 1
tunnelmate-agent -c "$STATE_DIR/agent.conf"