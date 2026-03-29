// vm-gateway - simple forwarding gateway
// Purpose: expose GET /api/service1 and GET /api/service2 and forward to configured services

const express = require('express');
const os = require('os');

const app = express();
const PORT = process.env.PORT || 3000;

// Read service targets from environment variables. Defaults favour local testing,
// but see .env.example for typical VirtualBox host-only IPs.
const SERVICE1_HOST = process.env.SERVICE1_HOST || '127.0.0.1';
const SERVICE1_PORT = process.env.SERVICE1_PORT || '3001';
const SERVICE2_HOST = process.env.SERVICE2_HOST || '127.0.0.1';
const SERVICE2_PORT = process.env.SERVICE2_PORT || '3002';

async function forwardTo(url, res) {
  try {
    const r = await fetch(url);
    const data = await r.json();
    return res.json(data);
  } catch (err) {
    console.error('Gateway error forwarding to', url, err.message);
    return res.status(502).json({ error: 'Bad Gateway', details: err.message });
  }
}

app.get('/api/service1', (req, res) => {
  const target = `http://${SERVICE1_HOST}:${SERVICE1_PORT}/service1`;
  console.log(`Gateway: forwarding /api/service1 -> ${target}`);
  return forwardTo(target, res);
});

app.get('/api/service2', (req, res) => {
  const target = `http://${SERVICE2_HOST}:${SERVICE2_PORT}/service2`;
  console.log(`Gateway: forwarding /api/service2 -> ${target}`);
  return forwardTo(target, res);
});

app.listen(PORT, () => {
  console.log(`vm-gateway listening on port ${PORT}`);
  console.log(`Targets: service1=${SERVICE1_HOST}:${SERVICE1_PORT}, service2=${SERVICE2_HOST}:${SERVICE2_PORT}`);
});