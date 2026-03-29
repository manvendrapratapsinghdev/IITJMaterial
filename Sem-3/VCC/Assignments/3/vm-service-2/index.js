// vm-service-2 - minimal Express service
// Purpose: expose GET /service2 returning service name, hostname, and timestamp

const express = require('express');
const os = require('os');

// Helper: find first non-internal IPv4 address for this host.
function getHostIp() {
  const nets = os.networkInterfaces();
  for (const name of Object.keys(nets)) {
    for (const net of nets[name]) {
      if (net.family === 'IPv4' && !net.internal) {
        return net.address;
      }
    }
  }
  return '127.0.0.1';
}

const app = express();
const PORT = process.env.PORT || 3002;

app.get('/service2', (req, res) => {
  const payload = {
    service: 'vm-service-2',
    hostname: os.hostname(),
    hostIp: getHostIp(),
    timestamp: new Date().toISOString()
  };
  res.json(payload);
});

app.listen(PORT, () => {
  console.log(`vm-service-2 listening on port ${PORT}`);
});