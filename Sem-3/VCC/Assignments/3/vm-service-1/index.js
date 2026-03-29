// vm-service-1 - minimal Express service
// Purpose: expose GET /service1 returning service name, hostname, and timestamp

const express = require('express');
const os = require('os');

// Helper: find first non-internal IPv4 address for this host.
function getHostIp() {
  const nets = os.networkInterfaces();
  for (const name of Object.keys(nets)) {
    for (const net of nets[name]) {
      // prefer IPv4 and non-internal addresses
      if (net.family === 'IPv4' && !net.internal) {
        return net.address;
      }
    }
  }
  return '127.0.0.1';
}

const app = express();
const PORT = process.env.PORT || 3001;

app.get('/service1', (req, res) => {
  const payload = {
    service: 'vm-service-1',
    hostname: os.hostname(),
    hostIp: getHostIp(),
    timestamp: new Date().toISOString()
  };
  res.json(payload);
});

app.listen(PORT, () => {
  console.log(`vm-service-1 listening on port ${PORT}`);
});