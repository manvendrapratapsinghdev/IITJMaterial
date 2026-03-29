# vm-service-1

What it does
- A minimal Express-based microservice intended to run inside a VM.
- Exposes a single endpoint: `GET /service1` which returns service metadata.

Install

```bash
npm install
```

Run

```bash
npm start
```

Example request

```bash
curl http://localhost:3001/service1
```

Response format (JSON)

```json
{
  "service": "vm-service-1",
  "hostname": "<vm-hostname>",
  "hostIp": "<host-ip-address>",
  "timestamp": "2026-01-27T...Z"
}
```