# vm-gateway

What it does
- A minimal Express-based API gateway that forwards requests to backend VM services.
- Endpoints:
  - `GET /api/service1` → forwards to Service 1
  - `GET /api/service2` → forwards to Service 2

Configuration
- Service endpoints are configurable via environment variables. See `.env.example` for VirtualBox host-only example IPs.

Install

```bash
npm install
```

Run (local test pointing to local services)

```bash
# example: run gateway locally while services run on localhost
SERVICE1_HOST=127.0.0.1 SERVICE1_PORT=3001 SERVICE2_HOST=127.0.0.1 SERVICE2_PORT=3002 npm start
```

Run (using .env file)

```bash
# copy .env.example to .env and edit IPs if needed
cp .env.example .env
# then start the gateway in an environment that loads .env (e.g. using direnv or systemd)
npm start
```

Example requests

```bash
curl http://localhost:3000/api/service1
curl http://localhost:3000/api/service2
```

Notes
- The gateway intentionally contains no business logic: it merely forwards and relays JSON responses.
- It uses the global `fetch` API (available in recent Node.js releases). Ensure Node LTS >= 18.