# Cloud API Deployment

The GitHub Pages site is static. Deploy `backend/` separately as a CPU API and
set `VITE_API_URL` to its fixed HTTPS address before building the frontend.

## Oracle Cloud Always Free

1. Create an Always Free Ubuntu VM and allow inbound TCP 80/443 in its security
   list. Oracle registration may require a payment card for identity validation,
   but the eligible VM has no monthly charge within free-tier limits.
2. Install Docker and clone this repository on the VM.
3. Build and run the API from the repository root:

```bash
docker build -f backend/Dockerfile -t 4g-traffic-api .
docker run -d --restart unless-stopped --name 4g-traffic-api -p 8000:8000 4g-traffic-api
```

4. Put Caddy or Nginx in front of port 8000 to provide HTTPS at a fixed domain.
   Verify `https://api.example.com/api/health` before changing the frontend.
5. Copy `react-app/.env.production.example` to the untracked production env
   file, set the real API URL, run `npm.cmd run build`, then publish `docs/` or
   the configured Pages build output.

## Scope and limits

- The CPU API serves validated benchmark curves and generic CPU forecasting.
- Full 23-model retraining remains on the local GPU workstation; do not expose
  its training endpoint publicly until a queue, authentication, and rate limits
  are added.
- Keep model checkpoints and large experimental outputs out of the container
  unless they are needed for a reviewed 4G inference model.
