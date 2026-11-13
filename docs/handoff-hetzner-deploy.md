# Handoff: run the Splat Studio interface on a Hetzner server

## Goal

Run `space/app.py` (the Gradio UI) on a private Hetzner VPS (2 vCPU) instead of, or next to, the Hugging Face Space. Reach it over HTTPS with a password. GPU work stays on Modal and storage stays in the private HF dataset. Neither changes.

## Is 2 vCPU enough? Yes, if the server has at least 4 GB RAM

The server does no GPU work. Reconstruction, text-to-object (FLUX + TRELLIS) and restyling run as Modal functions (`modal_app/app.py`). The UI only calls them through `src/splat_studio/modal_client.py`.

What does run on the server, and what it costs:

| Work | Where | Cost |
|---|---|---|
| Gradio UI, job polling | `space/app.py`, `space/jobs.py` | Light. 2 vCPU is plenty. |
| Upload checks (`ffprobe` on the video) | `space/validation.py` | Under a second per upload. |
| Upload to and download from the HF dataset | `HubStore` in `space/jobs.py` | Network-bound. Videos up to 500 MB. |
| Merging a generated object into a scene | `editing/objects.py: merge_files` | **Memory-heavy.** It loads the whole scene PLY into numpy. A real scene was 367 MB, so expect a 1–1.5 GB peak. |
| Viewer point sampling and the preview camera | `space/viewer.py`, `editing/preview.py` | Moderate. It reads the `.splat` file once per load. |

Limits:

- **RAM:** the server has 4 GB but already runs other services. What matters is the memory still free for Splat Studio: about 300–500 MB when idle, plus up to 1.5 GB during an object merge, so roughly **2 GB of headroom**. Measure it first (step 0). If less than 2 GB is free, add swap (step 2). The swap file then catches the merge peak instead of the kernel killing a process. The OOM killer could pick one of the *other* services, not Splat Studio.
- **Disk:** `scene_cache/` keeps a full copy of every scene version it downloads, at about 370 MB each. The 40 GB default disk fills after roughly 80–100 versions. Step 7 adds a cleanup job.
- **Concurrency:** expect slowdowns with more than 2–3 people editing at the same moment. That is fine for private use.

## Before starting

Facts the person doing this needs:

1. The Hetzner plan's RAM and disk size (`free -h`, `df -h` on the server).
2. A domain or subdomain pointing at the server's IPv4 address (A record). Without one, use plain `http://<ip>`; step 5 explains the trade-off.
3. The secrets (they come from the owner's accounts, never from the repo): `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `HF_TOKEN` (write access), `SPLAT_STUDIO_DATASET`.
4. `modal deploy modal_app/app.py` must already have run from the dev machine. The server only calls the deployed functions.

The working tree has uncommitted work (prompt log, preview, viewer changes), and `openspec/changes/redesign-space-gui` is planned. Deploy from a committed, pushed state. Otherwise the server runs something different from the local copy.

## Steps

Assumes Ubuntu 24.04 and root SSH access.

### 0. Measure what the existing services use

The server already runs other things, so check the headroom before installing anything:

```bash
free -h                                   # look at the "available" column
ps aux --sort=-rss | head -15             # biggest memory users
swapon --show                             # existing swap, if any
df -h /                                   # free disk
ss -tlnp                                  # ports in use (7860, 80, 443?)
```

- **available ≥ 2.5 GB:** fine as is. Add swap anyway (step 2) as a safety net.
- **available 1.5–2.5 GB:** works, but swap (step 2) is required. Object merges on large scenes will be slower while they use swap.
- **available < 1.5 GB:** the merge will hit swap hard. Consider a larger plan, or ask for the code change listed under *Open questions*.

If ports 80/443 are already taken by another web server (nginx, Apache, Caddy), do not install a second one in step 5. Add a site/virtual host for Splat Studio to the existing server instead, proxying to `127.0.0.1:7860` with websocket support and basic auth. If 7860 is taken, set `GRADIO_SERVER_PORT=7861` in the service file (step 6) and in the proxy.

### 1. Base packages and a service user

```bash
apt update && apt -y upgrade
apt -y install python3 python3-venv python3-pip ffmpeg git ufw
adduser --system --group --home /opt/splat splat
```

Ubuntu 24.04 ships Python 3.12. The project needs 3.11 or newer, so this version works.

### 2. Swap (recommended on this shared server)

Skip if `swapon --show` already lists at least 2 GB of swap.

```bash
fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

### 3. Code and Python environment

```bash
sudo -u splat git clone <repo-url> /opt/splat/app
cd /opt/splat/app
sudo -u splat python3 -m venv /opt/splat/venv
sudo -u splat /opt/splat/venv/bin/pip install -r space/requirements.txt
```

`space/app.py` adds `src/` to `sys.path` itself, so the package does not need to be installed. If the repo is private, use a deploy key or a GitHub fine-grained token with read-only access.

### 4. Secrets

Create `/opt/splat/app/.env`. The app reads it at startup (`load_env_file` in `space/app.py`).

```
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
HF_TOKEN=...
SPLAT_STUDIO_DATASET=<owner>/splat-studio-data
```

```bash
chown splat:splat /opt/splat/app/.env && chmod 600 /opt/splat/app/.env
```

Without `SPLAT_STUDIO_DATASET`, the app falls back to `./local_data`. Reconstructions cannot finish in that mode, so check that this variable is set.

### 5. Reverse proxy with HTTPS and a password (Caddy)

**The app has no login of its own.** Anyone who reaches it can start paid Modal GPU jobs. Do not expose port 7860 directly.

Gradio listens on `127.0.0.1:7860` by default (`launch()` sets no `server_name`). Caddy sits in front of it and adds HTTPS and basic auth:

```bash
apt -y install caddy
caddy hash-password   # enter the password; copy the hash it prints
```

`/etc/caddy/Caddyfile`:

```
splat.example.com {
    basic_auth {
        robin <hash-from-above>
    }
    reverse_proxy 127.0.0.1:7860
}
```

```bash
systemctl reload caddy
ufw allow OpenSSH && ufw allow 80 && ufw allow 443 && ufw enable
```

Caddy gets a Let's Encrypt certificate automatically and proxies websockets and server-sent events, which Gradio needs. It sets no upload size limit, so 500 MB videos pass through. Without a domain, write `:80 { ... }` instead. The password is then sent unencrypted, so only do this for a short test.

On Caddy versions older than 2.8, the directive is spelled `basicauth`.

### 6. Run as a service (systemd)

`/etc/systemd/system/splat.service`:

```ini
[Unit]
Description=Splat Studio UI
After=network-online.target
Wants=network-online.target

[Service]
User=splat
WorkingDirectory=/opt/splat/app
ExecStart=/opt/splat/venv/bin/python space/app.py
Restart=on-failure
Environment=GRADIO_ANALYTICS_ENABLED=False
MemoryHigh=1800M
MemoryMax=2200M
OOMScoreAdjust=500

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload && systemctl enable --now splat
journalctl -u splat -f
```

The working directory matters: `local_data/` and `scene_cache/` are relative paths.

The memory lines protect the other services on the server. `MemoryHigh` makes Splat Studio swap and slow down before it crowds them out. `MemoryMax` is a hard cap. `OOMScoreAdjust=500` makes the kernel kill Splat Studio, not the other services, if memory still runs out. If step 0 showed little headroom, lower both limits. A merge that hits the cap fails, and systemd restarts the app.

### 7. Disk cleanup for `scene_cache/`

The HF dataset is the source of truth, so the cache can be deleted. A daily cron job as `splat` that removes cache files older than 7 days:

```
0 4 * * * find /opt/splat/app/scene_cache -type f -mtime +7 -delete
```

### 8. Updating

```bash
cd /opt/splat/app && sudo -u splat git pull
sudo -u splat /opt/splat/venv/bin/pip install -r space/requirements.txt
systemctl restart splat
```

If the Modal functions changed, also run `modal deploy modal_app/app.py` from the dev machine. The server and Modal must run the same version of `jobs.py` and `modal_app/app.py`.

## Checks when done

1. `https://<domain>` asks for the password, then shows the UI.
2. `curl -I http://<ip>:7860` from outside fails (the port is not exposed).
3. Upload a short test video. The job goes from "queued" to "running" (Modal started), and later the viewer shows the scene.
4. Add an object to that scene. The merge succeeds without an OOM: `journalctl -u splat` shows no `Killed`, and `dmesg | grep -i oom` is empty.
5. Reboot the server. The UI comes back by itself.

## Open questions for the owner

- The plan has 4 GB RAM shared with other services. How much is free (step 0)?
- If headroom is tight: make `merge_files` (`editing/objects.py`) and `read_ply` (`reconstruction/export.py`) use less memory by memory-mapping the PLY (`np.memmap`) instead of loading it whole. That would cut the merge peak by about half. It is a code change in this repo, not a server setting.
- Which domain or subdomain should it use?
- Should the HF Space stay up next to the server, or be paused? Both can run at once against the same dataset and Modal app.
- Deploy now, or after the `redesign-space-gui` change lands?
