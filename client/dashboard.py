import asyncio
import html
import json
import time
from typing import Any

import cv2
from aiohttp import web


class DashboardServer:
    def __init__(self, config: dict[str, Any]):
        self.enabled = bool(config["enabled"])
        self.host = str(config["host"])
        self.port = int(config["port"])
        self.path = str(config["path"])
        self.stream_path = str(config["stream_path"])
        self.map_stream_path = str(config["map_stream_path"])
        self.stats_path = str(config["stats_path"])
        self.health_path = str(config["health_path"])
        self.stats_poll_ms = int(config["stats_poll_ms"])
        self.jpeg_quality = int(config["jpeg_quality"])
        self.window_title = str(config["window_title"])

        self._condition = asyncio.Condition()
        self._latest_jpegs: dict[str, bytes | None] = {"camera": None, "map": None}
        self._frame_seqs: dict[str, int] = {"camera": 0, "map": 0}
        self._stats: dict[str, Any] = {
            "client": {},
            "server": {},
            "webrtc": {},
            "ros2": {},
            "robots": [],
            "updated_at_unix_ns": time.time_ns(),
        }
        self._runner: web.AppRunner | None = None

    async def start(self):
        if not self.enabled:
            return
        app = web.Application()
        app.router.add_get(self.path, self._index)
        app.router.add_get(self.stream_path, self._camera_stream)
        app.router.add_get(self.map_stream_path, self._map_stream)
        app.router.add_get(self.stats_path, self._stats_handler)
        app.router.add_get(self.health_path, self._healthz)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()

    async def stop(self):
        if self._runner is not None:
            await self._runner.cleanup()

    async def update_frame(self, frame):
        await self._update_stream_frame("camera", frame)

    async def update_map_frame(self, frame):
        await self._update_stream_frame("map", frame)

    async def _update_stream_frame(self, stream_name: str, frame):
        if not self.enabled:
            return
        ok, jpg = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return
        async with self._condition:
            self._latest_jpegs[stream_name] = jpg.tobytes()
            self._frame_seqs[stream_name] += 1
            self._condition.notify_all()

    async def update_stats(self, stats: dict[str, Any]):
        if not self.enabled:
            return
        async with self._condition:
            self._stats = stats
            self._stats["updated_at_unix_ns"] = time.time_ns()

    async def _index(self, request):
        return web.Response(
            text=self._html(),
            content_type="text/html",
            headers={"Cache-Control": "no-store"},
        )

    async def _healthz(self, request):
        return web.json_response({"ok": True})

    async def _stats_handler(self, request):
        return web.json_response(
            self._stats,
            dumps=lambda payload: json.dumps(payload, separators=(",", ":")),
            headers={"Cache-Control": "no-store"},
        )

    async def _camera_stream(self, request):
        return await self._stream(request, "camera")

    async def _map_stream(self, request):
        return await self._stream(request, "map")

    async def _stream(self, request, stream_name: str):
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-store",
                "Pragma": "no-cache",
            },
        )
        await response.prepare(request)
        last_seq = -1
        try:
            while True:
                async with self._condition:
                    await self._condition.wait_for(
                        lambda: self._latest_jpegs[stream_name] is not None
                        and self._frame_seqs[stream_name] != last_seq
                    )
                    jpg = self._latest_jpegs[stream_name]
                    last_seq = self._frame_seqs[stream_name]

                await response.write(
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
                    + jpg
                    + b"\r\n"
                )
        except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
            return response

    def _html(self):
        title = html.escape(self.window_title)
        stats_path = html.escape(self.stats_path)
        stream_path = html.escape(self.stream_path)
        map_stream_path = html.escape(self.map_stream_path)
        stats_poll_ms = int(self.stats_poll_ms)
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #101217;
      --panel: #181c23;
      --panel-2: #202630;
      --line: #303846;
      --text: #edf2f7;
      --muted: #9aa6b2;
      --green: #4fd18b;
      --amber: #ffc857;
      --red: #ff6b6b;
      --blue: #65a9ff;
      --cyan: #5eead4;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .shell {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 380px;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 14px;
      padding: 14px;
      min-height: 100vh;
    }}
    .topbar {{
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      align-items: center;
      gap: 12px;
      padding: 9px 12px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
    }}
    .feed-panel, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .feed-panel {{
      display: grid;
      grid-template-rows: minmax(0, 1fr);
      min-width: 0;
    }}
    h1, h2 {{
      margin: 0;
      font-size: 15px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      padding: 5px 9px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--panel-2);
      color: var(--muted);
      white-space: nowrap;
    }}
    .dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--muted);
    }}
    .ok .dot {{ background: var(--green); }}
    .warn .dot {{ background: var(--amber); }}
    .bad .dot {{ background: var(--red); }}
    .top-stats {{
      min-width: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 18px;
      overflow: hidden;
    }}
    .stat-line {{
      min-width: 0;
      display: flex;
      align-items: baseline;
      gap: 10px;
      white-space: nowrap;
    }}
    .stat-title {{
      color: var(--cyan);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
    }}
    .metric {{
      display: inline-flex;
      align-items: baseline;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
    }}
    .metric strong {{
      color: var(--text);
      font-size: 13px;
      font-weight: 760;
    }}
    .feed-wrap {{
      min-height: 0;
      display: grid;
      place-items: center;
      background: #080a0f;
    }}
    .feed {{
      width: 100%;
      height: 100%;
      max-height: calc(100vh - 72px);
      object-fit: contain;
      display: block;
    }}
    .side {{
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 14px;
      min-width: 0;
    }}
    .map-wrap {{
      display: grid;
      place-items: center;
      padding: 10px;
      background: #eef2f5;
    }}
    .map-feed {{
      width: 100%;
      aspect-ratio: 7 / 5;
      object-fit: contain;
      display: block;
    }}
    .panel h2 {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #151922;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      background: #151922;
    }}
    .robots {{
      overflow: auto;
    }}
    .state-ok {{ color: var(--green); }}
    .state-warn {{ color: var(--amber); }}
    .state-bad {{ color: var(--red); }}
    @media (max-width: 1100px) {{
      .shell {{ grid-template-columns: 1fr; }}
      .topbar {{ grid-template-columns: 1fr; }}
      .top-stats {{ justify-content: flex-start; flex-wrap: wrap; }}
      .pills {{ justify-content: flex-start; }}
      .side {{ grid-template-rows: auto auto; }}
      .feed {{ max-height: none; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <h1>{title}</h1>
      <div class="top-stats">
        <div class="stat-line">
          <span class="stat-title">Client</span>
          <span class="metric">Camera <strong id="camera-fps">0.0</strong></span>
          <span class="metric">Frame <strong id="frame-id">0</strong></span>
          <span class="metric">Sent <strong id="frames-sent">0</strong></span>
          <span class="metric">Dropped <strong id="frames-dropped">0</strong></span>
        </div>
        <div class="stat-line">
          <span class="stat-title">Server</span>
          <span class="metric">Infer <strong id="infer-ms">0 ms</strong></span>
          <span class="metric">E2E <strong id="e2e-ms">0 ms</strong></span>
          <span class="metric">Objects <strong id="objects">0</strong></span>
          <span class="metric">Contours <strong id="contours">0</strong></span>
        </div>
      </div>
      <div class="pills">
        <span id="webrtc-pill" class="pill warn"><span class="dot"></span><span>WebRTC</span></span>
        <span id="ros2-pill" class="pill warn"><span class="dot"></span><span>ROS2</span></span>
        <span id="cal-pill" class="pill warn"><span class="dot"></span><span>Calibration</span></span>
      </div>
    </header>
    <section class="feed-panel">
      <div class="feed-wrap">
        <img class="feed" src="{stream_path}" alt="Live annotated camera feed">
      </div>
    </section>
    <aside class="side">
      <section class="panel">
        <h2>Map</h2>
        <div class="map-wrap">
          <img class="map-feed" src="{map_stream_path}" alt="Live field map">
        </div>
      </section>
      <section class="panel robots">
        <h2>Robots</h2>
        <table>
          <thead>
            <tr><th>Name</th><th>Pose</th><th>Age</th><th>ROS2</th></tr>
          </thead>
          <tbody id="robot-rows"></tbody>
        </table>
      </section>
    </aside>
  </main>
  <script>
    const statsPath = "{stats_path}";
    const $ = (id) => document.getElementById(id);
    function fmt(value, digits = 1) {{
      const number = Number(value || 0);
      return number.toFixed(digits);
    }}
    function pill(id, state, text) {{
      const el = $(id);
      el.classList.remove("ok", "warn", "bad");
      el.classList.add(state);
      el.querySelector("span:last-child").textContent = text;
    }}
    function stateClass(ok, warn = false) {{
      if (ok) return "ok";
      return warn ? "warn" : "bad";
    }}
    async function refresh() {{
      try {{
        const response = await fetch(statsPath, {{ cache: "no-store" }});
        const stats = await response.json();
        const client = stats.client || {{}};
        const server = stats.server || {{}};
        const webrtc = stats.webrtc || {{}};
        const ros2 = stats.ros2 || {{}};
        $("camera-fps").textContent = fmt(client.camera_fps);
        $("frame-id").textContent = client.frame_id ?? 0;
        $("frames-sent").textContent = webrtc.frames_sent ?? 0;
        $("frames-dropped").textContent = webrtc.frames_dropped ?? 0;
        $("infer-ms").textContent = `${{fmt(server.inference_ms)}} ms`;
        $("e2e-ms").textContent = `${{fmt(server.e2e_ms)}} ms`;
        $("objects").textContent = server.objects ?? 0;
        $("contours").textContent = server.contours ?? 0;

        const rtcState = webrtc.connection_state || "new";
        pill("webrtc-pill", stateClass(rtcState === "connected", rtcState === "connecting" || rtcState === "new"), `WebRTC ${{rtcState}}`);
        const rosState = ros2.state || "off";
        pill("ros2-pill", stateClass(rosState === "running", rosState === "disabled"), `ROS2 ${{rosState}}`);
        const calibrated = Boolean(client.calibrated);
        pill("cal-pill", stateClass(calibrated, true), calibrated ? "Calibration locked" : "Calibration waiting");

        const rows = (stats.robots || []).map((robot) => {{
          const visibleClass = robot.visible ? "state-ok" : "state-warn";
          const pose = robot.visible
            ? `${{fmt(robot.x, 2)}}, ${{fmt(robot.y, 2)}}, ${{fmt(robot.theta_deg, 0)}} deg`
            : "not visible";
          const age = robot.last_seen_age_ms == null ? "n/a" : `${{fmt(robot.last_seen_age_ms, 0)}} ms`;
          return `<tr>
            <td><span class="${{visibleClass}}">${{robot.name}}</span></td>
            <td>${{pose}}</td>
            <td>${{age}}</td>
            <td>${{robot.ros_namespace || ""}}</td>
          </tr>`;
        }}).join("");
        $("robot-rows").innerHTML = rows || `<tr><td colspan="4">No configured robots</td></tr>`;
      }} catch (err) {{
        pill("webrtc-pill", "bad", "Stats offline");
      }}
    }}
    refresh();
    setInterval(refresh, {stats_poll_ms});
  </script>
</body>
</html>"""
