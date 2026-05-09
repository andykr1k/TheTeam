import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from common.webrtc_transport import WebRTCSignalingServer
from config import load_config
from server.segmentation import Sam3Segmenter


CONFIG = load_config()
SERVER_CONFIG = CONFIG["server"]
SAM3_CONFIG = SERVER_CONFIG["sam3"]
WEBRTC_CONFIG = SERVER_CONFIG["webrtc"]


class PeerSegmentationSession:
    def __init__(self, peer, segmenter: Sam3Segmenter, max_workers: int):
        self.peer = peer
        self.segmenter = segmenter
        self.session = segmenter.create_session()
        self.latest = None
        self.latest_lock = asyncio.Lock()
        self.stop = asyncio.Event()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.session_frame_id = 0
        self.processed_frames = 0
        self.started_unix_ns = time.time_ns()
        self.task = asyncio.create_task(self._inference_sender())

    async def submit_frame(
        self,
        packet,
        jpg_bytes: bytes,
        recv_unix_ns: int,
        recv_monotonic_ns: int,
    ):
        async with self.latest_lock:
            self.latest = (packet, jpg_bytes, recv_unix_ns, recv_monotonic_ns)

    async def close(self):
        self.stop.set()
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        self.executor.shutdown(wait=False, cancel_futures=True)

    async def _inference_sender(self):
        loop = asyncio.get_running_loop()
        while not self.stop.is_set():
            async with self.latest_lock:
                item = self.latest
                self.latest = None

            if item is None:
                await asyncio.sleep(0.001)
                continue

            packet, jpg_bytes, recv_unix_ns, recv_monotonic_ns = item
            queue_lag_ms = (time.monotonic_ns() - recv_monotonic_ns) / 1_000_000.0
            stats = {
                "peer_id": self.peer.peer_id,
                "processed_frames": self.processed_frames,
                "session_frame_id": self.session_frame_id,
                "queue_lag_ms": queue_lag_ms,
                "uptime_s": (time.time_ns() - self.started_unix_ns) / 1_000_000_000.0,
            }

            infer = partial(
                self.segmenter.infer_jpeg,
                self.session,
                packet,
                self.session_frame_id,
                jpg_bytes,
                recv_unix_ns,
                recv_monotonic_ns,
                stats,
            )
            result = await loop.run_in_executor(self.executor, infer)
            self.session_frame_id += 1
            self.processed_frames += 1
            result.server_stats.update(
                {
                    "processed_frames": self.processed_frames,
                    "session_frame_id": self.session_frame_id,
                }
            )
            await self.peer.send_contours(result)


async def main():
    segmenter = Sam3Segmenter(SAM3_CONFIG)
    segmenter.load()
    sessions: dict[str, PeerSegmentationSession] = {}

    async def on_peer(peer):
        sessions[peer.peer_id] = PeerSegmentationSession(
            peer,
            segmenter,
            int(SAM3_CONFIG["inference_workers"]),
        )
        print(f"webrtc peer connected: {peer.peer_id}")

    async def on_frame(peer, packet, frame_bytes, recv_unix_ns, recv_monotonic_ns):
        session = sessions.get(peer.peer_id)
        if session is None:
            return
        await session.submit_frame(packet, frame_bytes, recv_unix_ns, recv_monotonic_ns)

    async def on_peer_closed(peer):
        session = sessions.pop(peer.peer_id, None)
        if session is not None:
            await session.close()
        print(f"webrtc peer closed: {peer.peer_id}")

    signaling = WebRTCSignalingServer(WEBRTC_CONFIG, on_peer, on_frame, on_peer_closed)
    await signaling.start()
    print(
        "webrtc signaling listening "
        f"http://{WEBRTC_CONFIG['signaling_host']}:{WEBRTC_CONFIG['signaling_port']}"
        f"{WEBRTC_CONFIG['signaling_path']}"
    )
    try:
        await asyncio.Future()
    finally:
        await signaling.stop()


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
