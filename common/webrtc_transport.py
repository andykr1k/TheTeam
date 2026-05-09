import asyncio
import math
import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from aiohttp import ClientSession, web
from aiortc import RTCConfiguration, RTCDataChannel, RTCIceServer, RTCPeerConnection
from aiortc import RTCSessionDescription

from common.messages import ContourPacket, FramePacket, json_loads


CHUNK_HEADER = struct.Struct("!QII")


def ice_servers_from_config(raw_servers: list[dict[str, Any]] | None):
    servers = []
    for raw_server in raw_servers or []:
        urls = raw_server.get("urls")
        if not urls:
            continue
        servers.append(
            RTCIceServer(
                urls=urls,
                username=raw_server.get("username"),
                credential=raw_server.get("credential"),
            )
        )
    return servers


@dataclass(slots=True)
class TransportStats:
    connection_state: str = "new"
    frames_sent: int = 0
    frames_received: int = 0
    frames_dropped: int = 0
    contours_sent: int = 0
    contours_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    buffered_amount: int = 0
    last_error: str = ""
    connected_at_unix_ns: int = 0
    updated_at_unix_ns: int = field(default_factory=time.time_ns)

    def to_dict(self):
        return {
            "connection_state": self.connection_state,
            "frames_sent": self.frames_sent,
            "frames_received": self.frames_received,
            "frames_dropped": self.frames_dropped,
            "contours_sent": self.contours_sent,
            "contours_received": self.contours_received,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "buffered_amount": self.buffered_amount,
            "last_error": self.last_error,
            "connected_at_unix_ns": self.connected_at_unix_ns,
            "updated_at_unix_ns": self.updated_at_unix_ns,
        }


class FrameAssembler:
    def __init__(self, max_pending_frames: int = 16):
        self.pending: dict[int, dict[str, Any]] = {}
        self.max_pending_frames = max_pending_frames

    def start(self, payload: dict[str, Any]):
        packet = FramePacket.from_start_message(payload)
        if packet.chunks <= 0:
            raise ValueError("frame packet must declare at least one chunk")
        self.pending[packet.frame_id] = {
            "packet": packet,
            "chunks": [None] * packet.chunks,
            "received": 0,
            "created_monotonic_ns": time.monotonic_ns(),
        }
        self._trim_pending()

    def add_chunk(self, message: bytes) -> tuple[FramePacket, bytes] | None:
        if len(message) <= CHUNK_HEADER.size:
            return None

        frame_id, chunk_index, chunk_count = CHUNK_HEADER.unpack_from(message)
        entry = self.pending.get(frame_id)
        if entry is None:
            return None

        packet: FramePacket = entry["packet"]
        if chunk_count != packet.chunks or chunk_index >= packet.chunks:
            return None

        if entry["chunks"][chunk_index] is None:
            entry["chunks"][chunk_index] = bytes(memoryview(message)[CHUNK_HEADER.size :])
            entry["received"] += 1

        if entry["received"] != packet.chunks:
            return None

        self.pending.pop(frame_id, None)
        payload = b"".join(entry["chunks"])
        if packet.payload_size and len(payload) != packet.payload_size:
            return None
        return packet, payload

    def _trim_pending(self):
        while len(self.pending) > self.max_pending_frames:
            oldest_frame_id = min(
                self.pending,
                key=lambda frame_id: self.pending[frame_id]["created_monotonic_ns"],
            )
            self.pending.pop(oldest_frame_id, None)


class WebRTCPeer:
    def __init__(self, pc: RTCPeerConnection, peer_id: str):
        self.pc = pc
        self.peer_id = peer_id
        self.frame_channel: RTCDataChannel | None = None
        self.contour_channel: RTCDataChannel | None = None
        self.stats = TransportStats()
        self.closed = asyncio.Event()

    async def send_contours(self, packet: ContourPacket):
        if self.contour_channel is None or self.contour_channel.readyState != "open":
            self.stats.frames_dropped += 1
            self.stats.updated_at_unix_ns = time.time_ns()
            return False

        message = packet.to_message()
        self.contour_channel.send(message)
        self.stats.contours_sent += 1
        self.stats.bytes_sent += len(message.encode("utf-8"))
        self.stats.updated_at_unix_ns = time.time_ns()
        return True

    async def close(self):
        if not self.closed.is_set():
            self.closed.set()
        await self.pc.close()


class WebRTCSignalingServer:
    def __init__(
        self,
        config: dict[str, Any],
        on_peer: Callable[[WebRTCPeer], Awaitable[None] | None],
        on_frame: Callable[[WebRTCPeer, FramePacket, bytes, int, int], Awaitable[None] | None],
        on_peer_closed: Callable[[WebRTCPeer], Awaitable[None] | None],
    ):
        self.config = config
        self.host = str(config["signaling_host"])
        self.port = int(config["signaling_port"])
        self.path = str(config["signaling_path"])
        self.health_path = str(config["health_path"])
        self.frame_channel_label = str(config["frame_channel"])
        self.contour_channel_label = str(config["contour_channel"])
        self.max_pending_frame_assemblies = int(config["max_pending_frame_assemblies"])
        self.on_peer = on_peer
        self.on_frame = on_frame
        self.on_peer_closed = on_peer_closed
        self.peers: set[WebRTCPeer] = set()
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None

    async def start(self):
        app = web.Application()
        app.router.add_post(self.path, self._handle_offer)
        app.router.add_get(self.health_path, self._handle_healthz)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()

    async def stop(self):
        for peer in list(self.peers):
            await self._close_peer(peer)
        if self.runner is not None:
            await self.runner.cleanup()

    async def serve_forever(self):
        await self.start()
        await asyncio.Future()

    async def _handle_healthz(self, request):
        return web.json_response({"ok": True, "peers": len(self.peers)})

    async def _handle_offer(self, request):
        params = await request.json()
        peer_id = uuid.uuid4().hex[:12]
        configuration = RTCConfiguration(
            iceServers=ice_servers_from_config(self.config.get("ice_servers", []))
        )
        pc = RTCPeerConnection(configuration=configuration)
        peer = WebRTCPeer(pc, peer_id)
        assembler = FrameAssembler(max_pending_frames=self.max_pending_frame_assemblies)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            peer.stats.connection_state = pc.connectionState
            peer.stats.updated_at_unix_ns = time.time_ns()
            if pc.connectionState == "connected" and not peer.stats.connected_at_unix_ns:
                peer.stats.connected_at_unix_ns = time.time_ns()
            if pc.connectionState in {"failed", "closed", "disconnected"}:
                await self._close_peer(peer)

        @pc.on("datachannel")
        def on_datachannel(channel):
            if channel.label == self.frame_channel_label:
                peer.frame_channel = channel
                self._configure_frame_channel(peer, channel, assembler)
            elif channel.label == self.contour_channel_label:
                peer.contour_channel = channel

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        )
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        self.peers.add(peer)
        result = self.on_peer(peer)
        if asyncio.iscoroutine(result):
            await result

        return web.json_response(
            {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "peer_id": peer.peer_id,
            }
        )

    def _configure_frame_channel(
        self,
        peer: WebRTCPeer,
        channel: RTCDataChannel,
        assembler: FrameAssembler,
    ):
        @channel.on("message")
        def on_message(message):
            try:
                if isinstance(message, str):
                    payload = json_loads(message)
                    if payload.get("type") == "frame_start":
                        assembler.start(payload)
                    return

                completed = assembler.add_chunk(message)
                peer.stats.bytes_received += len(message)
                peer.stats.updated_at_unix_ns = time.time_ns()
                if completed is None:
                    return

                packet, frame_bytes = completed
                peer.stats.frames_received += 1
                recv_unix_ns = time.time_ns()
                recv_monotonic_ns = time.monotonic_ns()
                result = self.on_frame(
                    peer,
                    packet,
                    frame_bytes,
                    recv_unix_ns,
                    recv_monotonic_ns,
                )
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as exc:
                peer.stats.last_error = str(exc)
                peer.stats.updated_at_unix_ns = time.time_ns()

    async def _close_peer(self, peer: WebRTCPeer):
        if peer.closed.is_set():
            return
        peer.closed.set()
        if peer in self.peers:
            self.peers.discard(peer)
            result = self.on_peer_closed(peer)
            if asyncio.iscoroutine(result):
                await result
        await peer.pc.close()


class WebRTCClient:
    def __init__(
        self,
        client_webrtc_config: dict[str, Any],
        server_webrtc_config: dict[str, Any],
        on_contours: Callable[[ContourPacket], Awaitable[None] | None],
    ):
        self.client_config = client_webrtc_config
        self.server_config = server_webrtc_config
        self.offer_url = str(client_webrtc_config["server_offer_url"])
        self.frame_label = str(server_webrtc_config["frame_channel"])
        self.contour_label = str(server_webrtc_config["contour_channel"])
        self.chunk_size_bytes = int(server_webrtc_config["chunk_size_bytes"])
        self.buffer_limit_bytes = int(client_webrtc_config["datachannel_buffer_limit_bytes"])
        self.connect_timeout_s = float(client_webrtc_config["connect_timeout_s"])
        self.on_contours = on_contours

        self.pc: RTCPeerConnection | None = None
        self.frame_channel: RTCDataChannel | None = None
        self.contour_channel: RTCDataChannel | None = None
        self.stats = TransportStats()
        self.closed = asyncio.Event()
        self._open = asyncio.Event()

    async def connect(self):
        configuration = RTCConfiguration(
            iceServers=ice_servers_from_config(self.client_config.get("ice_servers", []))
        )
        self.pc = RTCPeerConnection(configuration=configuration)
        self.closed.clear()
        self._open.clear()

        self.frame_channel = self.pc.createDataChannel(
            self.frame_label,
            ordered=bool(self.server_config["frame_ordered"]),
            maxRetransmits=self.server_config.get("frame_max_retransmits"),
        )
        self.contour_channel = self.pc.createDataChannel(self.contour_label, ordered=True)
        self._configure_channels()

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            self.stats.connection_state = self.pc.connectionState
            self.stats.updated_at_unix_ns = time.time_ns()
            if self.pc.connectionState == "connected" and not self.stats.connected_at_unix_ns:
                self.stats.connected_at_unix_ns = time.time_ns()
            if self.pc.connectionState in {"failed", "closed", "disconnected"}:
                self.closed.set()

        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        async with ClientSession() as session:
            async with session.post(
                self.offer_url,
                json={
                    "sdp": self.pc.localDescription.sdp,
                    "type": self.pc.localDescription.type,
                },
            ) as response:
                response.raise_for_status()
                answer = await response.json()

        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )
        await asyncio.wait_for(self._open.wait(), timeout=self.connect_timeout_s)
        return self

    def _configure_channels(self):
        @self.frame_channel.on("open")
        def on_frame_open():
            self._open.set()

        @self.contour_channel.on("message")
        def on_contour_message(message):
            try:
                if not isinstance(message, str):
                    return
                payload = json_loads(message)
                if payload.get("type") != "contours":
                    return
                packet = ContourPacket.from_payload(payload)
                self.stats.contours_received += 1
                self.stats.bytes_received += len(message.encode("utf-8"))
                self.stats.updated_at_unix_ns = time.time_ns()
                result = self.on_contours(packet)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as exc:
                self.stats.last_error = str(exc)
                self.stats.updated_at_unix_ns = time.time_ns()

    def is_ready(self):
        return (
            self.pc is not None
            and self.frame_channel is not None
            and self.frame_channel.readyState == "open"
            and self.stats.connection_state in {"connected", "connecting", "new"}
        )

    def buffered_amount(self):
        if self.frame_channel is None:
            return 0
        return int(getattr(self.frame_channel, "bufferedAmount", 0))

    async def send_frame(self, packet: FramePacket, payload: bytes):
        if not self.is_ready():
            self.stats.frames_dropped += 1
            self.stats.updated_at_unix_ns = time.time_ns()
            return False

        buffered = self.buffered_amount()
        self.stats.buffered_amount = buffered
        if buffered > self.buffer_limit_bytes:
            self.stats.frames_dropped += 1
            self.stats.updated_at_unix_ns = time.time_ns()
            return False

        chunk_count = max(1, math.ceil(len(payload) / self.chunk_size_bytes))
        packet.payload_size = len(payload)
        packet.chunks = chunk_count

        self.frame_channel.send(packet.to_start_message())
        sent_bytes = 0
        for chunk_index in range(chunk_count):
            start = chunk_index * self.chunk_size_bytes
            chunk = payload[start : start + self.chunk_size_bytes]
            message = CHUNK_HEADER.pack(packet.frame_id, chunk_index, chunk_count) + chunk
            self.frame_channel.send(message)
            sent_bytes += len(message)

        self.stats.frames_sent += 1
        self.stats.bytes_sent += sent_bytes
        self.stats.buffered_amount = self.buffered_amount()
        self.stats.updated_at_unix_ns = time.time_ns()
        return True

    async def wait_closed(self):
        await self.closed.wait()

    async def close(self):
        self.closed.set()
        if self.pc is not None:
            await self.pc.close()
