from fastapi import WebSocket
from typing import Dict, List
import json
import logging

logger = logging.getLogger("astraatmos.ws")

class ConnectionManager:
    def __init__(self):
        # task_id -> list of websockets
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        if task_id not in self.active_connections:
            self.active_connections[task_id] = []
        self.active_connections[task_id].append(websocket)
        logger.info(f"Client connected to task {task_id}. Total: {len(self.active_connections[task_id])}")

    def disconnect(self, websocket: WebSocket, task_id: str):
        if task_id in self.active_connections:
            if websocket in self.active_connections[task_id]:
                self.active_connections[task_id].remove(websocket)
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]
        logger.info(f"Client disconnected from task {task_id}")

    async def broadcast_to_task(self, task_id: str, message: dict):
        if task_id in self.active_connections:
            # Create a copy of the list to avoid issues during iteration if disconnects happen
            targets = self.active_connections[task_id][:]
            for connection in targets:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send WS message to task {task_id}: {e}")
                    # We don't remove here to avoid mutating list; the disconnect will handle it or next broadcast will fail too

# Global instance
manager = ConnectionManager()
