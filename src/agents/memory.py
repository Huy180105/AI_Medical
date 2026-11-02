import json
from typing import Any

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RedisMemoryStore:
    def __init__(self, redis_url: str | None = None, ttl_seconds: int | None = None) -> None:
        self.redis_url = redis_url or Config.REDIS_URL
        self.ttl_seconds = ttl_seconds or Config.AGENT_MEMORY_TTL_SECONDS
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import redis

                self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)
                self._client.ping()
            except Exception as exc:
                logger.warning("Redis memory is unavailable: %s", exc)
                self._client = False
        return self._client if self._client is not False else None

    def append(self, key: str, value: dict[str, Any]) -> None:
        client = self.client
        if client is None:
            return
        client.rpush(key, json.dumps(value, ensure_ascii=True))
        client.expire(key, self.ttl_seconds)

    def get_all(self, key: str) -> list[dict[str, Any]]:
        client = self.client
        if client is None:
            return []
        return [json.loads(item) for item in client.lrange(key, 0, -1)]

    def set_json(self, key: str, value: dict[str, Any]) -> None:
        client = self.client
        if client is None:
            return
        client.setex(key, self.ttl_seconds, json.dumps(value, ensure_ascii=True))

    def get_json(self, key: str) -> dict[str, Any] | None:
        client = self.client
        if client is None:
            return None
        value = client.get(key)
        return json.loads(value) if value else None


class ConversationMemory:
    def __init__(self, store: RedisMemoryStore | None = None) -> None:
        self.store = store or RedisMemoryStore()

    def add_turn(self, session_id: str, turn: dict[str, Any]) -> None:
        self.store.append(f"conversation:{session_id}", turn)

    def get_turns(self, session_id: str) -> list[dict[str, Any]]:
        return self.store.get_all(f"conversation:{session_id}")


class MedicalContextMemory:
    def __init__(self, store: RedisMemoryStore | None = None) -> None:
        self.store = store or RedisMemoryStore()

    def save_context(self, session_id: str, context: dict[str, Any]) -> None:
        self.store.set_json(f"medical_context:{session_id}", context)

    def get_context(self, session_id: str) -> dict[str, Any] | None:
        return self.store.get_json(f"medical_context:{session_id}")


class PatientHistoryMemory:
    def __init__(self, store: RedisMemoryStore | None = None) -> None:
        self.store = store or RedisMemoryStore()

    def add_event(self, patient_id: str, event: dict[str, Any]) -> None:
        self.store.append(f"patient_history:{patient_id}", event)

    def get_history(self, patient_id: str) -> list[dict[str, Any]]:
        return self.store.get_all(f"patient_history:{patient_id}")
