"""
session.py

The Session entity: one conversation belonging to one device. No
behavior lives here beyond simple data — logic (eviction, persistence,
budget) is deliberately kept in separate collaborators (SRP).
"""

from dataclasses import dataclass, field, asdict
import time


@dataclass
class Session:
    session_id: str
    device_name: str = "unknown device"
    system_prompt: str = "You are a helpful assistant."
    history: list = field(default_factory=list)  # list[{"role":.., "content":..}]
    summary: str = ""
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        return cls(**data)

    def build_messages(self) -> list:
        msgs = [{"role": "system", "content": self.system_prompt}]
        if self.summary:
            msgs.append({"role": "system",
                         "content": f"[Earlier conversation summary]: {self.summary}"})
        msgs.extend(self.history)
        return msgs
