# browser_agent_persistent_json.py
import asyncio
import json
import os
import sqlite3
import uuid
import datetime
from typing import Optional

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

# >>> OFFICIAL message <-> JSON helpers (from the docs) <<<
from pydantic_core import to_jsonable_python
from pydantic_ai.messages import ModelMessagesTypeAdapter

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "sessions.db")


# ---------------------------
# SQLite persistence layer
# ---------------------------
class SessionStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    data TEXT NOT NULL
                )
                """
            )
            con.commit()

    def create_session(self) -> str:
        sid = str(uuid.uuid4())
        created_at = datetime.datetime.now(datetime.UTC).isoformat()
        # start with an empty message array (as JSON)
        with self._connect() as con:
            con.execute(
                "INSERT INTO sessions (id, created_at, data) VALUES (?, ?, ?)",
                (sid, created_at, "[]"),
            )
            con.commit()
        return sid

    def list_sessions(self):
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, created_at, data FROM sessions ORDER BY created_at DESC"
            ).fetchall()
        out = []
        for r in rows:
            try:
                turns = len(json.loads(r["data"]))
            except Exception:
                turns = 0
            out.append({"id": r["id"], "created_at": r["created_at"], "turns": turns})
        return out

    def load_messages(self, session_id: str):
        """Return a list[ModelMessage] using ModelMessagesTypeAdapter."""
        with self._connect() as con:
            row = con.execute(
                "SELECT data FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if not row:
            return []
        # stored JSON -> python (list of plain dicts) -> Model messages
        python_data = json.loads(row["data"])
        return ModelMessagesTypeAdapter.validate_python(python_data)

    def save_messages(self, session_id: str, messages) -> None:
        """
        Save list[ModelMessage] using the official adapter:
        messages -> jsonable python -> json string -> DB
        """
        as_python = to_jsonable_python(messages)
        payload = json.dumps(as_python, ensure_ascii=False)
        with self._connect() as con:
            con.execute(
                "UPDATE sessions SET data = ? WHERE id = ?", (payload, session_id)
            )
            con.commit()


class BrowserAgent:
    def __init__(self):
        self.store = SessionStore()
        self._bootstrap_agent()
        self.session_id: Optional[str] = None
        self.history = []  # will be list[ModelMessage]

    def _bootstrap_agent(self):
        browser_mcp_server = MCPServerStdio("npx", args=["@browsermcp/mcp@latest"])
        self.agent = Agent(
            model="google-gla:gemini-2.0-flash",
            toolsets=[browser_mcp_server],
        )

    def _choose_session(self):
        sessions = self.store.list_sessions()
        if not sessions:
            self.session_id = self.store.create_session()
            print(f"\nNo previous conversations. New session: {self.session_id}\n")
            return

        print("\nPrevious conversations:")
        for i, s in enumerate(sessions, 1):
            print(
                f"[{i}] {s['id']}  |  created_at: {s['created_at']}  |  turns: {s['turns']}"
            )
        print("[N] New conversation")

        while True:
            choice = input("\nSelect a conversation number or 'N' for new: ").strip()
            if choice.lower() == "n":
                self.session_id = self.store.create_session()
                print(f"\nNew session: {self.session_id}\n")
                return
            if choice.isdigit() and 1 <= int(choice) <= len(sessions):
                self.session_id = sessions[int(choice) - 1]["id"]
                # load as ModelMessage list via adapter
                self.history = self.store.load_messages(self.session_id)
                print(
                    f"\nResumed session: {self.session_id} (turns: {len(self.history)})\n"
                )
                return
            print("Invalid selection. Try again.")

    async def start_conversation(self):
        self._choose_session()
        if not self.session_id:
            self.session_id = self.store.create_session()

        async with self.agent:
            while True:
                try:
                    command = input("you: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting…")
                    break

                if command.lower() in {"exit", "quit"}:
                    print("Goodbye!")
                    break

                # Run with prior messages (ModelMessage list); get full history back
                result = await self.agent.run(command, message_history=self.history)

                # Update in-memory history from result using the library's canonical method
                self.history = result.all_messages()

                # Persist canonically using the adapter
                self.store.save_messages(self.session_id, self.history)

                print(f"Agent: {result.output}")


if __name__ == "__main__":
    agent = BrowserAgent()
    asyncio.run(agent.start_conversation())
