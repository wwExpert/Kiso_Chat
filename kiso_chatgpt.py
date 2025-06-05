import os
import asyncio
import threading
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
try:  # LangChain < 0.1.0
    from langchain.utilities import PythonREPL
except Exception:  # pragma: no cover - fallback for newer versions
    from langchain.tools.python.tool import PythonREPLTool as PythonREPL


class ChatApp:
    """Streamlit chat application using LangChain."""

    def __init__(self) -> None:
        self._init_session()
        self.params = {}
        self._setup_sidebar()
        self._setup_chain()

    def _init_session(self) -> None:
        if "chats" not in st.session_state:
            st.session_state["chats"] = {"default": []}
        if "current_chat" not in st.session_state:
            st.session_state["current_chat"] = "default"
        if "agent_result" not in st.session_state:
            st.session_state["agent_result"] = None
        if "agent_running" not in st.session_state:
            st.session_state["agent_running"] = False
        if "agentic_mode" not in st.session_state:
            st.session_state["agentic_mode"] = False

    def _setup_sidebar(self) -> None:
        with st.sidebar:
            st.title("KiSo - Chat")
            api_key = st.text_input("OpenAI API Key", type="password")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            st.divider()

            chat_id = st.text_input("Chat ID", value=st.session_state["current_chat"])
            model = st.selectbox("Choose a model:", ["gpt-4-1106-preview", "gpt-3.5-turbo-1106"])
            temperature = st.slider("Temperature", 0.0, 1.0, 1.0, 0.1)
            max_tokens = st.number_input("Max tokens", min_value=1, max_value=4096, value=250)
            top_p = st.slider("Top P", 0.0, 1.0, 1.0, 0.05)

            st.session_state["agentic_mode"] = st.checkbox("Agentic Mode", value=st.session_state["agentic_mode"])

            if st.button("New Chat :page_facing_up:"):
                st.session_state["chats"][chat_id] = []
            st.session_state["current_chat"] = chat_id

            self.params.update(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

    def _setup_chain(self) -> None:
        if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
            st.warning("API key missing. Enter your OpenAI API key in the sidebar.")
            self.llm = None
            return
        try:
            self.llm = ChatOpenAI(
                streaming=True,
                model=self.params["model"],
                temperature=self.params["temperature"],
                max_tokens=self.params["max_tokens"],
                top_p=self.params["top_p"],
            )
            self.chain = ConversationChain(llm=self.llm, verbose=False)
            self._setup_agent()
        except Exception as exc:
            st.error(f"Failed to initialize model: {exc}")
            self.llm = None

    def _setup_agent(self) -> None:
        """Initialize a simple agent with Python REPL capabilities."""
        try:
            repl = PythonREPL()
            tool = Tool(name="python", func=repl.run, description="Execute Python code")
            self.agent = initialize_agent([tool], self.llm, agent="zero-shot-react-description", verbose=False)
        except Exception as exc:
            st.error(f"Failed to initialize agent: {exc}")
            self.agent = None

    def _stream_response(self, prompt: str):
        for chunk in self.llm.stream(prompt):
            text = chunk.content
            if text:
                yield text

    async def _async_response(self, prompt: str) -> str:
        return await self.chain.apredict(input=prompt)

    def _display_messages(self) -> None:
        messages = st.session_state["chats"].setdefault(st.session_state["current_chat"], [])
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def _start_agent_task(self, task: str) -> None:
        """Run agent task in a background thread."""

        def run() -> None:
            try:
                result = self.agent.run(task)
            except Exception as exc:  # pragma: no cover - interactive
                result = f"Agent failed: {exc}"
            st.session_state["agent_result"] = result
            st.session_state["agent_running"] = False

        st.session_state["agent_running"] = True
        st.session_state["agent_result"] = None
        threading.Thread(target=run, daemon=True).start()

    def _display_agentic(self) -> None:
        """Render agent UI and show results when available."""
        if not self.agent:
            st.warning("Agent not configured")
            return

        if st.session_state["agent_running"]:
            st.info("Agent is working...")
        else:
            task = st.text_input("Agent Task", key="agent_task")
            if st.button("Run Agent") and task:
                self._start_agent_task(task)

        if st.session_state.get("agent_result"):
            st.success(st.session_state["agent_result"])

    def run(self) -> None:
        st.title("KiSo - Chat")
        st.divider()

        if not self.llm:
            return

        chat_id = st.session_state["current_chat"]
        messages = st.session_state["chats"].setdefault(chat_id, [])

        self._display_messages()

        if st.session_state.get("agentic_mode"):
            self._display_agentic()

        prompt = st.chat_input("Ask your question!")
        if prompt:
            with st.chat_message("user"):
                st.write(prompt)
            messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                try:
                    response = st.write_stream(self._stream_response(prompt))
                except Exception:
                    response = asyncio.run(self._async_response(prompt))
                    st.write(response)

            messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    ChatApp().run()
