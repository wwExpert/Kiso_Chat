import os
import asyncio
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI


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
        except Exception as exc:
            st.error(f"Failed to initialize model: {exc}")
            self.llm = None

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

    def run(self) -> None:
        st.title("KiSo - Chat")
        st.divider()

        if not self.llm:
            return

        chat_id = st.session_state["current_chat"]
        messages = st.session_state["chats"].setdefault(chat_id, [])

        self._display_messages()

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
