import os
import streamlit as st
import logging

# KORREKTUR: Import-Anweisungen für Klarheit und Kompatibilität überarbeitet.

# Import für ChatOpenAI
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    # Fallback für ältere langchain Versionen, die ChatOpenAI unter langchain.chat_models hatten
    from langchain.chat_models import ChatOpenAI

# KORREKTUR: PythonREPLTool Import direkt aus langchain_experimental
# Die Fehlermeldung des Benutzers zeigt, dass dies der richtige Ort ist.
try:
    from langchain_experimental.tools import PythonREPLTool
except ImportError:
    st.error(
        "Fehler beim Importieren von PythonREPLTool aus langchain_experimental. "
        "Stelle sicher, dass 'langchain-experimental' installiert ist: pip install langchain-experimental"
    )
    # Als Fallback, falls die obige Meldung nicht direkt zum Abbruch führt, None setzen.
    # Die App wird dann an anderer Stelle mit einer Warnung darauf hinweisen, dass der Agent nicht funktioniert.
    PythonREPLTool = None


# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from langchain.agents import Tool, initialize_agent # initialize_agent ist veraltet, aber vorerst beibehalten
from langchain_core.messages import HumanMessage, AIMessage


class ChatApp:
    """Streamlit chat application using LangChain."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self._log("Initializing ChatApp")
        self._init_session()
        self.params = {}  # Wird in _setup_sidebar mit den ersten Widget-Werten gefüllt
        self._setup_sidebar()
        # LLM und Agent werden initialisiert, wenn ein API-Key vorhanden ist.
        # _setup_sidebar ruft _setup_llm_and_agent auf, wenn Parameter initial gesetzt werden und Key vorhanden ist.
        # oder wenn der Key eingegeben wird. Ein direkter Aufruf hier ist nur nötig, falls
        # _setup_sidebar nicht zwingend _setup_llm_and_agent beim ersten Mal mit Env-Key aufruft.
        # Die aktuelle Logik in _setup_sidebar sollte das aber abdecken.
        if not hasattr(self, 'llm') and os.environ.get("OPENAI_API_KEY"): # Sicherstellen, dass initial geladen wird, falls Key in Env
             self._setup_llm_and_agent()

    def _log(self, message: str, level: int = logging.DEBUG) -> None:
        """Helper to store and output debug messages."""
        logger.log(level, message)
        st.session_state.setdefault("debug_logs", []).append(message)


    def _init_session(self) -> None:
        # Initialisiert den Session State für Streamlit
        self._log("Initializing session state")
        if "chats" not in st.session_state:
            st.session_state["chats"] = {"default": []}
        if "current_chat" not in st.session_state:
            st.session_state["current_chat"] = "default"
        if "agent_result" not in st.session_state:
            st.session_state["agent_result"] = None
        if "agentic_mode" not in st.session_state:
            st.session_state["agentic_mode"] = False
        if "llm_initialized_with_key" not in st.session_state: # Hinzugefügt für bessere Logik
            st.session_state["llm_initialized_with_key"] = False
        if "openai_api_key_value" not in st.session_state: # Für Persistenz des API-Key Feldes
            st.session_state["openai_api_key_value"] = os.environ.get("OPENAI_API_KEY", "")
        self._log("Session state initialized")


    def _setup_sidebar(self) -> None:
        # Erstellt die Sidebar für Einstellungen
        self._log("Setting up sidebar")
        with st.sidebar:
            st.title("KiSo - Chat")
            # API-Key Eingabe
            api_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.get("openai_api_key_value", os.environ.get("OPENAI_API_KEY", "")),
                type="password",
                key="api_key_input"
            )

            if api_key != st.session_state.get("openai_api_key_value"):
                st.session_state["openai_api_key_value"] = api_key
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key
                    self._log("API key updated")
                    st.session_state["llm_initialized_with_key"] = False # Erfordert Neuinitialisierung
                    # Rufen _setup_llm_and_agent nicht direkt hier auf, sondern lassen es von der Parameter-Logik unten handhaben
                else: # API Key wurde gelöscht
                    if "OPENAI_API_KEY" in os.environ:
                        del os.environ["OPENAI_API_KEY"]
                    self._log("API key removed")
                    self.llm = None
                    self.agent = None
                    st.session_state["llm_initialized_with_key"] = False
                st.rerun() # Rerun, um die Logik mit dem neuen Key-Status zu triggern

            st.divider()

            # Chat ID und Modelleinstellungen
            current_chat_id_val = st.session_state.get("current_chat", "default")
            chat_id_input = st.text_input("Chat ID", value=current_chat_id_val, key="chat_id_input")

            model = st.selectbox("Choose a model:", ["gpt-4.1-mini", "gpt-4.1"], key="model_select")
            temperature = st.slider("Temperature", 0.0, 1.0, 1.0, 0.1, key="temp_slider")
            max_tokens = st.number_input("Max tokens", min_value=1, max_value=4096, value=250, key="max_tokens_input")
            top_p = st.slider("Top P", 0.0, 1.0, 1.0, 0.05, key="top_p_slider")

            st.session_state["agentic_mode"] = st.checkbox("Agentic Mode", value=st.session_state.get("agentic_mode", False), key="agentic_mode_checkbox")

            if st.button("New Chat :page_facing_up:", key="new_chat_button"):
                st.session_state["chats"][chat_id_input] = []
                st.session_state["current_chat"] = chat_id_input
                st.session_state["agent_result"] = None
                self._log(f"New chat created: {chat_id_input}")
                st.rerun()

            if chat_id_input != current_chat_id_val: # current_chat_id_val ist der Wert *vor* diesem Rerun
                st.session_state["current_chat"] = chat_id_input
                if chat_id_input not in st.session_state["chats"]:
                    st.session_state["chats"][chat_id_input] = []
                self._log(f"Switched to chat: {chat_id_input}")
                st.rerun()

            new_params = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
            }

            # Konsistente Prüfung, ob eine Neuinitialisierung des LLM notwendig ist
            should_reinitialize_llm = (
                self.params != new_params or
                not getattr(self, 'llm', None) or
                (os.environ.get("OPENAI_API_KEY") and not st.session_state.get("llm_initialized_with_key", False))
            )

            if should_reinitialize_llm:
                self._log("Model parameters changed, reinitializing LLM")
                self.params.update(new_params) # self.params wird hier mit den neuesten Werten aktualisiert
                if os.environ.get("OPENAI_API_KEY"):
                    self._setup_llm_and_agent() # Ruft die Initialisierung mit den aktualisierten self.params auf
                # Falls kein API-Key, wird _setup_llm_and_agent intern nichts tun oder self.llm auf None setzen


    def _setup_llm_and_agent(self) -> None:
        # Initialisiert das LLM und den Agenten.
        # Diese Methode wird aufgerufen, wenn:
        # 1. Die App startet und ein API-Key in der Umgebung ist (via __init__).
        # 2. Ein API-Key im Sidebar (neu) eingegeben oder geändert wird.
        # 3. Modellparameter im Sidebar geändert werden und ein API-Key vorhanden ist.
        # self.params enthält die zu verwendenden Parameter.

        self._log("Setting up LLM and agent")
        if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
            if self.llm is not None or self.agent is not None: # Nur zurücksetzen, wenn sie vorher existierten
                self.llm = None
                self.agent = None
            st.session_state["llm_initialized_with_key"] = False
            return

        try:
            # Parameter aus self.params holen (sollten von _setup_sidebar aktualisiert worden sein)
            current_model = self.params.get("model", "gpt-3.5-turbo-1106") # Fallback, falls self.params leer
            current_temp = self.params.get("temperature", 1.0)
            current_max_tokens = self.params.get("max_tokens", 250)
            current_top_p = self.params.get("top_p", 1.0)

            # KORREKTUR: Die detaillierte und fehleranfällige 'needs_reinitialization'-Prüfung
            # innerhalb dieser Methode wurde entfernt. Die Entscheidung zur Neuinitialisierung
            # wird von der aufrufenden Logik (hauptsächlich in _setup_sidebar) getroffen,
            # die self.params aktualisiert und dann diese Methode aufruft.
            # Wir erstellen das LLM-Objekt hier immer neu, wenn die Methode aufgerufen wird
            # und ein API-Key vorhanden ist. Das vereinfacht die Logik und vermeidet Attributfehler.

            self._log(f"Initializing ChatOpenAI with model {current_model}")
            self.llm = ChatOpenAI(
                streaming=True,
                model=current_model,
                temperature=current_temp,
                max_tokens=current_max_tokens,
                top_p=current_top_p,
                model_kwargs={} # Sicherstellen, dass keine doppelten Parameter übergeben werden
            )
            self._setup_agent()  # Agent muss nach LLM neu initialisiert werden
            st.session_state["llm_initialized_with_key"] = True
            self._log("LLM and agent initialized")

        except Exception as exc:
            self._log(f"Failed to initialize model: {exc}", level=logging.ERROR)
            st.error(f"Failed to initialize model: {exc}")
            self.llm = None
            self.agent = None
            st.session_state["llm_initialized_with_key"] = False


    def _setup_agent(self) -> None:
        """Initialize a simple agent with Python REPL capabilities."""
        self._log("Initializing agent")
        if not self.llm:
            self.agent = None
            return
        
        if PythonREPLTool is None:
            # Die Fehlermeldung zum Import wurde bereits beim Importversuch angezeigt.
            # Hier stellen wir sicher, dass der Agent nicht initialisiert wird.
            self.agent = None
            return

        try:
            python_repl_tool_instance = PythonREPLTool()
            tool = Tool(
                name="python_repl",
                func=python_repl_tool_instance.run,
                description="Execute Python code and return the result. Useful for calculations or code execution."
            )
            self.agent = initialize_agent(
                [tool],
                self.llm,
                agent="zero-shot-react-description",
                verbose=False,
                handle_parsing_errors=True
            )
            self._log("Agent initialized")
        except Exception as exc:
            self._log(f"Failed to initialize agent: {exc}", level=logging.ERROR)
            st.error(f"Failed to initialize agent: {exc}")
            self.agent = None


    def _stream_response(self, messages_history: list):
        self._log("Streaming response from LLM")
        if not self.llm:
            yield "LLM ist nicht initialisiert. Bitte API Key im Sidebar eingeben und Parameter prüfen."
            return

        langchain_messages = []
        for msg in messages_history:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        try:
            for chunk in self.llm.stream(langchain_messages):
                text = chunk.content
                if text:
                    yield text
        except Exception as e:
            self._log(f"Error while streaming response: {e}", level=logging.ERROR)
            yield f"Ein Fehler ist beim Streamen der Antwort aufgetreten: {e}"


    def _display_messages(self) -> None:
        current_chat_id = st.session_state.get("current_chat", "default")
        self._log(f"Displaying messages for chat: {current_chat_id}")
        # Sicherstellen, dass der Chat-Eintrag existiert
        if current_chat_id not in st.session_state["chats"]:
            st.session_state["chats"][current_chat_id] = []

        messages_to_display = st.session_state["chats"].get(current_chat_id, [])
        for message in messages_to_display:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


    def _start_agent_task(self, task: str) -> None:
        self._log(f"Starting agent task: {task}")
        if not self.agent:
            st.error("Agent ist nicht initialisiert. Agentic Mode nicht möglich.")
            return

        st.session_state["agent_result"] = None  # Ergebnis zurücksetzen
        with st.spinner("Agent is working..."):
            try:
                result = self.agent.run(task)
            except Exception as exc:
                self._log(f"Agent task failed: {exc}", level=logging.ERROR)
                result = f"Agent failed: {exc}"

        st.session_state["agent_result"] = result
        self._log("Agent task completed")


    def _display_agentic(self) -> None:
        self._log("Displaying agentic controls")
        if not self.agent:
            # Die Warnung über PythonREPLTool wird in _setup_agent behandelt, falls der Import fehlschlägt.
            # Hier eine allgemeinere Warnung, falls der Agent aus anderen Gründen nicht da ist.
            if PythonREPLTool is not None: # Nur warnen, wenn das Tool theoretisch verfügbar wäre
                st.warning("Agent nicht konfiguriert. Bitte API Key prüfen.")
            return # Wenn PythonREPLTool None ist, wurde die spezifische Warnung bereits ausgegeben.

        task = st.text_input("Agent Task", key="agent_task_input")
        if st.button("Run Agent", key="run_agent_button") and task:
            self._log("Run Agent button pressed")
            self._start_agent_task(task)

        if st.session_state.get("agent_result") is not None:
            st.success(f"Agent Result: {st.session_state['agent_result']}")


    def run(self) -> None:
        self._log("Running main app loop")
        st.title("KiSo - Chat")
        st.divider()

        if not self.llm or not st.session_state.get("llm_initialized_with_key", False):
            st.warning("LLM nicht initialisiert. Bitte gib deinen OpenAI API Key im Sidebar ein.")

        current_chat_id = st.session_state.get("current_chat", "default")
        if current_chat_id not in st.session_state["chats"]:
            st.session_state["chats"][current_chat_id] = []
        
        messages = st.session_state["chats"][current_chat_id]

        self._display_messages()

        if st.session_state.get("agentic_mode", False):
            if self.llm and self.agent: # Agentic UI nur anzeigen, wenn LLM und Agent bereit sind
                 self._display_agentic()
            elif self.llm and PythonREPLTool is not None and not self.agent : # LLM da, Tool da, aber Agent nicht initialisiert
                 st.warning("Agent konnte nicht initialisiert werden. Agentic Mode eingeschränkt.")
            # Fall: PythonREPLTool is None -> Warnung bereits in _setup_agent/_display_agentic

        if self.llm and st.session_state.get("llm_initialized_with_key", False):
            prompt = st.chat_input("Ask your question!", key=f"chat_input_{current_chat_id}")
            if prompt:
                messages.append({"role": "user", "content": prompt})
                self._log(f"User prompt received: {prompt}")
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    full_response_content = []
                    final_response = ""
                    try:
                        response_stream = self._stream_response(messages)
                        full_response_placeholder = st.empty()
                        for chunk in response_stream:
                            full_response_content.append(chunk)
                            full_response_placeholder.markdown("".join(full_response_content))

                        final_response = "".join(full_response_content)
                        if not final_response and not any("Fehler" in str(chunk) for chunk in full_response_content):
                            final_response = "Keine Antwort erhalten." # Fallback, falls Stream leer ist
                            st.warning(final_response)
                    except Exception as e:
                        self._log(f"Error while generating response: {e}", level=logging.ERROR)
                        st.error(f"Ein Fehler ist beim Anzeigen der Antwort aufgetreten: {e}")
                        final_response = f"Fehler beim Verarbeiten der Antwort: {e}"

                messages.append({"role": "assistant", "content": final_response})
                self._log("Assistant response stored")
                st.rerun()

        if st.session_state.get("debug_logs"):
            with st.expander("Debug Logs"):
                st.code("\n".join(st.session_state["debug_logs"]))


if __name__ == "__main__":
    app = ChatApp()
    app.run()
