import json

import streamlit as st

from src.agents.support_agent import SupportAgent, SupportState
from langchain_core.messages import HumanMessage


class Finbot:
    def __init__(self, agent: SupportAgent):
        self.agent = agent

        if 'messages' not in st.session_state:
            st.session_state['messages'] = []

    def session(self):
        # apply_global_theme()

        self._render_header()
        self._render_settings()
        self._render_messages()
        self._render_input()

    def _render_header(self):
        header_col, status_col = st.columns([4, 1])
        with header_col:
            st.markdown(
                """
                <div class="farm-header">
                    <span class="farm-header-icon">🌾</span>
                    <div>
                        <h1 style="margin: 0;">Finbot</h1>
                        <p style="margin: 0; color: #A5D6A7;">Rivers & Roots — Ask about ponds, batches, feed, harvest, and more.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with status_col:
            online = st.session_state['settings']['Finbot']['online']
            status_class = "status-online" if online else "status-offline"
            status_text = "Online" if online else "Offline"
            st.markdown(
                f"<div class='farm-card' style='text-align: center;'>"
                f"<div style='font-size: 0.8rem; color: #A5D6A7;'>Mode</div>"
                f"<div class='{status_class}'>{status_text}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    def _render_settings(self):
        with st.expander("⚙️ Finbot Settings", expanded=False):
            settings = st.session_state['settings']['Finbot']

            col1, col2 = st.columns(2)
            with col1:
                online = st.toggle(
                    "Online mode",
                    value=settings['online'],
                    help="When online, Finbot can search the web. When offline, it only uses internal farm documents and data.",
                )
            with col2:
                temperature = st.slider(
                    "Creativity (temperature)",
                    min_value=0.0,
                    max_value=1.0,
                    value=settings['temperature'],
                    step=0.05,
                    help="Lower values give more factual, deterministic answers. Higher values are more creative.",
                )

            if st.button("Apply Settings", use_container_width=True):
                settings['online'] = online
                settings['temperature'] = temperature
                st.session_state['settings']['Finbot'] = settings
                st.success("Settings applied.")

            st.caption(
                "Online mode lets the assistant search the web when internal farm data is insufficient. "
                "Offline mode keeps answers restricted to your uploaded documents and datasets."
            )

    def _render_messages(self):
        if not st.session_state['messages']:
            st.markdown(
                """
                <div class='farm-card' style='margin-bottom: 1rem;'>
                    <b>👋 Welcome to Finbot Assistant!</b><br>
                    Ask me about your business's finance, I can forcast too.
                </div>
                """,
                unsafe_allow_html=True,
            )

        for message in st.session_state['messages']:
            with st.chat_message(message['role']):
                if message.get('file', None):
                    st.markdown(f"{message['text']} (File: {message['file'].name})")
                else:
                    st.markdown(message['text'])

    def _render_input(self):
        if prompt := st.chat_input(
            "Ask about your farm...",
            key="user_input",
            accept_audio=False,
            accept_file=True,
        ):
            text = prompt.text
            file = prompt.files[0] if len(prompt.files) else None
            self.send_message(text, file)

    def send_message(self, text, file=None):
        query = {'role': 'user', 'text': text}
        if file:
            query['file'] = file
        st.session_state['messages'].append(query)

        with st.chat_message(query['role']):
            st.markdown(query['text'])
            if file:
                st.caption(f"Attached: {file.name}")

        tool_placeholder = st.empty()
        tool_placeholder.markdown("🌱 Thinking...")

        with st.spinner():
            response_placeholder = st.empty()
            events = self.agent.stream(
                SupportState(messages=[HumanMessage(content=text)]),
                save=False,
            )
            full_response = ""

            for chunk in self.parse_event_stream(events):
                if chunk['type'] == 'tool_call':
                    tool_use = chunk['content']
                    tool_placeholder.markdown(f"🔧 Using tool: **{tool_use}**")
                    
                if chunk['type'] == 'text':
                    full_response += chunk['content']
                    response_placeholder.markdown(full_response + "▌")
                    tool_placeholder.markdown("🌾 Responding...")

            response_placeholder.markdown(full_response)
            tool_placeholder.empty()

            st.session_state['messages'].append({'role': 'assistant', 'text': full_response})
            st.rerun()

    def parse_event_stream(self, events: bytes):
        for chunk in events:
            if isinstance(chunk, (bytearray, bytes)):
                chunk = chunk.decode('utf-8')

            marker = "data:"
            if chunk.startswith(marker):
                data = chunk[len(marker):].strip()
                if data:
                    try:
                        response = json.loads(data)
                        yield response
                    except json.JSONDecodeError:
                        pass
