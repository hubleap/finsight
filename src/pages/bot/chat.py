import streamlit as st
from src.agents.support_agent import SupportAgent
from src.pages.bot.finbot import Finbot
from src.models.workflow import WorkflowSettings

def main():
    settings = st.session_state['settings']['Finbot']
    agent = SupportAgent(settings=WorkflowSettings(online=settings['online'], temperature=settings['temperature']))
    chatbot = Finbot(agent)
    chatbot.session()

if __name__ == '__main__':
    main()