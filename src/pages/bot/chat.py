import streamlit as st
from src.agents.support_agent import get_support_agent
from src.pages.bot.finbot import Finbot
from src.models.workflow import WorkflowSettings
from src.lib.knowledge import Knowledge
from src.lib.storage import Storage

def main():
    settings = st.session_state['settings']['Finbot']
    km = Knowledge('finsight')
    store = Storage()
    agent = get_support_agent(settings=WorkflowSettings(online=settings['online'], temperature=settings['temperature']))
    
    chatbot = Finbot(agent, km, store)
    chatbot.session()

if __name__ == '__main__':
    main()