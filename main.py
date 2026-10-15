import streamlit as st


def setup():
    if 'settings' not in st.session_state:
        st.session_state['settings'] = {
            'Finbot': {
                'online': True,
                'temperature': 0.7,
                'model': 'gemma4:31b-cloud',
            },
        }


def run():
    st.set_page_config(layout='wide')
    pages = [
        st.Page('src/pages/bot/chat.py', title='Chat', default=True),
        st.Page('src/pages/dashboard.py', title='Dashboard', default=False)
    ]

    with st.sidebar:
        st.markdown("## 💬 Finsight")
        st.caption("Finance Analytics")

        if st.button("Chat", width='stretch',type="secondary", icon=':material/forum:'):
            st.switch_page('src/pages/bot/chat.py')

        if st.button("Dashboard", width='stretch',type="secondary", icon=':material/dashboard:'):
            st.switch_page('src/pages/dashboard.py')

    pg = st.navigation(pages, position='hidden')
    pg.run()


def main():
    setup()
    run()

if __name__ == "__main__":
    main()
