import streamlit as st
import requests
import uuid
from BIO import process_query
# Page configuration
st.set_page_config(
    page_title="Rafiq - Your Biology Assistant",
    page_icon="🧬",
    layout="wide",
)

# 🎨 Colors
LIGHT_GREEN = "#8ee3c5"
LIGHT_BLUE = "#7dc4e4"

# 🖋 Custom CSS
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    .title {{
        font-family: 'Inter', sans-serif;
        font-size: 48px;
        color: {LIGHT_GREEN};
        text-align: center;
        font-weight: bold;
    }}

    .chat-message {{
        background-color: #f7f7f7;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        font-family: 'Inter', sans-serif;
        font-size: 18px;
    }}

    .user-input {{
        font-family: 'Inter', sans-serif;
        font-size: 16px;
        color: black;
    }}

    .sidebar-header {{
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 10px;
        color: {LIGHT_BLUE};
    }}

    </style>
""", unsafe_allow_html=True)

# ✅ Session initialization
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {}  # {chat_id: [(role, msg), ...]}
if "current_chat_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.current_chat_id = new_id
    st.session_state.all_chats[new_id] = []
if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = {}
if "edit_title" not in st.session_state:
    st.session_state.edit_title = False
if "new_title" not in st.session_state:
    st.session_state.new_title = ""

# 📌 Functions
def get_current_chat():
    return st.session_state.all_chats.get(st.session_state.current_chat_id, [])

def add_to_current_chat(role, message):
    st.session_state.all_chats[st.session_state.current_chat_id].append((role, message))

def generate_title_from_text(text):
    return text.split("?")[0][:50] if "?" in text else " ".join(text.split()[:8])

def delete_chat(chat_id):
    if chat_id in st.session_state.all_chats:
        del st.session_state.all_chats[chat_id]
        if chat_id in st.session_state.chat_titles:
            del st.session_state.chat_titles[chat_id]
        
        # If the current chat is deleted, move to another chat or create a new one
        if chat_id == st.session_state.current_chat_id:
            if st.session_state.all_chats:
                st.session_state.current_chat_id = next(iter(st.session_state.all_chats))
            else:
                new_id = str(uuid.uuid4())
                st.session_state.current_chat_id = new_id
                st.session_state.all_chats[new_id] = []
                st.session_state.chat_titles[new_id] = "New Chat"

def toggle_edit_title():
    st.session_state.edit_title = not st.session_state.edit_title
    if st.session_state.edit_title:
        st.session_state.new_title = st.session_state.chat_titles.get(st.session_state.current_chat_id, "")

def save_title():
    if st.session_state.new_title.strip():
        st.session_state.chat_titles[st.session_state.current_chat_id] = st.session_state.new_title
    st.session_state.edit_title = False

# 🎯 Sidebar
st.sidebar.markdown("<div class='sidebar-header'>📂 Chat History</div>", unsafe_allow_html=True)

# New chat button at the top of sidebar
if st.sidebar.button("➕ New Chat"):
    new_id = str(uuid.uuid4())
    st.session_state.all_chats[new_id] = []
    st.session_state.chat_titles[new_id] = "New Chat"
    st.session_state.current_chat_id = new_id

# Display all chats
chat_names = list(st.session_state.all_chats.keys())

# Use separate buttons for each chat instead of radio button
st.sidebar.markdown("### Chats")
for chat_id in chat_names:
    col1, col2, col3 = st.sidebar.columns([5, 1, 1])
    
    # Chat selection button
    if col1.button(st.session_state.chat_titles.get(chat_id, "Chat"), key=f"select_{chat_id}"):
        st.session_state.current_chat_id = chat_id
    
    # Rename chat button
    if col2.button("✏", key=f"rename_{chat_id}"):
        st.session_state.edit_title = True
        st.session_state.new_title = st.session_state.chat_titles.get(chat_id, "")
        st.session_state.current_chat_id = chat_id
    
    # Delete chat button
    if col3.button("🗑", key=f"delete_{chat_id}"):
        delete_chat(chat_id)
        st.rerun()

# 🟢 Main title
st.markdown("<div class='title'>Rafiq</div>", unsafe_allow_html=True)

# Rename current chat
if st.session_state.edit_title:
    col1, col2 = st.columns([3, 1])
    st.session_state.new_title = col1.text_input("Rename Chat:", value=st.session_state.new_title, key="title_input")
    if col2.button("Save"):
        save_title()
        st.rerun()

# 🧠 User input
question = st.text_input("✏ Write Your Biology Question Here", key="user_input", placeholder="EX: What is the function of mitochondria?", label_visibility="collapsed")

if st.button("🚀 Send Question"):
    if question.strip():
        add_to_current_chat("user", question)

        try:
            with st.spinner("Generating Your Answer 🔎..."):
                # ✅ Directly call RAG model
                answer = process_query(question)
        except Exception as e:
            answer = f"⚠️ Error running model: {e}"

        add_to_current_chat("bot", answer)

        # 🔤 Generate automatic title for the chat
        chat_id = st.session_state.current_chat_id
        if chat_id not in st.session_state.chat_titles or not st.session_state.chat_titles[chat_id] or st.session_state.chat_titles[chat_id] == "New Chat":
            title = generate_title_from_text(question)  # Use question instead of answer to generate title
            st.session_state.chat_titles[chat_id] = title

    else:
        st.warning("⚠ Please enter a question before sending.", icon="⚠️")

# 💬 Display current chat
st.divider()
for sender, msg in get_current_chat():
    if sender == "user":
        st.markdown(f"<div class='chat-message' style='background-color:{LIGHT_BLUE}; color:black'>👤 {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-message' style='background-color:{LIGHT_GREEN}'>🤖 {msg}</div>", unsafe_allow_html=True)