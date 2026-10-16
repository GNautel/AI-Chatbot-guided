# =====================================================================
# CHATBOT STAGE 2: MEMORY, PERSONAS AND A CLEAR BUTTON
# =====================================================================
# What is new compared with Stage 1:
#   1. MEMORY    - the conversation is saved in st.session_state and the
#                  whole history is sent to the model on every call.
#   2. PERSONAS  - the user picks a persona (study tutor, translator,
#                  interview coach). A persona is just a different
#                  "system" message.
#   3. CLEAR     - a button in the sidebar empties the conversation.
#   4. LIMIT     - only the latest messages are sent, to control cost.
#
# How to run:
#   1. Put your key in a file called .env  ->  OPENROUTER_API_KEY=sk-or-v1-...
#   2. In the terminal:  streamlit run chatbot_stage2_memory_personas.py
# =====================================================================

# ---------- 1. IMPORTS ----------
import os                           # read environment variables (our API key)
import streamlit as st              # builds the web page
from openai import OpenAI           # the library used to talk to the model
from dotenv import load_dotenv      # loads the .env file

# ---------- 2. SET UP THE CONNECTION TO THE MODEL ----------
load_dotenv()                       # make the values in .env available

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",     # send requests to OpenRouter
    api_key=os.getenv("OPENROUTER_API_KEY")      # key from the .env file
)

# ---------- 3. SETTINGS ----------
# A dictionary of personas. The KEY is the name shown in the menu.
# The VALUE is the system message sent to the model.
# Same model, same code; only this instruction changes the behaviour.
PERSONAS = {
    "Study tutor": "You are a patient tutor. Explain step by step, use simple examples, and end by checking whether the student understood.",
    "Translator (English and Japanese)": "You are a translator. Translate English into Japanese and Japanese into English. Give the translation first, then one short note on tone or nuance.",
    "Interview coach": "You are a job interview coach. Ask one interview question at a time, wait for the answer, then give brief feedback and ask the next question.",
    "Tour guide": "You are a pro tour guide who has travelled the world. Answer questions about any country, city, or tourist attraction. Give practical tips and interesting facts.",
}

# Every message we send costs tokens, and long conversations can exceed the
# model's context window. So we only send the latest 10 messages.
# Trade-off: the bot forgets anything older than this window.
MAX_HISTORY_MESSAGES = 10


# ---------- 4. A FUNCTION THAT ASKS THE MODEL ----------
def ask_model(history, persona):
    """Send the conversation so far to the model.

    history : list of dictionaries like {"role": "user", "content": "Hello"}
    persona : the name of the chosen persona (a key of PERSONAS)
    Returns : (reply text, finish_reason)
    """

    # The system message always goes FIRST. It is rebuilt on every call
    # from the persona the user has selected right now.
    messages = [{"role": "system", "content": PERSONAS[persona]}]

    # history[-10:] means "the last 10 items of the list".
    # We add them after the system message. This is the memory: the model
    # sees the earlier questions and answers every time.
    messages += history[-MAX_HISTORY_MESSAGES:]

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,          # system message + recent conversation
        temperature=0.5,            # 0 = predictable, 1 = more creative
        max_tokens=1500             # generous: some free models spend part of it on hidden thinking
    )

    choice = response.choices[0]
    return choice.message.content, choice.finish_reason


# ---------- 5. THE PAGE (runs again from the top on every rerun) ----------
st.title("Chatbot with Memory and Personas")

# --- Sidebar controls: they appear in the left panel ---

# A drop-down menu. It returns the name of the selected persona.
# list(PERSONAS.keys()) turns the dictionary keys into the menu options.
persona = st.sidebar.selectbox("Persona", list(PERSONAS.keys()))

# A button that returns True only on the rerun right after it is clicked.
# We empty the saved conversation when that happens.
if st.sidebar.button("Clear chat"):
    st.session_state.messages = []

# --- MEMORY: st.session_state ---
# A normal variable such as  messages = []  would be reset to empty on
# every rerun. st.session_state is kept for as long as the browser tab
# stays open, so it can hold the conversation.
# The "if" makes sure we create the empty list only ONCE, the first time.
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- REDRAW THE OLD MESSAGES ---
# Because the page reruns from the top, we must draw every saved message
# again each time. Each message is a dictionary with a "role" ("user" or
# "assistant") and a "content" (the text).
for message in st.session_state.messages:
    with st.chat_message(message["role"]):      # bubble for that role
        st.write(message["content"])

# --- THE CHAT BOX ---
# Returns the typed text when the user presses Enter, otherwise None.
user_message = st.chat_input("Type your message")

if user_message:

    # Step 1: save the new user message in the history.
    st.session_state.messages.append({"role": "user", "content": user_message})

    # Step 2: show it on the page right away.
    with st.chat_message("user"):
        st.write(user_message)

    # Step 3: ask the model, sending the WHOLE history (not just the
    # latest message). This is what gives the bot its memory.
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply, finish_reason = ask_model(st.session_state.messages, persona)

        if not reply:
            # The model sent nothing back. Show an error and remove the
            # unanswered user message so the history stays consistent.
            st.error(f"Empty response (finish_reason: {finish_reason}). Try again.")
            st.session_state.messages.pop()
        else:
            st.write(reply)

            # Step 4: save the assistant reply too, so that on the next
            # turn the model can see both sides of the conversation.
            st.session_state.messages.append({"role": "assistant", "content": reply})

# ---------- THINGS TO TRY ----------
# 1. Ask a question, then a follow-up such as "Give me a simpler example."
#    The bot now understands, because it receives the earlier messages.
# 2. Change the persona in the sidebar and ask the same question again.
# 3. Press "Clear chat" and repeat the follow-up. The memory is gone.
# 4. Change MAX_HISTORY_MESSAGES to 2 and see what the bot forgets.
