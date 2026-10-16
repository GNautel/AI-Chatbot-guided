# =====================================================================
# CHATBOT STAGE 1: A BASIC CHATBOT (NO MEMORY)
# =====================================================================
# What this app does:
#   The user types a message, the model answers it, and the answer is
#   shown on the page. Each message is answered on its own, so the model
#   does NOT remember what was said before. We fix that in Stage 2.
#
# How to run:
#   1. Put your key in a file called .env  ->  OPENROUTER_API_KEY=sk-or-v1-...
#   2. In the terminal:  streamlit run chatbot_stage1_basic.py
# =====================================================================

# ---------- 1. IMPORTS ----------
import os                           # lets us read environment variables (our API key)
import streamlit as st              # Streamlit builds the web page for us
from openai import OpenAI           # the OpenAI library; OpenRouter works with it too
from dotenv import load_dotenv      # reads the .env file so the key is not in the code

# ---------- 2. SET UP THE CONNECTION TO THE MODEL ----------
# Read the .env file and make its values available through os.getenv().
load_dotenv()

# Create the client that sends our messages to OpenRouter.
# base_url points the OpenAI library at OpenRouter instead of OpenAI.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")   # the key comes from the .env file
)


# ---------- 3. A FUNCTION THAT ASKS THE MODEL ----------
def ask_model(user_message):
    """Send ONE user message to the model and return (reply, finish_reason)."""

    # chat.completions.create sends our messages and waits for the reply.
    response = client.chat.completions.create(
        model="openrouter/free",    # OpenRouter picks a free model for us

        # "messages" is the whole conversation we send to the model.
        # Here it contains only two items, which is why the bot has no memory:
        #   - the "system" message sets the behaviour of the assistant
        #   - the "user" message is what the person just typed
        messages=[
            {"role": "system", "content": "You are a helpful assistant for university students."},
            {"role": "user", "content": user_message}
        ],

        temperature=0.5,            # 0 = predictable, 1 = more creative
        max_tokens=1500             # maximum length of the reply
        # Why 1500? openrouter/free can choose a "reasoning" model that uses
        # part of this limit to think silently. A small limit can leave no
        # room for the visible answer, and the reply comes back empty.
    )

    # The reply is inside response.choices[0]. We take the text (content)
    # and also the finish_reason, which tells us WHY the model stopped.
    choice = response.choices[0]
    return choice.message.content, choice.finish_reason


# ---------- 4. THE PAGE (everything below runs on every rerun) ----------
# Streamlit reruns this whole file from top to bottom each time the user
# does something, such as sending a message.
st.title("Basic Chatbot")
st.write("Ask a question. Then try a follow-up that depends on your first one.")

# st.chat_input shows a message box at the bottom of the page.
# It returns the text when the user presses Enter, otherwise None.
user_message = st.chat_input("Type your message")

# This block only runs when the user has sent a message.
if user_message:

    # Draw a chat bubble for the user and show their message inside it.
    with st.chat_message("user"):
        st.write(user_message)

    # Draw a chat bubble for the assistant.
    with st.chat_message("assistant"):

        # Show a spinner while we wait for the model.
        with st.spinner("Thinking..."):
            reply, finish_reason = ask_model(user_message)

        # If the reply is empty, show a clear error instead of a blank bubble.
        if not reply:
            st.error(f"Empty response (finish_reason: {finish_reason}). Try again.")
        else:
            st.write(reply)

# ---------- THINGS TO NOTICE ----------
# 1. Ask "What is a transformer model?" and then "Give me a simpler example
#    of that." The bot does not know what "that" means.
# 2. The earlier chat bubbles disappear after each message. Streamlit reran
#    the file, and normal variables are reset on every rerun.
# Both problems are solved in chatbot_stage2_memory_personas.py.
