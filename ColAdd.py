import os
#Slack
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request

# Langchain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.schema.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain

#Chroma
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

chroma_client = chromadb.PersistentClient(path="vectordb", settings = Settings(allow_reset=True))

openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-ada-002")
model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name)

# GET OR CREATE PERSISTENT COLLECTION
collection = chroma_client.get_or_create_collection(name="nextcollection",embedding_function=openai_ef)
#collection = chroma_client.create_collection(name="nextcollection", embedding_function=openai_ef)

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set Slack API credentials
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)

# Initialize the Flask app
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])

# Store conversation history
conversation_history = []
# ID of the channel you want to send the message to
channel_id = "C06367B1E2G" #eirikgpt channel

messages_texts = []
messages_timestamps = []

try:
    # Fetch the channel history
    result = slack_client.conversations_history(channel=channel_id)
    messages = result['messages']

    # Iterate over each message
    for message in messages:
        # Initialize a variable to hold the concatenated text and timestamp string for each message
        full_text = message.get("text", "")
        full_timestamps = message.get("ts", "")

        # Check for thread replies in the message
        if "thread_ts" in message:
            # Fetch the thread replies
            thread_replies = slack_client.conversations_replies(channel=channel_id, ts=message["thread_ts"])
            replies = thread_replies['messages']

            # Skip the first message because it is the original message already included
            for reply in replies[1:]:
                # Append each reply's text to the full_text string
                full_text += " " + reply.get("text", "")
                full_timestamps += " " + reply.get("ts", "")

        # Append the full text of the message and its replies to the messages_texts list
        messages_texts.append(full_text)
        # Append the full timestamps string to the messages_timestamps list
        messages_timestamps.append(full_timestamps)

        collection.add(
            documents=messages_texts,
            ids = messages_timestamps
        )

except SlackApiError as e:
    print(f"Error fetching messages: {e}")
