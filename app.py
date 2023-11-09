import os
# Slack
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
from langchain.chains import RetrievalQA
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Chroma
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings


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

# Initialize the Slack client with bot token
slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])

# slackbot function
def slackbot(user_input):
    persist_directory = 'vectordb' #My local dir /Users/eirikodegaard/Desktop/NextGPT/NextGPT/vectordb
    chroma_client = chromadb.PersistentClient(path=persist_directory, settings = Settings(allow_reset=True))

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-ada-002")
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model_name)

    # GET OR CREATE PERSISTENT COLLECTION
    collection = chroma_client.get_or_create_collection(name="nextcollection",embedding_function=openai_ef)

    # ID of the channel you want to send the message to
    channel_id = "C06367B1E2G" #eirikgpt channel

    lc = Chroma(
        client=chroma_client,
        collection_name="nextcollection",
        embedding_function=OpenAIEmbeddings(),
    )
    
    question = user_input
    # Search Chroma for similar string
    docs = lc.similarity_search(question,k=1)
    #print(docs[0].page_content)
    human = docs[0].page_content
    
    # Prompt for the LLM receiving the content from Chroma
    # Making it strictly not answer outside of Chroma context
    template = """
    Text for Reference:
    {human}

    Instructions:
    Using only the information from the provided text, answer the questions that follow. Do not infer or use any external knowledge or database to inform your answers, including any information about geographical locations, entities, or other subjects not explicitly mentioned in the text. If you can't find the answer, say "Sorry, I could not find and answer based on the messages in this channel.".

    Question:
    {question}

    Answer in a sentence:

    Based on the messages in this channel: [Your answer]


    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = question
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=llm, prompt=chat_prompt)
    response = chain.run({'human': human, 'question': question})
    print(response)
    return response

def get_bot_user_id():
    """
    Get the bot user ID using the Slack API.
    Returns:
        str: The bot user ID.
    """
    try:
        response = slack_client.auth_test()
        return response["user_id"]
    except SlackApiError as e:
        print(f"Error: {e}")

# Listen for mention in Slack
@app.event("app_mention")   
def handle_mentions(body, say):
    """
    Event listener for mentions in Slack.
    When the bot is mentioned, this function processes the text and sends a response.

    Args:
        body (dict): The event data received from Slack.
        say (callable): A function for sending a response to the channel.
    """
    text = body["event"]["text"]

    mention = f"<@{SLACK_BOT_USER_ID}>"
    text = text.replace(mention, "").strip()

    response = slackbot(text)
    say(response)

# Route for handling slack events
# This is called for when updating Event Subscriptions with NGROK
@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    print("EVENTS")
    """
    Route for handling Slack events.
    This function passes the incoming HTTP request to the SlackRequestHandler for processing.

    Returns:
        Response: The result of handling the request.
    """
    return handler.handle(request)


# Run the Flask app
if __name__ == "__main__":
    flask_app.run()