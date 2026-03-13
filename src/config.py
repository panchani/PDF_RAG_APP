import yaml
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def load_config():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

embedding_model = HuggingFaceEmbeddings(
    model_name=config["embedding"]["model_name"],
    cache_folder=config["embedding"]["cache_folder"]
)

def get_embedding_model():
    return embedding_model