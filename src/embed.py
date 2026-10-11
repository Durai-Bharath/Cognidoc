from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.environ["OPENROUTER_API_KEY"],
)

embedding = client.embeddings.create(
  
  model="liquid/lfm-2.5-embedding-350m:free",
  input="Your text string goes here",
  # input: ["text1", "text2", "text3"] # batch embeddings also supported!
  encoding_format="float"
)
print(embedding.data[0].embedding)