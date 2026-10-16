import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
  azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
  api_key=os.environ["AZURE_OPENAI_API_KEY"],
  api_version=os.environ["OPENAI_API_VERSION"],
)

response = client.chat.completions.create(
  model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
  messages=[
          {
            "role": "user",
            "content": "How many r's are in the word 'strawberry'?"
          }
        ]
)

response = response.choices[0].message

# messages = [
#   {"role": "user", "content": "How many r's are in the word 'strawberry'?"},
#   {
#     "role": "assistant",
#     "content": response.content,
#     "reasoning_details": response.reasoning_details  # Pass back unmodified
#   },
#   {"role": "user", "content": "Are you sure? Think carefully."}
# ]



print(response.content)
