# %%
from dotenv import load_dotenv
load_dotenv()

# %%
import boto3
client = boto3.client('comprehend')

# %%
mytxt = """My name is Joe Smith and I was born in 1988 and I am 33 years old.
I work at Predictive Hacks and my email is joe.smith@predictivehacks.com. 
I live in Athens, Greece. My phone number is 623 12 34 567 and my bank account is 123-123-567-888"""
response = client.detect_pii_entities(
    Text= mytxt,
    LanguageCode='en'
)
# get the response
response

clean_text = mytxt
# reversed to not modify the offsets of other entities when substituting
for NER in reversed(response['Entities']):
    clean_text = clean_text[:NER['BeginOffset']] + NER['Type'] + clean_text[NER['EndOffset']:]
print(clean_text)


# %%
