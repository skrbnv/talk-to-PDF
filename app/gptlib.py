import openai
from typing import Any, List
from time import sleep

class GPTInterface():
    def __init__(self, config: dict) -> None:
        if 'API_KEY' not in config.keys():
            raise Exception('Missing API key in YAML config')
        self.api_key          = config['API_KEY']
        openai.api_key        = self.api_key
        self.model            = config['model']
        self.model_embeddings = config['model_embeddings']
        self.log              = []
        self.metadata         = ""
        #print([el['id'] for el in openai.Model.list()['data']])

    def get_metadata(self):
        return self.metadata

    def refresh_metadata(self, data):
        self.metadata = data

    def update_metadata(self, filename, text, retry=3):
        message = [{
            "role"   : "user",
            "content": f"Using filename '{filename}' and context below, try to extract document title and author(s) in following format ####METADATA\n##Author(s):\n##Title\n\nContext: {text}"
        }]
        for _ in range(retry):
            try:
                response = openai.ChatCompletion.create(model=self.model, messages=message, temperature=0)
                raw_text = response['choices'][0]['message']['content']
                if "####METADATA\n" in raw_text:
                    raw_text = raw_text.split('####METADATA\n')[1:]
                self.metadata = raw_text
                return raw_text
            except:
                continue
        else:
            return f"Chat API error: unable to get response after {retry} tries"

    def compute_embedding(self, string, retry=3, delay=0):
        for _ in range(retry):
            try:
                sleep(delay)
                response = openai.Embedding.create(
                    model=self.model_embeddings,
                    input=string
                    )
                return response['data'][0]['embedding']
            except:
                continue
        else:
            raise Exception('Cannot compute embeddings')

    def reset_log(self):
        self.log = []

    def logger(self, role: str, message: str):
        self.log.append({
            "role": role,
            "content": message
        })
        return True
    
    def generate_chat_sequence(self, context=[]):
        if len(context) > 0:
            inject = ""
            for el in context:
                inject += f"Page(s) {el['pages']}: {el['text']}\n"
            messages = [
                {
                    "role": "system",
                    "content": f"You are a helpful assitant. Answer user questions using document metadata: {self.metadata} and extracts from document: \n{inject}",
                }
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assitant. Your task is to answer user questions using all your knowledge.",
                }
            ]
        for record in self.log:
            messages.append({
                "role": record['role'],
                "content": record['content']
            })
        return messages

    def chat_request(self, sequence, retry=3):
        for _ in range(retry):
            try:
                response = openai.ChatCompletion.create(model=self.model, messages=sequence, temperature=0.7)
                return response['choices'][0]['message']['content']
            except:
                continue
        else:
            return f"Chat API error: unable to get response after {retry} tries"


    def chat(self, msg: str, context=[], retry=3) -> Any:
        if msg == "":
            return self.log
        self.logger("user", msg)
        sequence = self.generate_chat_sequence(context)
        reply = self.chat_request(sequence, retry=retry)
        self.logger("assistant", reply)
        return self.log