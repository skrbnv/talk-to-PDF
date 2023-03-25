import hashlib
import re
import os
import pickle
from transformers import GPT2Tokenizer


class TextUtils():
    def __init__(self) -> None:
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def token_count(self, string):
        return len(self.tokenizer(string)['input_ids'])

    def split_paragraph(self, paragraph, target_count):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)
        output = []
        text, token_count = "", 0        
        for s in sentences:
            tokens = self.token_count(s)
            if token_count + tokens > target_count:
                if token_count > 0:
                    output.append(text)
                    text, token_count = s, tokens
                else:
                    output.append(s)
            else:
                token_count += tokens
                text += ' '+s
        else:
            output.append(text.strip())
        return output                    

    def extract_text(self, element):
        pattern = r'(?<=\B[^\W\d])[ ]*-\n[ ]*(?=[^\W\d]\B)'
        raw = element.get_text().strip()
        text = re.sub(pattern, '', raw)
        text = text.replace('\n', ' ')
        text = re.sub(' +', ' ', text)
        return text



def calculate_md5(f):
    file_hash = hashlib.md5()
    while chunk := f.read(8192):
        file_hash.update(chunk)
    return file_hash.hexdigest()


def load_from_cache(checksum, cache_dir):
    if os.path.isfile(os.path.join(cache_dir, checksum)):
        with open(os.path.join(cache_dir, checksum), 'rb') as f:
            data = pickle.load(f)
        return data
    return None

def save_to_cache(checksum, cache_dir, data):
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    with open(os.path.join(cache_dir, checksum), 'wb') as f:
        data = pickle.dump(data, f)
