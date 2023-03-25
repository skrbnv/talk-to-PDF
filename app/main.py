import io
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams
from fastapi import FastAPI, UploadFile, File, Response, Request
from app.gptlib import GPTInterface
from app.utils import calculate_md5, load_from_cache, save_to_cache, TextUtils
from yaml import safe_load
from scipy.spatial import distance
from fastapi.templating import Jinja2Templates

app = FastAPI()
with open("config.yaml", "r") as cfgf:
    config = safe_load(cfgf)
gpti = GPTInterface(config)
tui = TextUtils()
global_context = {}
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.jinja", 
                                      {
                                        "request": request, 
                                        "uploaded": False if global_context == {} else True 
                                        }
        )


@app.get("/message/")
def proceed_message(query: str = ""):
    if len(global_context) == 0:
        return Response('No global context defined, probably PDF was never uploaded', 400)
    if query != "":
        embeddings = [el['embedding'] for el in global_context.values()]
        evalvec = gpti.compute_embedding(query)
        distances = [distance.cosine(evalvec, embedding) for embedding in embeddings]
        ixs = sorted(range(len(distances)), key=distances.__getitem__)
        keys = [list(global_context.keys())[ix] for ix in ixs]
        response = {}
        context = [global_context[key] for key in keys[:config['top_k_extracts']]]
    else:
        context = []
    response = gpti.chat(query, context)
    metadata = gpti.get_metadata()
    return {
        "query": query,
        "response": response,
        "extracts": [el['text'] for el in context],
        "metadata": metadata
    }


@app.post("/upload")
def upload_file(pdf_file: UploadFile = File(...)):
    global global_context
    md5 = calculate_md5(pdf_file.file)
    raw = load_from_cache(md5, config['cache_dir'])
    if raw is not None:
        data = raw['data']
        metadata = raw['metadata']
        global_context = data
        gpti.refresh_metadata(metadata)
        return data
    pdf_file.file.seek(0)
    paragraphs = {}
    for page_num, page_layout in enumerate(extract_pages(io.BytesIO(pdf_file.file.read()), laparams=LAParams())):
        for pos, element in enumerate(page_layout):
            if hasattr(element, "get_text"):
                if len(extracted_text := tui.extract_text(element)) > 0:
                    if tui.token_count(extracted_text) > config['target_chunk_size']:
                        extracted_texts = tui.split_paragraph(extracted_text, config['target_chunk_size'])
                    else:
                        extracted_texts = [extracted_text]
                for et in extracted_texts:
                    paragraphs[len(paragraphs)] = {
                        'text'  : et.strip(),
                        'tokens': tui.token_count(et.strip()), 
                        'page'  : page_num,
                        'pos'   : pos,
                        'bbox'  : element.bbox
                    }

    text_lookup = ""
    for p in paragraphs.values():
        text_lookup += ' '+p['text']
        if len(text_lookup) > 400:
            break
    gpti.update_metadata(pdf_file.filename, text_lookup.strip())
    data = {}
    roll, tokens, pages = "", 0, []
    for p in paragraphs.values():
        # if possible to stack
        if tokens + p['tokens'] < config['target_chunk_size']:
            roll += ' '+p['text']
            tokens += p['tokens']
            pages.append(p['page']) 
        else:
            # if not possible
            # if we already have something in cache
            if tokens > 0:
                data[len(data)] = {
                    'text'      : roll.strip(),
                    'pages'     : list(set(pages)),
                    'tokens'     : tui.token_count(roll.strip())
                }
            # put new in cache
            roll, tokens, pages = p['text'], p['tokens'], [p['page']]


    # collapse if sum is closer to target chunk size
    for key in (keys := list(data.keys())):
        if key in keys and key+1 in keys:
            d_k1 = abs(config['target_chunk_size'] - data[key]['tokens'])
            d_k2 = abs(config['target_chunk_size'] - data[key+1]['tokens'])
            sum_d = abs(data[key]['tokens']+data[key+1]['tokens']-config['target_chunk_size'])
            if sum_d<int((d_k1+d_k2)/2):
                data[key]['text'] = data.get(key)['text']+' '+data.get(key+1)['text']
                data[key]['tokens'] = data.get(key)['tokens']+data.get(key+1)['tokens']
                data[key]['pages'] = list(set(data.get(key)['pages']+data.get(key+1)['pages']))
                keys.remove(key+1)
                del data[key+1]

    # compute embeddings
    #for key in data.keys():
    #    data[key]['embedding'] = gpti.compute_embedding(data[key]['text'], delay=config['request_delay_embeddings'])
    # save to cache
    save_to_cache(md5, config['cache_dir'], {
        "data": data,
        #"metadata": gpti.get_metadata()
        })
    global_context = data
    gpti.reset_log()
    return data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)