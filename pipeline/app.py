from flask import Flask, render_template, request
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer
import torch
import requests

app = Flask(__name__)

data = {
    "text": "",
    "entity_list": []
}

model_path = "./model/SC"
model_SC = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer_SC = AutoTokenizer.from_pretrained(model_path)
model_SC.eval()

model_path = "./model/NER"
model_NER = AutoModelForTokenClassification.from_pretrained(model_path)
tokenizer_NER = AutoTokenizer.from_pretrained(model_path)

def section_classifier(text):
    inputs = tokenizer_SC(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_SC(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class

def start_pipeline():
    data_list = []
    for i in data["text"]:
        predicted_class = section_classifier(i)
        if predicted_class == 1:
            data_list.append(i)

    print(data_list)

    text_NER = ""
    for j in data_list:
        text_NER = text_NER + j

    # Process the text using a tokenizer
    inputs = tokenizer_NER(
        text_NER,
        return_tensors="pt",
        truncation=True,
        padding=True,
        return_offsets_mapping=True
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    offset_mapping = inputs["offset_mapping"]

    model_NER.eval()
    with torch.no_grad():
        outputs = model_NER(input_ids, attention_mask=attention_mask)

    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()

    label_map = {0: 'O', 1: 'B-HPO_TERM', 2: 'I-HPO_TERM'}

    tokens = tokenizer_NER.convert_ids_to_tokens(input_ids[0])
    current_entity = None
    entities = []

    for idx, (token, pred) in enumerate(zip(tokens, predictions)):
        label = label_map[pred]

        # Skip special tokens (such as [CLS], [SEP], [PAD])
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        # Handle entity labels
        if label.startswith("B-"):
            entity_type = label[2:]
            start, end = offset_mapping[0][idx]
            current_entity = {
                "entity": entity_type,
                "start": start,
                "end": end,
                "word": text_NER[start:end]
            }
            entities.append(current_entity)
        elif label.startswith("I-") and current_entity:
            current_entity["end"] = offset_mapping[0][idx][1]
            current_entity["word"] = text_NER[current_entity["start"]:current_entity["end"]]

    for entity in entities:
        print(f"Entity: {entity['word']}, Type: {entity['entity']}, Position: ({entity['start']}, {entity['end']})")

        url = "https://api-v3.monarchinitiative.org/v3/api/search"
        q = entity['word']
        url = url + "?q="+q+"&category=biolink%3APhenotypicFeature&limit=20&offset=0"

        response = requests.get(url)

        if response.status_code == 200:
            data_res = response.json()
            print("success！")
            hpo_list = data_res['items']
            print(hpo_list)
        else:
            print(f"Request failed！code: {response.status_code}")

        temp = {
            "entity": entity['word'],
            "hpo_list": hpo_list,
        }

        data["entity_list"].append(temp)


@app.route('/')
def home():
    return render_template('index.html', data=data)

@app.route('/input', methods=['POST'])
def get_data():
    text = request.form.get('text')
    if text:
        lines = text.splitlines()
        data["text"] = lines
        data["entity_list"] = []
        start_pipeline()
    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)



