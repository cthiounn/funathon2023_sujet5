import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)
model.load_state_dict(torch.load("ckpt_dares_ceren_bert_multi.pth", map_location=device))

st.text("Version : 1.0 ")

titre = st.text_input('Titre')
commentaire = st.text_area('Commentaire')

if titre and commentaire :
    text=f'TITRE: {titre}' ' |AND| ' + 'COMMENT: {commentaire}'
    inputs =tokenizer.encode_plus(
                text,
                None,
                add_special_tokens=True,
                max_length=128,
                padding="max_length",
                truncation=True
            )
    
    input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(1).permute(1,0)
    attention_mask = torch.tensor(inputs["attention_mask"]).unsqueeze(1).permute(1,0)
    token_type_ids = torch.tensor(inputs["token_type_ids"]).unsqueeze(1).permute(1,0)
    
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    
    loss = outputs.loss
    logits = outputs.logits
    
    _, predicted = torch.max(logits, dim=1)


    st.text('Note :' + '♥'*int(predicted[0]))

