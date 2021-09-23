import streamlit as st
import pandas as pd
import json

@st.cache
def get_data():
    with open('test_prediction_check.json') as f:
        predictions = json.load(f)
    pred_dict = {p['name']: p for p in predictions}

    return pred_dict

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

local_css("style.css")

#format the images list
categories = ['DISHES', 'TASTE_PROFILE', 'DIFFICULTY', 'MEALS_AND_COURSES', 'MENU', 'CUISINE', 'SPECIAL_DIETS']
sel = get_data()
option = st.selectbox('Pick a recipe', sel.keys())
threshold = st.slider("Set Threshold", min_value=0.00, max_value=1.0, step=0.01, value=0.5)

#sel2 = sel[['model', 'html', 'rank']].pivot(index='rank', columns="model", values="html")

#show the list of images as a dataframe
p = sel[option]
all_labels = []
tp = []
fn = []
fp = []
for c in categories:
    if c not in p:
        continue
    predictions = set()
    for conf, tag in zip(p[c]['confidences'], p[c]['possible_tags']):
        if float(conf) >= threshold:
            predictions.add(tag)
    tp.extend(list(set(p[c]['label']) & predictions))
    fp.extend(list(predictions - set(p[c]['label'])))
    fn.extend(list(set(p[c]['label']) - predictions))
    all_labels.extend(set(p[c]['label']))
st.markdown('**DESCRIPTION:**' +  str(p.get('description', 'None')))
st.markdown(f'**LABELS BY CONTENT TEAM:** {", ".join(all_labels)}')
st.markdown(f"**TRUE POSITIVES:** <span class='highlight green'> {', '.join(tp)} </span>", unsafe_allow_html=True)
st.markdown(f"**FALSE POSITIVES (we should not have predicted them):** <span class='highlight red'>{', '.join(fp)} </span>", unsafe_allow_html=True)
st.markdown(f"**FALSE NEGATIVES (we should've predicted them):** <span class='highlight red'>{', '.join(fn)} </span>", unsafe_allow_html=True)
st.markdown(f"**PRECISION:** {len(tp) / (len(tp) + len(fp))}")
st.markdown(f"**RECALL:** {len(tp) / (len(tp) + len(fn))}")