import streamlit as st

# 자주 틀리는 단어에 대한 사용자 정의 제안 사전
@st.cache_resource
def get_custom_suggestions():
    return {
        'tonite': ['tonight'],
        'tomorow': ['tomorrow'],
        'yestarday': ['yesterday'],
        'definately': ['definitely'],
        'recieve': ['receive'],
        'occurence': ['occurrence'],
        'calender': ['calendar'],
        'wierd': ['weird'],
        'alot': ['a lot'],
        'untill': ['until'],
        'thier': ['their'],
        'truely': ['truly'],
        'begining': ['beginning'],
        'beleive': ['believe'],
        'seperate': ['separate'],
        'goverment': ['government'],
        'neccessary': ['necessary'],
        'occasionaly': ['occasionally'],
        'independant': ['independent'],
        'basicly': ['basically'],
        'wich': ['which'],
        'wont': ["won't"],
        'cant': ["can't"],
        'dont': ["don't"],
        'isnt': ["isn't"],
        'didnt': ["didn't"],
        'couldnt': ["couldn't"],
        'shouldnt': ["shouldn't"],
        'wouldnt': ["wouldn't"],
        'doesnt': ["doesn't"],
        'wasnt': ["wasn't"],
        'werent': ["weren't"],
        'havent': ["haven't"],
        'hasnt': ["hasn't"],
        'hadnt': ["hadn't"],
        'arent': ["aren't"]
    } 