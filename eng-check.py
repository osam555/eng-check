import streamlit as st
import spacy
import transformers

# 페이지 설정 (가장 먼저 호출해야 함)
st.set_page_config(
    page_title="영작문 자동 첨삭 시스템",
    page_icon="📝",
    layout="wide"
)

# 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from collections import Counter
import re
import io
import random
from datetime import datetime
from textblob import TextBlob
import asyncio
import edge_tts
import tempfile
import os
import base64

# NLTK 라이브러리 및 데이터 처리
import nltk
from nltk.corpus import stopwords

# 변환기 모듈 존재 여부 확인
has_transformers = 'transformers' in globals()

# 대체 맞춤법 검사 라이브러리 설정
try:
    from spellchecker import SpellChecker
    spell = SpellChecker()
    has_spellchecker = True
except ImportError:
    has_spellchecker = False
    try:
        # 대체 맞춤법 검사 라이브러리로 enchant 시도
        import enchant
        has_enchant = True
    except ImportError:
        has_enchant = False
    #st.info("맞춤법 검사 라이브러리가 설치되지 않았습니다. TextBlob을 사용한 기본 맞춤법 검사만 제공됩니다.")

# NLTK 필요 데이터 다운로드 (Streamlit Cloud에서도 작동하도록 ssl 검증 무시)
try:
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    st.warning(f"NLTK 데이터 다운로드 중 오류가 발생했습니다: {e}")

# 정규식 패턴 컴파일 캐싱
@st.cache_resource
def get_compiled_patterns():
    return {
        'sentence_split': re.compile(r'(?<=[.!?])\s+'),
        'word_tokenize': re.compile(r'\b[\w\'-]+\b'),
        'punctuation': re.compile(r'[.,!?;:"]')
    }

# 수정된 sent_tokenize 함수 (NLTK 의존성 제거)
def custom_sent_tokenize(text):
    if not text:
        return []
    
    # 정규식 기반 문장 분할기
    # 온점, 느낌표, 물음표 뒤에 공백이 오는 패턴을 기준으로 분할
    patterns = get_compiled_patterns()
    sentences = patterns['sentence_split'].split(text)
    # 빈 문장 제거
    return [s.strip() for s in sentences if s.strip()]

# 수정된 word_tokenize 함수 (NLTK 의존성 제거)
def custom_word_tokenize(text):
    if not text:
        return []
    
    # 효과적인 정규식 패턴으로 단어 토큰화
    # 축약형(I'm, don't 등), 소유격(John's), 하이픈 단어(well-known) 등을 처리
    patterns = get_compiled_patterns()
    words = patterns['word_tokenize'].findall(text)
    # 구두점
    punctuation = patterns['punctuation'].findall(text)
    
    tokens = []
    tokens.extend(words)
    tokens.extend(punctuation)
    
    return tokens

# 세션 상태 초기화
if 'history' not in st.session_state:
    st.session_state.history = []

# 현재 선택된 탭 추적 (student_page의 경우)
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = 0  # 기본 탭은 0(영작문 검사)

# 맞춤법 사전 초기화
@st.cache_resource
def get_spell_checker():
    if has_spellchecker:
        return spell
    elif has_enchant:
        try:
            return enchant.Dict("en_US")
        except:
            return None
    return None

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

# 문법 검사 함수 (TextBlob 사용)
def check_grammar(text):
    if not text.strip():
        return []
    
    errors = []
    
    try:
        blob = TextBlob(text)
        
        # 문장 단위로 분석
        sentences = custom_sent_tokenize(text)
        
        # 맞춤법 검사
        checker = get_spell_checker()
        words = custom_word_tokenize(text)
        
        # 사용자 정의 제안 사전 로드
        custom_suggestions = get_custom_suggestions()
        
        # 문장의 시작 위치를 추적하기 위한 변수
        offset = 0
        
        for sentence in sentences:
            # 간단한 문법 검사 (TextBlob의 sentiment 사용)
            try:
                sentence_blob = TextBlob(sentence)
                
                # 문장의 단어 분석
                for word in custom_word_tokenize(sentence):
                    if word.isalpha():  # 알파벳 단어만 확인
                        # 맞춤법 확인 (enchant가 있는 경우에만)
                        if checker and has_enchant and not checker.check(word):
                            # 사용자 정의 제안이 있는지 먼저 확인
                            word_lower = word.lower()
                            if word_lower in custom_suggestions:
                                suggestions = custom_suggestions[word_lower]
                            else:
                                # 기존 라이브러리의 제안 사용
                                suggestions = checker.suggest(word)[:3]  # 최대 3개 제안
                            
                            word_offset = text.find(word, offset)
                            
                            if word_offset != -1:
                                errors.append({
                                    "offset": word_offset,
                                    "errorLength": len(word),
                                    "message": f"맞춤법 오류: '{word}'",
                                    "replacements": suggestions
                                })
                        # PySpellChecker 사용 (enchant 대신)
                        elif checker and 'has_spellchecker' in globals() and has_spellchecker and word.lower() not in checker:
                            # 사용자 정의 제안이 있는지 먼저 확인
                            word_lower = word.lower()
                            if word_lower in custom_suggestions:
                                suggestions = custom_suggestions[word_lower]
                            else:
                                # PySpellChecker 제안 사용
                                suggestions = [checker.correction(word)] + list(checker.candidates(word))[:2]
                            
                            word_offset = text.find(word, offset)
                            
                            if word_offset != -1:
                                errors.append({
                                    "offset": word_offset,
                                    "errorLength": len(word),
                                    "message": f"맞춤법 오류: '{word}'",
                                    "replacements": suggestions
                                })
                        # TextBlob을 사용한 맞춤법 검사 대안 (다른 라이브러리가 없는 경우)
                        elif not checker:
                            try:
                                # 사용자 정의 제안이 있는지 먼저 확인
                                word_lower = word.lower()
                                if word_lower in custom_suggestions:
                                    corrected = custom_suggestions[word_lower][0]  # 첫 번째 제안 사용
                                    word_offset = text.find(word, offset)
                                    if word_offset != -1:
                                        errors.append({
                                            "offset": word_offset,
                                            "errorLength": len(word),
                                            "message": f"맞춤법 오류: '{word}'",
                                            "replacements": custom_suggestions[word_lower]
                                        })
                                else:
                                    # TextBlob을 사용한 기존 로직
                                    word_blob = TextBlob(word)
                                    corrected = str(word_blob.correct())
                                    if corrected != word and len(word) > 3:  # 짧은 단어는 무시
                                        word_offset = text.find(word, offset)
                                        if word_offset != -1:
                                            errors.append({
                                                "offset": word_offset,
                                                "errorLength": len(word),
                                                "message": f"맞춤법 오류: '{word}'",
                                                "replacements": [corrected]
                                            })
                            except Exception as e:
                                # TextBlob 맞춤법 검사 오류 무시
                                pass
            except Exception as e:
                # 문장 분석 중 오류 발생시 해당 문장 건너뛰기
                pass
                
            # 다음 문장의 검색을 위해 오프셋 업데이트
            offset += len(sentence)
    except Exception as e:
        st.error(f"텍스트 분석 중 오류가 발생했습니다: {e}")
    
    return errors

# 텍스트 통계 분석 함수
def analyze_text(text):
    if not text.strip():
        return {
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'avg_sentence_length': 0,
            'vocabulary_size': 0
        }
    
    sentences = custom_sent_tokenize(text)
    words = custom_word_tokenize(text)
    words = [word.lower() for word in words if re.match(r'\w+', word)]
    
    word_count = len(words)
    sentence_count = len(sentences)
    avg_word_length = sum(len(word) for word in words) / max(1, word_count)
    avg_sentence_length = word_count / max(1, sentence_count)
    vocabulary_size = len(set(words))
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': round(avg_word_length, 2),
        'avg_sentence_length': round(avg_sentence_length, 2),
        'vocabulary_size': vocabulary_size
    }

# 어휘 분석 함수
def analyze_vocabulary(text):
    if not text.strip():
        return {
            'word_freq': {},
            'pos_dist': {}
        }
    
    words = custom_word_tokenize(text.lower())
    words = [word for word in words if re.match(r'\w+', word)]
    
    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    # 단어 빈도 계산
    word_freq = Counter(filtered_words).most_common(20)
    
    return {
        'word_freq': dict(word_freq)
    }

# 어휘 다양성 점수 계산
def calculate_lexical_diversity(text):
    words = custom_word_tokenize(text.lower())
    words = [word for word in words if re.match(r'\w+', word)]
    
    if len(words) == 0:
        return 0
    
    return len(set(words)) / len(words)

# 단어 빈도 시각화
def plot_word_frequency(word_freq):
    if not word_freq:
        return None
    
    df = pd.DataFrame(list(word_freq.items()), columns=['단어', '빈도'])
    df = df.sort_values('빈도', ascending=True)
    
    fig = px.bar(df.tail(10), x='빈도', y='단어', orientation='h', 
                title='상위 10개 단어 빈도')
    fig.update_layout(height=400)
    
    return fig

# 기본 단어 셋 정의
@st.cache_resource
def default_vocabulary_sets():
    basic_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it'}
    intermediate_words = {'achieve', 'consider', 'determine', 'establish', 'indicate'}
    advanced_words = {'arbitrary', 'cognitive', 'encompass', 'facilitate', 'implicit'}
    return {'basic': basic_words, 'intermediate': intermediate_words, 'advanced': advanced_words}

# 학술 단어 목록
@st.cache_resource
def get_academic_word_list():
    # 학술 단어 목록 (예시)
    return {'analyze', 'concept', 'data', 'environment', 'establish', 'evident', 
            'factor', 'interpret', 'method', 'principle', 'process', 'research', 
            'significant', 'theory', 'variable'}

# 단어 빈도 데이터 로드
@st.cache_resource
def get_word_frequency_data():
    # 영어 단어 빈도 데이터 (예시)
    common_words = {'the': 0.05, 'be': 0.04, 'to': 0.03, 'of': 0.025, 'and': 0.02}
    return common_words

# 고급 동의어 사전
@st.cache_resource
def get_advanced_synonyms():
    return {
        # 기본 형용사
        'good': ['exemplary', 'exceptional', 'impeccable', 'outstanding', 'superb', 'commendable'],
        'bad': ['detrimental', 'deplorable', 'egregious', 'lamentable', 'abysmal', 'substandard'],
        'big': ['immense', 'formidable', 'monumental', 'colossal', 'substantial', 'extensive'],
        'small': ['minuscule', 'negligible', 'infinitesimal', 'diminutive', 'minute', 'marginal'],
        'happy': ['euphoric', 'exuberant', 'ecstatic', 'jubilant', 'delighted', 'elated'],
        'sad': ['despondent', 'crestfallen', 'dejected', 'disconsolate', 'melancholic', 'woeful'],
        'important': ['imperative', 'indispensable', 'paramount', 'pivotal', 'consequential', 'significant'],
        'difficult': ['formidable', 'insurmountable', 'Herculean', 'arduous', 'challenging', 'demanding'],
        'easy': ['effortless', 'rudimentary', 'facile', 'straightforward', 'uncomplicated', 'elementary'],
        'beautiful': ['resplendent', 'breathtaking', 'sublime', 'exquisite', 'magnificent', 'captivating'],
        
        # 추가 형용사
        'interesting': ['intriguing', 'captivating', 'compelling', 'engrossing', 'fascinating', 'riveting'],
        'boring': ['tedious', 'monotonous', 'mundane', 'insipid', 'dull', 'unengaging'],
        'smart': ['brilliant', 'astute', 'sagacious', 'ingenious', 'erudite', 'perspicacious'],
        'stupid': ['obtuse', 'vacuous', 'inane', 'fatuous', 'imbecilic', 'absurd'],
        'fast': ['expeditious', 'prompt', 'accelerated', 'swift', 'rapid', 'nimble'],
        'slow': ['languorous', 'leisurely', 'sluggish', 'plodding', 'unhurried', 'dilatory'],
        
        # 자주 사용되는 동사
        'say': ['articulate', 'pronounce', 'proclaim', 'assert', 'expound', 'enunciate'],
        'think': ['contemplate', 'ponder', 'deliberate', 'ruminate', 'cogitate', 'muse'],
        'see': ['observe', 'perceive', 'discern', 'witness', 'behold', 'scrutinize'],
        'use': ['utilize', 'employ', 'implement', 'leverage', 'harness', 'apply'],
        'make': ['construct', 'fabricate', 'forge', 'produce', 'generate', 'devise'],
        'get': ['acquire', 'obtain', 'procure', 'attain', 'secure', 'garner']
    }

# 고급 표현 패턴
@st.cache_resource
def get_advanced_phrases():
    return {
        # 기본 표현 고급화
        r'\bi think\b': ['I postulate that', 'I am of the conviction that', 'It is my considered opinion that', 'I firmly believe that', 'From my perspective', 'I have come to the conclusion that'],
        r'\bi like\b': ['I am particularly enamored with', 'I hold in high regard', 'I find great merit in', 'I am deeply appreciative of', 'I have a profound affinity for', 'I derive considerable pleasure from'],
        r'\bi want\b': ['I aspire to', 'I am inclined towards', 'My inclination is toward', 'I earnestly desire', 'I have a vested interest in', 'My objective is to'],
        r'\blots of\b': ['a plethora of', 'an abundance of', 'a multitude of', 'a substantial amount of', 'a considerable quantity of', 'a significant number of'],
        r'\bmany of\b': ['a preponderance of', 'a substantial proportion of', 'a significant contingent of', 'a notable segment of', 'a sizable fraction of', 'a considerable percentage of'],
        
        # 문장 시작 부분 향상
        r'^In my opinion\b': ['From my perspective', 'According to my assessment', 'Based on my evaluation', 'In my estimation', 'As I perceive it', 'In my considered judgment'],
        r'^I agree\b': ['I concur with the assessment that', 'I am in complete accord with', 'I share the sentiment that', 'I am aligned with the view that', 'I endorse the position that', 'I subscribe to the notion that'],
        r'^I disagree\b': ['I take exception to', 'I contest the assertion that', 'I must respectfully differ with', 'I cannot reconcile myself with', 'I find myself at variance with', 'I am compelled to challenge the idea that'],
        
        # 한국어 학습자가 자주 사용하는 문구 대체
        r'\bit is important to\b': ['it is imperative to', 'it is essential to', 'it is crucial to', 'it is of paramount importance to', 'it is a fundamental necessity to', 'it is a critical requirement to'],
        r'\bin conclusion\b': ['in summation', 'to synthesize the aforementioned points', 'in culmination', 'as a final observation', 'to encapsulate the preceding discussion', 'as the logical denouement'],
        r'\bfor example\b': ['as an illustrative case', 'to cite a pertinent instance', 'as a demonstrative example', 'to exemplify this concept', 'as a representative case in point', 'to elucidate through a specific example']
    }

def rewrite_similar_level(text):
    """비슷한 수준으로 텍스트 재작성 - 간단한 동의어 교체"""
    sentences = custom_sent_tokenize(text)
    rewritten = []
    
    # 간단한 동의어 사전
    synonyms = {
        'good': ['nice', 'fine', 'decent'],
        'bad': ['poor', 'unfortunate', 'unpleasant'],
        'big': ['large', 'sizable', 'substantial'],
        'small': ['little', 'tiny', 'slight'],
        'happy': ['glad', 'pleased', 'content'],
        'sad': ['unhappy', 'down', 'blue']
    }
    
    for sentence in sentences:
        words = custom_word_tokenize(sentence)
        new_words = []
    
        for word in words:
            word_lower = word.lower()
            # 20% 확률로 동의어 교체 시도
            if word_lower in synonyms and random.random() < 0.2:
                replacement = random.choice(synonyms[word_lower])
                # 대문자 보존
                if word[0].isupper():
                    replacement = replacement.capitalize()
                new_words.append(replacement)
            else:
                new_words.append(word)
        
        rewritten.append(' '.join(new_words))
    
    return ' '.join(rewritten)

def rewrite_improved_level(text):
    """약간 향상된 수준으로 텍스트 재작성 - 중급 동의어로 대체하고 구문 개선"""
    sentences = custom_sent_tokenize(text)
    rewritten = []
    
    # 중급 동의어 사전
    intermediate_synonyms = {
        'good': ['beneficial', 'favorable', 'quality'],
        'bad': ['negative', 'inferior', 'flawed'],
        'big': ['significant', 'considerable', 'extensive'],
        'small': ['minimal', 'limited', 'minor'],
        'say': ['state', 'mention', 'express'],
        'think': ['believe', 'consider', 'reflect'],
        'important': ['essential', 'significant', 'crucial']
    }
    
    # 개선할 구문 패턴
    phrase_patterns = {
        r'\bi think\b': ['In my opinion', 'I believe', 'From my perspective'],
        r'\ba lot\b': ['considerably', 'significantly', 'substantially'],
        r'\bvery\b': ['notably', 'particularly', 'especially']
    }
    
    for sentence in sentences:
        # 동의어 교체
        words = custom_word_tokenize(sentence)
        new_words = []
    
        for word in words:
            word_lower = word.lower()
            # 30% 확률로 중급 동의어 교체 시도
            if word_lower in intermediate_synonyms and random.random() < 0.3:
                replacement = random.choice(intermediate_synonyms[word_lower])
                # 대문자 보존
                if word[0].isupper():
                    replacement = replacement.capitalize()
                new_words.append(replacement)
            else:
                new_words.append(word)
        
        improved = ' '.join(new_words)
        
        # 구문 패턴 개선
        for pattern, replacements in phrase_patterns.items():
            if re.search(pattern, improved, re.IGNORECASE) and random.random() < 0.4:
                improved = re.sub(pattern, random.choice(replacements), improved, flags=re.IGNORECASE)
        
        rewritten.append(improved)
    
    return ' '.join(rewritten)

def rewrite_advanced_level(text):
    """고급 수준으로 텍스트 재작성 - 고급 어휘와 표현으로 변환"""
    # 문장 토큰화
    sentences = custom_sent_tokenize(text)
    advanced_sentences = []
    
    # 고급 동의어 및 표현 가져오기
    advanced_synonyms = get_advanced_synonyms()
    advanced_phrases = get_advanced_phrases()
    
    for sentence in sentences:
        # 동의어 교체
        for word, replacements in advanced_synonyms.items():
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, sentence, re.IGNORECASE) and random.random() < 0.5:
                replacement = random.choice(replacements)
                # 원래 단어가 대문자로 시작하면 대체어도 대문자로 시작
                if re.search(pattern, sentence).group(0)[0].isupper():
                    replacement = replacement.capitalize()
                sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
        
        # 고급 표현으로 교체
        for pattern, replacements in advanced_phrases.items():
            if re.search(pattern, sentence, re.IGNORECASE) and random.random() < 0.6:
                replacement = random.choice(replacements)
                sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
        
        advanced_sentences.append(sentence)
    
    # 재작성된 텍스트 반환
    return ' '.join(advanced_sentences)

# 고급 재작성 함수
def advanced_rewrite_text(text, level='advanced'):
    # 변환기 모델이 없으면 규칙 기반 방식으로 폴백
    if not has_transformers:
        return rewrite_advanced_level(text)
    
    try:
        # 간단한 규칙 기반 고급 재작성 (transformers 모델 없을 때)
        return rewrite_advanced_level(text)
    except Exception as e:
        st.warning(f"고급 재작성 처리 중 오류: {e}")
        return rewrite_advanced_level(text)

def rewrite_text(text, level="similar"):
    """
    입력 텍스트를 지정된 레벨에 따라 재작성합니다.
    
    Parameters:
    - text: 재작성할 텍스트
    - level: 재작성 레벨 ("similar", "improved", "advanced" 중 하나)
    
    Returns:
    - 재작성된 텍스트
    """
    if not text:
        return ""
    
    # 레벨에 따라 적절한 재작성 함수 호출
    if level == "similar":
        return rewrite_similar_level(text)
    elif level == "improved":
        return rewrite_improved_level(text)
    elif level == "advanced":
        if has_transformers:
            return advanced_rewrite_text(text, level)
        else:
            return rewrite_advanced_level(text)
    else:
        return text  # 기본값은 원본 텍스트 반환

# 학생 페이지
def evaluate_vocabulary_level(text):
    # 온라인 데이터셋에서 어휘 로드
    vocabulary_sets = default_vocabulary_sets()
    
    # 온라인 데이터셋 로드 시도
    try:
        import requests
        
        # 영어 단어 빈도 데이터 다운로드
        word_freq_url = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/en/en_50k.txt"
        response = requests.get(word_freq_url)
        if response.status_code == 200:
            # 단어 빈도 데이터 파싱 (형식: "단어 빈도")
            lines = response.text.splitlines()
            words = [line.split()[0] for line in lines if ' ' in line]
            
            # 빈도에 따라 단어 분류
            total_words = len(words)
            basic_cutoff = int(total_words * 0.2)  # 상위 20%
            intermediate_cutoff = int(total_words * 0.5)  # 상위 20~50%
            
            basic_words = set(words[:basic_cutoff])
            intermediate_words = set(words[basic_cutoff:intermediate_cutoff])
            advanced_words = set(words[intermediate_cutoff:])
            
            vocabulary_sets = {'basic': basic_words, 'intermediate': intermediate_words, 'advanced': advanced_words}
    except Exception as e:
        pass
    
    words = custom_word_tokenize(text.lower())
    words = [word for word in words if re.match(r'\w+', word)]
    
    word_set = set(words)
    
    basic_count = len(word_set.intersection(vocabulary_sets['basic']))
    intermediate_count = len(word_set.intersection(vocabulary_sets['intermediate']))
    advanced_count = len(word_set.intersection(vocabulary_sets['advanced']))
    
    total = basic_count + intermediate_count + advanced_count
    if total == 0:
        return {'basic': 0, 'intermediate': 0, 'advanced': 0}
    
    return {
        'basic': basic_count / max(1, total),
        'intermediate': intermediate_count / max(1, total),
        'advanced': advanced_count / max(1, total)
    }

# 학생 페이지
def show_student_page():
    st.title("영작문 자동 첨삭 시스템 - 학생")
    
    # 로그아웃 버튼
    if st.button("로그아웃", key="student_logout"):
        st.session_state.user_type = None
        st.rerun()
    
    # 탭 인덱스를 세션 상태에서 가져옴
    tab_index = st.session_state.selected_tab
    
    # 탭 생성 - selected_tab에 따라 초기 선택 
    tabs = st.tabs(["영작문 검사", "영작문 재작성", "내 작문 기록"])
    
    # 현재 선택된 탭을 보여줌
    with tabs[tab_index]:
        pass

    # 영작문 검사 탭
    with tabs[0]:
        st.subheader("영작문 입력")
        
        # 영작문 입력과 듣기 버튼을 같은 행에 배치
        input_col, listen_col = st.columns([3, 1])
        
        with input_col:
            user_text = st.text_area("아래에 영어 작문을 입력하세요", height=200, key="text_tab1")
        
        with listen_col:
            # 음성 생성/재생 버튼
            st.markdown("<br><br>", unsafe_allow_html=True)  # 버튼 위치 조정을 위한 공백
            
            # 사용자 입력 텍스트의 해시값 계산 (변경 시 자동 갱신용)
            if user_text:
                text_hash = hash(user_text)
                audio_key = f"audio_tab1_{text_hash}"
                
                # 세션 상태에 음성 파일 경로가 없으면 초기화
                if audio_key not in st.session_state:
                    st.session_state[audio_key] = None
                
                # 토글 상태 관리
                if f"{audio_key}_playing" not in st.session_state:
                    st.session_state[f"{audio_key}_playing"] = False
                
                # 토글 버튼 생성
                if st.session_state[audio_key] is None:
                    if st.button("📢 영작문 듣기", key=f"generate_audio_tab1", use_container_width=True):
                        if user_text.strip():  # 텍스트가 있는 경우에만 실행
                            with st.spinner("음성 파일을 생성 중입니다..."):
                                try:
                                    # 음성 파일 생성
                                    voice_model = "en-US-JennyNeural"  # 기본 Jenny 음성 사용
                                    
                                    # 임시 파일 경로 생성
                                    temp_dir = tempfile.gettempdir()
                                    audio_file_path = os.path.join(temp_dir, f"speech_tab1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
                                    
                                    # 비동기 함수 실행
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    audio_path = loop.run_until_complete(text_to_speech(user_text, voice_model, audio_file_path))
                                    loop.close()
                                    
                                    # 세션 상태에 오디오 파일 경로 저장
                                    st.session_state[audio_key] = audio_path
                                    st.session_state[f"{audio_key}_playing"] = True
                                    st.experimental_rerun()
                                except Exception as e:
                                    st.error(f"음성 생성 중 오류가 발생했습니다: {e}")
                        else:
                            st.warning("텍스트를 먼저 입력해주세요.")
                else:
                    # 토글 버튼 로직
                    button_label = "⏹️ 음성 정지" if st.session_state[f"{audio_key}_playing"] else "▶️ 음성 재생"
                    if st.button(button_label, key=f"toggle_audio_tab1", use_container_width=True):
                        # 토글 상태 변경
                        st.session_state[f"{audio_key}_playing"] = not st.session_state[f"{audio_key}_playing"]
                        st.experimental_rerun()
                    
                    # 오디오 플레이어 표시 (현재 페이지 위치에 표시)
                    if st.session_state[f"{audio_key}_playing"]:
                        audio_path = st.session_state[audio_key]
                        if os.path.exists(audio_path):
                            # 음성 플레이어 표시
                            audio_html = get_audio_player_html(audio_path, loop_count=5)
                            st.markdown(audio_html, unsafe_allow_html=True)
        
        # 분석 버튼 행
        col1, col2 = st.columns([3, 1])
        
        # 모든 분석을 한 번에 실행하는 버튼
        with col1:
            if st.button("전체 분석하기", use_container_width=True, key="analyze_button"):
                if not user_text:
                    st.warning("텍스트를 입력해주세요.")
                else:
                    # 텍스트 통계 분석
                    stats = analyze_text(user_text)
                
                    # 문법 오류 검사
                    try:
                        grammar_errors = check_grammar(user_text)
                    except Exception as e:
                        st.error(f"문법 검사 중 오류가 발생했습니다: {e}")
                        grammar_errors = []
                    
                    # 어휘 분석
                    vocab_analysis = analyze_vocabulary(user_text)
                    
                    # 어휘 다양성 점수
                    diversity_score = calculate_lexical_diversity(user_text)
                    
                    # 어휘 수준 평가
                    vocab_level = evaluate_vocabulary_level(user_text)
                    
                    # 세션 상태에 결과 저장
                    if 'analysis_results' not in st.session_state:
                        st.session_state.analysis_results = {}
                    
                    st.session_state.analysis_results = {
                        'stats': stats,
                        'grammar_errors': grammar_errors,
                        'vocab_analysis': vocab_analysis,
                        'diversity_score': diversity_score,
                        'vocab_level': vocab_level,
                        'original_text': user_text  # 원본 텍스트도 저장
                    }
                    
                    # 기록에 저장
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.history.append({
                        'timestamp': timestamp,
                        'text': user_text,
                        'error_count': len(grammar_errors) if grammar_errors else 0
                    })
                    
                    st.success("분석이 완료되었습니다! 아래 탭에서 결과를 확인하세요.")
                    
                    # 분석이 완료되었음을 표시하는 플래그
                    st.session_state.analysis_completed = True
                    st.rerun()  # 재실행하여 버튼 표시 업데이트
        
        # 재작성 추천 버튼 추가
        with col2:
            # 분석 결과가 있는 경우에만 버튼 표시
            if 'analysis_results' in st.session_state and 'original_text' in st.session_state.analysis_results:
                if st.button("✨ 영작문 재작성 추천 ✨", 
                          key="rewrite_recommendation",
                          use_container_width=True,
                          help="분석 결과를 바탕으로 영작문을 더 좋은 표현으로 재작성해보세요!",
                          type="primary"):
                    # 텍스트를 세션 상태에 저장
                    st.session_state.copy_to_rewrite = st.session_state.analysis_results['original_text']
                    
                    # 재작성 탭으로 전환하기 위해 selected_tab 업데이트
                    st.session_state.selected_tab = 1  # 1은 영작문 재작성 탭
                    
                    # 사용자에게 안내 메시지 표시
                    st.success("텍스트가 복사되었습니다. 재작성 탭으로 이동합니다!")
                    st.balloons()  # 시각적 효과 추가
                    
                    # 페이지 새로고침 (탭 전환을 위해)
                    st.rerun()
        
        # 결과 표시를 위한 탭
        result_tab1, result_tab2, result_tab3 = st.tabs(["문법 검사", "어휘 분석", "텍스트 통계"])
        
        with result_tab1:
            if 'analysis_results' in st.session_state and 'grammar_errors' in st.session_state.analysis_results:
                grammar_errors = st.session_state.analysis_results['grammar_errors']
                
                if grammar_errors:
                    st.subheader("문법 오류 목록")
                    
                    error_data = []
                    for error in grammar_errors:
                        error_data.append({
                            "오류": user_text[error['offset']:error['offset'] + error['errorLength']],
                            "오류 내용": error['message'],
                            "수정 제안": error['replacements']
                        })
                    
                    st.dataframe(pd.DataFrame(error_data))
                else:
                    st.success("문법 오류가 없습니다!")
                
                # 음성 다운로드 버튼 표시 (기존 버튼 위치에는 다운로드만 유지)
                audio_key = f"audio_tab1_{hash(st.session_state.analysis_results['original_text'])}" if 'original_text' in st.session_state.analysis_results else None
                if audio_key and audio_key in st.session_state and st.session_state[audio_key]:
                    audio_path = st.session_state[audio_key]
                    if os.path.exists(audio_path):
                        with st.expander("음성 파일 다운로드"):
                            with open(audio_path, "rb") as f:
                                audio_bytes = f.read()
                            
                            st.download_button(
                                label="음성 다운로드",
                                data=audio_bytes,
                                file_name=f"audio_essay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
                                mime="audio/wav"
                            )
        
        with result_tab2:
            if 'analysis_results' in st.session_state and 'vocab_analysis' in st.session_state.analysis_results:
                vocab_analysis = st.session_state.analysis_results['vocab_analysis']
                diversity_score = st.session_state.analysis_results['diversity_score']
                vocab_level = st.session_state.analysis_results['vocab_level']
                        
                # 단어 빈도 시각화 - 에러 방지를 위한 예외 처리 추가
                if vocab_analysis and 'word_freq' in vocab_analysis and vocab_analysis['word_freq']:
                    fig = plot_word_frequency(vocab_analysis['word_freq'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("단어 빈도 분석을 위한 데이터가 충분하지 않습니다.")
                
                # 어휘 다양성 점수
                st.metric("어휘 다양성 점수", f"{diversity_score:.2f}")
                
                # 어휘 수준 평가
                if vocab_level:
                    level_df = pd.DataFrame({
                        '수준': ['기초', '중급', '고급'],
                        '비율': [vocab_level['basic'], vocab_level['intermediate'], vocab_level['advanced']]
                    })
                    
                    fig = px.pie(level_df, values='비율', names='수준', 
                                title='어휘 수준 분포')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("어휘 수준 평가를 위한 데이터가 충분하지 않습니다.")
        
        with result_tab3:
            if 'analysis_results' in st.session_state and 'stats' in st.session_state.analysis_results:
                stats = st.session_state.analysis_results['stats']
                    
                st.subheader("텍스트 통계")
                    
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("단어 수", stats['word_count'])
                    st.metric("문장 수", stats['sentence_count'])
                with col2:
                    st.metric("평균 단어 길이", stats['avg_word_length'])
                    st.metric("평균 문장 길이 (단어)", stats['avg_sentence_length'])
                    
                    st.metric("어휘 크기 (고유 단어 수)", stats['vocabulary_size'])
                    
                    # 게이지 차트로 표현하기
                    progress_col1, progress_col2 = st.columns(2)
                with progress_col1:
                        # 평균 문장 길이 게이지 (적정 영어 문장 길이: 15-20 단어)
                    sentence_gauge = min(1.0, stats['avg_sentence_length'] / 20)
                    st.progress(sentence_gauge)
                    st.caption(f"문장 길이 적정성: {int(sentence_gauge * 100)}%")
                    
                with progress_col2:
                        # 어휘 다양성 게이지
                    vocab_ratio = stats['vocabulary_size'] / max(1, stats['word_count'])
                    st.progress(min(1.0, vocab_ratio * 2))  # 0.5 이상이면 100%
                    st.caption(f"어휘 다양성: {int(min(1.0, vocab_ratio * 2) * 100)}%")
    
    # 영작문 재작성 탭
    with tabs[1]:
        st.subheader("영작문 재작성")
        
        # 왼쪽 열: 입력 및 옵션
        left_col, right_col = st.columns(2)
        
        with left_col:
            # 분석 탭에서 넘어온 경우 해당 텍스트를 자동으로 로드
            default_text = ""
            if 'copy_to_rewrite' in st.session_state:
                default_text = st.session_state.copy_to_rewrite
                st.success("분석 결과 텍스트가 로드되었습니다!")
                # 한 번 사용 후 임시 변수로 옮겨 저장
                st.session_state.copy_to_rewrite_temp = default_text
                del st.session_state.copy_to_rewrite
            elif 'copy_to_rewrite_temp' in st.session_state:
                default_text = st.session_state.copy_to_rewrite_temp
            
            rewrite_text_input = st.text_area("아래에 영어 작문을 입력하세요", 
                                            value=default_text,
                                            height=200, 
                                            key="text_tab2")
            
            level_option = st.radio(
                "작문 수준 선택",
                options=["비슷한 수준", "약간 높은 수준", "고급 수준"],
                horizontal=True
            )
            
            level_map = {
                "비슷한 수준": "similar",
                "약간 높은 수준": "improved",
                "고급 수준": "advanced"
            }
            
            if st.button("재작성하기"):
                if not rewrite_text_input:
                    st.warning("텍스트를 입력해주세요.")
                else:
                    level = level_map.get(level_option, "similar")
                    
                    # 재작성 처리
                    with st.spinner("텍스트를 재작성 중입니다..."):
                        rewritten_text = rewrite_text(rewrite_text_input, level)
                        
                        # 재작성된 텍스트를 세션 상태에 저장
                        if 'rewritten_text' not in st.session_state:
                            st.session_state.rewritten_text = {}
                        
                        st.session_state.rewritten_text[level] = rewritten_text
                        
                        # 기록에 추가
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.history.append({
                            'timestamp': timestamp,
                            'text': rewrite_text_input,
                            'action': f"재작성 ({level_option})"
                        })
        
        with right_col:
            st.subheader("재작성 결과")
            
            if 'rewritten_text' in st.session_state and st.session_state.rewritten_text:
                level = level_map.get(level_option, "similar")
                
                if level in st.session_state.rewritten_text:
                    rewritten = st.session_state.rewritten_text[level]
                    st.text_area("재작성된 텍스트", value=rewritten, height=250, key="rewritten_result")
                    
                    # 음성 옵션 추가
                    st.subheader("본문 읽기 옵션")
                    voice_options = {
                        "Jenny (여성, 미국)": "en-US-JennyNeural",
                        "Guy (남성, 미국)": "en-US-GuyNeural",
                        "Aria (여성, 영국)": "en-GB-SoniaNeural"
                    }
                    selected_voice = st.selectbox(
                        "음성 선택",
                        options=list(voice_options.keys()),
                        key="voice_selection"
                    )
                    
                    # 재작성 텍스트 다운로드 및 음성 변환 버튼
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 텍스트 다운로드 버튼
                        if rewritten:
                            text_output = io.BytesIO()
                            text_output.write(rewritten.encode('utf-8'))
                            text_output.seek(0)
                            
                            st.download_button(
                                label="텍스트 다운로드",
                                data=text_output,
                                file_name=f"rewritten_text_{level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                    
                    with col2:
                        # 음성 파일 다운로드 버튼
                        if rewritten:
                            if st.button("음성 파일 생성", key="generate_speech"):
                                with st.spinner("음성 파일을 생성 중입니다..."):
                                    # 선택된 음성 모델 가져오기
                                    voice_model = voice_options[selected_voice]
                                    
                                    # 임시 파일 경로 생성
                                    temp_dir = tempfile.gettempdir()
                                    audio_file_path = os.path.join(temp_dir, f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
                                    
                                    # 비동기 함수 실행을 위한 런타임 설정
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    audio_path = loop.run_until_complete(text_to_speech(rewritten, voice_model, audio_file_path))
                                    
                                    # 세션 상태에 오디오 파일 경로 저장
                                    st.session_state.audio_path = audio_path
                                    st.success("음성 파일이 생성되었습니다!")
                                    st.experimental_rerun()  # 재실행하여 오디오 플레이어 표시
            
                    # 오디오 플레이어 표시
                    if 'audio_path' in st.session_state and os.path.exists(st.session_state.audio_path):
                        st.subheader("본문 듣기")
                        
                        # 오디오 재생 상태 관리
                        if 'audio_playing' not in st.session_state:
                            st.session_state.audio_playing = True
                        
                        # 본문 듣기 토글 버튼
                        play_col, download_col = st.columns([3, 1])
                        
                        with play_col:
                            # 토글 버튼 로직
                            button_label = "⏹️ 음성 정지" if st.session_state.audio_playing else "▶️ 음성 재생"
                            if st.button(button_label, key="toggle_audio"):
                                # 토글 상태 변경
                                st.session_state.audio_playing = not st.session_state.audio_playing
                                st.experimental_rerun()
                            
                            # 현재 상태에 따라 오디오 플레이어 표시
                            if st.session_state.audio_playing:
                                audio_html = get_audio_player_html(st.session_state.audio_path, loop_count=5)
                                st.markdown(audio_html, unsafe_allow_html=True)
                        
                        with download_col:
                            with open(st.session_state.audio_path, "rb") as f:
                                audio_bytes = f.read()
                            
                            # 음성 파일 다운로드 버튼
                            st.download_button(
                                label="음성 다운로드",
                                data=audio_bytes,
                                file_name=f"audio_{level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
                                mime="audio/wav"
                            )
            
                    # 원본과 재작성 텍스트 비교
                    if rewrite_text_input and rewritten:
                        st.subheader("원본 vs 재작성 비교")
                        
                        comparison_data = []
                        original_sentences = custom_sent_tokenize(rewrite_text_input)
                        rewritten_sentences = custom_sent_tokenize(rewritten)
                        
                        # 문장 단위로 비교 (더 짧은 리스트 기준)
                        for i in range(min(len(original_sentences), len(rewritten_sentences))):
                            comparison_data.append({
                                "원본": original_sentences[i],
                                "재작성": rewritten_sentences[i]
                            })
                        
                        if comparison_data:
                            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
            else:
                st.info("텍스트를 입력하고 재작성 버튼을 클릭하세요.")
    
    # 내 작문 기록 탭
    with tabs[2]:
        st.subheader("내 작문 기록")
        if not st.session_state.history:
            st.info("아직 기록이 없습니다.")
        else:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df)
            
            # 오류 수 추이 차트
            if len(history_df) > 1 and 'error_count' in history_df.columns:
                fig = px.line(history_df, x='timestamp', y='error_count', 
                            title='문법 오류 수 추이',
                            labels={'timestamp': '날짜', 'error_count': '오류 수'})
                st.plotly_chart(fig, use_container_width=True)

# 교사 페이지
def show_teacher_page():
    st.title("영작문 자동 첨삭 시스템 - 교사")
    
    # 로그아웃 버튼
    if st.button("로그아웃", key="teacher_logout"):
        st.session_state.user_type = None
        st.rerun()
    
    # 탭 생성
    tabs = st.tabs(["영작문 첨삭", "학습 대시보드"])
    
    # 영작문 첨삭 탭
    with tabs[0]:
        st.subheader("영작문 입력 및 첨삭")
        
        user_text = st.text_area("학생의 영어 작문을 입력하세요", height=200, key="teacher_text")
        
        col1, col2 = st.columns([3, 1])
        
        # 모든 분석을 한 번에 실행하는 버튼
        with col1:
            if st.button("전체 분석하기", key="teacher_analyze_all", use_container_width=True):
                if not user_text:
                    st.warning("텍스트를 입력해주세요.")
                else:
                    # 문법 오류 검사
                    try:
                        grammar_errors = check_grammar(user_text)
                    except Exception as e:
                        st.error(f"문법 검사 중 오류가 발생했습니다: {e}")
                        grammar_errors = []
                    
                    # 어휘 분석
                    vocab_analysis = analyze_vocabulary(user_text)
                    
                    # 어휘 다양성 점수
                    diversity_score = calculate_lexical_diversity(user_text)
                    
                    # 어휘 수준 평가
                    vocab_level = evaluate_vocabulary_level(user_text)
                    
                    # 텍스트 통계 분석
                    stats = analyze_text(user_text)
                    
                    # 세션 상태에 결과 저장
                    if 'teacher_analysis_results' not in st.session_state:
                        st.session_state.teacher_analysis_results = {}
                    
                    st.session_state.teacher_analysis_results = {
                        'stats': stats,
                        'grammar_errors': grammar_errors,
                        'vocab_analysis': vocab_analysis,
                        'diversity_score': diversity_score,
                        'vocab_level': vocab_level,
                        'original_text': user_text  # 원본 텍스트도 저장
                    }
                    
                    st.success("분석이 완료되었습니다! 아래 탭에서 결과를 확인하세요.")
        
        # 재작성 추천 버튼 추가
        with col2:
            # 분석 결과가 있는 경우에만 버튼 표시
            if 'teacher_analysis_results' in st.session_state and 'original_text' in st.session_state.teacher_analysis_results:
                if st.button("✨ 영작문 재작성 추천 ✨", 
                          key="teacher_rewrite_recommendation",
                          use_container_width=True,
                          help="분석 결과를 바탕으로 학생의 영작문을 더 좋은 표현으로 재작성해보세요!",
                          type="primary"):
                    # 교사 첨삭 영역에 모범 답안 작성용으로 추가
                    # 재작성 함수를 사용하여 즉시 고급 수준으로 재작성
                    original_text = st.session_state.teacher_analysis_results['original_text']
                    rewritten_text = rewrite_text(original_text, "advanced")
                    
                    # 첨삭 노트에 재작성된 텍스트 추가
                    if 'feedback_template' not in st.session_state:
                        st.session_state.feedback_template = "다음은 학생 작문을 고급 수준으로 재작성한 예시입니다:\n\n" + rewritten_text
                    else:
                        st.session_state.feedback_template += "\n\n추천 모범 예시:\n" + rewritten_text
                    
                    st.success("고급 수준으로 재작성된 텍스트가 첨삭 노트에 추가되었습니다.")
                    st.balloons()  # 시각적 효과 추가
        
        # 결과 표시를 위한 탭
        result_tab1, result_tab2, result_tab3 = st.tabs(["문법 검사", "어휘 분석", "텍스트 통계"])
        
        with result_tab1:
            if 'teacher_analysis_results' in st.session_state and 'grammar_errors' in st.session_state.teacher_analysis_results:
                grammar_errors = st.session_state.teacher_analysis_results['grammar_errors']
                        
                if grammar_errors:
                            st.subheader("문법 오류 목록")
                            
                            error_data = []
                            for error in grammar_errors:
                                error_data.append({
                            "오류": user_text[error['offset']:error['offset'] + error['errorLength']],
                            "오류 내용": error['message'],
                            "수정 제안": error['replacements']
                                })
                            
                            st.dataframe(pd.DataFrame(error_data))
                else:
                            st.success("문법 오류가 없습니다!")
            
        with result_tab2:
            if 'teacher_analysis_results' in st.session_state and 'vocab_analysis' in st.session_state.teacher_analysis_results:
                vocab_analysis = st.session_state.teacher_analysis_results['vocab_analysis']
                diversity_score = st.session_state.teacher_analysis_results['diversity_score']
                vocab_level = st.session_state.teacher_analysis_results['vocab_level']
                        
                # 단어 빈도 시각화 - 에러 방지를 위한 예외 처리 추가
                if vocab_analysis and 'word_freq' in vocab_analysis and vocab_analysis['word_freq']:
                    fig = plot_word_frequency(vocab_analysis['word_freq'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("단어 빈도 분석을 위한 데이터가 충분하지 않습니다.")
                
                # 어휘 다양성 점수
                st.metric("어휘 다양성 점수", f"{diversity_score:.2f}")
                
                # 어휘 수준 평가
                if vocab_level:
                    level_df = pd.DataFrame({
                        '수준': ['기초', '중급', '고급'],
                        '비율': [vocab_level['basic'], vocab_level['intermediate'], vocab_level['advanced']]
                    })
                    
                    fig = px.pie(level_df, values='비율', names='수준', 
                                title='어휘 수준 분포')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("어휘 수준 평가를 위한 데이터가 충분하지 않습니다.")
        
        with result_tab3:
            if 'teacher_analysis_results' in st.session_state and 'stats' in st.session_state.teacher_analysis_results:
                stats = st.session_state.teacher_analysis_results['stats']
                    
                st.subheader("텍스트 통계")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("단어 수", stats['word_count'])
                    st.metric("문장 수", stats['sentence_count'])
                with col2:
                    st.metric("평균 단어 길이", stats['avg_word_length'])
                    st.metric("평균 문장 길이 (단어)", stats['avg_sentence_length'])
                    
                st.metric("어휘 크기 (고유 단어 수)", stats['vocabulary_size'])
        
        # 교사 전용 기능
        st.subheader("첨삭 및 피드백")
        
        # 첨삭 노트 템플릿
        feedback_default = "다음 사항을 중점적으로 개선해보세요:\n1. \n2. \n3. "
        if 'feedback_template' in st.session_state:
            feedback_default = st.session_state.feedback_template
        
        feedback = st.text_area("첨삭 노트", 
                             value=feedback_default, 
                             height=200,  # 더 높게 조정
                 key="feedback_template")
        
        # 세션 상태 업데이트
        st.session_state.feedback_template = feedback
        
        # 점수 입력
        score_col1, score_col2, score_col3 = st.columns(3)
        with score_col1:
            grammar_score = st.slider("문법 점수", 0, 10, 5)
        with score_col2:
            vocab_score = st.slider("어휘 점수", 0, 10, 5)
        with score_col3:
            content_score = st.slider("내용 점수", 0, 10, 5)
        
        total_score = (grammar_score + vocab_score + content_score) / 3
        st.metric("종합 점수", f"{total_score:.1f} / 10")
        
        # 저장 옵션
        if st.button("첨삭 결과 저장 (Excel)"):
            if not user_text:
                st.warning("텍스트를 입력해주세요.")
            else:
                # 문법 오류 검사
                grammar_errors = check_grammar(user_text)
                
                # 저장할 데이터 생성
                error_data = []
                for error in grammar_errors:
                    error_data.append({
                        "오류": user_text[error['offset']:error['offset'] + error['errorLength']],
                        "오류 내용": error['message'],
                        "수정 제안": str(error['replacements']),
                        "위치": f"{error['offset']}:{error['offset'] + error['errorLength']}"
                    })
                
                # 어휘 분석
                vocab_analysis = analyze_vocabulary(user_text)
                
                # 통계 분석
                stats = analyze_text(user_text)
                
                # 여러 시트가 있는 Excel 파일 생성
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    # 원본 텍스트 시트
                    pd.DataFrame({"원본 텍스트": [user_text]}).to_excel(writer, sheet_name="원본 텍스트", index=False)
                    
                    # 오류 데이터 시트
                    pd.DataFrame(error_data).to_excel(writer, sheet_name="문법 오류", index=False)
                    
                    # 통계 시트
                    stats_df = pd.DataFrame({
                        "항목": ["단어 수", "문장 수", "평균 단어 길이", "평균 문장 길이", "어휘 크기"],
                        "값": [stats['word_count'], stats['sentence_count'], stats['avg_word_length'], 
                             stats['avg_sentence_length'], stats['vocabulary_size']]
                    })
                    stats_df.to_excel(writer, sheet_name="통계", index=False)
                    
                    # 평가 점수 시트
                    score_df = pd.DataFrame({
                        "평가 항목": ["문법", "어휘", "내용", "종합 점수"],
                        "점수": [grammar_score, vocab_score, content_score, total_score]
                    })
                    score_df.to_excel(writer, sheet_name="평가 점수", index=False)
                    
                    # 피드백 시트
                    pd.DataFrame({"첨삭 피드백": [feedback]}).to_excel(writer, sheet_name="피드백", index=False)
                
                # 다운로드 버튼
                excel_buffer.seek(0)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="Excel 다운로드",
                    data=excel_buffer,
                    file_name=f"첨삭결과_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
    
    # 학습 대시보드 탭
    with tabs[1]:
        st.subheader("학습 대시보드")
        
        # 샘플 학생 데이터 (실제로는 DB에서 가져와야 함)
        sample_data = {
            "날짜": ["2023-01-01", "2023-01-08", "2023-01-15", "2023-01-22", "2023-01-29"],
            "학생_A": [7, 6, 8, 7, 9],
            "학생_B": [5, 6, 6, 7, 6],
            "학생_C": [8, 8, 7, 9, 9],
            "학생_D": [4, 5, 6, 6, 7]
        }
        
        scores_df = pd.DataFrame(sample_data)
        
        # 학생별 점수 추이
        fig = px.line(scores_df, x="날짜", y=["학생_A", "학생_B", "학생_C", "학생_D"],
                     title="학생별 영작문 점수 추이",
                     labels={"value": "점수", "variable": "학생"})
        st.plotly_chart(fig, use_container_width=True)
        
        # 최근 점수 분포
        latest_scores = scores_df.iloc[-1, 1:].values
        students = scores_df.columns[1:]
        
        fig = px.bar(x=students, y=latest_scores, 
                    title="최근 영작문 점수 분포",
                    labels={"x": "학생", "y": "점수"})
        st.plotly_chart(fig, use_container_width=True)
        
        # 평균 오류 유형 (샘플 데이터)
        error_types = {
            "오류 유형": ["문장 구조", "시제", "관사", "전치사", "대명사", "철자", "구두점"],
            "빈도": [45, 30, 25, 20, 15, 10, 5]
        }
        
        error_df = pd.DataFrame(error_types)
        
        fig = px.pie(error_df, values="빈도", names="오류 유형",
                    title="평균 오류 유형 분포")
        st.plotly_chart(fig, use_container_width=True)

# 메인 함수
def main():
    # 제목 및 소개
    # st.title("영작문 자동 첨삭 시스템")
    st.markdown("""
    이 애플리케이션은 학생들의 영작문을 자동으로 첨삭하고 피드백을 제공합니다.
    """)
    
    # 직접 학생 페이지로 이동
    show_student_page()

if __name__ == "__main__":
    main()

@st.cache_resource
def load_vocabulary_datasets():
    # 온라인 소스에서 데이터셋 다운로드 (실제 작동하는 URL로 수정)
    word_freq_url = "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/en/en_50k.txt"
    
    try:
        import requests
        
        # 영어 단어 빈도 데이터 다운로드
        response = requests.get(word_freq_url)
        if response.status_code == 200:
            # 단어 빈도 데이터 파싱 (형식: "단어 빈도")
            lines = response.text.splitlines()
            words = [line.split()[0] for line in lines if ' ' in line]
            
            # 빈도에 따라 단어 분류
            total_words = len(words)
            basic_cutoff = int(total_words * 0.2)  # 상위 20%
            intermediate_cutoff = int(total_words * 0.5)  # 상위 20~50%
            
            basic_words = set(words[:basic_cutoff])
            intermediate_words = set(words[basic_cutoff:intermediate_cutoff])
            advanced_words = set(words[intermediate_cutoff:])
            
            return {'basic': basic_words, 'intermediate': intermediate_words, 'advanced': advanced_words}
    except Exception as e:
        st.warning(f"온라인 데이터셋 다운로드 중 오류가 발생했습니다: {e}")
    
    # 다운로드 실패시 내장 데이터셋 사용
    return default_vocabulary_sets()

def evaluate_advanced_vocabulary(text):
    words = custom_word_tokenize(text.lower())
    
    # 단어 빈도 기반 평가
    word_frequencies = get_word_frequency_data()  # 단어별 빈도 데이터 로드
    
    rare_words = [w for w in words if w in word_frequencies and word_frequencies[w] < 0.001]
    
    # 학술 단어 목록 가져오기
    academic_word_list = get_academic_word_list()
    academic_words = [w for w in words if w in academic_word_list]
    
    vocab_score = (len(rare_words) * 2 + len(academic_words)) / max(len(words), 1)
    return vocab_score

# 텍스트를 음성으로 변환하는 함수 추가 (라인 360 이후에 추가)
async def text_to_speech(text, voice="en-US-JennyNeural", output_file=None):
    """
    텍스트를 음성으로 변환하고 파일로 저장합니다.
    
    Parameters:
    - text: 음성으로 변환할 텍스트
    - voice: 음성 모델 (기본값: 'en-US-JennyNeural')
    - output_file: 출력 파일 경로 (None인 경우 임시 파일 생성)
    
    Returns:
    - 음성 파일 경로
    """
    if not text:
        return None
    
    # 출력 파일이 지정되지 않은 경우 임시 파일 생성
    if output_file is None:
        temp_dir = tempfile.gettempdir()
        output_file = os.path.join(temp_dir, f"speech_{random.randint(1000, 9999)}.wav")
    
    # 텍스트를 음성으로 변환하고 파일로 저장
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)
    
    return output_file

# 음성 파일을 HTML 오디오 요소로 변환하는 함수
def get_audio_player_html(audio_path, loop_count=5, autoplay=True):
    """
    음성 파일을 재생할 수 있는 HTML 오디오 플레이어를 생성합니다.
    
    Parameters:
    - audio_path: 음성 파일 경로
    - loop_count: 반복 재생 횟수 (기본값: 5)
    - autoplay: 자동 재생 여부 (기본값: True)
    
    Returns:
    - HTML 코드 문자열
    """
    if not audio_path or not os.path.exists(audio_path):
        return ""
    
    # 파일을 base64로 인코딩
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    
    audio_b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
        <audio id="audio-player" controls {' loop' if loop_count > 1 else ''} {' autoplay' if autoplay else ''}>
            <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
            Your browser does not support the audio element.
        </audio>
        <script>
            var audioPlayer = document.getElementById('audio-player');
            var playCount = 0;
            var maxPlays = {loop_count};
            
            audioPlayer.addEventListener('ended', function() {{
                playCount++;
                if (playCount < maxPlays) {{
                    audioPlayer.play();
                }}
            }});
        </script>
    """
    return audio_html
