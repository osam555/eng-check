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

# NLTK 라이브러리 및 데이터 처리
import nltk
from nltk.corpus import stopwords

# 변환기 모듈 존재 여부 확인
has_transformers = 'transformers' in globals()

# 대체 맞춤법 검사 라이브러리 설정
try:
    import enchant
    has_enchant = True
except ImportError:
    has_enchant = False
    try:
        # 대체 맞춤법 검사 라이브러리 시도
        from spellchecker import SpellChecker
        spell = SpellChecker()
        has_spellchecker = True
    except ImportError:
        has_spellchecker = False
    #st.info("맞춤법 검사 라이브러리(enchant)가 설치되지 않았습니다. TextBlob을 사용한 기본 맞춤법 검사만 제공됩니다.")

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

# 맞춤법 사전 초기화
@st.cache_resource
def get_spell_checker():
    if has_enchant:
        try:
            return enchant.Dict("en_US")
        except:
            return None
    elif 'has_spellchecker' in globals() and has_spellchecker:
        return spell
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

# 학생 페이지
def evaluate_vocabulary_level(text):
    # 온라인 데이터셋에서 어휘 로드
    vocabulary_sets = load_vocabulary_datasets()
    
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
    
    tabs = st.tabs(["영작문 검사", "영작문 재작성", "내 작문 기록"])
    
    # 영작문 검사 탭
    with tabs[0]:
        st.subheader("영작문 입력")
        user_text = st.text_area("아래에 영어 작문을 입력하세요", height=200, key="text_tab1")
        
        col1, col2 = st.columns([3, 1])
        
        # 모든 분석을 한 번에 실행하는 버튼
        with col1:
            if st.button("전체 분석하기", use_container_width=True, key="analyze_button"):
                    if not user_text:
                        st.warning("텍스트를 입력해주세요.")
                    else:
                    # 텍스트 통계 분석
                        stats = analyze_text(user_text)
                    
                        try:
                            # 문법 오류 검사
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
                    
                    # 사용자에게 안내 메시지 표시
                    st.success("텍스트가 복사되었습니다. 상단의 '영작문 재작성' 탭을 클릭하세요!")
                    st.balloons()  # 시각적 효과 추가
        
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
        
        with result_tab2:
            if 'analysis_results' in st.session_state and 'vocab_analysis' in st.session_state.analysis_results:
                vocab_analysis = st.session_state.analysis_results['vocab_analysis']
                diversity_score = st.session_state.analysis_results['diversity_score']
                vocab_level = st.session_state.analysis_results['vocab_level']
                        
                # 단어 빈도 시각화
                fig = plot_word_frequency(vocab_analysis['word_freq'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                # 어휘 다양성 점수
                st.metric("어휘 다양성 점수", f"{diversity_score:.2f}", 
                         delta="높을수록 다양한 어휘 사용")
                
                # 어휘 수준 평가
                level_df = pd.DataFrame({
                    '수준': ['기초', '중급', '고급'],
                    '비율': [vocab_level['basic'], vocab_level['intermediate'], vocab_level['advanced']]
                })
                
                fig = px.pie(level_df, values='비율', names='수준', 
                            title='어휘 수준 분포',
                            color_discrete_sequence=px.colors.sequential.Viridis)
                st.plotly_chart(fig, use_container_width=True)
        
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
                    
                    # 재작성 텍스트 복사 기능 대신 다운로드 기능 제공
                    if rewritten:
                        output = io.BytesIO()
                        output.write(rewritten.encode('utf-8'))
                        output.seek(0)
                        
                        st.download_button(
                            label="재작성 텍스트 다운로드",
                            data=output,
                            file_name=f"rewritten_text_{level}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
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
                        try:
                            # 문법 오류 검사
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
                        
                # 단어 빈도 시각화
                fig = plot_word_frequency(vocab_analysis['word_freq'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                # 어휘 다양성 점수
                st.metric("어휘 다양성 점수", f"{diversity_score:.2f}")
                
                # 어휘 수준 평가
                level_df = pd.DataFrame({
                    '수준': ['기초', '중급', '고급'],
                    '비율': [vocab_level['basic'], vocab_level['intermediate'], vocab_level['advanced']]
                })
                
                fig = px.pie(level_df, values='비율', names='수준', 
                            title='어휘 수준 분포')
                st.plotly_chart(fig, use_container_width=True)
        
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
    st.title("영작문 자동 첨삭 시스템")
    st.markdown("""
    이 애플리케이션은 학생들의 영작문을 자동으로 첨삭하고 피드백을 제공합니다.
    학생은 텍스트를 입력하고 자동 분석 결과를 확인할 수 있습니다.
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

@st.cache_resource
def load_rewrite_model():
    if not has_transformers:
        return None, None
        
    try:
        # T5 또는 GPT 기반 모델 로드
        model_name = "t5-base"  # 또는 "gpt2-medium"
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.warning(f"모델 로드 중 오류 발생: {e}")
        return None, None

def advanced_rewrite_text(text, level='advanced'):
    tokenizer, model = load_rewrite_model()
    if not tokenizer or not model:
        # 폴백: 기존 규칙 기반 방식 사용
        return rewrite_advanced_level(text)
        
    try:
        import torch
        prefix = f"paraphrase to {level} level: "
        inputs = tokenizer(prefix + text, return_tensors="pt", max_length=512, truncation=True)
        
        outputs = model.generate(
            inputs.input_ids, 
            max_length=512,
            temperature=0.8,  # 창의성 조절
            num_return_sequences=1
        )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        st.warning(f"AI 모델 처리 중 오류: {e}. 대체 방법으로 진행합니다.")
        # 폴백: 기존 규칙 기반 방식 사용
        return rewrite_advanced_level(text)

def context_aware_rewrite(text, subject_area="general"):
    # 주제별 어휘 데이터셋 로드
    domain_vocabulary = load_domain_vocabulary(subject_area)
    
    sentences = custom_sent_tokenize(text)
    rewritten = []
    
    # 문맥 유지하며 주제 관련 어휘로 강화
    for i, sentence in enumerate(sentences):
        # 이전/다음 문장 참조해서 문맥 유지
        prev_context = sentences[i-1] if i > 0 else ""
        next_context = sentences[i+1] if i < len(sentences)-1 else ""
        
        enhanced = enhance_sentence_with_domain(
            sentence, 
            domain_vocabulary, 
            prev_context, 
            next_context
        )
        rewritten.append(enhanced)
    
    return " ".join(rewritten)

def transform_grammar_structure(sentence, level):
    # 문법적 복잡성 레벨 매핑
    complexity_patterns = {
        "intermediate": [
            # 단순 시제 → 완료 시제
            (r'\b(do|does|did)\b', 'have done'),
            # 능동태 → 수동태 변환
            (r'(\w+)\s+(\w+ed|s)\s+(\w+)', r'\3 was \2ed by \1')
        ],
        "advanced": [
            # 단순 조건문 → 가정법 변환
            (r'If\s+(\w+)\s+(\w+),\s+(\w+)\s+(\w+)', r'Were \1 to \2, \3 would \4'),
            # 분사구문 도입
            (r'(\w+)\s+(\w+ed|s)\s+and\s+(\w+)', r'\1 \2ed, \3ing')
        ]
    }
    
    # 문법 구조 변환 적용
    for pattern, replacement in complexity_patterns.get(level, []):
        sentence = re.sub(pattern, replacement, sentence)
    
    return sentence

def maintain_topic_coherence(sentences):
    # 주제어 추출
    topic_words = extract_topic_words(sentences)
    
    # 주제 일관성 강화
    coherent_sentences = []
    for sentence in sentences:
        # 문장이 주제와 연관되어 있는지 확인하고 보강
        if not has_topic_reference(sentence, topic_words):
            # 주제 연결어 추가
            sentence = add_topic_reference(sentence, topic_words)
        coherent_sentences.append(sentence)
    
    # 전체 흐름 개선
    add_transition_phrases(coherent_sentences)
    
    return coherent_sentences

def genre_specific_rewrite(text, genre):
    """
    특정 장르(에세이, 비즈니스 이메일, 학술 논문 등)에 맞는 스타일로 재작성
    """
    genre_styles = {
        "academic": {
            "phrases": ["It can be argued that", "The evidence suggests that", 
                       "This study examines", "The findings indicate"],
            "tone": "formal",
            "sentence_length": "long"
        },
        "business": {
            "phrases": ["I am writing to inquire about", "I would like to request", 
                      "Please find attached", "I look forward to hearing from you"],
            "tone": "professional",
            "sentence_length": "medium"
        },
        "creative": {
            "phrases": ["Imagine a world where", "The air was thick with", 
                      "In that moment", "Everything changed when"],
            "tone": "descriptive", 
            "sentence_length": "varied"
        }
    }
    
    style = genre_styles.get(genre, genre_styles["academic"])
    return apply_genre_style(text, style)

def optimize_complexity(text, target_level):
    # 현재 텍스트 복잡도 분석
    current_complexity = analyze_text_complexity(text)
    
    # 타겟 레벨과 현재 복잡도 비교
    if current_complexity < target_level:
        # 어휘 고급화
        text = enhance_vocabulary(text)
        # 문장 구조 복잡화
        text = add_complexity(text)
    elif current_complexity > target_level:
        # 간결화
        text = simplify_text(text)
    
    return text

def culturally_appropriate_rewrite(text, target_culture="american"):
    # 문화적 특성에 맞는 표현 데이터셋
    cultural_expressions = {
        "american": {
            "idioms": ["hit the nail on the head", "ballpark figure"],
            "references": ["Super Bowl", "Thanksgiving"],
            "measurements": "imperial"  # feet, pounds
        },
        "british": {
            "idioms": ["bob's your uncle", "chin up"],
            "references": ["BBC", "Bank Holiday"],
            "measurements": "metric"  # meters, kilograms
        }
    }
    
    culture_data = cultural_expressions.get(target_culture, cultural_expressions["american"])
    return adapt_to_culture(text, culture_data)

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

# 기본 단어 셋 정의
@st.cache_resource
def default_vocabulary_sets():
    basic_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it'}
    intermediate_words = {'achieve', 'consider', 'determine', 'establish', 'indicate'}
    advanced_words = {'arbitrary', 'cognitive', 'encompass', 'facilitate', 'implicit'}
    return {'basic': basic_words, 'intermediate': intermediate_words, 'advanced': advanced_words}

# 주제별 어휘 데이터셋 로드
@st.cache_resource
def load_domain_vocabulary(subject_area="general"):
    domain_vocabs = {
        "general": ["discuss", "explain", "describe", "analyze"],
        "business": ["market", "strategy", "investment", "revenue"],
        "science": ["hypothesis", "experiment", "theory", "analysis"],
        "technology": ["innovation", "interface", "algorithm", "platform"]
    }
    return domain_vocabs.get(subject_area, domain_vocabs["general"])

# 주제어 추출
def extract_topic_words(sentences, top_n=3):
    all_words = []
    for sentence in sentences:
        words = custom_word_tokenize(sentence.lower())
        words = [w for w in words if re.match(r'\w+', w) and w not in stopwords.words('english')]
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    return [word for word, _ in word_counts.most_common(top_n)]

# 문장이 주제와 연관되어 있는지 확인
def has_topic_reference(sentence, topic_words):
    words = custom_word_tokenize(sentence.lower())
    return any(topic in words for topic in topic_words)

# 주제 연결어 추가
def add_topic_reference(sentence, topic_words):
    if not topic_words:
        return sentence
    
    topic = random.choice(topic_words)
    reference_phrases = [
        f"Regarding {topic}, ",
        f"In terms of {topic}, ",
        f"Concerning {topic}, "
    ]
    
    return random.choice(reference_phrases) + sentence

# 전환 문구 추가
def add_transition_phrases(sentences):
    if len(sentences) <= 1:
        return sentences
    
    transitions = [
        "Furthermore, ", "Moreover, ", "In addition, ",
        "Consequently, ", "Therefore, ", "Thus, ",
        "On the other hand, ", "However, ", "Nevertheless, "
    ]
    
    for i in range(1, len(sentences)):
        if random.random() < 0.4:  # 40% 확률로 전환 문구 추가
            sentences[i] = random.choice(transitions) + sentences[i]
    
    return sentences

# 문장 복잡도 분석
def analyze_text_complexity(text):
    sentences = custom_sent_tokenize(text)
    words = custom_word_tokenize(text)
    
    # 평균 문장 길이
    avg_sentence_length = len(words) / max(len(sentences), 1)
    
    # 긴 단어(6자 이상) 비율
    long_words = [w for w in words if len(w) >= 6]
    long_word_ratio = len(long_words) / max(len(words), 1)
    
    # 복합 문장 비율 (and, but, because 등 포함)
    complex_markers = ['and', 'but', 'because', 'however', 'therefore', 'although', 'since']
    complex_sentences = sum(1 for s in sentences if any(marker in custom_word_tokenize(s.lower()) for marker in complex_markers))
    complex_ratio = complex_sentences / max(len(sentences), 1)
    
    # 복잡도 점수 (0~1)
    complexity = (avg_sentence_length / 25 + long_word_ratio + complex_ratio) / 3
    return min(max(complexity, 0), 1)  # 0과 1 사이로 제한

# 어휘 고급화
def enhance_vocabulary(text):
    words = custom_word_tokenize(text)
    enhanced = []
    
    # 기본 단어에 대한 고급 대체어
    enhancements = {
        'good': 'excellent',
        'bad': 'detrimental',
        'big': 'substantial',
        'small': 'minimal',
        'happy': 'euphoric',
        'sad': 'depressed',
        'important': 'crucial',
        'difficult': 'formidable',
        'easy': 'facile',
        'beautiful': 'resplendent'
    }
    
    for word in words:
        word_lower = word.lower()
        if word_lower in enhancements and random.random() < 0.7:
            replacement = enhancements[word_lower]
            if word[0].isupper():
                replacement = replacement.capitalize()
            enhanced.append(replacement)
        else:
            enhanced.append(word)
    
    return ' '.join(enhanced)

# 문장 구조 복잡화
def add_complexity(text):
    sentences = custom_sent_tokenize(text)
    complex_sentences = []
    
    for sentence in sentences:
        # 문장 길이에 따라 다른 전략 적용
        if len(custom_word_tokenize(sentence)) < 10:
            # 짧은 문장은 수식어구 추가
            modifiers = [
                "Interestingly, ", "Specifically, ", "Notably, ",
                "In this context, ", "From this perspective, "
            ]
            sentence = random.choice(modifiers) + sentence
        else:
            # 긴 문장은 구조 변경
            if sentence.startswith("I "):
                sentence = sentence.replace("I ", "The author ")
            elif random.random() < 0.3:
                # 30% 확률로 수동태로 변경 시도
                if " is " in sentence:
                    parts = sentence.split(" is ")
                    if len(parts) >= 2:
                        sentence = parts[1] + " is " + parts[0]
        
        complex_sentences.append(sentence)
    
    return ' '.join(complex_sentences)

# 문장 간소화
def simplify_text(text):
    sentences = custom_sent_tokenize(text)
    simplified_sentences = []
    
    for sentence in sentences:
        words = custom_word_tokenize(sentence)
        if len(words) > 20:  # 긴 문장 분할
            middle = len(words) // 2
            first_half = ' '.join(words[:middle])
            second_half = ' '.join(words[middle:])
            simplified_sentences.append(first_half + '.')
            simplified_sentences.append(second_half)
        else:
            simplified_sentences.append(sentence)
    
    return ' '.join(simplified_sentences)

# 장르별 스타일 적용
def apply_genre_style(text, style):
    sentences = custom_sent_tokenize(text)
    styled_sentences = []
    
    # 첫 문장에 스타일 구문 추가
    if sentences and random.random() < 0.7:
        phrases = style.get("phrases", [])
        if phrases:
            sentences[0] = random.choice(phrases) + " " + sentences[0].lower()
    
    # 나머지 문장에 대한 스타일 적용
    for i, sentence in enumerate(sentences):
        if i == 0:
            styled_sentences.append(sentence)
            continue
        
        # 스타일에 따른 문장 길이 조정
        if style.get("sentence_length") == "long" and len(custom_word_tokenize(sentence)) < 10:
            # 짧은 문장을 더 길게
            modifiers = ["furthermore", "additionally", "consequently", "in this context"]
            sentence = random.choice(modifiers) + ", " + sentence.lower()
        elif style.get("sentence_length") == "short" and len(custom_word_tokenize(sentence)) > 15:
            # 긴 문장을 분할
            words = custom_word_tokenize(sentence)
            middle = len(words) // 2
            first_half = ' '.join(words[:middle])
            second_half = ' '.join(words[middle:])
            styled_sentences.append(first_half + '.')
            sentence = second_half
        
        styled_sentences.append(sentence)
    
    return ' '.join(styled_sentences)

# 문화적 맥락에 맞게 적응
def adapt_to_culture(text, culture_data):
    sentences = custom_sent_tokenize(text)
    culturally_adapted = []
    
    for sentence in sentences:
        # 10% 확률로 문화적 관용구 추가
        if random.random() < 0.1 and culture_data.get("idioms"):
            idiom = random.choice(culture_data.get("idioms"))
            sentence = sentence + " " + idiom + "."
        
        # 측정 단위 변환
        if culture_data.get("measurements") == "imperial":
            sentence = re.sub(r'(\d+)\s*km', lambda m: f"{float(m.group(1)) * 0.621:.1f} miles", sentence)
            sentence = re.sub(r'(\d+)\s*kg', lambda m: f"{float(m.group(1)) * 2.205:.1f} pounds", sentence)
        elif culture_data.get("measurements") == "metric":
            sentence = re.sub(r'(\d+)\s*miles', lambda m: f"{float(m.group(1)) * 1.609:.1f} km", sentence)
            sentence = re.sub(r'(\d+)\s*pounds', lambda m: f"{float(m.group(1)) * 0.454:.1f} kg", sentence)
        
        culturally_adapted.append(sentence)
    
    return ' '.join(culturally_adapted)

# 문장 단위 도메인 강화 
def enhance_sentence_with_domain(sentence, domain_vocabulary, prev_context="", next_context=""):
    words = custom_word_tokenize(sentence)
    domain_enhanced = []
    
    for word in words:
        word_lower = word.lower()
        # 일반적인 단어를 도메인 관련 단어로 교체 (20% 확률)
        if word_lower in ['good', 'great', 'important', 'interesting'] and random.random() < 0.2:
            domain_word = random.choice(domain_vocabulary)
            domain_enhanced.append(domain_word)
        else:
            domain_enhanced.append(word)
    
    enhanced_sentence = ' '.join(domain_enhanced)
    
    # 문맥을 고려한 연결 구문 추가
    if prev_context and not any(sentence.lower().startswith(x) for x in ['however', 'moreover', 'furthermore']):
        connectors = ['Furthermore', 'Moreover', 'In addition', 'Subsequently']
        enhanced_sentence = f"{random.choice(connectors)}, {enhanced_sentence.lower()}"
    
    return enhanced_sentence

# 고급 재작성을 위한 동의어 및 문구 사전
@st.cache_resource
def get_advanced_synonyms():
    return {
        # 기존 단어
        'good': ['exemplary', 'exceptional', 'impeccable', 'superb', 'outstanding', 'excellent'],
        'bad': ['detrimental', 'deplorable', 'egregious', 'substandard', 'inadequate', 'unfavorable'],
        'big': ['immense', 'formidable', 'monumental', 'substantial', 'enormous', 'extensive'],
        'small': ['minuscule', 'negligible', 'infinitesimal', 'minute', 'diminutive', 'marginal'],
        'happy': ['euphoric', 'exuberant', 'ecstatic', 'elated', 'jubilant', 'overjoyed'],
        'sad': ['despondent', 'crestfallen', 'dejected', 'melancholic', 'disheartened', 'disconsolate'],
        'important': ['imperative', 'indispensable', 'paramount', 'pivotal', 'crucial', 'vital'],
        'difficult': ['formidable', 'insurmountable', 'herculean', 'arduous', 'demanding', 'laborious'],
        'easy': ['effortless', 'rudimentary', 'facile', 'straightforward', 'uncomplicated', 'simple'],
        'beautiful': ['resplendent', 'breathtaking', 'sublime', 'exquisite', 'magnificent', 'stunning'],
        
        # 추가 단어
        'interesting': ['fascinating', 'compelling', 'intriguing', 'captivating', 'engaging', 'absorbing'],
        'boring': ['tedious', 'monotonous', 'mundane', 'insipid', 'dull', 'unengaging'],
        'smart': ['astute', 'perspicacious', 'erudite', 'brilliant', 'sagacious', 'ingenious'],
        'stupid': ['obtuse', 'imperceptive', 'injudicious', 'imprudent', 'misguided', 'inept'],
        'nice': ['amiable', 'congenial', 'benevolent', 'affable', 'cordial', 'genial'],
        'angry': ['indignant', 'irate', 'incensed', 'furious', 'exasperated', 'vexed'],
        'scared': ['apprehensive', 'trepidatious', 'daunted', 'alarmed', 'disquieted', 'perturbed'],
        'funny': ['humorous', 'comical', 'amusing', 'hilarious', 'entertaining', 'witty'],
        'useless': ['ineffectual', 'fruitless', 'unproductive', 'futile', 'unavailing', 'inefficacious'],
        'useful': ['advantageous', 'beneficial', 'serviceable', 'practical', 'valuable', 'constructive'],
        
        # 동사
        'say': ['articulate', 'proclaim', 'assert', 'express', 'pronounce', 'convey'],
        'look': ['observe', 'scrutinize', 'examine', 'survey', 'inspect', 'perceive'],
        'make': ['construct', 'fabricate', 'produce', 'generate', 'establish', 'formulate'],
        'get': ['acquire', 'obtain', 'procure', 'attain', 'secure', 'gain'],
        'take': ['appropriate', 'seize', 'acquire', 'grasp', 'adopt', 'employ'],
        'give': ['bestow', 'confer', 'impart', 'provide', 'allocate', 'distribute'],
        'find': ['discover', 'locate', 'detect', 'uncover', 'identify', 'ascertain'],
        'tell': ['relate', 'divulge', 'disclose', 'reveal', 'narrate', 'recount'],
        'ask': ['inquire', 'interrogate', 'query', 'question', 'solicit', 'request'],
        'feel': ['experience', 'perceive', 'sense', 'discern', 'intuit', 'apprehend'],
        
        # 자주 쓰는 부사
        'very': ['exceedingly', 'immensely', 'tremendously', 'extraordinarily', 'remarkably', 'profoundly'],
        'really': ['genuinely', 'authentically', 'truly', 'undeniably', 'veritably', 'legitimately'],
        'just': ['precisely', 'specifically', 'exactly', 'particularly', 'solely', 'exclusively'],
        'so': ['consequently', 'therefore', 'thus', 'accordingly', 'hence', 'subsequently'],
        'quite': ['considerably', 'substantially', 'significantly', 'markedly', 'notably', 'decidedly'],
        'too': ['excessively', 'overly', 'unduly', 'inordinately', 'immoderately', 'disproportionately']
    }

@st.cache_resource
def get_advanced_phrases():
    return {
        # 기본 표현
        r'\bi think\b': ['I postulate that', 'I am of the conviction that', 'It is my considered opinion that', 
                         'I hold the perspective that', 'My assessment suggests that', 'I would posit that'],
        r'\bi like\b': ['I am particularly enamored with', 'I hold in high regard', 'I find great merit in', 
                       'I am especially fond of', 'I have a distinct preference for', 'I am drawn to'],
        r'\bi want\b': ['I aspire to', 'I am inclined towards', 'My inclination is toward', 
                       'I seek to', 'It is my desire to', 'I am motivated to'],
        r'\blots of\b': ['a plethora of', 'an abundance of', 'a multitude of', 
                        'a substantial quantity of', 'a considerable amount of', 'numerous instances of'],
        r'\bmany of\b': ['a preponderance of', 'a substantial proportion of', 'a significant contingent of', 
                        'a notable fraction of', 'a considerable number of', 'a substantial segment of'],
                        
        # 추가 기본 표현
        r'\bI need\b': ['I require', 'I necessitate', 'It is imperative for me to', 
                       'I find it essential to', 'I am in need of', 'I deem it necessary to'],
        r'\bI believe\b': ['I am convinced that', 'I maintain that', 'I subscribe to the notion that', 
                         'I adhere to the principle that', 'My conviction is that', 'I firmly hold that'],
        r'\bI know\b': ['I am cognizant of', 'I am well-versed in', 'I am thoroughly familiar with', 
                       'I possess comprehensive knowledge of', 'I am well-acquainted with', 'I am fully aware of'],
        r'\bI agree\b': ['I concur with', 'I am in accordance with', 'I align myself with', 
                        'I share the sentiment that', 'I endorse the view that', 'I am of one mind with'],
        r'\bI disagree\b': ['I take issue with', 'I contest the notion that', 'I diverge from the view that', 
                          'I am at variance with', 'I oppose the perspective that', 'I cannot reconcile myself with'],
        
        # 시작 문구 개선
        r'^In conclusion': ['To synthesize the aforementioned points', 'Drawing all aspects into consideration', 
                           'Upon final analysis', 'As a culmination of these insights', 'In summation', 'To consolidate the arguments presented'],
        r'^Moreover': ['Furthermore, it warrants emphasis that', 'In addition to the foregoing', 
                      'Beyond what has been established', 'As an extension of this reasoning', 'To augment these considerations', 'Building upon this foundation'],
        r'^However': ['Nonetheless, it must be acknowledged that', 'Conversely, one must consider that', 
                     'Despite this assertion', 'In contrast to this perspective', 'Yet, it remains evident that', 'Notwithstanding these factors'],
        r'^First': ['As an initial consideration', 'Foremost among these factors', 
                   'The primary aspect to consider is', 'To begin this analysis', 'The foundational element is', 'At the forefront of this discussion'],
        r'^For example': ['To illustrate this concept', 'As exemplified by', 
                         'A pertinent instance of this phenomenon is', 'Consider, for instance', 'A demonstrative case is', 'By way of illustration'],
        
        # 한국 학습자 특화 표현 개선
        r'\bI will do\b': ['I shall undertake', 'I intend to proceed with', 'I commit myself to', 
                          'I will diligently execute', 'I will dedicate myself to accomplishing', 'I will systematically address'],
        r'\bvery much\b': ['substantially', 'considerably', 'significantly', 
                          'to a marked degree', 'to a notable extent', 'remarkably'],
        r'\bI studied\b': ['I engaged in scholarly pursuit of', 'I devoted time to examining', 
                          'I conducted an academic exploration of', 'I undertook a systematic study of', 'I applied myself to learning', 'I dedicated myself to mastering'],
        r'\bIt is important\b': ['It is imperative', 'It is of paramount significance', 
                                'It holds critical importance', 'It is fundamentally essential', 'It remains a crucial consideration', 'It constitutes a vital element']
    }

@st.cache_resource
def get_domain_specific_phrases(domain='academic'):
    """
    특정 영역(학술, 비즈니스, 일상 등)에 맞는 어휘 및 문구를 제공합니다.
    """
    phrases = {
        'academic': {
            r'\bshow\b': ['demonstrate', 'indicate', 'establish', 'reveal', 'manifest', 'elucidate'],
            r'\btopic\b': ['subject matter', 'field of inquiry', 'area of study', 'focus of research', 'domain of investigation'],
            r'\bresearch\b': ['scholarly investigation', 'academic inquiry', 'empirical study', 'systematic exploration', 'scientific examination'],
            r'\bdata\b': ['empirical evidence', 'quantitative measurements', 'statistical information', 'research findings', 'collected observations'],
            r'\bmethod\b': ['methodological approach', 'analytical framework', 'procedural technique', 'investigative protocol', 'systematic procedure'],
            r'^The paper discusses': ['This scholarly work examines', 'The present study analyzes', 'This research investigates', 'This academic treatise addresses', 'The current investigation explores'],
            r'^This study aims to': ['The objective of this research is to', 'This investigation seeks to', 'The purpose of this scholarly inquiry is to', 'This academic examination endeavors to', 'This analysis attempts to'],
            r'^Results show': ['The empirical findings indicate', 'The data demonstrates', 'The research outcomes reveal', 'The analytical results suggest', 'The experimental evidence confirms']
        },
        'business': {
            r'\bmoney\b': ['financial resources', 'capital', 'monetary assets', 'funds', 'financial means'],
            r'\bcostumer\b': ['client', 'patron', 'consumer', 'end-user', 'purchaser'],
            r'\bboss\b': ['executive director', 'chief executive officer', 'senior manager', 'supervisor', 'administrative head'],
            r'\bjob\b': ['professional role', 'career position', 'occupational function', 'employment capacity', 'professional responsibility'],
            r'\bmeeting\b': ['strategic gathering', 'corporate assembly', 'professional conference', 'business consultation', 'executive session'],
            r'^I want to apply': ['I wish to submit my candidacy', 'I am interested in pursuing a position', 'I would like to express my interest in', 'I am seeking to secure a role', 'I aspire to join your organization as'],
            r'^Please find attached': ['I have enclosed for your consideration', 'Attached herewith is', 'I submit for your review', 'Kindly find accompanying this correspondence', 'For your perusal, I have included'],
            r'^We need to discuss': ['It would be beneficial to address', 'We should consider deliberating on', 'I propose we convene to examine', 'It is imperative we confer regarding', 'A consultation is warranted concerning']
        },
        'conversation': {
            r'\bnice to meet you\b': ['pleased to make your acquaintance', 'delighted to encounter you', 'it is a pleasure to be introduced', 'charmed to meet you', 'gratified by our meeting'],
            r'\bthanks\b': ['I appreciate your assistance', 'I am grateful for', 'my sincere gratitude for', 'I extend my thanks for', 'I would like to express my appreciation for'],
            r'\bsorry\b': ['I apologize for', 'I regret that', 'please accept my apologies for', 'I must express my regret regarding', 'I am remorseful about'],
            r'\bbye\b': ['farewell', 'until we meet again', 'I bid you adieu', 'I look forward to our next encounter', 'I must now take my leave'],
            r'\bfriend\b': ['companion', 'confidant', 'associate', 'acquaintance', 'ally'],
            r'^How are you': ['I trust you are faring well', 'How do you find yourself today', 'I hope this day finds you in good spirits', 'May I inquire after your wellbeing', 'I trust all is well with you'],
            r'^I hope you': ['It is my sincere wish that you', 'I earnestly anticipate that you', 'My expectations are that you', 'I look forward to you', 'My desire is that you'],
            r'^I'm writing to': ['I am corresponding to', 'The purpose of my communication is to', 'I am reaching out to', 'This message serves to', 'I am taking the liberty of contacting you to']
        }
    }
    
    return phrases.get(domain, phrases['academic'])

def enhance_patterns_from_corpus():
    patterns = get_advanced_phrases()  # 기본 패턴
    
    # 학술 문헌에서 추출한 추가 패턴
    academic_patterns = {
        r'\bshow that\b': ['demonstrate that', 'indicate that', 'reveal that', 'establish that', 'evidence that'],
        r'\bthe result is\b': ['the outcome suggests', 'findings indicate', 'this yields', 'the data confirms', 'empirical evidence shows'],
        r'\bin recent years\b': ['in contemporary scholarship', 'in the current academic landscape', 'in modern research', 'in present-day studies', 'in the evolving literature'],
        r'\bthe purpose of this study\b': ['the objective of this investigation', 'this research aims to', 'the goal of this analysis', 'this inquiry seeks to', 'the focus of this examination'],
        r'\bprevious research\b': ['extant literature', 'prior investigations', 'established scholarship', 'existing studies', 'antecedent research'],
        r'\bneed more research\b': ['warrant further investigation', 'necessitate additional scholarly inquiry', 'require more comprehensive examination', 'demand deeper academic exploration', 'call for extended analysis'],
        r'\bthis paper discusses\b': ['this study examines', 'this investigation addresses', 'this research analyzes', 'this work explores', 'this scholarly effort evaluates'],
        r'\bit is clear that\b': ['evidence unequivocally suggests that', 'it is demonstrably evident that', 'findings conclusively indicate that', 'it is empirically verifiable that', 'data substantiates the conclusion that']
    }
    
    patterns.update(academic_patterns)
    return patterns

# 영작문 재작성 기능 수정 (다양한 수준으로 변환)
def rewrite_text(text, level='similar'):
    """
    학생이 작성한 영어 텍스트를 지정된 수준으로 재작성합니다.
    
    Parameters:
    - text: 원본 텍스트
    - level: 'similar' (비슷한 수준), 'improved' (약간 더 높은 수준), 'advanced' (고급 수준)
    
    Returns:
    - 재작성된 텍스트
    """
    if not text.strip():
        return ""
    
    try:
        # 고급 AI 모델 사용 가능한 경우
        if level == 'advanced' and has_transformers:
            try:
                return advanced_rewrite_text(text, level)
            except Exception as e:
                st.warning(f"고급 모델 사용 중 오류: {e}. 대체 방법으로 진행합니다.")
        
        # 문장 단위로 분석 및 재작성
        sentences = custom_sent_tokenize(text)
        rewritten_sentences = []
        
        for sentence in sentences:
            # 원본 문장 구존 보존
            if level == 'similar':
                rewritten = rewrite_similar_level(sentence)
            # 약간 더 높은 수준으로 개선
            elif level == 'improved':
                rewritten = rewrite_improved_level(sentence)
            # 고급 수준으로 변환
            elif level == 'advanced':
                rewritten = rewrite_advanced_level(sentence)
            else:
                rewritten = sentence
            
            rewritten_sentences.append(rewritten)
        
        # 문장 구조 변환 패턴 추가
        structure_transformations = {
            r'^I believe': ['In my opinion', 'From my perspective', 'I am of the view that'],
            r'^There is': ['There exists', 'We can observe', 'It is evident that there is'],
            r'^It is important': ['It is crucial', 'It is essential', 'A key consideration is'],
        }
        
        # 재작성 함수에 적용
        for i, sentence in enumerate(rewritten_sentences):
            for pattern, replacements in structure_transformations.items():
                if re.search(pattern, sentence):
                    replacement = random.choice(replacements)
                    rewritten_sentences[i] = re.sub(pattern, replacement, sentence)
        
        # 주제 일관성 강화 및 흐름 개선 (advanced 모드에서만)
        if level == 'advanced':
            rewritten_sentences = maintain_topic_coherence(rewritten_sentences)
        
        return ' '.join(rewritten_sentences)
    except Exception as e:
        st.error(f"텍스트 재작성 중 오류가 발생했습니다: {e}")
        return text

# 고급 수준으로 재작성
def rewrite_advanced_level(sentence):
    # 고급 어휘로 대체
    advanced_synonyms = get_advanced_synonyms()
    advanced_phrases = get_advanced_phrases()
    
    # 문장 구조 개선 패턴
    structure_improvements = {
        r'^I am ': ['Being ', 'As someone who is '],
        r'^I have ': ['Having ', 'Possessing '],
        r'^This is ': ['This constitutes ', 'This represents '],
        r'^There are ': ['There exist ', 'One can observe '],
        r'^It is ': ['It remains ', 'It stands as '],
    }
    
    # 우선 단어 수준 개선
    words = custom_word_tokenize(sentence)
    result = []
    
    for word in words:
        word_lower = word.lower()
        if word_lower in advanced_synonyms and random.random() < 0.8:  # 80% 확률로 대체
            synonyms = advanced_synonyms[word_lower]
            replacement = random.choice(synonyms)
            
            # 대문자 보존
            if word[0].isupper():
                replacement = replacement.capitalize()
            
            result.append(replacement)
        else:
            result.append(word)
    
    advanced_text = ' '.join(result)
    
    # 문구 패턴 개선
    for pattern, replacements in advanced_phrases.items():
        if re.search(pattern, advanced_text, re.IGNORECASE):
            replacement = random.choice(replacements)
            advanced_text = re.sub(pattern, replacement, advanced_text, flags=re.IGNORECASE)
    
    # 문장 구조 개선
    for pattern, replacements in structure_improvements.items():
        if re.search(pattern, advanced_text):
            if random.random() < 0.7:  # 70% 확률로 구조 변경
                replacement = random.choice(replacements)
                advanced_text = re.sub(pattern, replacement, advanced_text)
    
    return advanced_text

# 비슷한 수준으로 재작성
def rewrite_similar_level(sentence):
    # 간단한 동의어 대체만 수행
    basic_synonyms = {
        'good': ['nice', 'fine', 'decent'],
        'bad': ['poor', 'negative', 'unpleasant'],
        'big': ['large', 'sizable', 'substantial'],
        'small': ['little', 'tiny', 'minor'],
        'happy': ['glad', 'pleased', 'content'],
        'sad': ['unhappy', 'upset', 'down']
    }
    
    words = custom_word_tokenize(sentence)
    result = []
    
    for word in words:
        word_lower = word.lower()
        if word_lower in basic_synonyms and random.random() < 0.3:  # 30% 확률로 대체
            synonyms = basic_synonyms[word_lower]
            replacement = random.choice(synonyms)
            
            # 대문자 보존
            if word[0].isupper():
                replacement = replacement.capitalize()
            
            result.append(replacement)
        else:
            result.append(word)
    
    return ' '.join(result)

# 약간 더 높은 수준으로 재작성
def rewrite_improved_level(sentence):
    # 중급 어휘로 대체
    intermediate_synonyms = {
        'good': ['excellent', 'outstanding', 'superb'],
        'bad': ['inferior', 'substandard', 'inadequate'],
        'big': ['enormous', 'extensive', 'considerable'],
        'small': ['diminutive', 'slight', 'limited'],
        'happy': ['delighted', 'thrilled', 'overjoyed'],
        'sad': ['depressed', 'miserable', 'gloomy']
    }
    
    # 기본 문구 개선
    improved_phrases = {
        r'\bI think\b': ['I believe', 'In my opinion', 'I consider'],
        r'\bI like\b': ['I enjoy', 'I appreciate', 'I am fond of'],
        r'\bI want\b': ['I desire', 'I wish', 'I would like'],
        r'\blots of\b': ['numerous', 'many', 'plenty of'],
        r'\bvery\b': ['extremely', 'particularly', 'significantly']
    }
    
    # 우선 단어 수준 개선
    words = custom_word_tokenize(sentence)
    result = []
    
    for word in words:
        word_lower = word.lower()
        if word_lower in intermediate_synonyms and random.random() < 0.5:  # 50% 확률로 대체
            synonyms = intermediate_synonyms[word_lower]
            replacement = random.choice(synonyms)
            
            # 대문자 보존
            if word[0].isupper():
                replacement = replacement.capitalize()
            
            result.append(replacement)
        else:
            result.append(word)
    
    improved_text = ' '.join(result)
    
    # 문구 패턴 개선
    for pattern, replacements in improved_phrases.items():
        if re.search(pattern, improved_text, re.IGNORECASE):
            replacement = random.choice(replacements)
            improved_text = re.sub(pattern, replacement, improved_text, flags=re.IGNORECASE)
    
    return improved_text

@st.cache_resource(ttl=86400)
def load_expanded_suggestions():
    try:
        # 외부 소스에서 데이터 로드 또는 로컬 파일 사용
        import json
        with open('data/expanded_suggestions.json', 'r') as f:
            return json.load(f)
    except:
        return get_custom_suggestions()  # 폴백 옵션

@st.cache_resource
def get_expanded_vocabulary_levels():
    # A1~C2 레벨별 단어 데이터 로드
    vocab_url = "https://raw.githubusercontent.com/openlanguagedata/cefr-english/main/cefr_wordlist.csv"
    try:
        import pandas as pd
        df = pd.read_csv(vocab_url)
        return {
            'basic': set(df[df['level'].isin(['A1', 'A2'])]['word']),
            'intermediate': set(df[df['level'].isin(['B1', 'B2'])]['word']),
            'advanced': set(df[df['level'].isin(['C1', 'C2'])]['word'])
        }
    except:
        return default_vocabulary_sets()

def get_enriched_synonyms(word, level):
    try:
        from nltk.corpus import wordnet
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        
        # 난이도에 따라 적절한 동의어 필터링
        return synonyms if synonyms else get_fallback_synonyms(word, level)
    except:
        return get_fallback_synonyms(word, level)

def get_fallback_synonyms(word, level):
    if level == 'advanced':
        synonyms_dict = get_advanced_synonyms()
    elif level == 'improved':
        synonyms_dict = {
            'good': ['excellent', 'outstanding', 'superb'],
            'bad': ['inferior', 'substandard', 'inadequate'],
            'big': ['enormous', 'extensive', 'considerable'],
            'small': ['diminutive', 'slight', 'limited'],
            'happy': ['delighted', 'thrilled', 'overjoyed'],
            'sad': ['depressed', 'miserable', 'gloomy']
        }
    else:  # similar/basic level
        synonyms_dict = {
            'good': ['nice', 'fine', 'decent'],
            'bad': ['poor', 'negative', 'unpleasant'],
            'big': ['large', 'sizable', 'substantial'],
            'small': ['little', 'tiny', 'minor'],
            'happy': ['glad', 'pleased', 'content'],
            'sad': ['unhappy', 'upset', 'down']
        }
    
    return synonyms_dict.get(word.lower(), [word])  # 기본값으로 원래 단어 반환

def comprehensive_vocabulary_analysis(text):
    # 기본 평가
    basic_eval = evaluate_vocabulary_level(text)
    
    # 추가 측정: Type-Token Ratio, Lexical Density, Academic Word Usage
    words = custom_word_tokenize(text.lower())
    if not words:
        return basic_eval
        
    # 어휘 밀도(내용어 비율)
    content_words = [w for w in words if w not in stopwords.words('english')]
    lexical_density = len(content_words) / len(words)
    
    # 학술어 비율
    academic_words = get_academic_word_list()
    academic_ratio = len([w for w in words if w in academic_words]) / len(words)
    
    return {**basic_eval, 'lexical_density': lexical_density, 'academic_ratio': academic_ratio}
