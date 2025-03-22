import streamlit as st

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
from datetime import datetime
from textblob import TextBlob

# NLTK 라이브러리 및 데이터 처리
import nltk
from nltk.corpus import stopwords

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

# 맞춤법 검사 라이브러리를 조건부로 임포트
try:
    import enchant
    has_enchant = True
except ImportError:
    has_enchant = False
    st.info("맞춤법 검사 라이브러리(enchant)가 설치되지 않았습니다. TextBlob을 사용한 기본 맞춤법 검사만 제공됩니다.")

# 수정된 sent_tokenize 함수 (NLTK 의존성 제거)
def custom_sent_tokenize(text):
    if not text:
        return []
    
    # 정규식 기반 문장 분할기
    # 온점, 느낌표, 물음표 뒤에 공백이 오는 패턴을 기준으로 분할
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # 빈 문장 제거
    return [s.strip() for s in sentences if s.strip()]

# 수정된 word_tokenize 함수 (NLTK 의존성 제거)
def custom_word_tokenize(text):
    if not text:
        return []
    
    # 효과적인 정규식 패턴으로 단어 토큰화
    # 축약형(I'm, don't 등), 소유격(John's), 하이픈 단어(well-known) 등을 처리
    tokens = []
    # 기본 단어(알파벳 + 숫자 + 아포스트로피 + 하이픈)
    words = re.findall(r'\b[\w\'-]+\b', text)
    # 구두점
    punctuation = re.findall(r'[.,!?;:"]', text)
    
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
    return None

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
                        if checker and not checker.check(word):
                            suggestions = checker.suggest(word)[:3]  # 최대 3개 제안
                            word_offset = text.find(word, offset)
                            
                            if word_offset != -1:
                                errors.append({
                                    "offset": word_offset,
                                    "errorLength": len(word),
                                    "message": f"맞춤법 오류: '{word}'",
                                    "replacements": suggestions
                                })
                        # TextBlob을 사용한 맞춤법 검사 대안 (enchant가 없는 경우)
                        elif not checker and has_enchant == False:
                            try:
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

# 어휘 수준 평가 (간단한 버전)
def evaluate_vocabulary_level(text):
    # 영어 단어 수준을 나타내는 샘플 데이터 (실제로는 더 큰 데이터셋 필요)
    basic_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 
                  'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but'}
    intermediate_words = {'achieve', 'consider', 'determine', 'establish', 'indicate', 'occur',
                        'participate', 'predict', 'provide', 'recognize', 'resolve', 'specific', 
                        'therefore', 'utilize', 'aspect', 'concept', 'context', 'diverse'}
    advanced_words = {'arbitrary', 'cognitive', 'encompass', 'facilitate', 'fundamental', 'implicit',
                     'intricate', 'legitimate', 'paradigm', 'phenomenon', 'pragmatic', 'scrutinize',
                     'sophisticated', 'subsequent', 'synthesis', 'theoretical', 'underlying'}
    
    words = custom_word_tokenize(text.lower())
    words = [word for word in words if re.match(r'\w+', word)]
    
    word_set = set(words)
    
    basic_count = len(word_set.intersection(basic_words))
    intermediate_count = len(word_set.intersection(intermediate_words))
    advanced_count = len(word_set.intersection(advanced_words))
    
    total = basic_count + intermediate_count + advanced_count
    if total == 0:
        return {'basic': 0, 'intermediate': 0, 'advanced': 0}
    
    return {
        'basic': basic_count / max(1, total),
        'intermediate': intermediate_count / max(1, total),
        'advanced': advanced_count / max(1, total)
    }

# 어휘 수준 평가 (간단한 버전)
def evaluate_vocabulary_level_simple(text):
    # 영어 단어 수준을 나타내는 샘플 데이터 (실제로는 더 큰 데이터셋 필요)
    basic_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 
                  'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but'}
    intermediate_words = {'achieve', 'consider', 'determine', 'establish', 'indicate', 'occur',
                        'participate', 'predict', 'provide', 'recognize', 'resolve', 'specific', 
                        'therefore', 'utilize', 'aspect', 'concept', 'context', 'diverse'}
    advanced_words = {'arbitrary', 'cognitive', 'encompass', 'facilitate', 'fundamental', 'implicit',
                     'intricate', 'legitimate', 'paradigm', 'phenomenon', 'pragmatic', 'scrutinize',
                     'sophisticated', 'subsequent', 'synthesis', 'theoretical', 'underlying'}
    
    words = custom_word_tokenize(text.lower())
    words = [word for word in words if re.match(r'\w+', word)]
    
    word_set = set(words)
    
    basic_count = len(word_set.intersection(basic_words))
    intermediate_count = len(word_set.intersection(intermediate_words))
    advanced_count = len(word_set.intersection(advanced_words))
    
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
    
    tab1, tab2 = st.tabs(["영작문 검사", "내 작문 기록"])
    
    with tab1:
        st.subheader("영작문 입력")
        user_text = st.text_area("아래에 영어 작문을 입력하세요", height=200)
        
        col1, col2 = st.columns(2)
        
        with col1:
            grammar_tab, vocab_tab = st.tabs(["문법 검사", "어휘 분석"])
            
            with grammar_tab:
                if st.button("문법 검사하기"):
                    if not user_text:
                        st.warning("텍스트를 입력해주세요.")
                    else:
                        # 문법 오류 검사
                        grammar_errors = check_grammar(user_text)
                        
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
                            
                            # 기록에 저장
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.history.append({
                                'timestamp': timestamp,
                                'text': user_text,
                                'error_count': len(grammar_errors)
                            })
                        else:
                            st.success("문법 오류가 없습니다!")
                            
                            # 기록에 저장
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.history.append({
                                'timestamp': timestamp,
                                'text': user_text,
                                'error_count': 0
                            })
            
            with vocab_tab:
                if st.button("어휘 분석하기"):
                    if not user_text:
                        st.warning("텍스트를 입력해주세요.")
                    else:
                        # 어휘 분석
                        vocab_analysis = analyze_vocabulary(user_text)
                        
                        # 단어 빈도 시각화
                        fig = plot_word_frequency(vocab_analysis['word_freq'])
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 어휘 다양성 점수
                        diversity_score = calculate_lexical_diversity(user_text)
                        st.metric("어휘 다양성 점수", f"{diversity_score:.2f}", 
                                 delta="높을수록 다양한 어휘 사용")
                        
                        # 어휘 수준 평가
                        vocab_level = evaluate_vocabulary_level(user_text)
                        level_df = pd.DataFrame({
                            '수준': ['기초', '중급', '고급'],
                            '비율': [vocab_level['basic'], vocab_level['intermediate'], vocab_level['advanced']]
                        })
                        
                        fig = px.pie(level_df, values='비율', names='수준', 
                                    title='어휘 수준 분포',
                                    color_discrete_sequence=px.colors.sequential.Viridis)
                        st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if st.button("텍스트 분석하기"):
                if not user_text:
                    st.warning("텍스트를 입력해주세요.")
                else:
                    # 텍스트 통계 분석
                    stats = analyze_text(user_text)
                    
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
    
    with tab2:
        st.subheader("내 작문 기록")
        if not st.session_state.history:
            st.info("아직 기록이 없습니다.")
        else:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df)
            
            # 오류 수 추이 차트
            if len(history_df) > 1:
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
    
    tab1, tab2 = st.tabs(["영작문 첨삭", "학습 대시보드"])
    
    with tab1:
        st.subheader("영작문 입력 및 첨삭")
        
        user_text = st.text_area("학생의 영어 작문을 입력하세요", height=200)
        
        col1, col2 = st.columns(2)
        
        with col1:
            grammar_tab, vocab_tab = st.tabs(["문법 검사", "어휘 분석"])
            
            with grammar_tab:
                if st.button("문법 검사하기", key="teacher_grammar"):
                    if not user_text:
                        st.warning("텍스트를 입력해주세요.")
                    else:
                        # 문법 오류 검사
                        grammar_errors = check_grammar(user_text)
                        
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
            
            with vocab_tab:
                if st.button("어휘 분석하기", key="teacher_vocab"):
                    if not user_text:
                        st.warning("텍스트를 입력해주세요.")
                    else:
                        # 어휘 분석
                        vocab_analysis = analyze_vocabulary(user_text)
                        
                        # 단어 빈도 시각화
                        fig = plot_word_frequency(vocab_analysis['word_freq'])
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 어휘 다양성 점수
                        diversity_score = calculate_lexical_diversity(user_text)
                        st.metric("어휘 다양성 점수", f"{diversity_score:.2f}")
                        
                        # 어휘 수준 평가
                        vocab_level = evaluate_vocabulary_level(user_text)
                        level_df = pd.DataFrame({
                            '수준': ['기초', '중급', '고급'],
                            '비율': [vocab_level['basic'], vocab_level['intermediate'], vocab_level['advanced']]
                        })
                        
                        fig = px.pie(level_df, values='비율', names='수준', 
                                    title='어휘 수준 분포')
                        st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if st.button("텍스트 분석하기", key="teacher_analysis"):
                if not user_text:
                    st.warning("텍스트를 입력해주세요.")
                else:
                    # 텍스트 통계 분석
                    stats = analyze_text(user_text)
                    
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
        feedback = st.text_area("첨삭 노트", 
                 value="다음 사항을 중점적으로 개선해보세요:\n1. \n2. \n3. ", 
                 height=100,
                 key="feedback_template")
        
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
    
    with tab2:
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
