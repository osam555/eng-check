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
import requests

# 자체 제작한 custom_suggestions 모듈 import
try:
    from custom_suggestions import get_custom_suggestions
except ImportError:
    # 모듈을 import 할 수 없는 경우를 대비한 기본 함수
    @st.cache_resource
    def get_custom_suggestions():
        return {
            'tonite': ['tonight'],
            'tomorow': ['tomorrow'],
            'yestarday': ['yesterday'],
            'definately': ['definitely'],
            'recieve': ['receive'],
            'wierd': ['weird'],
            'alot': ['a lot'],
            'untill': ['until'],
            'thier': ['their'],
            # 축소된 버전
        }

# LanguageTool API 임포트 시도
try:
    import language_tool_python
    has_languagetool = True
except ImportError:
    has_languagetool = False
    print("language-tool-python이 설치되어 있지 않습니다. 기본 문법 검사 기능을 사용합니다.")

# GrammarBot API 임포트 시도
try:
    from grammarbot import GrammarBotClient
    has_grammarbot = True
except ImportError:
    has_grammarbot = False
    print("grammarbot이 설치되어 있지 않습니다. 대체 문법 검사 기능을 사용합니다.")

# Gramformer 임포트 시도 (문법 교정 모델)
try:
    from gramformer import Gramformer
    has_gramformer = True
except (ImportError, ModuleNotFoundError):
    has_gramformer = False
    print("gramformer가 설치되어 있지 않습니다. 대체 문법 교정 기능을 사용합니다.")
    
    # gramformer가 없는 경우 더미 클래스 구현
    class DummyGramformer:
        def __init__(self, models=1, use_gpu=False):
            pass
            
        def correct(self, text, max_candidates=1):
            return [text]  # 입력 텍스트 그대로 반환
    
    # 실제 클래스가 없을 때 사용할 더미 클래스 정의
    Gramformer = DummyGramformer

# 텍스트를 음성으로 변환하는 함수
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

# 비동기 함수를 동기식으로 호출하는 래퍼 함수
def sync_text_to_speech(text, voice="en-US-JennyNeural", output_file=None):
    """
    text_to_speech 함수를 동기식으로 호출하는 래퍼 함수
    
    Parameters:
    - text: 음성으로 변환할 텍스트
    - voice: 음성 모델 (기본값: 'en-US-JennyNeural')
    - output_file: 출력 파일 경로 (None인 경우 임시 파일 생성)
    
    Returns:
    - 음성 파일 경로
    """
    # 비동기 함수 실행을 위한 런타임 설정
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        audio_path = loop.run_until_complete(text_to_speech(text, voice, output_file))
        return audio_path
    finally:
        loop.close()

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

# 맞춤법 검사기 초기화 함수
def get_spell_checker():
    """
    사용 가능한 맞춤법 검사기를 로드합니다.
    여러 라이브러리를 시도하고 사용 가능한 첫 번째 검사기를 반환합니다.
    """
    # PyEnchant 사용 시도
    if 'has_enchant' in globals() and has_enchant:
        try:
            return enchant.Dict("en_US")
        except Exception as e:
            print(f"PyEnchant 초기화 오류: {e}")
    
    # PySpellChecker 사용 시도
    if 'has_spellchecker' in globals() and has_spellchecker:
        try:
            return SpellChecker()
        except Exception as e:
            print(f"PySpellChecker 초기화 오류: {e}")
    
    # 둘 다 실패한 경우 None 반환
            return None

# LanguageTool 검사기 초기화 함수
def get_language_tool():
    """
    LanguageTool 검사기를 초기화하고 반환합니다.
    """
    try:
        import language_tool_python
        return language_tool_python.LanguageTool('en-US')
    except Exception as e:
        st.error(f"LanguageTool 초기화 오류: {str(e)}")
        return None

# GrammarBot 검사기 초기화 함수
def get_grammar_bot():
    """
    GrammarBot 검사기를 초기화하고 반환합니다.
    """
    if has_grammarbot:
        try:
            # GrammarBot 클라이언트 생성
            return GrammarBotClient()
        except Exception as e:
            print(f"GrammarBot 초기화 오류: {e}")
    
    return None

# Gramformer 초기화 함수
def get_gramformer():
    """
    Gramformer 문법 교정 모델을 초기화하고 반환합니다.
    """
    if has_gramformer:
        try:
            # Gramformer 모델 로드 (문법 교정용)
            return Gramformer(models=1, use_gpu=False)  # CPU 모드
        except Exception as e:
            print(f"Gramformer 초기화 오류: {e}")
    
    return None

# 한국인이 자주 범하는 영어 오류 패턴 정의
def get_korean_english_error_patterns():
    """
    한국인이 영어를 배울 때 자주 범하는 오류 패턴을 반환합니다.
    """
    return [
        # 관사 오류
        (r'\b([aeiou]\w+)\b', 'an \1', r'\ba ([aeiou]\w+)\b', "모음으로 시작하는 단어 앞에는 'a' 대신 'an'을 사용해야 합니다"),
        (r'\ban ([^aeiou\W]\w+)\b', 'a \1', r'\ban ([^aeiou\W]\w+)\b', "자음으로 시작하는 단어 앞에는 'an' 대신 'a'를 사용해야 합니다"),
        
        # 단수/복수 오류
        (r'\b(one|a|an|each|every|this) (\w+s)\b', '\1 \2', r'\b(one|a|an|each|every|this) (\w+s)\b', "'one', 'a', 'an', 'each', 'every', 'this' 뒤에는 단수형을 사용해야 합니다"),
        (r'\b(many|several|few|these|those|two|three|four|five) (\w+)(?<!s)\b', '\1 \2s', r'\b(many|several|few|these|those|two|three|four|five) (\w+)(?<!s)\b', "'many', 'several', 'few', 'these', 'those', 숫자 뒤에는 복수형을 사용해야 합니다"),
        
        # 소유격 오류
        (r'\b(\w+)s\'s\b', "\1s'", r'\b(\w+)s\'s\b', "복수형 소유격은 's가 아닌 '만 붙여야 합니다"),
        
        # 전치사 오류
        (r'\bin (\w+ )?(weekend|morning|evening|night|spring|summer|fall|autumn|winter)\b', 'on \1\2', r'\bin (\w+ )?(weekend|morning|evening|night|spring|summer|fall|autumn|winter)\b', "날짜나 시간대는 'in' 대신 'on'을 사용해야 합니다"),
        (r'\bon (January|February|March|April|May|June|July|August|September|October|November|December|next month|last month|this month)\b', 'in \1', r'\bon (January|February|March|April|May|June|July|August|September|October|November|December|next month|last month|this month)\b', "월 이름에는 'on' 대신 'in'을 사용해야 합니다"),
        (r'\bon (yesterday|today|tomorrow)\b', 'by \1', r'\bon (yesterday|today|tomorrow)\b', "'on' 대신 'by'를 사용해야 합니다"),
        
        # 시제 오류
        (r'\b(yesterday|last week|last month|last year) I (go|come|do|have|are|is|eat|drink|sleep|wake|see|drive)\b', '\1 I \2ed', r'\b(yesterday|last week|last month|last year) I (go|come|do|have|are|is|eat|drink|sleep|wake|see|drive)\b', "과거를 나타내는 표현 뒤에는 과거형 동사를 사용해야 합니다"),
        (r'\b(ago) I (go|come|do|have|are|is|eat|drink|sleep|wake|see|drive)\b', '\1 I \2ed', r'\b(ago) I (go|come|do|have|are|is|eat|drink|sleep|wake|see|drive)\b', "'ago' 뒤에는 과거형 동사를 사용해야 합니다"),
        
        # 누락된 주어
        (r'^\s*([A-Z]\w*\s+){0,3}(is|am|are|was|were|have|has|had|do|does|did|can|could|will|would|shall|should|may|might|must)\b', 'Subject \1\2', r'^\s*([A-Z]\w*\s+){0,3}(is|am|are|was|were|have|has|had|do|does|did|can|could|will|would|shall|should|may|might|must)\b', "문장에 주어가 필요합니다"),
        
        # 중복 단어
        (r'\b(\w+)\s+\1\b', '\1', r'\b(\w+)\s+\1\b', "중복된 단어가 있습니다"),
        
        # be 동사 누락
        (r'\bI|He|She|It|They|We|You (\w+ing)\b', 'I am|He is|She is|It is|They are|We are|You are \1', r'\bI|He|She|It|They|We|You (\w+ing)\b', "진행형에서 be 동사가 필요합니다"),
        
        # 주어-동사 일치 오류
        (r'\bhe|she|it (are|were|have)\b', 'he|she|it is|was|has', r'\bhe|she|it (are|were|have)\b', "3인칭 단수 주어에는 3인칭 단수 동사가 필요합니다"),
        (r'\bI|we|you|they (is|was|has)\b', 'I|we|you|they am|are|were|have', r'\bI|we|you|they (is|was|has)\b', "해당 주어에 맞는 동사가 필요합니다"),
        
        # 조동사 후 동사원형 누락
        (r'\b(can|could|will|would|shall|should|may|might|must) (am|is|are|was|were|have|has|had)\b', '\1 be|have', r'\b(can|could|will|would|shall|should|may|might|must) (am|is|are|was|were|have|has|had)\b', "조동사 뒤에는 동사 원형을 사용해야 합니다")
    ]

# 한국인이 자주 범하는 영어 오류를 체크하는 함수
def check_korean_english_errors(text):
    """
    한국인이 자주 범하는 영어 오류를 체크합니다.
    """
    errors = []
    patterns = get_korean_english_error_patterns()
    
    # 각 오류 패턴을 검사
    for pattern in patterns:
        if len(pattern) == 4:  # (검색 패턴, 교정 제안, 오류 패턴, 설명)
            search_pattern, correction, error_pattern, description = pattern
            matches = re.finditer(error_pattern, text, re.IGNORECASE)
            
            for match in matches:
                # 오류 정보 수집
                start = match.start()
                end = match.end()
                error_text = match.group(0)
                
                # 교정 제안 생성 (단순히 제안만 함)
                suggestion = re.sub(search_pattern, correction, error_text, flags=re.IGNORECASE)
                
                errors.append({
                    "offset": start,
                    "errorLength": len(error_text),
                    "message": f"한국인 학습자 일반 오류: {description}",
                    "replacements": [suggestion],
                    "source": "KoreanErrRule"
                })
    
    return errors

# GrammarBot API를 사용한 문법 검사 함수
def check_grammar_with_grammarbot(text):
    """
    GrammarBot API를 사용하여 문법을 검사합니다.
    """
    errors = []
    client = get_grammar_bot()
    
    if client and text.strip():
        try:
            result = client.check(text)
            
            for match in result.matches:
                # 오류 정보 추출
                offset = match.offset
                length = match.length
                message = match.message
                rule = match.rule
                
                # 수정 제안 추출
                replacements = []
                for suggestion in match.replacements:
                    replacements.append(suggestion)
                
                errors.append({
                    "offset": offset,
                    "errorLength": length,
                    "message": message,
                    "replacements": replacements,
                    "source": f"GrammarBot:{rule}"
                })
        except Exception as e:
            print(f"GrammarBot API 호출 중 오류: {e}")
    
    return errors

# Gramformer를 사용한 문법 교정 함수
def correct_grammar_with_gramformer(text):
    """
    Gramformer를 사용하여 문법을 교정합니다.
    """
    gf = get_gramformer()
    corrected_text = text
    
    if gf and text.strip():
        try:
            # 문장 단위로 교정
            sentences = custom_sent_tokenize(text)
            corrected_sentences = []
            
            for sentence in sentences:
                corrected = gf.correct(sentence, max_candidates=1)
                if corrected:
                    corrected_sentences.append(corrected[0])
                else:
                    corrected_sentences.append(sentence)
            
            # 교정된 문장들을 다시 합침
            corrected_text = ' '.join(corrected_sentences)
        except Exception as e:
            print(f"Gramformer 모델 사용 중 오류: {e}")
    
    return corrected_text

# GrammarCheck.io API를 사용한 문법 검사 함수
def check_grammar_with_grammarcheck_api(text):
    """
    GrammarCheck.io API를 사용하여 문법을 검사합니다.
    """
    errors = []
    
    if not text.strip():
        return errors
    
    try:
        # GrammarCheck.io API 엔드포인트
        api_url = "https://api.grammarcheck.io/v1/check"
        
        # API 요청 파라미터
        payload = {
            "text": text,
            "language": "en-US"
        }
        
        # API 요청 헤더 (필요할 경우 API 키 추가)
        headers = {
            "Content-Type": "application/json"
        }
        
        # API 요청 보내기
        response = requests.post(api_url, json=payload, headers=headers)
        
        # 응답 처리
        if response.status_code == 200:
            result = response.json()
            
            if "matches" in result:
                for match in result["matches"]:
                    # 오류 정보 추출
                    offset = match.get("offset", 0)
                    length = match.get("length", 0)
                    message = match.get("message", "문법 오류")
                    
                    # 수정 제안 추출
                    replacements = match.get("replacements", [])
                    
                    errors.append({
                        "offset": offset,
                        "errorLength": length,
                        "message": message,
                        "replacements": replacements,
                        "source": "GrammarCheck.io"
                    })
        else:
            print(f"GrammarCheck.io API 요청 실패: {response.status_code}")
    except Exception as e:
        print(f"GrammarCheck.io API 사용 중 오류: {e}")
    
    return errors

# 문법 검사 함수 개선
def check_grammar(text):
    """여러 엔진을 사용하여 문법을 체크합니다."""
    if not text.strip():
        return []
    
    all_errors = []
    
    # 한국어 특화 오류 패턴 체크
    korean_english_errors = check_korean_english_errors(text)
    all_errors.extend(korean_english_errors)
    
    # 패턴 기반 추가 검사 (자주 발생하는 오류)
    pattern_errors = check_additional_patterns(text)
    all_errors.extend(pattern_errors)
    
    # TextBlob 문법 체크 사용
    if has_textblob:
        try:
            textblob_errors = check_grammar_with_textblob(text)
            all_errors.extend(textblob_errors)
        except Exception as e:
            st.error(f"TextBlob 문법 검사 오류: {str(e)}")
    
    # LanguageTool 검사
    if has_languagetool:
        try:
            tool = get_language_tool()
            languagetool_errors = tool.check(text)
            
            for error in languagetool_errors:
                all_errors.append({
                    'message': error.message,
                    'offset': error.offset,
                    'length': error.errorLength,
                    'replacements': error.replacements,
                    'rule': error.ruleId,
                    'context': text[max(0, error.offset - 20):min(len(text), error.offset + error.errorLength + 20)]
                })
        except Exception as e:
            st.error(f"LanguageTool 오류: {str(e)}")
    
    # GrammarBot 검사
    if has_grammarbot and not all_errors:
        try:
            grammarbot_errors = check_grammar_with_grammarbot(text)
            all_errors.extend(grammarbot_errors)
        except Exception as e:
            st.error(f"GrammarBot 오류: {str(e)}")
    
    # 철자 검사 (SpellChecker)
    try:
        spell = get_spell_checker()
        words = custom_word_tokenize(text)
        misspelled = spell.unknown(words)
        
        for word in misspelled:
            # 커스텀 제안 확인
            custom_suggestions = get_custom_suggestions()
            if word.lower() in custom_suggestions:
                suggestions = custom_suggestions[word.lower()]
            else:
                suggestions = spell.candidates(word)
            
            # 단어 위치 찾기
            word_start = text.find(word)
            if word_start == -1:
                continue
                
            all_errors.append({
                'message': f"철자 오류: '{word}'",
                'offset': word_start,
                'length': len(word),
                'replacements': list(suggestions),
                'rule': 'SPELLING',
                'context': text[max(0, word_start - 20):min(len(text), word_start + len(word) + 20)]
            })
    except Exception as e:
        st.error(f"철자 검사 오류: {str(e)}")
    
    # Gramformer 검사 추가 (전체 문장 교정)
    if has_gramformer:
        try:
            corrected_text = correct_grammar_with_gramformer(text)
            if corrected_text and corrected_text != text:
                all_errors.append({
                    'message': "문법 교정 제안",
                    'offset': 0,
                    'length': len(text),
                    'replacements': [corrected_text],
                    'rule': 'GRAMFORMER_CORRECTION',
                    'context': text
                })
        except Exception as e:
            st.error(f"Gramformer 오류: {str(e)}")
    
    # 결과를 정렬: 오프셋 기준
    all_errors.sort(key=lambda x: x['offset'])
    
    # 중복 제거: 같은 위치에 있는 오류 중 가장 유용한 것만 유지
    filtered_errors = []
    error_positions = set()
    
    for error in all_errors:
        # 일관성을 위해 'length'와 'errorLength' 모두 확인
        length = error.get('length', error.get('errorLength', 0))
        position = (error.get('offset', 0), length)
        if position not in error_positions:
            # 'length'가 없으면 'errorLength'를 'length'로 복사
            if 'length' not in error and 'errorLength' in error:
                error['length'] = error['errorLength']
            error_positions.add(position)
            filtered_errors.append(error)
    
    return filtered_errors
    
# 추가 패턴 검사 함수 추가
def check_additional_patterns(text):
    """한국인 학습자가 자주 범하는 오류 패턴을 검사합니다."""
    errors = []
    
    # 문장들로 분리
    sentences = custom_sent_tokenize(text)
    current_pos = 0
    
    for sentence in sentences:
        # 전치사 누락 패턴: "impeachment the" -> "impeachment of the"
        pattern_impeachment = r'impeachment\s+the'
        matches = re.finditer(pattern_impeachment, sentence, re.IGNORECASE)
        for match in matches:
            start_in_sentence = match.start()
            length = match.end() - match.start()
            start_in_text = text.find(sentence, current_pos) + start_in_sentence
            
            if start_in_text >= 0:
                                errors.append({
                    'message': "전치사 누락: 'impeachment the' → 'impeachment of the'",
                    'offset': start_in_text,
                    'length': length,
                    'replacements': ['impeachment of the'],
                    'rule': 'MISSING_PREPOSITION',
                    'context': sentence
                })
        
        # 불완전 문장 패턴: "has serious" 다음에 명사가 없는 경우
        pattern_incomplete = r'has\s+serious\s*$'
        matches = re.finditer(pattern_incomplete, sentence, re.IGNORECASE)
        for match in matches:
            start_in_sentence = match.start()
            length = match.end() - match.start()
            start_in_text = text.find(sentence, current_pos) + start_in_sentence
            
            if start_in_text >= 0:
                                        errors.append({
                    'message': "불완전 문장: 명사가 필요합니다. 'has serious' → 'has serious consequences'",
                    'offset': start_in_text,
                    'length': length,
                    'replacements': ['has serious consequences', 'has serious implications', 'has serious effects'],
                    'rule': 'INCOMPLETE_SENTENCE',
                    'context': sentence
                })
        
        # 기타 전치사 누락 패턴들
        # "related to" 다음에 "the"가 오는 경우 체크
        pattern_related = r'related\s+the'
        matches = re.finditer(pattern_related, sentence, re.IGNORECASE)
        for match in matches:
            start_in_sentence = match.start()
            length = match.end() - match.start()
            start_in_text = text.find(sentence, current_pos) + start_in_sentence
            
            if start_in_text >= 0:
                                            errors.append({
                    'message': "전치사 누락: 'related the' → 'related to the'",
                    'offset': start_in_text,
                    'length': length,
                    'replacements': ['related to the'],
                    'rule': 'MISSING_PREPOSITION',
                    'context': sentence
                })
                
        # 대응하는 to-be 동사가 없는 경우
        pattern_missing_verb = r'(the\s+\w+(?:\s+\w+){0,3})\s+(?:very|so|quite|extremely)\s+(\w+)(?:\s+(?:and|but|or)\s+(?:very|so|quite|extremely)\s+(\w+))?(?:\s*[,.]|\s+(?:that|which|who)|\s*$)'
        matches = re.finditer(pattern_missing_verb, sentence, re.IGNORECASE)
        for match in matches:
            start_in_sentence = match.start()
            length = match.end() - match.start()
            start_in_text = text.find(sentence, current_pos) + start_in_sentence
            
            subject = match.group(1)
            adjective = match.group(2)
            
            if start_in_text >= 0:
                errors.append({
                    'message': f"동사 누락: '{subject} {adjective}' → '{subject} is {adjective}'",
                    'offset': start_in_text,
                    'length': length,
                    'replacements': [f"{subject} is {adjective}"],
                    'rule': 'MISSING_VERB',
                    'context': sentence
                })
        
        # 현재 위치 업데이트
        current_pos += len(sentence)
    
    return errors

# 문법 오류 시각화 및 표시를 위한 함수
def display_grammar_errors(text, errors):
    """문법 오류를 시각화하여 표시합니다."""
    if not errors:
        return text, []
    
    # 오류가 있는 경우 강조 표시를 위한 HTML 생성
    html_parts = []
    last_end = 0
    error_details = []
    
    # 오프셋 기준으로 오류 정렬
    errors = sorted(errors, key=lambda x: x['offset'])
    
    for i, error in enumerate(errors):
        # 키 오류 방지를 위한 기본값 설정
        offset = error.get('offset', 0)
        length = error.get('length', 0)  # 'errorLength'에서 'length'로 변경
        message = error.get('message', '알 수 없는 오류')
        replacements = error.get('replacements', [])
        
        # 오류 앞의 텍스트 추가
        html_parts.append(text[last_end:offset])
        
        # 오류 텍스트 추출
        error_text = text[offset:offset+length]
        
        # 오류 번호 생성
        error_num = i + 1
        
        # 오류 텍스트를 강조 표시 (툴팁 포함)
        html_parts.append(f'<span class="grammar-error" title="오류 {error_num}: {message}" '
                          f'style="background-color: #ffcccc; text-decoration: underline wavy red;">'
                          f'{error_text}</span>')
        
        # 다음 시작 위치 설정
        last_end = offset + length
        
        # 오류 세부 정보 저장 (사이드바 표시용)
        error_details.append({
            "id": error_num,
            "text": error_text,
            "message": message,
            "replacements": replacements
        })
    
    # 마지막 오류 이후의 텍스트 추가
    html_parts.append(text[last_end:])
    
    # 최종 HTML 생성
    highlighted_text = ''.join(html_parts)
    
    return highlighted_text, error_details

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
    """
    입력 텍스트를 약간 향상된 수준으로 재작성합니다.
    """
    if not text:
        return ""
    
    # 문장으로 분할
    sentences = custom_sent_tokenize(text)
    improved_sentences = []
    
    # 향상된 동의어
    improved_synonyms = {
        # 일반 형용사
        "good": ["excellent", "great", "wonderful", "remarkable", "favorable"],
        "bad": ["poor", "unfavorable", "negative", "substandard", "inadequate"],
        "big": ["large", "substantial", "considerable", "significant", "extensive"],
        "small": ["minor", "modest", "limited", "slight", "minimal"],
        
        # 일반 동사
        "said": ["stated", "mentioned", "noted", "expressed", "communicated"],
        "make": ["create", "produce", "generate", "develop", "establish"],
        "get": ["obtain", "acquire", "gain", "attain", "procure"],
        "use": ["utilize", "employ", "apply", "implement", "leverage"],
        
        # 추가 형용사
        "important": ["significant", "essential", "crucial", "vital", "fundamental"],
        "interesting": ["engaging", "captivating", "intriguing", "fascinating", "compelling"],
        "difficult": ["challenging", "demanding", "arduous", "strenuous", "complex"],
        "easy": ["straightforward", "uncomplicated", "effortless", "simple", "manageable"]
    }
    
    for sentence in sentences:
        # 문장 내 단어 처리
        words = custom_word_tokenize(sentence)
        for i, word in enumerate(words):
            word_lower = word.lower()
            
            # 동의어 교체 (확률적으로 교체)
            if word_lower in improved_synonyms and random.random() < 0.4:
                replacement = random.choice(improved_synonyms[word_lower])
                
                # 원래 단어가 대문자로 시작하면 교체 단어도 대문자로 시작
                if word[0].isupper() and isinstance(replacement, str):
                    replacement = replacement.capitalize()
            
                words[i] = replacement
        
        # 문장 재구성
        improved_sentence = ' '.join(words)
        improved_sentences.append(improved_sentence)
    
    # 재작성된 텍스트 반환
    return ' '.join(improved_sentences)

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
                                    
                                    # 동기식 래퍼 함수를 사용하여 음성 파일 생성
                                    audio_path = sync_text_to_speech(user_text, voice_model, audio_file_path)
                                    
                                    # 세션 상태에 오디오 파일 경로 저장
                                    st.session_state[audio_key] = audio_path
                                    st.session_state[f"{audio_key}_playing"] = True
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"음성 생성 중 오류가 발생했습니다: {str(e)}")
                        else:
                            st.warning("텍스트를 먼저 입력해주세요.")
                else:
                    # 토글 버튼 로직
                    button_label = "⏹️ 음성 정지" if st.session_state[f"{audio_key}_playing"] else "▶️ 음성 재생"
                    if st.button(button_label, key=f"toggle_audio_tab1", use_container_width=True):
                        # 토글 상태 변경
                        st.session_state[f"{audio_key}_playing"] = not st.session_state[f"{audio_key}_playing"]
                        st.rerun()
                    
                    # 오디오 플레이어 표시 (현재 페이지 위치에 표시)
                    if st.session_state[f"{audio_key}_playing"]:
                        audio_html = get_audio_player_html(st.session_state[audio_key], loop_count=5)
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
                original_text = st.session_state.analysis_results['original_text']
                
                # 새로운 함수를 사용하여 문법 오류 표시
                highlighted_text, error_details = display_grammar_errors(original_text, grammar_errors)
                
                # 오류 표시 섹션
                st.markdown("## 문법 오류 분석")
                # 강조 표시된 텍스트 출력
                st.markdown(highlighted_text, unsafe_allow_html=True)
                
                # 오류 세부 사항 표시
                if error_details:
                    st.markdown("### 오류 세부 사항")
                    for error in error_details:
                        with st.expander(f"오류 {error['id']}: {error['text']}"):
                            st.write(f"**메시지:** {error['message']}")
                            if error['replacements']:
                                # 문자열 변환 과정 추가
                                suggestions = []
                                for r in error['replacements'][:3]:  # 최대 3개만 표시
                                    if isinstance(r, str):
                                        suggestions.append(r)
                                st.write(f"**수정 제안:** {', '.join(suggestions)}")
                
                # 오류 통계
                st.write(f"총 {len(grammar_errors)}개의 문법/맞춤법 오류가 발견되었습니다.")
                
                # 음성 다운로드 버튼 표시
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
                fig = None  # fig 변수를 명시적으로 초기화
                if vocab_analysis and 'word_freq' in vocab_analysis and vocab_analysis['word_freq']:
                    fig = plot_word_frequency(vocab_analysis['word_freq'])
                
                # fig 변수 확인 후 시각화
                if fig is not None:
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
                                    try:
                                        # 선택된 음성 모델 가져오기
                                        voice_model = voice_options[selected_voice]
                                        
                                        # 임시 파일 경로 생성
                                        temp_dir = tempfile.gettempdir()
                                        audio_file_path = os.path.join(temp_dir, f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
                                        
                                        # 동기식 래퍼 함수를 사용하여 음성 파일 생성
                                        audio_path = sync_text_to_speech(rewritten, voice_model, audio_file_path)
                                        
                                        # 세션 상태에 오디오 파일 경로 저장
                                        st.session_state.audio_path = audio_path
                                        st.success("음성 파일이 생성되었습니다!")
                                        st.rerun()  # 재실행하여 오디오 플레이어 표시
                                    except Exception as e:
                                        st.error(f"음성 생성 중 오류가 발생했습니다: {str(e)}")
            
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
                                st.rerun()
                            
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
                original_text = st.session_state.teacher_analysis_results['original_text']
                
                # 새로운 함수를 사용하여 문법 오류 표시
                highlighted_text, error_details = display_grammar_errors(original_text, grammar_errors)
                
                # 오류 표시 섹션
                st.markdown("## 문법 오류 분석")
                # 강조 표시된 텍스트 출력
                st.markdown(highlighted_text, unsafe_allow_html=True)
                
                # 오류 세부 사항 표시
                if error_details:
                    st.markdown("### 오류 세부 사항")
                    for error in error_details:
                        with st.expander(f"오류 {error['id']}: {error['text']}"):
                            st.write(f"**메시지:** {error['message']}")
                            if error['replacements']:
                                # 문자열 변환 과정 추가
                                suggestions = []
                                for r in error['replacements'][:3]:  # 최대 3개만 표시
                                    if isinstance(r, str):
                                        suggestions.append(r)
                                st.write(f"**수정 제안:** {', '.join(suggestions)}")
                
                # 오류 통계
                st.write(f"총 {len(grammar_errors)}개의 문법/맞춤법 오류가 발견되었습니다.")
                
                # 음성 다운로드 버튼 표시
                audio_key = f"audio_tab1_{hash(st.session_state.teacher_analysis_results['original_text'])}" if 'original_text' in st.session_state.teacher_analysis_results else None
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
            if 'teacher_analysis_results' in st.session_state and 'vocab_analysis' in st.session_state.teacher_analysis_results:
                vocab_analysis = st.session_state.teacher_analysis_results['vocab_analysis']
                diversity_score = st.session_state.teacher_analysis_results['diversity_score']
                vocab_level = st.session_state.teacher_analysis_results['vocab_level']
                        
                # 단어 빈도 시각화 - 에러 방지를 위한 예외 처리 추가
                fig = None  # fig 변수를 명시적으로 초기화
                if vocab_analysis and 'word_freq' in vocab_analysis and vocab_analysis['word_freq']:
                    fig = plot_word_frequency(vocab_analysis['word_freq'])
                
                # fig 변수 확인 후 시각화
                if fig is not None:
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
                        "오류": user_text[error['offset']:error['offset'] + error.get('length', error.get('errorLength', 0))],
                        "오류 내용": error['message'],
                        "수정 제안": str(error['replacements']),
                        "위치": f"{error['offset']}:{error['offset'] + error.get('length', error.get('errorLength', 0))}"
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

# TextBlob 문법 체크 기능 추가
try:
    from textblob import TextBlob
    has_textblob = True
except ImportError:
    has_textblob = False
    print("textblob이 설치되어 있지 않습니다. 기본 문법 검사 기능을 사용합니다.")

# TextBlob을 사용한 문법 체크 함수 추가
def check_grammar_with_textblob(text):
    """TextBlob 라이브러리를 사용하여 문법을 체크합니다."""
    if not has_textblob:
        return []
    
    errors = []
    blob = TextBlob(text)
    
    sentences = custom_sent_tokenize(text)
    for i, sentence in enumerate(sentences):
        # 철자 검사
        sentence_blob = TextBlob(sentence)
        misspelled = sentence_blob.correct()
        
        if str(misspelled) != sentence:
            # 문장 내 단어별로 분석
            words = custom_word_tokenize(sentence)
            corrected_words = custom_word_tokenize(str(misspelled))
            
            # 길이가 다를 수 있으므로 최소 길이만큼 비교
            min_len = min(len(words), len(corrected_words))
            for j in range(min_len):
                if words[j] != corrected_words[j]:
                    # 시작 위치 계산
                    start_pos = text.find(sentence)
                    if start_pos == -1:
                        continue
                    
                    # 단어 시작 위치 계산 (간단한 추정)
                    word_start = start_pos + sentence.find(words[j])
                    if word_start == -1:
                        continue
                    
                    errors.append({
                        'message': f"철자 오류: '{words[j]}' → '{corrected_words[j]}'",
                        'offset': word_start,
                        'length': len(words[j]),
                        'replacements': [corrected_words[j]],
                        'rule': 'TEXTBLOB_SPELLING',
                        'context': sentence
                    })
    
    return errors
