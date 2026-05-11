# gemini_api.py
from __future__ import annotations

import os
import sys
import json
import base64
import queue
import threading
import random
import time
import re 
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Callable
import multiprocessing

from core.profile_manager import ProfileManager
from core.report_manager import ReportManager
from core.utils import (
    _get_relative_time_str, _extract_text, _get_env, 
    STAGES, build_opening_prompt, build_extract_prompt, 
    build_retry_prompt, build_next_prompt, 
    build_main_system_instruction, build_hidden_first_prompt
)
from media.audio_manager import AudioManager
from media.tts_manager import TTSManager 

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from vision.vision_brain import RobotBrain

try:
    from dotenv import load_dotenv
    if os.path.exists(".env.local"):
        load_dotenv(dotenv_path=".env.local")
    else:
        load_dotenv()
except Exception:
    pass

from pynput import keyboard
import google.generativeai as genai

PROFILE_DB_FILE = "user_profiles.json"
MODEL_NAME = _get_env("MODEL_NAME", "gemini-3.1-flash-lite")
GREETING_TEXT = _get_env("GREETING_TEXT", "안녕하세요! 모티입니다.")
FAREWELL_TEXT = _get_env("FAREWELL_TEXT", "도움이 되었길 바라요. 언제든 다시 불러주세요.")
ENABLE_GREETING = _get_env("ENABLE_GREETING", "1") not in ("0", "false", "False")


# --- 메인 PressToTalk 클래스 (컨트롤러) ---
class PressToTalk:
    def __init__(self,
                emotion_queue: Optional[queue.Queue] = None,
                subtitle_queue: Optional[multiprocessing.Queue] = None, 
                stop_event: Optional[threading.Event] = None,
                shared_state: Optional[dict] = None,
                mouth_event_queue: Optional[queue.Queue] = None,
                brain_instance = None,
                perform_head_nod_cb: Optional[Callable[[int], None]] = None,
                ): 
        
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key or not api_key.strip():
            print("❗ GOOGLE_API_KEY가 없습니다."); sys.exit(1)

        genai.configure(api_key=api_key)
        self.MODEL_NAME = MODEL_NAME
        self.model = genai.GenerativeModel(MODEL_NAME)
        self.chat = self.model.start_chat(history=[])
        
        self.interview_stage_index = -1  
        self.temp_user_info = {}
        
        self.is_setting_up_main_chat = False 
        self.session_active = False 

        self.main_chat_cooldown_until = 0

        self.current_user_name = None
        self.profile_db_file = PROFILE_DB_FILE
        self.initial_chat_summary = "아직 기록된 내용이 없습니다."
        self.initial_last_seen_str = "기록 없음"
        self.session_history = []

        self.emotion_queue = emotion_queue
        self.subtitle_queue = subtitle_queue
        self.stop_event = stop_event or threading.Event()
        
        self.brain = brain_instance
        self.last_logged_in_user = None

        self.mouth_event_queue = mouth_event_queue
        self.listening_enabled = threading.Event() 
        self.mouth_listener_thread = None 
        
        self.last_activity_time = 0
        self.current_listener = None

        self.busy_lock = threading.Lock()
        self.busy_signals = 0

        self.perform_head_nod_cb = perform_head_nod_cb
        self.nodding_thread = None
        self.stop_nodding_event = threading.Event()

        self.tts = TTSManager()
        self.tts.subtitle_queue = subtitle_queue
        self.tts.start()

        self.audio = AudioManager() 
        self._print_intro()

        self.profile_manager = ProfileManager(self)
        self.profile_manager.init_db()
        
        self.report_manager = ReportManager(self.MODEL_NAME)
        
        if ENABLE_GREETING:
            self._speak_and_subtitle(GREETING_TEXT)
            if self.emotion_queue: self.emotion_queue.put("NEUTRAL")

        self.shared_state = shared_state

    def _process_and_speak_gemini_response(self, text: str):
        emotion = "NEUTRAL"
        emotion_match = re.search(r'\[EMOTION\](.*?)\[/EMOTION\]', text, re.DOTALL)
        if emotion_match:
            emotion = emotion_match.group(1).strip().upper()
        
        if self.emotion_queue:
            if emotion in ["HAPPY", "SAD", "ANGRY", "SURPRISED", "TENDER", "NEUTRAL"]:
                self.emotion_queue.put(emotion)
            else:
                self.emotion_queue.put("NEUTRAL")

        clean_text = re.sub(r'\[EMOTION\].*?\[/EMOTION\]', '', text, flags=re.DOTALL).strip()
        clean_text = clean_text.replace('*', '')
        
        if clean_text:
            self._speak_and_subtitle(clean_text)
            self.tts.wait() 
        else:
            print("⚠️ [경고] 모티가 뱉을 텍스트가 비어있습니다!")
            
        return clean_text

    def _process_and_speak_stream(self, response_stream, user_text=""):
        buffer = ""
        terminators = ['.', '!', '?', '\n']
        header_parsed = False
        speak_text_full = ""
        ts = datetime.now().strftime("%H:%M:%S")

        for chunk in response_stream:
            if not chunk.text: continue
            buffer += chunk.text

            if not header_parsed:
                if "[/EMOTION]" in buffer:
                    emotion_match = re.search(r'\[EMOTION\](.*?)\[/EMOTION\]', buffer, re.DOTALL)
                    if emotion_match:
                        extracted_emotion = emotion_match.group(1).strip().upper()
                        print(f"[{ts}] [Emotion] {extracted_emotion}")
                        if self.emotion_queue:
                            valid_emotions = ["HAPPY", "SAD", "ANGRY", "SURPRISED", "TENDER", "NEUTRAL"]
                            if extracted_emotion in valid_emotions:
                                self.emotion_queue.put(extracted_emotion)
                            else:
                                self.emotion_queue.put("NEUTRAL")
                    
                    buffer = buffer.split("[/EMOTION]")[-1].lstrip()
                    header_parsed = True
                elif len(buffer) > 100: 
                    header_parsed = True
            
            if header_parsed:
                while any(t in buffer for t in terminators):
                    first_term_idx = min([buffer.find(t) for t in terminators if t in buffer])
                    sentence = buffer[:first_term_idx+1].strip()
                    buffer = buffer[first_term_idx+1:]
                    sentence = sentence.replace('*', '').strip()
                    
                    if sentence:
                        print(f"[{ts}] 🗣️ 말하기: {sentence}")
                        self.tts.speak(sentence)
                        if self.subtitle_queue: self.subtitle_queue.put(sentence)
                        speak_text_full += sentence + " "

        if buffer.strip():
            sentence = buffer.replace('*', '').strip()
            if sentence:
                if not header_parsed and "[EMOTION]" in sentence:
                    sentence = re.sub(r'\[EMOTION\].*?\[/EMOTION\]', '', sentence).strip()
                
                if sentence:
                    print(f"[{ts}] 🗣️ 마지막 말하기: {sentence}")
                    self.tts.speak(sentence)
                    if self.subtitle_queue: self.subtitle_queue.put(sentence)
                    speak_text_full += sentence + " "

        speak_text_full = speak_text_full.strip()

        if user_text and speak_text_full:
            new_history = list(self.chat.history)
            new_history.append({'role': 'user', 'parts': [user_text]})
            new_history.append({'role': 'model', 'parts': [speak_text_full]})
            self.chat = self.chat.model.start_chat(history=new_history)
            
            log_entry = f"User: {user_text} | Moti: {speak_text_full}"
            self.session_history.append(log_entry)
            print(f"📝 대화 메모리 기록 (현재 {len(self.session_history)}턴 쌓임)")

    def raise_busy_signal(self):
        with self.busy_lock:
            self.busy_signals += 1

    def lower_busy_signal(self):
        with self.busy_lock:
            self.busy_signals = max(0, self.busy_signals - 1)
            if self.busy_signals == 0:
                self.last_activity_time = time.time()

    def _listening_nod_worker(self):
        start_wait = random.uniform(0.5, 1.5)
        interrupted = self.stop_nodding_event.wait(timeout=start_wait)
        if interrupted: return

        while not self.stop_nodding_event.is_set():
            reps = 2 if random.random() < 0.3 else 1
            if callable(self.perform_head_nod_cb):
                try: threading.Thread(target=self.perform_head_nod_cb, args=(reps,), daemon=True).start()
                except Exception: pass
            
            wait_time = random.uniform(1.5, 4.0)
            if self.stop_nodding_event.wait(timeout=wait_time): break

    def _mouth_listener_worker(self):
        while not self.stop_event.is_set():
            try:
                msg = self.mouth_event_queue.get(timeout=0.2) 
                if msg == "START_RECORDING":
                    if self.listening_enabled.is_set() and self.busy_signals == 0:
                        self._start_recording()
                elif msg == "STOP_RECORDING":
                    self._stop_recording_and_transcribe()
            except queue.Empty: continue 
            except Exception as e: print(f"❌ Mouth listener error: {e}")

    def _speak_and_subtitle(self, text_data: str | dict):
        if not text_data: return
        text_to_display = text_data.get("text", "") if isinstance(text_data, dict) else str(text_data).strip()
        if not text_to_display: return

        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] 🗣️ 모티: {text_to_display}")
        if self.subtitle_queue: self.subtitle_queue.put(text_to_display)
        self.tts.speak(text_data)

    def _print_intro(self):
        print("\n=== Gemini PTT (한동대 특화 인터뷰 + 공감 버전) ===")
        print("▶ 입 열기로 대화 시작 → ESC로 종료")
        print(f"▶ MODEL={MODEL_NAME}")
        print("----------------------------------------------------------------\n")

    def _start_recording(self):
        if self.audio.recording: return
        if self.emotion_queue: self.emotion_queue.put("LISTENING") 

        self.last_activity_time = time.time()
        print("🎙️ 녹음 시작...")
        self.audio.start_recording()

        if callable(self.perform_head_nod_cb) and (self.nodding_thread is None or not self.nodding_thread.is_alive()):
            self.stop_nodding_event.clear()
            self.nodding_thread = threading.Thread(target=self._listening_nod_worker, daemon=True)
            self.nodding_thread.start()

    def _stop_recording_and_transcribe(self):
        if not self.audio.recording: return
        if self.emotion_queue: self.emotion_queue.put("THINKING") 
        self.last_activity_time = time.time()
        print("⏹️ 녹음 종료, 전사 중...")
        
        self.stop_nodding_event.set()
        wav_bytes = self.audio.stop_recording()
        
        if not wav_bytes:
            if self.emotion_queue: self.emotion_queue.put("NEUTRAL") 
            return
            
        threading.Thread(target=self._transcribe_then_chat, args=(wav_bytes,), daemon=True).start()

    def _transcribe_then_chat(self, wav_bytes: bytes):
        self.raise_busy_signal()
        ts = datetime.now().strftime("%H:%M:%S")
        
        emotion_hint = "\n\n🚨[출력 필수 조건]: 답변의 맨 앞에 반드시 [EMOTION]감정[/EMOTION] 태그를 하나 작성하고, 이어서 생성된 대답 텍스트를 명확하게 작성하세요. (예: [EMOTION]HAPPY[/EMOTION] 반가워요! 몇 학년이신가요?)"

        try:
            b64 = base64.b64encode(wav_bytes).decode("ascii")

            transcribe_resp = self.model.generate_content([
                "첨부된 오디오에서 사용자가 한 말만 정확하게 텍스트로 받아적으세요. 대답이나 부연설명 절대 금지.",
                {"inline_data": {"mime_type": "audio/wav", "data": b64}}
            ])
            user_text = _extract_text(transcribe_resp).strip()
            print(f"[{ts}] 🗣️ 사용자 발화: {user_text}")

            if not user_text:
                if self.emotion_queue: self.emotion_queue.put("NEUTRAL")
                self.lower_busy_signal()
                return

            if self.interview_stage_index >= 0:
                current_stage_dict = STAGES[self.interview_stage_index]
                current_stage = current_stage_dict["key"]
                stage_desc = current_stage_dict["desc"]

                current_name = self.temp_user_info.get("이름", "사용자")
                if current_name and len(current_name) == 3 and current_name != "사용자":
                    current_name = current_name[1:]

                ext_prompt = build_extract_prompt(user_text, current_stage, stage_desc)
                ext_resp = self.model.generate_content(ext_prompt)
                extracted_val = _extract_text(ext_resp).strip()

                if "FAIL" in extracted_val.upper() or not extracted_val:
                    print(f"⚠️ [{current_stage}] 추출 실패: '{user_text}' -> 재질문 생성 중...")
                    retry_prompt = build_retry_prompt(current_name, current_stage, user_text) + emotion_hint
                    
                    retry_resp = self.model.generate_content(retry_prompt)
                    raw_text = _extract_text(retry_resp)
                    print(f"👉 [DEBUG] Retry 응답: {raw_text}")
                    self._process_and_speak_gemini_response(raw_text)
                else:
                    print(f"✅ [{current_stage}] 저장 완료: {extracted_val}")
                    self.temp_user_info[current_stage] = extracted_val
                    
                    if current_stage == "이름":
                        self.shared_state['current_user_name'] = self.temp_user_info["이름"]
                        self.last_logged_in_user = self.temp_user_info["이름"]

                    if self.interview_stage_index < len(STAGES) - 1:
                        next_stage = STAGES[self.interview_stage_index + 1]["key"]
                        print(f"👉 다음 단계 준비: {next_stage}")
                        next_prompt = build_next_prompt(
                            current_stage, next_stage, extracted_val,
                            self.temp_user_info.get("학년", ""),
                            current_name,
                            self.temp_user_info
                        ) + emotion_hint
                        
                        resp = self.model.generate_content(next_prompt)
                        raw_text = _extract_text(resp)
                        print(f"👉 [DEBUG] Next 응답: {raw_text}")
                        self._process_and_speak_gemini_response(raw_text)
                        
                        self.interview_stage_index += 1
                        print(f"✅ {current_stage} 처리 및 응답 완료! 다음 질문({next_stage}) 대기 상태 돌입.")
                    else:
                        print("🎉 인터뷰 완료! 본 대화 세팅 시작...")
                        self.is_setting_up_main_chat = True
                        self.listening_enabled.clear() 
                        self.interview_stage_index = -1
                        
                        final_name = self.temp_user_info.get("이름", "사용자")
                        self.shared_state['current_user_name'] = final_name
                        self.last_logged_in_user = final_name
                        
                        self._speak_and_subtitle(f"정보를 모두 기억했어요! {final_name}님을 알아볼 수 있게 10초 동안 카메라를 봐주세요.")
                        self.tts.wait()
                        
                        if self.emotion_queue: self.emotion_queue.put("SCANNING")
                        
                        self.shared_state['force_learning'] = True
                        self.shared_state['learning_target_name'] = final_name
                        time.sleep(10)
                        self.shared_state['force_learning'] = False
                        
                        # ====================================================================
                        # 🚨 [음성 인식 기반 - 데이터 자가 교정 무한 루프 추가]
                        # ====================================================================
                        while True:
                            check_name = self.temp_user_info.get("이름", "알 수 없음")
                            check_major = self.temp_user_info.get("전공", "알 수 없음")
                            check_mbti = self.temp_user_info.get("MBTI", "알 수 없음")
                            
                            if self.emotion_queue: self.emotion_queue.put("HAPPY")
                            summary_msg = f"얼굴 인식이 끝났어요! 제가 기억한 정보는 이름 {check_name}, {check_major} 전공, MBTI는 {check_mbti} 인데, 혹시 제가 맞게 기억했나요?"
                            self._speak_and_subtitle(summary_msg)
                            self.tts.wait()

                            self.listening_enabled.clear()
                            is_info_correct = self._quick_listen_for_yes_no(timeout=4.0)

                            if is_info_correct:
                                if self.emotion_queue: self.emotion_queue.put("HAPPY")
                                self._speak_and_subtitle("다행이네요! 이 정보대로 완벽하게 기억해둘게요.")
                                self.tts.wait()
                                break  # 정보가 맞다면 루프 탈출
                            else:
                                if self.emotion_queue: self.emotion_queue.put("SURPRISED")
                                self._speak_and_subtitle("아앗, 제가 잘못 알아들었나 봐요! 이름, 전공, MBTI 중 어느 부분이 틀리셨는지 알려주세요.")
                                self.tts.wait()
                                
                                field_to_fix = self._listen_and_get_correction_field(timeout=4.0)
                                
                                if field_to_fix in ["이름", "전공", "MBTI"]:
                                    if self.emotion_queue: self.emotion_queue.put("NEUTRAL")
                                    self._speak_and_subtitle(f"그렇군요! 올바른 {field_to_fix} 정보를 다시 알려주세요.")
                                    self.tts.wait()
                                    
                                    new_val = self._listen_and_get_new_value(field_to_fix, timeout=5.0)
                                    if new_val:
                                        self.temp_user_info[field_to_fix] = new_val
                                        if self.emotion_queue: self.emotion_queue.put("HAPPY")
                                        self._speak_and_subtitle(f"네, {new_val} 맞으시죠? 새롭게 수정했어요. 다시 한 번 확인해볼게요.")
                                        self.tts.wait()
                                    else:
                                        if self.emotion_queue: self.emotion_queue.put("SAD")
                                        self._speak_and_subtitle("제가 잘 듣지 못했어요. 처음부터 다시 확인해볼게요.")
                                        self.tts.wait()
                                else:
                                    if self.emotion_queue: self.emotion_queue.put("SAD")
                                    self._speak_and_subtitle("어떤 항목인지 잘 듣지 못했어요. 다시 한 번 확인해볼게요.")
                                    self.tts.wait()

                        # 루프를 무사히 통과했다면, 최종 이름 정보를 동기화합니다.
                        final_name = self.temp_user_info.get("이름", "사용자")
                        self.shared_state['current_user_name'] = final_name
                        self.last_logged_in_user = final_name
                        # ====================================================================

                        self.profile_manager.save_user_info(final_name, self.temp_user_info)
                        self.profile_manager.load_profile_for_chat(final_name)
                        
                        mbti = self.temp_user_info.get("MBTI", "")
                        saesae = self.temp_user_info.get("새새 인원", "")
                        major = self.temp_user_info.get("전공", "")
                        
                        hidden_prompt = build_hidden_first_prompt(final_name, mbti, saesae, major) + emotion_hint
                        resp = self.chat.send_message(hidden_prompt)
                        raw_text = _extract_text(resp)
                        print(f"👉 [DEBUG] 메인 대화 진입 응답: {raw_text}")
                        self._process_and_speak_gemini_response(raw_text)
                        
                        while not self.mouth_event_queue.empty():
                            try: self.mouth_event_queue.get_nowait()
                            except: pass
                            
                        self.listening_enabled.set()
                        
                        self.session_active = True
                        self.main_chat_cooldown_until = time.time() + 5.0
                        self.is_setting_up_main_chat = False
                        print("✅ 메인 대화 세팅 완벽 종료! [1:1 대화 고정 모드 활성화]")

            else:
                current_time_str = datetime.now().strftime("%Y년 %m월 %d일 %p %I시 %M분")
                prompt = (
                    f"현재 시간: {current_time_str}\n"
                    f"[USER]{user_text}[/USER]\n"
                    "위 사용자의 말에 대해 시스템 프롬프트(한동대 모티 가이드라인)를 엄격히 준수하여 대답하세요.\n"
                    "반드시 [EMOTION]감정[/EMOTION] 태그를 맨 앞에 출력하세요."
                )

                contents = list(self.chat.history)
                contents.append({"role": "user", "parts": [prompt]})
                
                print(f"[{ts}] [Gemini] 스트리밍 호출 시작...")
                response_stream = self.chat.model.generate_content(contents, stream=True)
                self._process_and_speak_stream(response_stream, user_text)

        except Exception as e:
            print(f"❌ 처리 실패: {e}\n")
            if self.emotion_queue: self.emotion_queue.put("NEUTRAL")
            
        finally:
            print("... TTS 대기 완료(finally) ...")
            self.tts.wait()
            if self.emotion_queue: self.emotion_queue.put("NEUTRAL")
            self.lower_busy_signal()

    def _flush_session_history(self, is_shutdown=False):
        if not self.session_history: return
        print("💾 대화 세션 전환. 기억을 정리하여 저장합니다...")
        full_conversation_log = "\n".join(self.session_history)
        
        current_user = self.last_logged_in_user or self.shared_state.get('current_user_name', 'Unknown')
        
        vitals_data = None
        if is_shutdown and current_user and current_user != "Unknown":
            print("\n" + "="*60)
            print(f"📊 [{current_user}]님의 상담 결과지 생성을 위한 추가 정보 입력")
            print("============================================================")
            try:
                avg_hr = input("👉 사용자의 평균 심박수 (예: 81.5): ")
                max_hr = input("👉 사용자의 최대 심박수 (예: 107.7): ")
                stress = input("👉 stress_level (예: 측정불가, 높음 등): ")
                mood = input("👉 mood_summary (예: 불안함, 데이터 부족 등): ")
                
                vitals_data = {
                    "avg_hr": avg_hr.strip(),
                    "max_hr": max_hr.strip(),
                    "stress": stress.strip(),
                    "mood": mood.strip()
                }
            except Exception as e:
                print(f"입력 중 오류 발생: {e}")
                vitals_data = None
            print("============================================================\n")

        threads = []
        
        if current_user and current_user != "Unknown":
            t1 = threading.Thread(target=self.report_manager.generate_and_save_reports, args=(current_user, full_conversation_log, self.temp_user_info, vitals_data))
            threads.append(t1)
            t1.start()

        if hasattr(self.profile_manager, "batch_update_summary"):
            t2 = threading.Thread(target=self.profile_manager.batch_update_summary, args=(full_conversation_log,))
            threads.append(t2)
            t2.start()

        self.session_history = []
        self.session_active = False 
        
        if is_shutdown:
            print("⏳ 백그라운드 데이터 저장(결과지 작성 및 기억 요약)을 기다립니다. 잠시만 대기해주세요...")
            for t in threads:
                t.join()
            print("✅ 모든 데이터 저장이 안전하게 완료되었습니다.")
    
    def _quick_listen_for_yes_no(self, timeout=3.0) -> bool:
        print(f"👂 [Yes/No] {timeout}초간 답변 듣기 시작...")
        if self.emotion_queue: self.emotion_queue.put("LISTENING")
        wav_bytes = self.audio.record_fixed_duration(timeout)
        if not wav_bytes: return False
            
        if self.emotion_queue: self.emotion_queue.put("THINKING")
        try:
            b64 = base64.b64encode(wav_bytes).decode("ascii")
            prompt = (
                "사용자의 오디오를 듣고 '긍정(Yes)'인지 '부정(No)'인지 판단하세요. "
                "사용자가 '네', '응', '맞아', '좋아', '그래', '어'라고 하면 긍정입니다. "
                "사용자가 '아니', '틀렸어', '아니요', '됐어', '싫어'라고 하거나 아무 말도 없으면 부정입니다. "
                "반드시 JSON으로만 출력하세요: {\"answer\": \"yes\"} 또는 {\"answer\": \"no\"}"
            )
            resp = self.model.generate_content([prompt, {"inline_data": {"mime_type": "audio/wav", "data": b64}}])
            txt = _extract_text(resp).lower()
            return '"yes"' in txt or "'yes'" in txt
        except Exception:
            return False 

    # 🚨 [신규 함수 1] 수정하고 싶은 "항목"을 추출하는 함수
    def _listen_and_get_correction_field(self, timeout=4.0) -> str:
        print("👂 [수정 항목 파악] 듣기 시작...")
        if self.emotion_queue: self.emotion_queue.put("LISTENING")
        wav_bytes = self.audio.record_fixed_duration(timeout)
        if not wav_bytes: return "none"
            
        if self.emotion_queue: self.emotion_queue.put("THINKING")
        try:
            b64 = base64.b64encode(wav_bytes).decode("ascii")
            prompt = (
                "사용자의 오디오를 듣고, 사용자가 '이름', '전공', 'MBTI' 중 어느 항목을 수정하고 싶어하는지 판단하세요. "
                "반드시 JSON으로만 출력하세요: {\"field\": \"이름\"} 또는 {\"field\": \"전공\"} 또는 {\"field\": \"MBTI\"}. "
                "명확하지 않거나 파악할 수 없다면 {\"field\": \"none\"} 으로 출력하세요."
            )
            resp = self.model.generate_content([prompt, {"inline_data": {"mime_type": "audio/wav", "data": b64}}])
            txt = _extract_text(resp)
            match = re.search(r'\{.*?\}', txt, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return data.get("field", "none")
            return "none"
        except Exception:
            return "none"

    # 🚨 [신규 함수 2] 새롭게 입력된 "값"을 추출하는 함수
    def _listen_and_get_new_value(self, field_name: str, timeout=5.0) -> str:
        print(f"👂 [새 {field_name} 값 파악] 듣기 시작...")
        if self.emotion_queue: self.emotion_queue.put("LISTENING")
        wav_bytes = self.audio.record_fixed_duration(timeout)
        if not wav_bytes: return ""
            
        if self.emotion_queue: self.emotion_queue.put("THINKING")
        try:
            b64 = base64.b64encode(wav_bytes).decode("ascii")
            prompt = (
                f"사용자의 오디오를 듣고, 새로운 '{field_name}' 값을 추출하세요. "
                "사용자의 인사말이나 조사, 불필요한 부연 설명 없이 오직 핵심 '값(단어)'만 딱 잘라서 텍스트로 출력하세요. "
                "(예: 사용자가 '조형민이야' 또는 '내 이름은 조형민'이라고 하면 '조형민'만 출력)"
            )
            resp = self.model.generate_content([prompt, {"inline_data": {"mime_type": "audio/wav", "data": b64}}])
            return _extract_text(resp).strip()
        except Exception:
            return ""

    def _on_press(self, key): pass

    def _on_release(self, key):
        if self.stop_event.is_set(): return False
        if key == keyboard.Key.esc:
            self.stop_event.set()
            self.stop_nodding_event.set() 
            if self.current_listener and self.current_listener.is_alive(): self.current_listener.stop()
            return False 

    def run(self):
        self.current_listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.current_listener.start()
        
        if self.mouth_event_queue:
            self.mouth_listener_thread = threading.Thread(target=self._mouth_listener_worker, daemon=True)
            self.mouth_listener_thread.start()

        self.last_activity_time = time.time()
        self.listening_enabled.clear()
        print("▶ 대화 세션을 시작합니다. (상시 대기 상태)")

        while not self.stop_event.is_set():
            time.sleep(0.1)

            if getattr(self, 'session_active', False):
                continue

            if self.shared_state:
                raw_name = self.shared_state.get('detected_user')
                
                if raw_name and raw_name not in ["Thinking...", None]:
                    stabilize_timeout = 1.0 
                    elapsed = 0.0
                    final_name = raw_name
                    
                    while elapsed < stabilize_timeout:
                        if self.stop_event.is_set(): break
                        current_name = self.shared_state.get('detected_user')
                        if current_name and current_name not in ["Unknown", "Thinking...", None]:
                            final_name = current_name
                            break
                        if current_name is None:
                            final_name = None
                            break
                        time.sleep(0.1); elapsed += 0.1
                    
                    if not final_name or final_name in ["Thinking...", None]: continue 
                    detected_name = final_name

                    if self.interview_stage_index >= 0 or self.is_setting_up_main_chat or time.time() < self.main_chat_cooldown_until:
                        continue

                    if detected_name != self.last_logged_in_user:
                        if detected_name == "Unknown":
                            if self.session_active:
                                #print("🛡️ [방어] 대화 도중 잠깐 Unknown 감지됨. 무시합니다.")
                                continue

                            print("🤖 새로운 Unknown 감지 -> 인터뷰 시작!")
                            self.raise_busy_signal()
                            
                            self.interview_stage_index = 0
                            self.temp_user_info = {"이름": "", "학년": "", "나이": "", "MBTI": "", "성별": "", "전공": "", "RC": "", "새새 인원": ""}
                            self.last_logged_in_user = "Unknown"
                            self.shared_state['current_user_name'] = "Unknown"

                            opening_prompt = build_opening_prompt() + "\n\n🚨[출력 필수 조건]: 답변의 맨 앞에 반드시 [EMOTION]감정[/EMOTION] 태그를 하나 작성하고, 이어서 생성된 대답 텍스트를 명확하게 작성하세요."
                            resp = self.model.generate_content(opening_prompt)
                            raw_text = _extract_text(resp)
                            
                            print(f"👉 [DEBUG] 첫 인사 응답: {raw_text}")
                            self._process_and_speak_gemini_response(raw_text)
                            
                            while not self.mouth_event_queue.empty():
                                try: self.mouth_event_queue.get_nowait()
                                except: pass
                            self.listening_enabled.set()
                            
                            self.lower_busy_signal()
                            
                        else:
                            print(f"🤖 아는 사람({detected_name}) 감지 -> 프로필 로드 및 대화 세팅")
                            self.raise_busy_signal()
                            
                            self.last_logged_in_user = detected_name
                            self.shared_state['current_user_name'] = detected_name
                            self.interview_stage_index = -1 
                            
                            self.profile_manager.load_profile_for_chat(detected_name)
                            
                            if self.emotion_queue: self.emotion_queue.put("HAPPY")
                            greeting_msg = f"{detected_name}님 안녕하세요! {detected_name}님을 더 잘 기억할 수 있게 얼굴 인식을 수행할까요?"
                            self._speak_and_subtitle(greeting_msg)
                            self.tts.wait()

                            self.listening_enabled.clear()
                            do_learning = self._quick_listen_for_yes_no(timeout=4.0)

                            if do_learning:
                                self._speak_and_subtitle("네! 10초 동안 카메라를 봐주세요.")
                                self.tts.wait()
                                
                                if self.emotion_queue: self.emotion_queue.put("SCANNING")
                                self.shared_state['force_learning'] = True
                                self.shared_state['learning_target_name'] = detected_name
                                time.sleep(10) 
                                self.shared_state['force_learning'] = False
                                
                                if self.emotion_queue: self.emotion_queue.put("HAPPY")
                                self._speak_and_subtitle("얼굴 데이터 업데이트 완료! 이제 대화를 시작해요!")
                                self.tts.wait()
                            else:
                                if self.emotion_queue: self.emotion_queue.put("HAPPY")
                                self._speak_and_subtitle("네, 바로 대화를 시작할게요.")
                                self.tts.wait()
                            
                            while not self.mouth_event_queue.empty():
                                try: self.mouth_event_queue.get_nowait()
                                except: pass
                            self.listening_enabled.set()
                            
                            self.session_active = True
                            self.main_chat_cooldown_until = time.time() + 5.0
                            self.lower_busy_signal()
                            print("✅ 아는 사람 세팅 완료! [1:1 대화 고정 모드 활성화]")

        print("PTT App 종료 절차 시작...")
        self._flush_session_history(is_shutdown=True)
        self.listening_enabled.clear()
        
        self.stop_nodding_event.set()
        if self.nodding_thread and self.nodding_thread.is_alive():
            self.nodding_thread.join(timeout=0.5)

        if self.current_listener and self.current_listener.is_alive(): self.current_listener.stop()
        if self.mouth_listener_thread and self.mouth_listener_thread.is_alive(): self.mouth_listener_thread.join(timeout=1.0)
        
        try: self.profile_manager.save_profile_at_exit()
        except Exception as e: print(f"❌ 종료 요약 저장 중 치명적 오류: {e}")

        try:
            if FAREWELL_TEXT: self.tts.speak(FAREWELL_TEXT)
        finally:
            self.tts.close_and_join(drain=True)
        print("PTT App 정상 종료")