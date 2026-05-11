"""
SparkLearn: Ignite Your Learning
AI Placement Preparation System — CPU compatible, local-first
"""

from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import os, json, base64, random, re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'sparklearn_secret_2024')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

APP_NAME      = "SparkLearn"
APP_FULL_NAME = "SparkLearn: Ignite Your Learning"

# ── AI: Ollama (primary) → Gemini (fallback) ──────────────────────────────────
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
OLLAMA_HOST    = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODEL   = os.getenv('OLLAMA_MODEL', 'qwen2:0.5b')
_gemini_model  = None

def _load_gemini():
    global _gemini_model
    if _gemini_model:
        return _gemini_model
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            _gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
            print("Gemini loaded")
            return _gemini_model
        except Exception as e:
            print(f"Gemini: {e}")
    return None

def _ollama_available():
    try:
        import urllib.request
        urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2)
        return True
    except:
        return False

def _ollama_generate(prompt, max_tokens=300):
    import urllib.request, json as _j
    payload = _j.dumps({"model": OLLAMA_MODEL, "prompt": prompt,
                        "stream": False,
                        "options": {"num_predict": max_tokens, "temperature": 0.7}}).encode()
    req  = urllib.request.Request(f"{OLLAMA_HOST}/api/generate", data=payload,
                                  headers={"Content-Type": "application/json"}, method="POST")
    resp = urllib.request.urlopen(req, timeout=90)
    return _j.loads(resp.read()).get("response", "")

def ai_generate(prompt, max_tokens=300):
    if _ollama_available():
        try:
            r = _ollama_generate(prompt, max_tokens)
            if r.strip():
                return r.strip()
        except Exception as e:
            print(f"  Ollama err: {e}")
    m = _load_gemini()
    if m:
        try:
            import google.generativeai as genai
            resp = m.generate_content(
                contents=prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens, temperature=0.7))
            return resp.text.strip()
        except Exception as e:
            return f"AI Error: {e}"
    return "AI unavailable. Run `ollama serve` or add GEMINI_API_KEY to .env"

def ai_json(prompt, max_tokens=600):
    raw = ai_generate(prompt, max_tokens)
    raw = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`')
    for pat in [r'\[.*\]', r'\{.*\}']:
        m = re.search(pat, raw, re.DOTALL)
        if m:
            try: return json.loads(m.group())
            except: pass
    return None

# ── TF-IDF Vector Search ───────────────────────────────────────────────────────
_vector_store = {}

def _build_vectors():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        print("scikit-learn missing - pip install scikit-learn")
        return
    for topic in TOPICS:
        try:
            with open(topic['file'], 'r') as f:
                text = f.read()
        except:
            continue
        words = text.split()
        chunks = [' '.join(words[i:i+200]) for i in range(0, len(words), 200) if words[i:i+200]]
        if not chunks:
            continue
        vec    = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        matrix = vec.fit_transform(chunks)
        _vector_store[topic['id']] = {'chunks': chunks, 'matrix': matrix, 'vectorizer': vec}
    print(f"Vectors built for {len(_vector_store)} topics")

def vector_search(query, topic_id=None, top_k=2):
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except:
        return ""
    candidates = [topic_id] if (topic_id and topic_id in _vector_store) else list(_vector_store.keys())
    results = []
    for tid in candidates:
        s     = _vector_store[tid]
        q_vec = s['vectorizer'].transform([query])
        sims  = cosine_similarity(q_vec, s['matrix'])[0]
        top   = np.argsort(sims)[::-1][:top_k]
        for idx in top:
            if sims[idx] > 0.05:
                results.append((sims[idx], s['chunks'][idx]))
    results.sort(key=lambda x: -x[0])
    return "\n\n".join(c for _, c in results[:top_k])

# ── Topic & Problem Data ───────────────────────────────────────────────────────
TOPICS = [
    {"id": "dbms",     "name": "DBMS",    "icon": "🗄️", "file": "content/dbms.txt",             "color": "#6366f1"},
    {"id": "sql",      "name": "SQL",      "icon": "📊", "file": "content/sql.txt",              "color": "#8b5cf6"},
    {"id": "dsa",      "name": "DSA",      "icon": "🌳", "file": "content/dsa.txt",              "color": "#06b6d4"},
    {"id": "oops",     "name": "OOP",      "icon": "🔷", "file": "content/oops.txt",             "color": "#10b981"},
    {"id": "cn",       "name": "Networks", "icon": "🌐", "file": "content/cn.txt",               "color": "#f59e0b"},
    {"id": "os",       "name": "OS",       "icon": "💻", "file": "content/os.txt",               "color": "#ef4444"},
    {"id": "aptitude", "name": "Aptitude", "icon": "🧮", "file": "content/aptitude.txt",         "color": "#ec4899"},
    {"id": "coding",   "name": "Coding",   "icon": "⚡", "file": "content/coding_questions.txt", "color": "#14b8a6"},
]

CODING_PROBLEMS = [
    {"id":1,"title":"Two Sum","difficulty":"Easy","topic":"Array",
     "description":"Return indices of two numbers that sum to target.",
     "examples":[{"input":"nums=[2,7,11,15], target=9","output":"[0,1]","explanation":"nums[0]+nums[1]=9"}],
     "constraints":["2<=nums.length<=10^4","Exactly one answer"],
     "starter_code":{"python":"def twoSum(nums, target):\n    pass",
                     "javascript":"function twoSum(nums, target) {}",
                     "java":"public int[] twoSum(int[] nums, int target){return new int[]{};}"},
     "test_cases":[{"input":"[2,7,11,15]\n9","expected":"[0, 1]"},{"input":"[3,2,4]\n6","expected":"[1, 2]"}]},
    {"id":2,"title":"Reverse String","difficulty":"Easy","topic":"String",
     "description":"Reverse a character array in-place.",
     "examples":[{"input":'["h","e","l","l","o"]',"output":'["o","l","l","e","h"]',"explanation":"Reversed"}],
     "constraints":["1<=s.length<=10^5"],
     "starter_code":{"python":"def reverseString(s):\n    pass",
                     "javascript":"function reverseString(s){}",
                     "java":"public void reverseString(char[] s){}"},
     "test_cases":[{"input":"hello","expected":"olleh"},{"input":"world","expected":"dlrow"}]},
    {"id":3,"title":"Valid Parentheses","difficulty":"Medium","topic":"Stack",
     "description":"Determine if bracket string is valid.",
     "examples":[{"input":'"()"',"output":"true","explanation":"Match"}],
     "constraints":["1<=s.length<=10^4"],
     "starter_code":{"python":"def isValid(s):\n    pass",
                     "javascript":"function isValid(s){return false;}",
                     "java":"public boolean isValid(String s){return false;}"},
     "test_cases":[{"input":"()","expected":"True"},{"input":"(]","expected":"False"}]},
    {"id":4,"title":"Fibonacci","difficulty":"Easy","topic":"Dynamic Programming",
     "description":"Compute F(n) = F(n-1)+F(n-2), F(0)=0, F(1)=1.",
     "examples":[{"input":"n=4","output":"3","explanation":"0,1,1,2,3"}],
     "constraints":["0<=n<=30"],
     "starter_code":{"python":"def fib(n):\n    pass","javascript":"function fib(n){}","java":"public int fib(int n){return 0;}"},
     "test_cases":[{"input":"0","expected":"0"},{"input":"10","expected":"55"}]},
    {"id":5,"title":"Binary Search","difficulty":"Medium","topic":"Binary Search",
     "description":"Search sorted array, return index or -1.",
     "examples":[{"input":"nums=[-1,0,3,5,9], target=9","output":"4","explanation":"index 4"}],
     "constraints":["Sorted ascending","All unique"],
     "starter_code":{"python":"def search(nums, target):\n    pass",
                     "javascript":"function search(nums,target){return -1;}",
                     "java":"public int search(int[] nums,int target){return -1;}"},
     "test_cases":[{"input":"[-1,0,3,5,9,12]\n9","expected":"4"},{"input":"[1,2,3]\n6","expected":"-1"}]},
    {"id":6,"title":"Maximum Subarray","difficulty":"Hard","topic":"Dynamic Programming",
     "description":"Find contiguous subarray with largest sum.",
     "examples":[{"input":"[-2,1,-3,4,-1,2,1,-5,4]","output":"6","explanation":"[4,-1,2,1]=6"}],
     "constraints":["1<=nums.length<=10^5"],
     "starter_code":{"python":"def maxSubArray(nums):\n    pass",
                     "javascript":"function maxSubArray(nums){}","java":"public int maxSubArray(int[] nums){return 0;}"},
     "test_cases":[{"input":"[-2,1,-3,4,-1,2,1,-5,4]","expected":"6"}]},
]

# ── Session helpers ────────────────────────────────────────────────────────────
def get_session_user():
    return session.get('user', {'name': 'Student', 'email': '', 'avatar': '👤'})

def init_session():
    session.setdefault('user', {'name': 'Student', 'email': '', 'avatar': '👤'})
    session.setdefault('stats', {
        'topics_completed': [], 'quiz_scores': [],
        'interview_sessions': 0, 'coding_solved': [],
        'total_study_time': 0, 'warnings': 0,
        'start_time': datetime.now().isoformat()
    })

# ── OpenCV ─────────────────────────────────────────────────────────────────────
face_cascade = eye_cascade = None
try:
    import cv2, numpy as np
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    if face_cascade.empty(): face_cascade = None
    else: print("Face detection loaded")
except Exception as e:
    print(f"OpenCV: {e}")

# ── Routes: Core ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    init_session()
    return render_template('index.html')

@app.route('/setup', methods=['GET','POST'])
def setup():
    if request.method == 'POST':
        name  = request.form.get('name','Student').strip() or 'Student'
        email = request.form.get('email','').strip()
        session['user'] = {'name': name, 'email': email, 'avatar': '👤'}
        init_session()
        return redirect(url_for('dashboard'))
    return render_template('setup.html', app_name=APP_NAME, app_full=APP_FULL_NAME)

@app.route('/dashboard')
def dashboard():
    init_session()
    return render_template('dashboard.html', user=get_session_user(),
        stats=session.get('stats',{}), topics=TOPICS,
        app_name=APP_NAME, active_page='dashboard')

# ── Study ─────────────────────────────────────────────────────────────────────
@app.route('/study')
def study():
    init_session()
    return render_template('study.html', user=get_session_user(), topics=TOPICS,
        app_name=APP_NAME, active_page='study')

@app.route('/subjects')
def subjects():
    return redirect(url_for('study'))

@app.route('/doubt', methods=['GET', 'POST'])
def doubt():
    init_session()
    answer = None
    question = ''
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        if question:
            answer = ai_generate(
                f"Answer this placement preparation doubt in 50-80 words: {question}",
                max_tokens=150
            )
    return render_template('doubt.html', user=get_session_user(), answer=answer,
        question=question, app_name=APP_NAME, active_page='doubt')

@app.route('/quit')
def quit():
    session.clear()
    return render_template('end.html', app_name=APP_NAME)

@app.route('/study/<topic_id>')
def study_topic(topic_id):
    init_session()
    topic = next((t for t in TOPICS if t['id'] == topic_id), None)
    if not topic: return redirect(url_for('study'))
    try:
        with open(topic['file']) as f: content = f.read()
    except: content = f"Content for {topic['name']} not found."
    stats = session.get('stats', {})
    if topic_id not in stats.get('topics_completed', []):
        stats.setdefault('topics_completed', []).append(topic_id)
        session['stats'] = stats
    return render_template('study_topic.html', user=get_session_user(), topic=topic,
        content=content, topics=TOPICS, app_name=APP_NAME, active_page='study')

# ── API: Doubt (50-100 word limit, vector-augmented) ──────────────────────────
@app.route('/api/ask_doubt', methods=['POST'])
def ask_doubt():
    data     = request.get_json()
    question = data.get('question','').strip()[:300]
    topic_id = data.get('topic_id')
    topic    = data.get('topic','General CS')
    if not question: return jsonify({'error': 'No question'}), 400
    ctx = vector_search(question, topic_id=topic_id, top_k=2)
    ctx_block = f"\nNotes:\n{ctx[:500]}\n" if ctx else ""
    prompt = (f"CS tutor. Topic:{topic}.{ctx_block}\n"
              f"Q:{question}\nAnswer in 50-80 words max. Be direct, include one example.")
    return jsonify({'answer': ai_generate(prompt, max_tokens=150)})

# ── Quiz ──────────────────────────────────────────────────────────────────────
@app.route('/quiz')
def quiz():
    init_session()
    return render_template('quiz.html', user=get_session_user(), topics=TOPICS,
        app_name=APP_NAME, active_page='quiz')

@app.route('/api/generate_quiz', methods=['POST'])
def generate_quiz_api():
    data       = request.get_json()
    topic_name = data.get('topic','General CS')
    count      = min(int(data.get('count',10)), 20)
    difficulty = data.get('difficulty','mixed')
    topic_id   = data.get('topic_id')
    ctx = vector_search(topic_name, topic_id=topic_id, top_k=2)
    ctx_note = f"\nBased on:\n{ctx[:400]}\n" if ctx else ""
    prompt = (f"Generate {count} MCQ on '{topic_name}', difficulty:{difficulty}.{ctx_note}\n"
              f"JSON array only:\n"
              f'[{{"question":"...","options":["A","B","C","D"],"correct":0,"explanation":"..."}}]')
    result = ai_json(prompt, max_tokens=900)
    if isinstance(result, list) and result:
        return jsonify({'questions': result[:count]})
    return jsonify({'error': 'Quiz generation failed'}), 500

@app.route('/api/save_quiz_score', methods=['POST'])
def save_quiz_score():
    data  = request.get_json()
    stats = session.get('stats', {})
    stats.setdefault('quiz_scores', []).append({
        'topic': data.get('topic'), 'score': data.get('score'),
        'total': data.get('total'), 'timestamp': datetime.now().isoformat()
    })
    session['stats'] = stats
    return jsonify({'saved': True})

# ── Interview: generate 5 resume-based questions ──────────────────────────────
@app.route('/api/generate_interview_questions', methods=['POST'])
def generate_interview_questions():
    data    = request.get_json()
    itype   = data.get('type','hr')
    resume  = data.get('resume','').strip()[:600]
    role    = data.get('role','Software Engineer')
    rb      = f"\nResume:\n{resume}\n" if resume else ""
    type_map = {
        'hr': "HR behavioural",
        'technical': "core CS technical (DSA/OOP/OS/DBMS)",
        'system_design': "system design scenario"
    }
    instr = type_map.get(itype, type_map['hr'])
    prompt = (f"Create exactly 5 {instr} interview questions for {role}.{rb}\n"
              f'Return JSON array of 5 strings ONLY: ["Q1?","Q2?","Q3?","Q4?","Q5?"]')
    result = ai_json(prompt, max_tokens=350)
    fallbacks = {
        'hr':           ["Tell me about yourself.",
                         "Describe a challenge you overcame.",
                         "Where do you see yourself in 5 years?",
                         "Why do you want this role?",
                         "Tell me about working effectively in a team."],
        'technical':    ["Explain OOP principles with examples.",
                         "What is Big-O notation? Give two examples.",
                         "How does a hash map work internally?",
                         "Explain process vs thread.",
                         "What is database normalization?"],
        'system_design':["Design a URL shortener.",
                         "Design a real-time chat system.",
                         "Design an API rate limiter.",
                         "How would you scale a DB to 10M users?",
                         "Design a push notification service."],
    }
    fb = fallbacks.get(itype, fallbacks['hr'])
    if isinstance(result, list) and len(result) >= 3:
        qs = [str(q) for q in result[:5]]
        while len(qs) < 5: qs.append(fb[len(qs)])
        return jsonify({'questions': qs})
    return jsonify({'questions': fb})

# ── Interview: evaluate one answer ────────────────────────────────────────────
@app.route('/api/evaluate_answer', methods=['POST'])
def evaluate_answer():
    data     = request.get_json()
    question = data.get('question','')
    answer   = data.get('answer','').strip()
    qtype    = data.get('type','hr')
    if len(answer.split()) < 8:
        return jsonify({'score':0,'overall':'Too short',
                        'strengths':[],'improvements':['Answer too brief'],'ideal_answer_hint':''})
    prompt = (f"Evaluate {qtype.upper()} answer (50-word max).\nQ:{question}\nA:{answer}\n"
              f'JSON:{{"score":0-10,"overall":"Excellent/Good/Average/Poor",'
              f'"strengths":["..."],"improvements":["..."],"ideal_answer_hint":"..."}}')
    result = ai_json(prompt, max_tokens=200)
    if isinstance(result, dict):
        stats = session.get('stats',{}); stats['interview_sessions'] = stats.get('interview_sessions',0)+1
        session['stats'] = stats
        return jsonify(result)
    return jsonify({'score':5,'overall':'Average','strengths':['Attempted'],'improvements':['More detail'],'ideal_answer_hint':''})

# ── Interview: final session scoring ─────────────────────────────────────────
@app.route('/api/score_interview_session', methods=['POST'])
def score_interview_session():
    data    = request.get_json()
    qa_list = data.get('qa_list', [])
    itype   = data.get('type', 'hr')
    role    = data.get('role', 'Software Engineer')
    if not qa_list: return jsonify({'error': 'No answers'}), 400
    avg = sum(qa.get('score',0) for qa in qa_list) / len(qa_list)
    summary = "\n".join(
        f"Q{i+1}:{qa['question'][:60]}\nA:{qa.get('answer','')[:80]}\nScore:{qa.get('score',0)}"
        for i,qa in enumerate(qa_list))
    prompt = (f"Interview debrief {role} ({itype}). Avg:{avg:.1f}/10.\n{summary}\n"
              f"60 words max. JSON:\n"
              f'{{"overall_score":{avg:.1f},"readiness":"Ready/Almost/Needs Work",'
              f'"top_strength":"...","key_improvement":"...","study_topics":["t1","t2"],'
              f'"motivational_note":"..."}}')
    result = ai_json(prompt, max_tokens=250)
    if isinstance(result, dict):
        result['overall_score'] = round(avg, 1)
        return jsonify(result)
    return jsonify({'overall_score':round(avg,1),'readiness':'Almost' if avg>=5 else 'Needs Work',
                    'top_strength':'Completed all questions','key_improvement':'Practice structure',
                    'study_topics':['Communication','Technical depth'],
                    'motivational_note':'Keep going — every session makes you sharper!'})

# ── Face Detection ────────────────────────────────────────────────────────────
@app.route('/api/detect_face', methods=['POST'])
def detect_face():
    try:
        import cv2, numpy as np
        data = request.get_json()
        if not data or 'image' not in data or face_cascade is None:
            return jsonify({'face':False,'eyes':0,'warning':False})
        img_data = data['image']
        if ',' in img_data: img_data = img_data.split(',')[1]
        arr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None: return jsonify({'face':False,'eyes':0,'warning':False})
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
        face_ok = len(faces) > 0
        eyes, away = 0, False
        if face_ok and eye_cascade is not None:
            x,y,w,h = faces[0]
            eyelist = eye_cascade.detectMultiScale(gray[y:y+h,x:x+w], 1.1, 3, minSize=(20,20))
            eyes = len(eyelist); away = eyes < 1
        stats = session.get('stats',{})
        if not face_ok or away: stats['warnings'] = stats.get('warnings',0)+1
        session['stats'] = stats
        return jsonify({'face':face_ok,'eyes':eyes,'looking_away':away,
                        'warning': not face_ok or away,
                        'warning_count':stats.get('warnings',0),'terminate':stats.get('warnings',0)>=15})
    except Exception as e:
        return jsonify({'face':False,'eyes':0,'warning':False,'error':str(e)})

@app.route('/api/reset_warnings', methods=['POST'])
def reset_warnings():
    s = session.get('stats',{}); s['warnings']=0; session['stats']=s
    return jsonify({'reset':True})

# ── Coding ────────────────────────────────────────────────────────────────────
@app.route('/coding')
def coding():
    init_session()
    return render_template('coding.html', user=get_session_user(), problems=CODING_PROBLEMS,
        app_name=APP_NAME, active_page='coding')

@app.route('/coding/<int:problem_id>')
def coding_problem(problem_id):
    init_session()
    p = next((x for x in CODING_PROBLEMS if x['id']==problem_id), None)
    if not p: return redirect(url_for('coding'))
    return render_template('coding_editor.html', user=get_session_user(), problem=p,
        problems=CODING_PROBLEMS, app_name=APP_NAME, active_page='coding')

@app.route('/api/run_code', methods=['POST'])
def run_code():
    data = request.get_json()
    code, language = data.get('code',''), data.get('language','python')
    test_cases = data.get('test_cases', [])
    if not code.strip(): return jsonify({'error':'No code'})
    results = []
    for tc in test_cases:
        prompt = (f"Simulate {language}:\n{code[:400]}\nInput:{tc.get('input','')}\n"
                  f"Expected:{tc.get('expected','')}\n"
                  f'JSON:{{"output":"...","passed":true/false,"error":null}}')
        r = ai_json(prompt, max_tokens=80)
        results.append(r if isinstance(r,dict) else {'output':'?','passed':False,'error':'Failed'})
    return jsonify({'results':results,'passed':sum(1 for r in results if r.get('passed')),'total':len(results)})

@app.route('/api/get_code_hint', methods=['POST'])
def get_code_hint():
    d = request.get_json()
    prompt = (f"40-word hint (no solution) for: {d.get('problem','')}\n"
              f"Code:\n{d.get('code','')[:300]}")
    return jsonify({'hint': ai_generate(prompt, max_tokens=80)})

@app.route('/api/review_code', methods=['POST'])
def review_code():
    d = request.get_json()
    prompt = (f"Review {d.get('language','python')} for '{d.get('problem','')}' in 50 words.\n"
              f"Code:\n{d.get('code','')[:400]}\n"
              f'JSON:{{"time_complexity":"O(?)","space_complexity":"O(?)",'
              f'"correctness_score":0-10,"code_quality_score":0-10,'
              f'"strengths":["..."],"improvements":["..."],"interview_feedback":"..."}}')
    r = ai_json(prompt, max_tokens=200)
    return jsonify(r if isinstance(r,dict) else {'interview_feedback':ai_generate(prompt,100),'time_complexity':'?','space_complexity':'?'})

@app.route('/api/mark_solved', methods=['POST'])
def mark_solved():
    d=request.get_json(); pid=d.get('problem_id')
    s=session.get('stats',{})
    if pid and pid not in s.get('coding_solved',[]): s.setdefault('coding_solved',[]).append(pid)
    session['stats']=s
    return jsonify({'saved':True})

# ── Performance ───────────────────────────────────────────────────────────────
@app.route('/performance')
def performance():
    init_session()
    stats=session.get('stats',{}); qs=stats.get('quiz_scores',[])
    avg=round(sum(s['score']/s['total']*100 for s in qs)/len(qs)) if qs else 0
    return render_template('performance.html', user=get_session_user(), stats=stats,
        topics=TOPICS, avg_quiz=avg, total_topics=len(TOPICS),
        app_name=APP_NAME, active_page='performance')

@app.route('/api/get_ai_feedback', methods=['POST'])
def get_ai_feedback():
    s=session.get('stats',{})
    prompt = (f"Placement readiness: {len(s.get('topics_completed',[]))}/{len(TOPICS)} topics, "
              f"{len(s.get('quiz_scores',[]))} quizzes, {len(s.get('coding_solved',[]))} coding, "
              f"{s.get('interview_sessions',0)} interviews. 60-word response. JSON:\n"
              f'{{"overall_readiness":"X%","strengths":["..."],"focus_areas":["..."],'
              f'"weekly_plan":["Day 1:...","Day 2:...","Day 3:..."],"motivational_message":"..."}}')
    r=ai_json(prompt, max_tokens=280)
    return jsonify(r if isinstance(r,dict) else {'overall_readiness':'50%','motivational_message':'Keep going!'})

# ── Interview page ────────────────────────────────────────────────────────────
@app.route('/interview')
def interview():
    init_session()
    return render_template('interview.html', user=get_session_user(),
        app_name=APP_NAME, active_page='interview')

# ── Mock / Resume / Companies ─────────────────────────────────────────────────
@app.route('/mock_test')
def mock_test():
    init_session()
    return render_template('mock_test.html', user=get_session_user(),
        app_name=APP_NAME, active_page='mock_test')

@app.route('/resume')
def resume():
    init_session()
    return render_template('resume.html', user=get_session_user(),
        app_name=APP_NAME, active_page='resume')

@app.route('/api/analyze_resume', methods=['POST'])
def analyze_resume():
    d=request.get_json(); text=d.get('text','').strip()[:1200]
    if len(text)<50: return jsonify({'error':'Paste more content'})
    prompt=(f"Score resume for placement (70-word max):\n{text}\n"
            f'JSON:{{"overall_score":0-100,"ats_score":0-100,"strengths":["..."],'
            f'"improvements":["..."],"keywords_missing":["..."],"action_items":["..."]}}')
    r=ai_json(prompt, max_tokens=300)
    return jsonify(r if isinstance(r,dict) else {'error':'Analysis failed'})

@app.route('/companies')
def companies():
    init_session()
    return render_template('companies.html', user=get_session_user(),
        app_name=APP_NAME, active_page='companies')

@app.route('/api/company_prep', methods=['POST'])
def company_prep():
    d=request.get_json(); company=d.get('company','').strip()[:60]; role=d.get('role','SWE')
    if not company: return jsonify({'error':'Company needed'})
    prompt=(f"Placement guide for {company}-{role} (60 words).\n"
            f'JSON:{{"company":"{company}","role":"{role}","difficulty":"Easy/Medium/Hard",'
            f'"rounds":["..."],"focus_topics":["..."],"sample_questions":["..."],'
            f'"tips":["..."],"package_range":"X-Y LPA","prep_timeline":"X weeks"}}')
    r=ai_json(prompt, max_tokens=350)
    return jsonify(r if isinstance(r,dict) else {'company':company,'tips':['Check AI connection']})

# ── Utility ───────────────────────────────────────────────────────────────────
@app.route('/api/stats')
def get_stats(): return jsonify(session.get('stats',{}))

@app.route('/api/update_time', methods=['POST'])
def update_time():
    d=request.get_json(); s=session.get('stats',{})
    s['total_study_time']=s.get('total_study_time',0)+d.get('seconds',0)
    session['stats']=s; return jsonify({'ok':True})

@app.route('/api/ai_status')
def ai_status():
    ol=_ollama_available()
    return jsonify({'ollama':ol,'ollama_model':OLLAMA_MODEL if ol else None,
                    'gemini':bool(GEMINI_API_KEY),
                    'active':'ollama' if ol else ('gemini' if GEMINI_API_KEY else 'none')})

# ── Boot ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    _build_vectors()
    port=int(os.getenv('PORT',5000))
    print(f"\n{APP_FULL_NAME} -> http://localhost:{port}")
    print(f"   Ollama : {OLLAMA_MODEL if _ollama_available() else 'not running (ollama serve)'}")
    print(f"   Gemini : {'ready' if GEMINI_API_KEY else 'no key'}")
    print(f"   OpenCV : {'ready' if face_cascade else 'unavailable'}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
