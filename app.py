from flask import Flask, render_template, request, session, redirect, url_for, send_file, jsonify
import google.generativeai as genai
from gtts import gTTS
import os
import cv2
import numpy as np
from datetime import datetime, timedelta
import base64
import random

from dotenv import load_dotenv
import os

load_dotenv()

# Replace 'GEMINI_API_KEY' with your actual Google Gemini API key from environment variable
api_key = os.getenv('GEMINI_API_KEY')

# Configure the Google Generative AI client
genai.configure(api_key=api_key)

# Initialize the model
model = genai.GenerativeModel('gemini-2.0-flash')

# List of placement preparation topics and corresponding files
placement_topics = [
    {
        "topic": "DBMS",
        "file": "dbms.txt"
    },
    {
        "topic": "SQL",
        "file": "sql.txt"
    },
    {
        "topic": "DSA",
        "file": "dsa.txt"
    },
    {
        "topic": "OOPS",
        "file": "oops.txt"
    },
    {
        "topic": "CN",
        "file": "cn.txt"
    },
    {
        "topic": "OS",
        "file": "os.txt"
    },
    {
        "topic": "Aptitude",
        "file": "aptitude.txt"
    },
    {
        "topic": "Coding Questions",
        "file": "coding_questions.txt"
    },
    # Add more topics and files as needed
]

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Sample dynamic data for progress, recommendations, mentor info
user_data = {
    "name": "Rahul",
    "topics_progress": {
        "DBMS": 30,
        "SQL": 70,
        "AI & ML": 20,
        "Generative AI": 10
    },
    "recommendations": [
        "Deep Dive into Normalization",
        "Generative AI Explained"
    ],
    "weekly_progress": 85,
    "mentor": {
        "name": "John Doe",
        "title": "Data Science Expert",
        "image_url": "https://randomuser.me/api/portraits/men/32.jpg"
    }
}

@app.route('/')
def dashboard():
    # Redirect to subjects page as the main learning interface
    return redirect(url_for('subjects'))

@app.route('/subjects')
def subjects():
    # Subject section showing all subjects and doubt section link
    subjects = [{"name": topic["topic"], "file": topic["file"]} for topic in placement_topics]

    # Initialize session variables for topic navigation
    if 'current_topic_index' not in session:
        session['current_topic_index'] = 0
    if 'show_doubts' not in session:
        session['show_doubts'] = False
    if 'response' not in session:
        session['response'] = ""
    if 'last_topic' not in session:
        session['last_topic'] = -1
    if 'doubt_response' not in session:
        session['doubt_response'] = ""
    if 'started' not in session:
        session['started'] = False
    if 'quiz_questions' not in session:
        session['quiz_questions'] = []
    if 'current_question_index' not in session:
        session['current_question_index'] = 0
    if 'quiz_score' not in session:
        session['quiz_score'] = 0
    if 'show_quiz' not in session:
        session['show_quiz'] = False

    # Get current topic content if started
    current_topic = None
    current_content = ""
    if session['started'] and session['current_topic_index'] < len(placement_topics):
        topic_info = placement_topics[session['current_topic_index']]
        current_topic = topic_info['topic']
        if session['last_topic'] != session['current_topic_index']:
            try:
                with open(topic_info["file"], 'r') as f:
                    current_content = f.read()
                session['response'] = current_content
                session['last_topic'] = session['current_topic_index']
            except Exception as e:
                current_content = f"Error reading file: {str(e)}"
                session['response'] = current_content
        else:
            current_content = session['response']

    # Get doubt response for current topic
    doubt_responses = session.get('doubt_responses', {})
    current_topic_index = session.get('current_topic_index', 0)
    current_topic_name = placement_topics[current_topic_index]['topic'] if current_topic_index < len(placement_topics) else 'General'
    doubt_response = doubt_responses.get(current_topic_name, "")

    return render_template('subjects_new.html',
                           user_name=user_data["name"],
                           subjects=subjects,
                           current_topic=current_topic,
                           current_content=current_content,
                           show_doubts=session['show_doubts'],
                           doubt_response=doubt_response,
                           show_quiz=session['show_quiz'],
                           quiz_questions=session['quiz_questions'],
                           current_question_index=session['current_question_index'],
                           quiz_score=session['quiz_score'],
                           active_page='subjects')

@app.route('/doubt', methods=['GET', 'POST'])
def doubt():
    doubt_response = session.get('doubt_response', "")
    if request.method == 'POST':
        doubt_text = request.form.get('doubt', '').strip()
        if doubt_text:
            try:
                response = model.generate_content(
                    contents=doubt_text,
                    generation_config=genai.types.GenerationConfig(max_output_tokens=500)
                )
                doubt_response = response.text
                session['doubt_response'] = doubt_response
            except Exception as e:
                doubt_response = f"Error: {str(e)}"
                session['doubt_response'] = doubt_response
    return render_template('doubt.html',
                           user_name=user_data["name"],
                           doubt_response=doubt_response,
                           active_page='doubt')

@app.route('/performance')
def performance():
    # Performance page showing time spent and face detection times
    usage_time = session.get('usage_time', 0)
    face_detected_times = session.get('face_detected_times', [])
    return render_template('performance.html',
                           user_name=user_data["name"],
                           usage_time=usage_time,
                           face_detected_times=face_detected_times,
                           active_page='performance')



@app.route('/quit')
def quit():
    # Save last topic index before clearing session
    if 'current_topic_index' in session:
        session['last_topic_index'] = session['current_topic_index']
    # Clear session and redirect to end page
    session.clear()
    return render_template('end.html')

# Update detect_presence to track face detection times and usage time
@app.before_request
def track_usage():
    if 'start_time' not in session:
        session['start_time'] = datetime.now().isoformat()
    if 'face_detected_times' not in session:
        session['face_detected_times'] = []

@app.route('/detect_presence', methods=['POST'])
def detect_presence():
    try:
        if face_cascade is None:
            return jsonify({'presence': False, 'alert': False, 'error': 'Face cascade not loaded'})

        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'presence': False, 'alert': False, 'error': 'No image data'})

        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'presence': False, 'alert': False, 'error': 'Invalid image'})

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        presence_detected = len(faces) > 0

        if presence_detected:
            session['alert_triggered'] = False
            # Record face detection time
            face_times = session.get('face_detected_times', [])
            face_times.append(datetime.now().isoformat())
            session['face_detected_times'] = face_times
        else:
            last_detection = session.get('face_detected_times', [])
            if last_detection:
                last_time = datetime.fromisoformat(last_detection[-1])
                if datetime.now() - last_time > timedelta(seconds=10):
                    session['alert_triggered'] = True
            else:
                session['alert_triggered'] = True

        # Calculate usage time
        start_time = datetime.fromisoformat(session['start_time'])
        usage_time = (datetime.now() - start_time).total_seconds()
        session['usage_time'] = usage_time

        return jsonify({'presence': presence_detected, 'alert': session.get('alert_triggered', False)})
    except Exception as e:
        print(f"Error in detect_presence: {str(e)}")
        return jsonify({'presence': False, 'alert': False, 'error': str(e)})

@app.route('/topics')
def topics():
    # Render topics page or section
    return render_template('topics.html',
                           user_name=user_data["name"],
                           topics_progress=user_data["topics_progress"])

@app.route('/progress')
def progress():
    # Render progress page or section
    return render_template('progress.html',
                           user_name=user_data["name"],
                           topics_progress=user_data["topics_progress"],
                           weekly_progress=user_data["weekly_progress"])

@app.route('/projects')
def projects():
    # Render projects page or section
    return render_template('projects.html',
                           user_name=user_data["name"])

@app.route('/mentorship')
def mentorship():
    # Render mentorship page or section
    return render_template('mentorship.html',
                           user_name=user_data["name"],
                           mentor=user_data["mentor"])

# Load OpenCV face cascade
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)
if face_cascade.empty():
    print(f"Error: Could not load face cascade from {face_cascade_path}")
    face_cascade = None
else:
    print("Face cascade loaded successfully")

@app.route('/next')
def next_topic():
    session['current_topic_index'] += 1
    return redirect(url_for('subjects'))

@app.route('/doubts')
def doubts():
    session['show_doubts'] = True
    return redirect(url_for('subjects'))

@app.route('/back')
def back():
    session['show_doubts'] = False
    return redirect(url_for('subjects'))

@app.route('/home')
def home():
    session.clear()
    return redirect(url_for('subjects'))

@app.route('/select_topic/<int:topic_index>')
def select_topic(topic_index):
    session['current_topic_index'] = topic_index
    session['started'] = True
    return redirect(url_for('subjects'))

@app.route('/resume_learning')
def resume_learning():
    last_topic_index = session.get('last_topic_index', 0)
    if last_topic_index >= 0 and last_topic_index < len(placement_topics):
        session['current_topic_index'] = last_topic_index
        session['started'] = True
    return redirect(url_for('subjects'))

@app.route('/submit_doubt', methods=['POST'])
def submit_doubt():
    doubt = request.form['doubt']
    current_topic_index = session.get('current_topic_index', 0)
    current_topic = placement_topics[current_topic_index]['topic'] if current_topic_index < len(placement_topics) else 'General'

    if doubt.strip():
        try:
            response = model.generate_content(
                contents=doubt,
                generation_config=genai.types.GenerationConfig(max_output_tokens=500)
            )
            # Store doubt response per subject/topic
            if 'doubt_responses' not in session:
                session['doubt_responses'] = {}
            doubt_responses = session['doubt_responses']
            doubt_responses[current_topic] = response.text
            session['doubt_responses'] = doubt_responses
        except Exception as e:
            if 'doubt_responses' not in session:
                session['doubt_responses'] = {}
            doubt_responses = session['doubt_responses']
            doubt_responses[current_topic] = f"Error: {str(e)}"
            session['doubt_responses'] = doubt_responses
    return redirect(url_for('subjects'))

@app.route('/read_aloud')
def read_aloud():
    text = session['response']
    tts = gTTS(text=text, lang='en')
    os.makedirs('static', exist_ok=True)
    audio_path = 'static/audio.mp3'
    tts.save(audio_path)
    return send_file(audio_path, as_attachment=False, mimetype='audio/mpeg')

@app.route('/read_doubt')
def read_doubt():
    text = session['doubt_response']
    tts = gTTS(text=text, lang='en')
    os.makedirs('static', exist_ok=True)
    audio_path = 'static/doubt_audio.mp3'
    tts.save(audio_path)
    return send_file(audio_path, as_attachment=False, mimetype='audio/mpeg')

@app.route('/quiz')
def quiz():
    """Render the quiz page"""
    import copy
    # Initialize quiz if not started
    if 'aptitude_quiz_questions' not in session or not session['aptitude_quiz_questions']:
        return redirect(url_for('generate_quiz'))
    
    current_quiz_type = session.get('current_quiz_type', 'aptitude')
    aptitude_questions = session.get('aptitude_quiz_questions', [])
    technical_questions = session.get('technical_quiz_questions', [])
    quiz_questions = aptitude_questions if current_quiz_type == 'aptitude' else technical_questions
    current_question_index = session.get('current_question_index', 0)
    quiz_score = session.get('quiz_score', 0)
    
    if quiz_questions and current_question_index < len(quiz_questions):
        question = copy.deepcopy(quiz_questions[current_question_index])
        total_questions = len(aptitude_questions) + len(technical_questions)
        progress_percent = ((current_question_index + 1) / total_questions) * 100

        # Shuffle options and adjust correct answer accordingly
        options = question['options']
        correct_answer = question['correct_answer']
        option_pairs = list(zip(['A', 'B', 'C', 'D'], options))
        random.shuffle(option_pairs)
        shuffled_options = [opt for _, opt in option_pairs]
        # Find new correct answer letter after shuffle
        for idx, (letter, opt) in enumerate(option_pairs):
            if letter == correct_answer:
                new_correct_answer = chr(ord('A') + idx)
                break
        question['options'] = shuffled_options
        question['correct_answer'] = new_correct_answer

        # Save shuffled question back to session for answer checking
        if current_quiz_type == 'aptitude':
            session['aptitude_quiz_questions'][current_question_index] = question
        else:
            session['technical_quiz_questions'][current_question_index] = question

        return render_template('quiz.html',
                               question=question,
                               current_question=current_question_index + 1,
                               total_questions=total_questions,
                               progress_percent=progress_percent,
                               quiz_score=quiz_score,
                               active_page='quiz')
    else:
        # Quiz completed
        return render_template('quiz_result.html',
                               quiz_score=quiz_score,
                               total_questions=len(aptitude_questions) + len(technical_questions),
                               active_page='quiz')

@app.route('/generate_quiz')
def generate_quiz():
    """Generate 5 aptitude and 5 technical questions using Gemini AI separately"""
    try:
        aptitude_prompt = """
        Generate 5 multiple choice aptitude questions for placement preparation.
        Format each question as:
        Question X: [Question text]
        A) [Option A]
        B) [Option B]
        C) [Option C]
        D) [Option D]
        Correct Answer: [Correct option letter]
        Return the questions in a structured format.
        """

        technical_prompt = """
        Generate 5 multiple choice technical questions for placement preparation.
        Format each question as:
        Question X: [Question text]
        A) [Option A]
        B) [Option B]
        C) [Option C]
        D) [Option D]
        Correct Answer: [Correct option letter]
        Return the questions in a structured format.
        """

        aptitude_response = model.generate_content(
            contents=aptitude_prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=1500)
        )
        technical_response = model.generate_content(
            contents=technical_prompt,
            generation_config=genai.types.GenerationConfig(max_output_tokens=1500)
        )

        aptitude_questions = parse_quiz_response(aptitude_response.text)
        technical_questions = parse_quiz_response(technical_response.text)

        session['aptitude_quiz_questions'] = aptitude_questions
        session['technical_quiz_questions'] = technical_questions
        session['current_quiz_type'] = 'aptitude'
        session['current_question_index'] = 0
        session['quiz_score'] = 0

    except Exception as e:
        print(f"Error generating quiz: {str(e)}")
        session['aptitude_quiz_questions'] = []
        session['technical_quiz_questions'] = []
        session['current_quiz_type'] = 'aptitude'
        session['current_question_index'] = 0
        session['quiz_score'] = 0

    return redirect(url_for('quiz'))

@app.route('/submit_quiz_answer', methods=['POST'])
def submit_quiz_answer():
    """Handle quiz answer submission and show result"""
    selected_answer = request.form.get('answer')
    question_index = session.get('current_question_index', 0)
    current_quiz_type = session.get('current_quiz_type', 'aptitude')

    if current_quiz_type == 'aptitude':
        quiz_questions = session.get('aptitude_quiz_questions', [])
    else:
        quiz_questions = session.get('technical_quiz_questions', [])

    if quiz_questions and question_index < len(quiz_questions):
        question = quiz_questions[question_index]
        is_correct = selected_answer == question['correct_answer']
        if is_correct:
            session['quiz_score'] = session.get('quiz_score', 0) + 1

        # Store result for display
        session['last_result'] = {
            'is_correct': is_correct,
            'selected_answer': selected_answer,
            'selected_text': question['options'][ord(selected_answer) - ord('A')] if selected_answer else '',
            'correct_answer': question['correct_answer'],
            'correct_text': question['options'][ord(question['correct_answer']) - ord('A')]
        }

    return redirect(url_for('quiz_result'))

@app.route('/quiz_result')
def quiz_result():
    """Show quiz result for the last answered question"""
    last_result = session.get('last_result')
    if not last_result:
        return redirect(url_for('quiz'))
    
    quiz_score = session.get('quiz_score', 0)
    total_questions = len(session.get('aptitude_quiz_questions', [])) + len(session.get('technical_quiz_questions', []))
    
    return render_template('quiz_result.html',
                           is_correct=last_result['is_correct'],
                           selected_answer=last_result['selected_answer'],
                           selected_text=last_result['selected_text'],
                           correct_answer=last_result['correct_answer'],
                           correct_text=last_result['correct_text'],
                           quiz_score=quiz_score,
                           total_questions=total_questions,
                           active_page='quiz')

@app.route('/next_question', methods=['POST'])
def next_question():
    """Move to the next question"""
    question_index = session.get('current_question_index', 0)
    current_quiz_type = session.get('current_quiz_type', 'aptitude')

    if current_quiz_type == 'aptitude':
        quiz_questions = session.get('aptitude_quiz_questions', [])
    else:
        quiz_questions = session.get('technical_quiz_questions', [])

    # Move to next question
    session['current_question_index'] = question_index + 1

    # If aptitude quiz is complete, switch to technical
    if session['current_question_index'] >= len(quiz_questions) and current_quiz_type == 'aptitude':
        session['current_quiz_type'] = 'technical'
        session['current_question_index'] = 0

    return redirect(url_for('quiz'))

@app.route('/reset_quiz')
def reset_quiz():
    """Reset quiz and go back to subjects"""
    session['show_quiz'] = False
    session['aptitude_quiz_questions'] = []
    session['technical_quiz_questions'] = []
    session['current_question_index'] = 0
    session['quiz_score'] = 0
    session['current_quiz_type'] = 'aptitude'
    return redirect(url_for('subjects'))

def parse_quiz_response(response_text):
    """Parse the Gemini AI response into structured quiz data"""
    questions = []
    lines = response_text.strip().split('\n')
    current_question = None
    options = []
    correct_answer = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for question start (more flexible)
        if line.lower().startswith('question') or (line[0].isdigit() and ':' in line):
            # Save previous question if exists
            if current_question and len(options) >= 4 and correct_answer:
                questions.append({
                    'question': current_question,
                    'options': options[:4],  # Ensure exactly 4 options
                    'correct_answer': correct_answer
                })

            # Start new question
            if ':' in line:
                current_question = line.split(':', 1)[1].strip()
            else:
                current_question = line
            options = []
            correct_answer = None

        # Check for options (more flexible)
        elif any(line.upper().startswith(f'{letter})') for letter in 'ABCD') or any(line.startswith(f'{letter}.') for letter in 'ABCD'):
            option_text = line[3:].strip() if line[1] == ')' else line[3:].strip() if line[1] == '.' else line[2:].strip()
            options.append(option_text)

        # Check for correct answer (more flexible)
        elif 'correct' in line.lower() and 'answer' in line.lower():
            # Extract the letter
            for letter in 'ABCD':
                if letter in line.upper():
                    correct_answer = letter
                    break
            if not correct_answer and ':' in line:
                answer_part = line.split(':', 1)[1].strip().upper()
                if answer_part in 'ABCD':
                    correct_answer = answer_part

    # Add the last question
    if current_question and len(options) >= 4 and correct_answer:
        questions.append({
            'question': current_question,
            'options': options[:4],
            'correct_answer': correct_answer
        })

    # If parsing failed, try a different approach
    if not questions:
        # Split by double newlines or other separators
        blocks = response_text.split('\n\n')
        for block in blocks:
            if 'question' in block.lower() and any(opt in block.upper() for opt in ['A)', 'B)', 'C)', 'D)']):
                lines = block.strip().split('\n')
                q_text = lines[0].strip()
                opts = []
                ans = None
                for l in lines[1:]:
                    l = l.strip()
                    if any(l.upper().startswith(f'{letter})') for letter in 'ABCD'):
                        opts.append(l[3:].strip())
                    elif 'correct' in l.lower():
                        for letter in 'ABCD':
                            if letter in l.upper():
                                ans = letter
                                break
                if len(opts) >= 4 and ans:
                    questions.append({
                        'question': q_text,
                        'options': opts[:4],
                        'correct_answer': ans
                    })

    return questions[:5]  # Return only first 5 questions

if __name__ == '__main__':
    app.run(debug=True)
