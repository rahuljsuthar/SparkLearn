# SparkLearn: Ignite Your Learning
An AI-Powered Web Application for Interactive Learning in Placement Preparation

## 🌐 Live Demo
[Try SparkLearn Live](https://sparklearn.onrender.com/)

SparkLearn is a Flask-based web application that harnesses the power of Google Gemini AI to deliver automated tutoring on various computer science and aptitude topics. This application generates questions, provides detailed explanations, and allows users to learn interactively.

### Requirements
- Python 3.6 or higher
- Google Gemini API key

### Installation
- Clone the repository: git clone https://github.com/absterjr/SparkLearn.git
- Install the required packages: pip install -r requirements.txt
- Set up the Google Gemini API key: https://aistudio.google.com/app/apikey
- Create a `.env` file in the root directory and add your API key:
  ```
  GEMINI_API_KEY=your_actual_api_key_here
  ```
- Run the application: python app.py

![alt text](https://github.com/absterjr/SparkLearn/blob/main/SparkLearn.png?raw=true)

Type Yes to continue on to the next topic.
#### Or you can use this command or your command line:
  - ##### (Linux, make sure you have python and git installed):
```bash
   git clone https://github.com/absterjr/SparkLearn.git && pip install -r requirements.txt && python3 env.py && python app.py
```
  - ##### (Windows, make sure you have python and git installed):
```powershell
  git clone https://github.com/absterjr/SparkLearn.git && pip install -r requirements.txt && python env.py && python app.py
```

### Usage
How It Works
- The script teaches machine learning topics.
- It generates questions and explanations.
- Users can ask questions and interact with the AI tutor.
- The script progresses to the next subtopic when the user demonstrates understanding.

You can see an example below.

![alt text](https://github.com/absterjr/SparkLearn/blob/main/Doubt.png?raw=true)


### Contributing
Contributions to SparkLearn are welcome! If you find a bug or have an idea for a new feature, please open an issue or submit a pull request.


#### Note: replace "YOUR_API_KEY" with your actual OpenAI API key.
The number of topics can be increased by simply adding the topic names into the list defined at the beginning.
