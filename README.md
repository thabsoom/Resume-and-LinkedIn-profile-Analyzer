# 🚀 AI Resume & LinkedIn Profile Analyzer

An AI-powered web application built with **Streamlit** that analyzes resumes and LinkedIn profiles, provides personalized improvement suggestions, and evaluates resume-job compatibility using **Natural Language Processing (NLP)** and **Machine Learning**.

## 📌 Features

### 🔹 LinkedIn Profile Analyzer

* Analyze your LinkedIn **About** section.
* Evaluate the quality of your listed skills.
* Score your profile out of 100.
* Receive actionable suggestions to improve your professional profile.
* Detect the use of strong action verbs.

### 🔹 Resume Analyzer

* Upload your resume in **PDF** format.
* Extract resume text automatically.
* Analyze resume quality and structure.
* Detect:

  * Companies & Organizations
  * Locations
* Generate an overall resume quality score.

### 🔹 AI ATS (Applicant Tracking System) Match Analysis

* Compare your resume with a job description.
* Measure:

  * Semantic Similarity
  * Keyword Coverage
  * Overall ATS Match Score
* Highlight:

  * ✅ Matched Skills
  * ❌ Missing Skills
* Provide recommendations to improve your chances of passing ATS screening.

---

## 🛠️ Technologies Used

* Python
* Streamlit
* SpaCy (Named Entity Recognition)
* Sentence Transformers
* KeyBERT
* Scikit-learn
* PyPDF2
* Plotly
* NumPy

---

## 📂 Project Structure

```
Resume-and-LinkedIn-profile-Analyzer/
│
├── analyzer.py         
├── requirements.txt     
├── README.md
└── assets/            
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/thabsoom/Resume-and-LinkedIn-profile-Analyzer.git
```

```bash
cd AI-Resume-LinkedIn-Analyzer
```

### 2. Create a virtual environment (Optional)

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the SpaCy model

```bash
python -m spacy download en_core_web_sm
```

### 5. Run the application

```bash
streamlit run analyzer.py
```

---

## 📖 How It Works

### LinkedIn Analyzer

1. Paste your **LinkedIn About** section.
2. Enter your skills (comma-separated).
3. Click **Analyze LinkedIn Profile**.
4. View:

   * Profile Score
   * Improvement Suggestions
   * Visual Score Dashboard

---

### Resume Analyzer

1. Upload your resume (PDF).
2. Paste the target Job Description.
3. Click **Analyze Resume for Job**.
4. View:

   * Resume Quality Score
   * Named Entity Recognition (Companies & Locations)
   * Semantic Similarity
   * Keyword Match Score
   * ATS Match Score
   * Skill Gap Dashboard

---

## 🧠 AI Techniques Used

### Keyword Extraction

Uses **KeyBERT** to extract the most important keywords from the job description.

### Semantic Similarity

Uses **Sentence Transformers (all-MiniLM-L6-v2)** to compare resume content with the job description using embedding similarity.

### Named Entity Recognition (NER)

Uses **SpaCy** to identify:

* Organizations
* Companies
* Locations

### ATS Scoring

The final ATS score combines:

```
Final Score =
60% Semantic Similarity
+
40% Keyword Match
```

---

## 📊 Output

The application generates:

* Resume Score
* LinkedIn Profile Score
* ATS Match Score
* Semantic Similarity
* Keyword Coverage
* Skill Gap Dashboard
* Improvement Suggestions
* Interactive Pie Charts

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push to GitHub

```bash
git push origin feature-name
```

5. Open a Pull Request

