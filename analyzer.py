# main code
import streamlit as st
import spacy
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.matcher import Matcher
import PyPDF2
import plotly.graph_objects as go
import re

# --- Streamlit Caching ---
@st.cache_resource
def load_spacy_model():
    """Load and cache the SpaCy model."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading SpaCy model 'en_core_web_sm'...")
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()
@st.cache_resource
def load_keybert():
    return KeyBERT(model=embedding_model)

kw_model = load_keybert()

# --- Page Configuration ---
st.set_page_config(page_title='AI Profile & Resume Analyzer', page_icon='🚀', layout='wide')

# --- Widen Sidebar CSS ---
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 375px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('🚀 100% Free AI Profile & Resume Analyzer')

# --- Helper Functions ---

def extract_text_from_pdf(file):
    text = ''
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text

# --- Keyword Lists ---

POWER_WORDS = [
    'develop', 'build', 'deploy', 'developed', 'managed', 'led', 'achieved', 
    'optimized', 'created', 'launched', 'improved', 'increased', 'negotiated',
    'implemented', 'automated', 'designed', 'built', 'authored', 'mentored', 
    'streamlined', 'trained', 'presented', 'resolved', 'manage', 'lead', 
    'achieve', 'optimize', 'create', 'launch', 'implement', 'automate', 
    'design', 'author', 'mentor', 'train', 'present', 'resolve', 'analyze', 
    'spearhead', 'direct'
]

# NEW: List of tech jargon to prevent NER confusion
TECH_JARGON = {
    'python', 'java', 'sql', 'c++', 'c#', 'javascript', 'react', 'reactjs', 'angular',
    'vue', 'node.js', 'nodejs', 'php', 'swift', 'kotlin', 'ruby', 'perl', 'go',
    'rust', 'scala', 'typescript', 'html', 'css', 'api', 'ml', 'ai', 'nlp',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'github', 'gitlab',
    'linux', 'unix', 'windows', 'macos', 'android', 'ios', 'firebase', 'mongodb',
    'mysql', 'postgresql', 'sqlite', 'django', 'flask', 'spring', '.net',
    'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy'
}


# ---  Keyword Extractor ---
def extract_keywords(text):

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1,2),
        stop_words='english',
        top_n=15
    )

    return [(kw[0].lower(), kw[1]) for kw in keywords]
# ---  Entity Recognition (NER) Function ---
def extract_entities(text):
    """Uses SpaCy's NER model, but filters out common tech jargon."""
    doc = nlp(text)
    entities = {
        "Companies / Organizations": set(),
        "Locations": set()
    }
    
    for ent in doc.ents:
        # Check if the entity is in our tech jargon list
        if ent.text.lower() not in TECH_JARGON:
            if ent.label_ == "ORG":
                entities["Companies / Organizations"].add(ent.text)
            elif ent.label_ == "GPE":
                entities["Locations"].add(ent.text)
            
    # Convert sets to lists for display
    return {
        "Companies / Organizations": list(entities["Companies / Organizations"]),
        "Locations": list(entities["Locations"])
    }
def semantic_similarity(resume_text, job_description):

    resume_embedding = embedding_model.encode(
        resume_text,
        convert_to_numpy=True
    )

    jd_embedding = embedding_model.encode(
        job_description,
        convert_to_numpy=True
    )

    similarity = cosine_similarity(
        [resume_embedding],
        [jd_embedding]
    )[0][0]

    return round(similarity * 100, 2)


def normalize_text(text):

    doc = nlp(text)

    tokens = []

    for token in doc:
        if not token.is_stop and not token.is_punct:
            tokens.append(token.lemma_.lower())

    return " ".join(tokens)

def keyword_match(resume_text, keyword_scores):

    resume_text_normalized = normalize_text(resume_text)

    found_keywords = []
    missing_keywords = []

    matched_weight = 0
    total_weight = 0

    for keyword, weight in keyword_scores:

        total_weight += weight

        normalized_keyword = normalize_text(keyword)

        if normalized_keyword in resume_text_normalized:
            found_keywords.append(keyword)
            matched_weight += weight
        else:
            missing_keywords.append(keyword)

    if total_weight == 0:
        return 0, [], []

    keyword_score = int((matched_weight / total_weight) * 100)

    return keyword_score, found_keywords, missing_keywords

def get_linkedin_feedback(about_text, skills_text):
    suggestions = []
    score = 0
    
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', about_text).lower()
    words = cleaned_text.split()
    
    if len(words) > 50: score += 30
    else: suggestions.append(f"* **About Section is too short:** Yours is {len(words)} words. Aim for 50+.")
    
    action_word_count = sum(1 for word in words if word in POWER_WORDS)
    if action_word_count >= 3: score += 30
    else: suggestions.append(f"* **Add Action Verbs:** Your 'About' section has {action_word_count} action verbs. Aim for 3 or more.")

    skills_list = [skill.strip().lower() for skill in skills_text.split(',')]
    skills_count = len(skills_list) if skills_text else 0

    if skills_count >= 5: score += 20
    if skills_count >= 10: score += 20
    if skills_count < 5: suggestions.append(f"* **Add more skills:** You listed {skills_count} skills. Aim for at least 5-10.")
        
    return suggestions, score

def get_resume_feedback(resume_text):
    suggestions = []
    score = 0
    
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', resume_text).lower()
    words = cleaned_text.split()
    
    if len(words) > 150: score += 20
    else: suggestions.append(f"* **Resume seems short:** Yours is {len(words)} words.")

    action_word_count = sum(1 for word in words if word in POWER_WORDS)
    
    if action_word_count >= 3: score += 30
    if action_word_count >= 5: score += 20
    if action_word_count < 3: suggestions.append(f"* **Add action verbs:** You used {action_word_count} action verbs. Aim for at least 3.")
    
    if 'education' in cleaned_text: score += 15
    else: suggestions.append("* **'Education' section not found.**")
        
    if 'experience' in cleaned_text: score += 15
    else: suggestions.append("* **'Experience' section not found.**")
        
    return suggestions, score

def create_score_chart(score, title):
    if score == 100:
        fig = go.Figure(go.Pie(values=[100], labels=['Perfect Score!'], marker_colors=['#00C853']))
    else:
        fig = go.Figure(go.Pie(
            values=[score, 100 - score],
            labels=['Your Score', 'Points to Improve'],
            marker_colors=['#00C853', '#FFD600'],
            hole=0.4, textinfo='percent'
        ))
    
    fig.update_layout(
        title_text=f'<b>{title}</b>',
        annotations=[dict(text=f'<b>{score}/100</b>', x=0.5, y=0.5, font_size=20, showarrow=False)],
        showlegend=False, height=300, margin=dict(t=50, b=0, l=0, r=0)
    )
    return fig

def get_summary_line(score, context="profile"):
    if score == 100: st.success(f"🎉 **Perfect! 100/100. Your {context} is in excellent shape!**")
    elif score >= 80: st.success(f"🚀 **Very Good! {score}/100. Your {context} is strong.**")
    elif score >= 60: st.warning(f"👍 **Good Start! {score}/100. Follow the tips below to improve.**")
    else: st.error(f"🌱 **Needs Work. {score}/100. Let's get this score up!**")

# --- Main App Logic ---

st.sidebar.title('Navigation')
app_mode = st.sidebar.radio(
    "Choose an Analyzer",
    ("LinkedIn Analyzer", "Resume Analyzer")
)

# --- LinkedIn Analyzer Page ---
if app_mode == "LinkedIn Analyzer":
    st.sidebar.header("LinkedIn Inputs")
    about_input = st.sidebar.text_area("Paste your 'About' Section", height=150)
    skills_input = st.sidebar.text_input("Paste your 'Skills' (comma-separated)", "e.g., Python, SQL, Streamlit")
    
    if st.sidebar.button("Analyze LinkedIn Profile"):
        if not about_input or not skills_input:
            st.sidebar.error("Please paste both your 'About' and 'Skills' sections.")
        else:
            st.header('LinkedIn Profile Analysis')
            with st.spinner('Analyzing your profile...'):
                suggestions, score = get_linkedin_feedback(about_input, skills_input)
                
                get_summary_line(score, "profile")
                fig = create_score_chart(score, 'LinkedIn Profile Score')
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader('Your Action Plan')
                if suggestions:
                    st.markdown('\n'.join(suggestions))
                else:
                    st.success("**✅ Your profile summary looks strong!**")

# --- Resume Analyzer Page ---
elif app_mode == "Resume Analyzer":
    st.sidebar.header("Resume Inputs")
    resume_file = st.sidebar.file_uploader('Upload your resume (PDF)', type='pdf', key="resume_uploader")
    job_desc_input = st.sidebar.text_area("Paste the Job Description", height=150)
    
    if st.sidebar.button("Analyze Resume for Job"):
        if resume_file and job_desc_input:
            st.header('Resume & Job Match Analysis')
            with st.spinner('Analyzing...'):
                resume_text = extract_text_from_pdf(resume_file)
                if resume_text:
                    # --- 1. General Resume Analysis ---
                    st.subheader("General Resume Score")
                    suggestions, score = get_resume_feedback(resume_text)
                    
                    get_summary_line(score, "resume")
                    fig = create_score_chart(score, 'General Resume Score')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader('General Action Plan')
                    if suggestions:
                        st.markdown('\n'.join(suggestions))
                    
                    st.divider()
                    
                    # --- 2. Key Entities Section (UPDATED) ---
                    st.subheader("Key Entities Detected (NER Scan)")
                    with st.spinner("Scanning for entities..."):
                        entities = extract_entities(resume_text)
                        
                        if not entities["Companies / Organizations"] and not entities["Locations"]:
                            st.info("No major entities (company names or locations) were automatically detected.")
                        else:
                            if entities["Companies / Organizations"]:
                                st.markdown("**Skills, Companies & Organizations:**")
                                st.markdown(f"```\n{', '.join(entities['Companies / Organizations'])}\n```")
                            if entities["Locations"]:
                                st.markdown("**Locations:**")
                                st.markdown(f"```\n{', '.join(entities['Locations'])}\n```")

                    st.divider()
                    
                    # --- 3. Job Match Analysis ---
                    st.subheader("AI ATS Match Analysis")
                    keyword_scores = extract_keywords(job_desc_input)
                    keyword_score, found_keywords, missing_keywords = keyword_match(resume_text,keyword_scores)
                    semantic_score = semantic_similarity(resume_text,job_desc_input)
                    final_score = int(0.6 * semantic_score + 0.4 * keyword_score)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Semantic Similarity", f"{semantic_score}%")
                    col2.metric("Keyword Coverage", f"{keyword_score}%")
                    col3.metric("Overall ATS Score", f"{final_score}%")
                    st.progress(final_score / 100)
                    get_summary_line(final_score, "job match")
                    fig_match = create_score_chart(final_score,"ATS Match Score")
                    st.plotly_chart(fig_match, use_container_width=True)
                    # ---------- Skill Gap Dashboard ----------
                    st.subheader("📊 Skill Gap Dashboard")
                    found_keywords = sorted(set(found_keywords))
                    missing_keywords = sorted(set(missing_keywords))
                    # Remove tiny/unhelpful words
                    found_keywords = [k for k in found_keywords if len(k) > 2]
                    missing_keywords = [k for k in missing_keywords if len(k) > 2]
                    left, right = st.columns(2)
                    with left:
                        st.success("✅ Matched Skills")
                        if found_keywords:
                            c1, c2 = st.columns(2)
                            for i, skill in enumerate(found_keywords):
                                if i % 2 == 0:
                                    c1.markdown(f":green-badge[{skill}]")
                                else:
                                    c2.markdown(f":green-badge[{skill}]")
                    with right:
                        st.error("❌ Missing Skills")
                        if missing_keywords:
                            c1, c2 = st.columns(2)
                            for i, skill in enumerate(missing_keywords[:10]):
                                if i % 2 == 0:
                                    c1.markdown(f":red-badge[{skill}]")
                                else:
                                    c2.markdown(f":red-badge[{skill}]")
                else:
                    st.error("Could not read text from the uploaded PDF.")
        else:
            st.sidebar.error("Please upload a resume AND paste a job description.")