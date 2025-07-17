# Full Project: Personality Prediction System via CV Analysis
import fitz  # PyMuPDF
import re
import pandas as pd
import nltk
import string
import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from collections import Counter

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Clean text returns both string and list of filtered words
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = re.findall(r'\b\w+\b', text)
    filtered_words = [word for word in words if word not in stop_words and word.isalpha()]
    return ' '.join(filtered_words), filtered_words

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def train_model():
    data = {
        'cv_text': [
            'Software engineer skilled in Java Python SQL',
            'Team leader experienced in sales and communication',
            'Creative designer with interest in storytelling and media',
            'Detail-oriented analyst experienced in Excel Tableau'
        ],
        'openness': [1, 0, 1, 0],
        'conscientiousness': [1, 0, 0, 1],
        'extraversion': [0, 1, 1, 0],
        'agreeableness': [0, 1, 1, 1],
        'neuroticism': [0, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    df['cv_text'] = df['cv_text'].apply(lambda x: clean_text(x)[0])

    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['cv_text'])

    models = {}
    for trait in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
        model = LogisticRegression()
        model.fit(X, df[trait])
        models[trait] = model

    joblib.dump(models, 'models.pkl')
    joblib.dump(tfidf, 'vectorizer.pkl')
    print("Models trained and saved.")

def predict_traits(text):
    cleaned_text_str, _ = clean_text(text)
    vectorizer = joblib.load('vectorizer.pkl')
    models = joblib.load('models.pkl')
    vect_text = vectorizer.transform([cleaned_text_str])

    predictions = {}
    for trait, model in models.items():
        prob = model.predict_proba(vect_text)[0][1]
        predictions[trait] = round(prob, 2)
    return predictions

def get_top_frequent_words(words, n=10):
    counter = Counter(words)
    return counter.most_common(n)

CAREER_OPPORTUNITIES = {
    'extroversion': ['Sales Manager', 'Public Relations Specialist', 'Event Coordinator', 'Marketing Professional'],
    'introversion': ['Writer', 'Research Scientist', 'Software Developer', 'Data Analyst'],
    'agreeableness': ['Counselor', 'Nurse', 'Social Worker', 'Teacher'],
    'conscientiousness': ['Project Manager', 'Accountant', 'Engineer', 'Quality Analyst'],
    'neuroticism': ['Therapist', 'Artist', 'Writer', 'Journalist'],
    'openness': ['Designer', 'Inventor', 'Researcher', 'Musician']
}

DEVELOPMENT_SUGGESTIONS = {
    'extroversion': 'Try balancing your social time with some quiet reflection to recharge.',
    'introversion': 'Practice engaging in small group conversations to build confidence.',
    'agreeableness': 'Set boundaries to avoid being taken advantage of while helping others.',
    'conscientiousness': 'Remember to allow flexibility and avoid perfectionism.',
    'neuroticism': 'Learn stress management techniques like mindfulness and deep breathing.',
    'openness': 'Explore new hobbies and challenge yourself with novel experiences.'
}

# Skill keywords by category (extend as needed)
SKILL_CATEGORIES = {
    "Technical Skills": ['python', 'java', 'c++', 'sql', 'excel', 'tableau', 'javascript', 'html', 'css', 'git', 'docker'],
    "Leadership Skills": ['leader', 'management', 'team lead', 'supervisor', 'coordinator', 'manager'],
    "Communication Skills": ['communication', 'presentation', 'negotiation', 'public speaking', 'writing', 'collaboration', 'interpersonal'],
    "Creative Skills": ['creative', 'design', 'storytelling', 'media', 'innovation', 'artistic'],
    "Analytical Skills": ['analytical', 'problem-solving', 'data analysis', 'research', 'critical thinking'],
}

def score_skill_categories(words):
    scores = {}
    words_set = set(words)
    for category, keywords in SKILL_CATEGORIES.items():
        count = sum(word in words_set for word in keywords)
        scores[category] = count
    return scores

def get_trait_description(trait, score):
    if score > 0.7:
        return "High"
    elif score > 0.4:
        return "Moderate"
    else:
        return "Low"

def suggest_careers(traits):
    careers = []
    for trait in traits:
        careers.extend(CAREER_OPPORTUNITIES.get(trait, []))
    return list(set(careers))

def suggest_development(traits):
    suggestions = []
    for trait in traits:
        suggestion = DEVELOPMENT_SUGGESTIONS.get(trait)
        if suggestion:
            suggestions.append(f"{trait.capitalize()}: {suggestion}")
    return suggestions

def main():
    st.title("🧠 Personality Prediction from CV and Career Insights")
    st.write("Upload your CV (PDF) to predict your Big Five personality traits, analyze skills, and get career suggestions.")

    # Sidebar options for interactivity
    st.sidebar.header("Options")
    show_frequent_words = st.sidebar.checkbox("Show Frequent Keywords", value=True)
    show_skill_scores = st.sidebar.checkbox("Show Skill Category Scores", value=True)
    show_career_suggestions = st.sidebar.checkbox("Show Career Suggestions", value=True)
    show_dev_suggestions = st.sidebar.checkbox("Show Personality Development Tips", value=True)

    uploaded_file = st.file_uploader("Choose a PDF resume", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner('Extracting and analyzing your resume...'):
            extracted_text = extract_text_from_pdf(uploaded_file)
            predictions = predict_traits(extracted_text)
            cleaned_text_str, words_list = clean_text(extracted_text)
            frequent_words = get_top_frequent_words(words_list)
            skill_scores = score_skill_categories(words_list)

        st.success("Analysis Completed!")

        st.subheader("Predicted Personality Traits")
        traits = list(predictions.keys())
        scores = list(predictions.values())
        df_chart = pd.DataFrame({"Trait": traits, "Score": scores})
        st.bar_chart(df_chart.set_index("Trait"))

        for trait, score in predictions.items():
            st.write(f"**{trait.title()}**: {score} → {get_trait_description(trait, score)}")

        dominant_traits = [trait for trait, score in predictions.items() if score > 0.4]

        if show_frequent_words:
            with st.expander("Top 10 Frequent Keywords in Resume"):
                for word, count in frequent_words:
                    st.write(f"{word} ({count} times)")

        if show_skill_scores:
            with st.expander("Skill Category Scores"):
                for category, score in skill_scores.items():
                    st.write(f"{category}: {score}")

        if dominant_traits:
            if show_career_suggestions:
                with st.expander("Career Suggestions Based on Personality"):
                    careers = suggest_careers(dominant_traits)
                    if careers:
                        for career in careers:
                            st.write(f"- {career}")
                    else:
                        st.write("No career suggestions available.")

            if show_dev_suggestions:
                with st.expander("Personality Development Suggestions"):
                    tips = suggest_development(dominant_traits)
                    if tips:
                        for tip in tips:
                            st.write(f"- {tip}")
                    else:
                        st.write("No development suggestions available.")
        else:
            st.warning("No strong personality traits detected for career or development suggestions.")

if __name__ == "__main__":
    # Uncomment to train model once
    # train_model()
    main()
