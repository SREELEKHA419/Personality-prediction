
#  Personality Prediction from Resume using NLP & Machine Learning 📄🔍

This project analyzes a resume (CV) using **Natural Language Processing (NLP)** and **Machine Learning** to predict a candidate’s **personality traits** and suggest **career insights**. It's designed to help recruiters or individuals gain a deeper understanding of personality profiles based solely on textual content.

---

##  Project Structure

```

📁 OUTPUTS/                        # Folder to store predictions or reports
📄 personality\_prediction\_app.py   # Main Python script for processing and prediction
📄 models.pkl                      # Trained ML model for personality prediction
📄 vectorizer.pkl                  # TF-IDF or Count Vectorizer for text transformation
📄 Sample Emily Johnson-resume.pdf # Sample resume for testing

````

---

##  How to Run the Project

###  Prerequisites

- Python 3.8 or higher
- Libraries: scikit-learn, pandas, numpy, PyPDF2 (or pdfplumber), joblib

###  Install Required Packages

```bash
pip install scikit-learn pandas numpy PyPDF2 joblib
````

> If you're using `pdfplumber` instead of `PyPDF2`, install it using:

```bash
pip install pdfplumber
```

###  Run the Script

```bash
python personality_prediction_app.py
```

Follow the prompts to upload/select a resume. The script will display predicted personality traits and potential career directions.

---

##  Features

* Extracts and processes text from PDF resumes.
* Uses a trained machine learning model to predict Big Five personality traits:

  * Openness
  * Conscientiousness
  * Extraversion
  * Agreeableness
  * Neuroticism
* Offers **career suggestions** based on the trait profile.

---

##  Model Details

* **Model**: Stored in `models.pkl`
* **Vectorizer**: Stored in `vectorizer.pkl` (used for transforming resume text)
* **Algorithm**: Could be Logistic Regression, Random Forest, or another classifier trained on labeled personality datasets.

---

##  Sample Output

```text
Predicted Personality Traits:
- Openness: High
- Conscientiousness: Medium
- Extraversion: Low
- Agreeableness: High
- Neuroticism: Low

Suggested Career Domains:
- Research Analyst
- Content Writer
- Software Developer
```

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

##  Acknowledgements

* This project uses open-source tools for NLP and ML.
* Inspired by real-world HRTech solutions for talent profiling.

---
##  Contact

For queries or contributions, feel free to connect via [GitHub](https://github.com/yourusername) or email.

---

 **To run this as a web app**, simply use Streamlit:

```bash
streamlit run personality_prediction_app.py


