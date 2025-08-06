# Resume Classification using Machine Learning

A machine learning project that classifies resumes into predefined job categories using natural language processing (NLP) techniques. It helps streamline the recruitment process by automatically analyzing and tagging resume data.

---

## Features

* Automatic classification of resumes into roles (React JS Developer, Workday, PeopleSoft, SQL Developer)
* Resume parsing using Named Entity Recognition (NER) in a separate web app
* Text preprocessing and cleaning
* TF-IDF feature extraction
* Model training and evaluation using scikit-learn
* Performance metrics for evaluation (accuracy, classification report, etc.)

---

## Tech Stack

* Python 3.x
* pandas
* NumPy
* scikit-learn
* spaCy (NER)
* Flask / Streamlit
* Jupyter Notebook

---

## Project Structure

```
├── Resume_Classification.ipynb   # Main notebook for ML classification
├── data/                         # Folder containing resume text files
├── models/                       # Saved models
├── web_app/                      # Folder for NER-based resume parsing app
├── README.md                     # Project documentation
```

---

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/resume-classification.git
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:

   ```bash
   jupyter notebook Resume_Classification.ipynb
   ```
4. Run all cells to preprocess the data, train the model, and see the results.

---

