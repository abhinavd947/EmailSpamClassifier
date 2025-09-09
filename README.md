# Email Spam Classifier 🚀

A machine learning project that classifies emails as **Spam** or **Not Spam** using Python, Flask, and TF-IDF + Naive Bayes. The project includes a web interface for real-time predictions and demonstrates practical application of ML and web development skills.

---

## Features

- Train a spam detection model using a sample email dataset
- Real-time spam prediction via a web interface built with Flask
- User-friendly input form with dynamic prediction results
- Model persistence using `pickle` for efficient loading

---

## Tech Stack

- Python
- Flask
- scikit-learn (TF-IDF, Naive Bayes)
- HTML/CSS (for web interface)
- pandas (data preprocessing)

---

## Getting Started

1. **Clone the repository**
git clone https://github.com/yourusername/email-spam-classifier.git
cd email-spam-classifier

2. **Install dependencies**
pip install -r requirements.txt

3. **Train the model**
python train_model.py

4. **Run the web app**
python app.py
Visit `http://127.0.0.1:5000` in your browser.

---

## Dataset

The project includes a small sample dataset (`data/sms_spam_sample.csv`) to demonstrate functionality. For more accurate results, replace it with a larger real-world email dataset.

---

## Usage

- Paste any email content in the web interface
- Click **Check**
- Receive instant feedback: `Spam` or `Not Spam`

