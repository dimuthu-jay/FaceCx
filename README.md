
# FaceCx - Real-Time Customer Experience Feedback Analysis System

**FaceCx** is an AI-powered customer experience feedback analysis system that uses facial features such as emotion and gender to provide real-time insights during customer interactions. This system is designed to assist businesses and advertisers in understanding how customers respond to advertisements, products, and services—beyond the limitations of traditional surveys and interviews.

---

## 🚀 Project Summary

Traditional feedback systems suffer from low response rates, biases, and limited insights. FaceCx addresses these issues by using machine learning (CNN) models to analyze facial expressions and demographic features in real-time. It provides businesses with actionable, analytical reports to support marketing and product decisions.

---

## 🧠 Features

- 🎭 Emotion Detection (Positive / Neutral / Negative)
- 🚻 Gender Detection (Male / Female)
- 🧑‍💻 Real-Time Feedback via Webcam
- 📊 Analytical Dashboard for Businesses
- 🔎 High Accuracy with FER-2013 and UTKFace Datasets

## 🗂️ Project Structure

```text
FaceCx/
├── model/
│   ├── emotion.ipynb
│   └── emotion_model.keras
│   ├── gender-original-retrain.ipynb
│   └── gender-original-retrain.keras
│ 
├── static/
│   ├── css/
│   └── img/
│   └── js/
│   
├── templates/
│   ├── About.html
│   ├── AnalyticalAnalysis.html
│   ├── CaptureStart.html
│   ├── Contact.html
│   ├── EmotionResults.html
│   ├── FemaleEmotionResults.html
│   ├── GenderResults.html
│   ├── Home.html
│   ├── MaleEmotionResults.html
│   ├── Results.html
│   ├── UserAccount.html
│   ├── Welcome.html
│   ├── login.html
│   ├── navbar.html
│   └── register.html
│ 
├── ui/
│   └── app.py
│
├── login-firebase-adminsdk.json
├── requirements.txt
└── README.md
```   
---

## ⚙️ How to Run

### 1️⃣ Installation

git clone https://github.com/dimuthuJayathuga/FaceCx.git
cd FaceCx
pip install -r requirements.txt

### 2️⃣ Create a Virtual Environment

-Create venv (Linux/macOS)
python3 -m venv venv

-On Windows
python -m venv venv

### 3️⃣ Activate the Environment

-On macOS/Linux
source venv/bin/activate

-On Windows
venv\Scripts\activate

### 4️⃣ Install Dependencies

pip install -r requirements.txt

### 5️⃣ Run Real-Time Feedback System

python app.py


---

## 📈 Model Performance

| Model           | Accuracy |
|----------------|----------|
| Emotion Model  | 90.2%    |
| Gender Model   | 84.7%    |

---

## 📚 Reference Datasets

- FER-2013 Emotion Dataset
- UTKFace Dataset for Age/Gender

---

## 👤 Author

**Dimuthu Jayathunga**  
BEng (Hons) Software Engineering – University of Westminster  
📧 Email: dimuthuchanaka06@gmail.com  
🔗 GitHub: ([github.com/dimuthuJayathunga](https://github.com/dimuthuJayathunga))

---

## 📌 Future Enhancements

- Age prediction integration
- Mobile app for on-the-go analysis
- Integration with marketing dashboards (Google Ads, Meta Ads)

---

## 📄 License

This project is open-source under the MIT License.
