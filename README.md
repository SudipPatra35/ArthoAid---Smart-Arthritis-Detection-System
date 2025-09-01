# 🩺 ArthoAid – Smart Arthritis Detection System  

ArthoAid is an AI-powered arthritis detection system designed to help in the early diagnosis of **Rheumatoid Arthritis (RA)** and **Osteoarthritis (OA)**.  
It combines **Machine Learning (Random Forest)** with **Deep Learning (CNN)** to:  
- Classify patients as **Healthy, RA, or OA** using clinical data  
- Detect and **grade the severity of OA** using knee X-ray images  

---

## 📖 Table of Contents  
- [About the Project](#-about-the-project)  
- [Problem Statement](#-problem-statement)  
- [Motivation](#-motivation)  
- [Objectives](#-objectives)  
- [System Architecture](#-system-architecture)  
- [Technologies Used](#-technologies-used)  
- [Workflow](#-workflow)  
- [Results](#-results)  
- [Future Enhancements](#-future-enhancements)  
- [Installation](#-installation)  
- [Contributors](#-contributors)  
- [License](#-license)  

---

## 🚑 About the Project  
Arthritis affects millions worldwide and is a leading cause of disability.  
ArthoAid provides a **fast, accurate, and accessible** solution, especially for rural areas where medical specialists are limited.  

- **ML model** → Detects type of arthritis (Healthy / RA / OA)  
- **CNN model** → Grades OA severity from X-rays  
- **Web app (Flask + HTML/CSS/JS)** → Easy-to-use interface for patients and doctors  

---

## ❗ Problem Statement  
- Manual diagnosis of arthritis is **time-consuming and error-prone**  
- Rural/remote areas lack **specialists and proper diagnostic tools**  
- X-ray reading varies among doctors → **inconsistent results**  
- No integrated system for **type detection + severity grading**  

---

## 🎯 Motivation  
- Increasing number of arthritis patients worldwide  
- Need for **early detection** to prevent severe disability  
- Reduce misdiagnosis and save doctor/patient time  
- Provide a **low-cost and accessible** healthcare solution  

---

## ✅ Objectives  
1. Classify patients as **Healthy, RA, or OA** using clinical data  
2. For OA patients, analyze **knee X-rays** to grade severity (0–4)  
3. Provide a **simple and affordable web-based platform**  
4. Reduce diagnostic errors and make healthcare more accessible  

---

## 🏗 System Architecture  
**Layers:**  
1. **Frontend:** HTML, CSS, JS (patient registration, X-ray upload, result display)  
2. **Backend:** Flask (Python) – Handles routes, connects models & DB  
3. **Database:** MySQL (patients, reports, results)  
4. **Machine Learning:** Random Forest (classification)  
5. **Deep Learning:** CNN (X-ray OA grading with TensorFlow/Keras)  

---

## 🛠 Technologies Used  
- **ML/DL:** Random Forest, CNN, TensorFlow, Scikit-learn  
- **Image Processing:** OpenCV, Pillow  
- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** Flask (Python)  
- **Database:** MySQL  
- **Deployment:** Flask + Gunicorn (future: Cloud/Heroku)  

---

## 🔄 Workflow  
1. Patient registers/logs in  
2. Enters clinical details (age, symptoms, RF, ESR, etc.)  
3. Random Forest model predicts: Healthy / RA / OA  
4. If RA → Recommend doctor consultation  
5. If OA → Upload knee X-ray  
6. CNN analyzes X-ray → Outputs severity grade (0–4)  
7. Results shown on dashboard with recommendations  

---

## 📊 Results  

### 🔹 Random Forest (Arthritis Classification)  
- **Accuracy:** 94%  
- **F1-Score:** 0.94  

### 🔹 CNN (OA Severity Grading)  
- **Accuracy:** 88%  
- **Weighted F1-Score:** 0.88  

Both models show strong real-world performance, reducing misdiagnosis and ensuring reliable predictions.  

---

## 🚀 Future Enhancements  
- RA severity detection (needs larger dataset)  
- Advanced AI models (Transformers, multimodal learning)  
- Doctor consultation (chat/video) inside the app  
- Mobile application for rural healthcare workers  
- Offline-friendly version for poor connectivity areas  
- Cloud deployment (AWS/Heroku) for scalability  
- Integration with hospital records & wearable devices  

---

## ⚙️ Installation  

### 1️⃣ Clone the repo & Install Dependencies  
      #Clone the repository
      git clone https://github.com/your-username/ArthoAid.git
      cd ArthoAid
        
      #Install dependencies
      pip install -r requirements.txt

---

