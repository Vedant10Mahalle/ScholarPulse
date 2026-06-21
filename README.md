# ScholarPulse — Intelligent Student Monitoring & Support System

![ScholarPulse](https://img.shields.io/badge/Status-Active-success.svg) ![Python](https://img.shields.io/badge/Python-3.x-blue.svg) ![Flask](https://img.shields.io/badge/Flask-Backend-black.svg) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)

**ScholarPulse** is an advanced, AI-driven web application designed to monitor student academic performance and behavioral well-being. Built with Python, Flask, and Scikit-Learn, it leverages a **Random Forest Classifier** to predict student dropout probabilities and subject-level pass rates. The system emphasizes **Explainable AI (XAI)**, providing educators with a transparent, interactive "What-If" simulator to understand exactly how different factors contribute to a student's risk profile.

## 🌟 Key Features

### 1. Faculty Mentor Dashboard & ML Inference Engine
* **Predictive Risk Analytics:** Real-time calculation of Dropout Probability, Academic Risk, and Behavioral Risk.
* **Explainable AI (SHAP-inspired):** Visualizes feature importance, showing exactly which metrics (e.g., Attendance, Stress, SGPA) are increasing or decreasing a student's risk.
* **Live What-If Simulator:** Interactive sliders allow mentors to tweak a student's metrics (e.g., "What if their attendance drops to 60%?") and watch the 100-tree Random Forest model re-evaluate the risk live.
* **Academic Data Management:** Secure, teacher-only interface for uploading subject marks and semester performance.

### 2. Gamified Student Portal
* **Daily Check-ins:** Students log their mood, stress levels, missed classes, and energy.
* **Streak System:** Encourages consistent check-ins with gamified streaks and motivational messages.
* **Academic Summary:** Students can view their performance, including an ML-generated "Pass Probability" for each subject based on their CIE (Continuous Internal Evaluation) marks.

### 3. Parent Portal
* **Risk Tracking:** Provides parents with a simplified view of their child's predictive risk analytics and academic performance.
* **Mentor Collaboration:** Allows parents to directly request formal evaluation meetings with the assigned faculty mentor.

### 4. Admin Control Panel
* **Live Model Retraining:** Admins can trigger live retraining of the global risk model or append new CSV data to update the subject-performance sub-model.
* **Bulk Data Management:** Upload students, teachers, and parents via CSV. 
* **Danger Zone:** Secure data clearing options for system resets.

## 🛠️ Technology Stack

* **Backend Framework:** Python, Flask, Werkzeug
* **Machine Learning Engine:** Scikit-Learn (Random Forest Classifier), Pandas, NumPy
* **Frontend:** HTML5, Vanilla CSS (Premium Glassmorphism & Pastel Aesthetic), Vanilla JavaScript
* **Data Visualization:** Chart.js (Trend analysis & SHAP impact graphs)
* **Data Persistence:** CSV-based lightweight database architecture (located in `/data`), designed for easy scaling to SQL databases like PostgreSQL or SQLite.

## 🚀 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/project-v.git
   cd project-v
   ```

2. **Create a virtual environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Data & Train Models**
   If the models are not pre-trained, run the training scripts to generate the `.pkl` files:
   ```bash
   python train_model.py
   python train_subject_model.py
   ```

5. **Run the Application**
   ```bash
   python app.py
   ```
   The system will be accessible at `http://127.0.0.1:5000/`.

## 📁 Repository Structure

```text
📦 project-v
 ┣ 📂 data/                   # CSV databases (students, teachers, marks, etc.)
 ┣ 📂 models/                 # Serialized ML models (risk_model.pkl, etc.)
 ┣ 📂 static/
 ┃ ┣ 📜 style.css             # Core design system & UI tokens
 ┃ ┗ 📜 script.js             # Frontend interactivity
 ┣ 📂 templates/              # HTML views (admin, teacher, student, parent)
 ┣ 📜 app.py                  # Core Flask backend & routing
 ┣ 📜 train_model.py          # Global risk model training pipeline
 ┣ 📜 train_subject_model.py  # Subject pass probability model training
 ┣ 📜 clean.py                # Utility script for sanitizing templates
 ┗ 📜 requirements.txt        # Python dependencies
```

## 🧠 Machine Learning Architecture

The system utilizes an ensemble learning approach via Scikit-Learn's `RandomForestClassifier`. 
* **Global Model (`train_model.py`):** Uses 7 core features (Attendance, SGPA, Stress, Missed Days, Mood, Energy, Check-in Streak) to classify students into risk tiers.
* **Subject Model (`train_subject_model.py`):** Uses Continuous Internal Evaluation (CIE) marks to predict the probability of passing the Semester End Examination (SEE).

---
*Built for the future of intelligent education management.*

## ☁️ Deployment (Decoupled Vercel + Render Architecture)

Because this application uses server-side Jinja templates alongside lightweight CSV databases, you can host the **Frontend on Vercel** (for speed and serverless execution) and the **Backend on Render** (for stateful data persistence).

### 1. Render Deployment (Backend & Data Storage)
Render hosts the master database (CSV files) and persists them using a persistent disk volume.
1. Connect your repository to a new **Web Service** on Render.
2. In the Render Dashboard under **Disk**, add a **Persistent Disk**:
   - **Mount Path:** `/app/data`
   - **Size:** `1 GB`
3. Under **Environment**, configure these variables:
   - `INTERNAL_API_KEY` = `your_secure_secret_token` (a secure random string)
   - `SMTP_EMAIL` = `your_email` (optional, for parent emails)
   - `SMTP_PASSWORD` = `your_password` (optional, Gmail App Password)

### 2. Vercel Deployment (Frontend / Serverless UI)
Vercel executes the serverless Flask app and retrieves/saves database data dynamically from the Render backend.
1. Deploy the same repository to **Vercel**.
2. Under project settings, configure these **Environment Variables**:
   - `RENDER_BACKEND_URL` = `https://your-render-service.onrender.com`
   - `INTERNAL_API_KEY` = `your_secure_secret_token` (must match the Render token)
   - `SMTP_EMAIL` and `SMTP_PASSWORD` = (same email credentials)
