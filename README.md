# 🌲 Green Vision Project | Machine Learning

## 📘 Overview  
**Green Vision** is a machine learning project designed to predict **forest cover types** based on cartographic variables from the **UCI Forest Cover Type dataset**.  
The project implements a complete data pipeline—from preprocessing to model deployment—using modern tools such as **FastAPI**, **MongoDB**, and **AWS S3**, along with a **responsive frontend interface** for real-time predictions.

---

## 🚀 Features
- 🔍 **Data Analysis & Preprocessing:**  
  Analyzed the *UCI Forest Cover Type dataset* using Python and implemented complete data preprocessing, training, and testing pipelines.

- 🗃️ **Database Integration:**  
  Integrated **MongoDB** as a storage system for raw data used during model training.

- ☁️ **Cloud Storage:**  
  Stored trained ML models and prediction outputs in **AWS S3 bucket** for reliable access and scalability.

- ⚙️ **Model Serving via FastAPI:**  
  Developed a **RESTful API using FastAPI** to serve the trained model and handle real-time inference requests.

- 💻 **Interactive Frontend:**  
  Built a **responsive frontend interface (HTML, CSS, JavaScript)** to interact with the FastAPI backend and visualize live forest cover predictions.

---

## 🧠 Tech Stack
| Category | Technologies |
|-----------|---------------|
| **Programming Languages** | Python, JavaScript |
| **Frameworks & Libraries** | FastAPI, Scikit-learn, Pandas, NumPy, Matplotlib |
| **Database** | MongoDB |
| **Cloud Services** | AWS S3 |
| **Frontend** | HTML, CSS, JavaScript |

---

## 🧩 Workflow
1. Store raw data in **MongoDB**  
2. Load and preprocess data using **Python (NumPy, Pandas)**  
3. Train ML model and save serialized model  
4. Serve the model using **FastAPI** as a RESTful API  
5. Build a responsive frontend to send requests to FastAPI  
6. Retrieve and display **real-time predictions**

---
