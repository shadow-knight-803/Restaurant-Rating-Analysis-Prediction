# 🍽️ Restaurant Rating Prediction

## 📌 Project Overview
The **Restaurant Rating Prediction** project is a machine learning application that predicts restaurant ratings based on various features such as location, cuisine type, pricing, and amenities. The project is built using **Python, scikit-learn, and Flask** for the web interface.

## 🏗️ Project Structure
The repository follows a modular design with clear separation of concerns:

```
├── src
│   ├── components      # Data ingestion, transformation, and model training modules
│   ├── pipelines       # Training and prediction pipelines
│   ├── utils           # Helper functions, logging, and exception handling
│   ├── __init__.py
│
├── artifacts           # Stores trained models and processed data
├── templates           # HTML templates for the web interface
├── static              # CSS and JavaScript files
├── app.py              # Flask web application
├── requirements.txt    # List of dependencies
├── README.md           # Project documentation
└── notebook            # Jupyter Notebooks for EDA and model training
```

## 📊 Dataset Overview
The dataset includes information about restaurants, such as:
- **Geographic coordinates** (latitude, longitude)
- **Country code & city**
- **Cuisine type**
- **Average cost for two**
- **Table booking & delivery availability**
- **Number of votes & rating text**
- **User reviews & aggregate ratings**

## ⚙️ Features & Implementation
### **1️⃣ Data Ingestion & Transformation**
- Loads the restaurant dataset and splits it into training and test sets.
- Handles **missing values** using imputation.
- Encodes **categorical features** and scales **numerical features**.
- Applies **feature engineering** techniques to enhance data quality.

### **2️⃣ Model Training**
- Trains multiple regression models:
  - **Linear Regression**
  - **Lasso & Ridge Regression**
  - **ElasticNet**
  - **Decision Tree & Random Forest**
- Evaluates models based on **R² score**.
- Saves the best-performing model for predictions.

### **3️⃣ Prediction Pipeline**
- Captures user inputs from the web interface.
- Applies the same preprocessing steps as training.
- Uses the trained model to predict restaurant ratings.

### **4️⃣ Web Application (Flask UI)**
- Users can input restaurant details via a **clean and interactive interface**.
- The application returns a **predicted rating** based on user inputs.
- Implements **error handling and logging** for debugging.

## 🚀 Installation & Setup
### **Step 1: Clone the Repository**
```bash
git clone https://github.com/yourusername/restaurant-rating-prediction.git
cd restaurant-rating-prediction
```

### **Step 2: Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 4: Run the Flask Application**
```bash
python app.py
```
Visit `http://127.0.0.1:5000/` in your browser to interact with the web application.

## 🛠️ Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, scikit-learn, Flask, Matplotlib, Seaborn
- **Web Development:** HTML, CSS, Bootstrap
- **Deployment:** Flask, Docker (optional), Cloud Platforms (Future scope)

## 📈 Results & Performance
- The best-performing model achieved an **R² score of ~98%** on test data.
- The application can predict restaurant ratings with high accuracy.
- Modular architecture ensures **scalability and maintainability**.

## 🔥 Future Enhancements
- **Deploying the application on AWS/GCP.**
- **Enhancing the dataset with real-time data collection.**
- **Integrating NLP for sentiment analysis of user reviews.**
- **Adding advanced ML techniques like XGBoost and Neural Networks.**

## 📌 Contributing
Contributions are welcome! Feel free to open issues and submit pull requests.

## 📄 License
This project is licensed under the **MIT License**.

---

📢 **Follow & Connect:** If you liked this project, give it a ⭐ on GitHub and connect with me on [LinkedIn](https://www.linkedin.com/in/aditya-kumar-arya/)!

🚀 **Happy Coding!** 🎯
