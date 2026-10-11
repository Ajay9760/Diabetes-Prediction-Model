# Diabetes Prediction Model 🩺
Link- https://diabetes-prediction-model-xv.streamlit.app/

An end-to-end Machine Learning web application designed to predict the likelihood of diabetes in patients based on key diagnostic measurements. Built with **Python**, **Scikit-Learn**, and **Streamlit** (or Flask, depending on your `app.py` implementation), this project includes trained model serialization and preprocessing pipelines.

---

## 📂 Project Structure

```text
├── diabetes (1).csv         # Dataset used for training and analysis
├── diabetes_model.pkl       # Pre-trained machine learning model
├── scaler.pkl               # Fitted standard scaler for feature normalization
├── app.py                   # Main web application interface script
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

📊 Dataset FeaturesThe model analyzes standard health metrics (derived typically from datasets like the Pima Indians Diabetes Database):Pregnancies: Number of times pregnant   Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test   Blood Pressure: Diastolic blood pressure (mm Hg)   Skin Thickness: Triceps skin fold thickness (mm)   Insulin: 2-Hour serum insulin ($\mu U/ml$)   BMI: Body mass index ($\text{weight in kg}/(\text{height in m})^2$)   Diabetes Pedigree Function: Diabetes pedigree function (genetic score)Age: Age (years)🚀 Getting StartedPrerequisitesMake sure you have Python 3.8 or higher installed on your system.InstallationClone the repository:Bashgit clone [https://github.com/Ajay9760/Diabetes-Prediction-Model.git](https://github.com/Ajay9760/Diabetes-Prediction-Model.git)
cd Diabetes-Prediction-Model
Install the required dependencies:Bashpip install -r requirements.txt
🖥️ Running the ApplicationStart the web application by running the app.py script:Bashpython app.py
(Note: If the application uses Streamlit, run streamlit run app.py instead).Open your browser and navigate to the local URL provided in your terminal (usually http://localhost:5000 or http://localhost:8501) to interact with the prediction interface.🛠️ Tech Stack & LibrariesLanguage: PythonMachine Learning: Scikit-Learn, NumPy, PandasModel Persistence: PickleWeb Framework: Streamlit / Flask👤 AuthorA.Ajay KumarGitHub: @Ajay9760
***

Feel free to let me know if you'd like to adjust any details, such as clarifying the exact web framework used in `app.py`!
