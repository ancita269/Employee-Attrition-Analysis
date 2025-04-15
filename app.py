
import pandas as pd
import sqlite3
import hashlib
import streamlit as st
import joblib
import numpy as np

# Load model & preprocessor
model = joblib.load('best_attrition_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Load and clean dataset
def clean_dataset(filepath):
    df = pd.read_csv(filepath)

    # Drop rows with too many missing values
    df = df.dropna(thresh=int(df.shape[1] * 0.8))  # keep rows with at least 80% valid values

    # Optionally, drop or fill NaNs
    df = df.fillna(method='ffill')  # forward fill as a quick fix

    return df

def init_db():
    conn = sqlite3.connect('attrition.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            probability REAL,
            prediction INTEGER,
            timestamp TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(username, probability, prediction):
    import datetime
    conn = sqlite3.connect('attrition.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (username, probability, prediction, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (username, probability, prediction, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_predictions():
    conn = sqlite3.connect('attrition.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions')
    rows = cursor.fetchall()
    conn.close()
    return rows

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def signup():
    st.sidebar.title("Sign Up")
    new_username = st.sidebar.text_input("New Username", key="signup_username")
    new_password = st.sidebar.text_input("New Password", type="password", key="signup_password")
    confirm_password = st.sidebar.text_input("Confirm Password", type="password", key="signup_confirm_password")
    new_role = st.sidebar.radio("Role", ["Employee", "HR"], key="signup_role")

    if st.sidebar.button("Sign Up"):
        if not new_username or not new_password or not confirm_password:
            st.sidebar.error("Please fill all fields.")
        elif new_password != confirm_password:
            st.sidebar.error("Passwords do not match.")
        else:
            conn = sqlite3.connect('attrition.db')
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (new_username,))
            if cursor.fetchone():
                st.sidebar.error("Username already exists.")
            else:
                password_hash = hash_password(new_password)
                cursor.execute('INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)',
                               (new_username, password_hash, new_role))
                conn.commit()
                conn.close()
                st.sidebar.success("Signup successful! Please login.")
                # Clear signup fields
                st.session_state.pop("signup_username", None)
                st.session_state.pop("signup_password", None)
                st.session_state.pop("signup_confirm_password", None)
                st.session_state.pop("signup_role", None)

def login():
    st.sidebar.title("Login")
    user_type = st.sidebar.radio("Who are you?", ["Employee", "HR"], key="login_role")
    username = st.sidebar.text_input("Username", key="login_username")
    password = st.sidebar.text_input("Password", type="password", key="login_password")

    if st.sidebar.button("Login"):
        if not username or not password:
            st.sidebar.error("Please enter username and password.")
        else:
            conn = sqlite3.connect('attrition.db')
            cursor = conn.cursor()
            cursor.execute('SELECT password_hash, role FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()
            conn.close()
            if user and user[0] == hash_password(password) and user[1] == user_type:
                st.session_state['user'] = username
                st.session_state['role'] = user_type
                st.rerun()
            else:
                st.sidebar.error("Invalid username, password, or role.")

def show_prediction_form():
    st.title("Employee Attrition Prediction")

    with st.form("prediction_form"):
        age = st.slider("Age", 18, 60)
        distance = st.slider("Distance From Home", 1, 30)
        income = st.number_input("Monthly Income", 1000, 20000)
        daily_rate = st.number_input("Daily Rate", 100, 1500)
        monthly_rate = st.number_input("Monthly Rate", 1000, 25000)
        hourly_rate = st.number_input("Hourly Rate", 10, 100)
        percent_hike = st.slider("Percent Salary Hike", 0, 50)
        num_companies = st.slider("Number of Companies Worked", 0, 10)
        total_years = st.slider("Total Working Years", 0, 40)
        years_at_company = st.slider("Years at Company", 0, 40)
        years_in_role = st.slider("Years in Current Role", 0, 20)
        years_since_promo = st.slider("Years Since Last Promotion", 0, 20)
        years_with_manager = st.slider("Years With Current Manager", 0, 20)
        training_times = st.slider("Training Times Last Year", 0, 10)

        travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
        education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manager", "Manufacturing Director", "Healthcare Representative", "Human Resources", "Technical Architect", "Other"])
        gender = st.radio("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        overtime = st.radio("OverTime", ["Yes", "No"])

        st.subheader("Ratings")
        job_satisfaction = st.slider("Job Satisfaction", 1, 4)
        relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4)
        environment_satisfaction = st.slider("Environment Satisfaction", 1, 4)
        job_involvement = st.slider("Job Involvement", 1, 4)
        work_life_balance = st.slider("Work Life Balance", 1, 4)
        performance_rating = st.slider("Performance Rating", 1, 4)
        stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3])


        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = pd.DataFrame({
            'Age': [age],
            'BusinessTravel': [travel],
            'Department': [department],
            'DistanceFromHome': [distance],
            'Education': [education],
            'EducationField': [education_field],
            'EnvironmentSatisfaction': [environment_satisfaction],
            'Gender': [gender],
            'HourlyRate': [hourly_rate],
            'JobInvolvement': [job_involvement],
            'JobLevel': [job_level],
            'JobRole': [job_role],
            'JobSatisfaction': [job_satisfaction],
            'MaritalStatus': [marital],
            'MonthlyIncome': [income],
            'MonthlyRate': [monthly_rate],
            'NumCompaniesWorked': [num_companies],
            'OverTime': [overtime],
            'PercentSalaryHike': [percent_hike],
            'PerformanceRating': [performance_rating],
            'RelationshipSatisfaction': [relationship_satisfaction],
            'StockOptionLevel': [stock_option],
            'TotalWorkingYears': [total_years],
            'TrainingTimesLastYear': [training_times],
            'WorkLifeBalance': [work_life_balance],
            'YearsAtCompany': [years_at_company],
            'YearsInCurrentRole': [years_in_role],
            'YearsSinceLastPromotion': [years_since_promo],
            'YearsWithCurrManager': [years_with_manager],
            'DailyRate': [daily_rate]
            })

            transformed = preprocessor.transform(input_data)
            pred = model.predict(transformed)[0]
            prob = model.predict_proba(transformed)[0][1]

            save_prediction(st.session_state['user'], prob, int(pred))
            st.success("Survey submitted!")

def show_hr_dashboard():
    st.title("📊 HR Dashboard - Prediction Results")
    results = get_predictions()
    if results:
        df = pd.DataFrame(results, columns=["ID", "Username", "Probability", "Prediction", "Timestamp"])
        df['Probability'] = df['Probability'].apply(lambda x: f"{x * 100:.4f}%")
        df['Prediction'] = df['Prediction'].map({1: 'High Risk', 0: 'Low Risk'})

        def color_risk(val):
            color = 'red' if val == 'High Risk' else 'green'
            return f'color: {color}'

        styled_df = df.style.applymap(color_risk, subset=['Prediction'])
        st.dataframe(styled_df)

    else:
        st.info("No predictions stored yet.")

def main():
    st.set_page_config(page_title="Employee Attrition System")
    init_db()

    if 'user' not in st.session_state:
        signup()
        login()
    else:
        role = st.session_state['role']
        st.sidebar.success(f"Logged in as {st.session_state['user']} ({role})")
        if st.sidebar.button("Logout"):
            del st.session_state['user']
            del st.session_state['role']
            st.rerun()

        if role == "Employee":
            show_prediction_form()
        elif role == "HR":
            tab1, tab2 = st.tabs(["🏠 Home", "📈 View Predictions"])
            with tab1:
                st.write("Welcome HR! You can review predictions in the other tab.")
            with tab2:
                show_hr_dashboard()

if __name__ == "__main__":
    main()
