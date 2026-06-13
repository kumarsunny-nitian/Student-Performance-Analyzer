import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Load dataset
df = pd.read_csv("data/student_data.csv")

# Load model
model = pickle.load(
    open("models/linear_regression_marks.pkl", "rb")
)

placement_model = pickle.load(
    open("models/logistic_regression_placement.pkl", "rb")
)

# Page title
st.title("🎓 Student Performance Analyzer")

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Visualizations",
        "Marks Prediction",
        "Placement Prediction",
        "About Project"
    ]
)

# ==========================
# Dashboard Page
# ==========================
if page == "Dashboard":

    total_students = len(df)
    avg_marks = round(df['Marks'].mean(), 2)
    placement_rate = round(df['Placed'].mean() * 100, 2)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Students", total_students)

    with col2:
        st.metric("Average Marks", avg_marks)

    with col3:
        st.metric("Placement Rate (%)", placement_rate)

    st.subheader("Dataset Preview")
    st.dataframe(df)

# ==========================
# Visualization Page
# ==========================
elif page == "Visualizations":

    st.subheader("Marks Distribution")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(df["Marks"], bins=10)

    ax.set_title("Distribution of Marks")
    ax.set_xlabel("Marks")
    ax.set_ylabel("Count")

    st.pyplot(fig)

# ==========================
# Marks Prediction Page
# ==========================
elif page == "Marks Prediction":

    st.subheader("🎯 Predict Student Marks")

    study_hours = st.number_input(
        "Study Hours",
        min_value=0,
        max_value=15,
        value=5
    )

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0,
        max_value=100,
        value=80
    )

    assignments = st.number_input(
        "Assignments Completed",
        min_value=0,
        max_value=15,
        value=8
    )

    iq = st.number_input(
        "IQ",
        min_value=50,
        max_value=150,
        value=110
    )

    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.5
    )

    if st.button("Predict Marks"):

        prediction = model.predict([[
            study_hours,
            attendance,
            assignments,
            iq,
            cgpa
        ]])

        st.success(
            f"Predicted Marks: {prediction[0]:.2f}"
        )





        # ==========================
# Placement Prediction Page
# ==========================
elif page == "Placement Prediction":

    st.subheader("💼 Placement Prediction")

    study_hours = st.number_input(
        "Study Hours",
        min_value=0,
        max_value=15,
        value=5,
        key="placement_study_hours"
    )

    attendance = st.number_input(
        "Attendance (%)",
        min_value=0,
        max_value=100,
        value=80,
        key="placement_attendance"
    )

    assignments = st.number_input(
        "Assignments Completed",
        min_value=0,
        max_value=15,
        value=8,
        key="placement_assignments"
    )

    

    iq = st.number_input(
        "IQ",
        min_value=50,
        max_value=150,
        value=110,
        key="placement_iq"
    )

    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.5,
        key="placement_cgpa"
    )

    marks = st.number_input(
    "Marks",
    min_value=0,
    max_value=100,
    value=75,
    key="placement_marks"
)

    if st.button("Predict Placement"):

        result = placement_model.predict([[
            study_hours,
            attendance,
            assignments,
            iq,
            cgpa,
            marks
        ]])

        if result[0] == 1:
            st.success("✅ Student is likely to be Placed")
        else:
            st.error("❌ Student is likely to be Not Placed")



        


# PASTE HERE 👇
elif page == "About Project":

    st.header("📘 About Project")

    st.write("""
    ### Student Performance Analyzer

    This project analyzes student academic performance using Machine Learning.

    ### Technologies Used
    - Python
    - Pandas
    - NumPy
    - Matplotlib
    - Scikit-Learn
    - Streamlit

    ### Machine Learning Models
    - Linear Regression (Marks Prediction)
    - Logistic Regression (Placement Prediction)
    - KNN Classification
    - K-Means Clustering

    ### Dataset Features
    - Study Hours
    - Attendance
    - Assignments
    - IQ
    - CGPA
    - Marks
    - Placement Status

    ### Project Developed By
    Sunny Kumar
    """)
