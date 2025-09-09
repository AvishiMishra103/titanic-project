import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("titanic.csv")

df = load_data()

# App Title
st.title("🚢 Titanic Dataset - EDA App")

# Show dataset preview
st.subheader("📋 Dataset Preview")
st.write(df.head())

# Dataset Info
st.subheader("🔎 Dataset Information")
st.write(f"Shape of dataset: {df.shape}")
st.write("Missing Values:")
st.write(df.isnull().sum())

# Sidebar filters
st.sidebar.header("Filter Data")
pclass_filter = st.sidebar.multiselect("Select Passenger Class:", df['Pclass'].unique())
sex_filter = st.sidebar.multiselect("Select Gender:", df['Sex'].unique())

filtered_df = df.copy()
if pclass_filter:
    filtered_df = filtered_df[filtered_df['Pclass'].isin(pclass_filter)]
if sex_filter:
    filtered_df = filtered_df[filtered_df['Sex'].isin(sex_filter)]

st.subheader("📊 Filtered Data Preview")
st.write(filtered_df.head())

# Survival Rate Analysis
st.subheader("🧑‍🤝‍🧑 Survival Rate")
survival_rate = filtered_df['Survived'].value_counts(normalize=True) * 100
st.bar_chart(survival_rate)

# Age Distribution
st.subheader("🎂 Age Distribution")
fig, ax = plt.subplots()
sns.histplot(filtered_df['Age'].dropna(), bins=30, kde=True, ax=ax)
st.pyplot(fig)

# Survival by Gender
st.subheader("🚻 Survival by Gender")
fig, ax = plt.subplots()
sns.countplot(data=filtered_df, x="Sex", hue="Survived", ax=ax)
st.pyplot(fig)

# Survival by Passenger Class
st.subheader("🎟️ Survival by Passenger Class")
fig, ax = plt.subplots()
sns.countplot(data=filtered_df, x="Pclass", hue="Survived", ax=ax)
st.pyplot(fig)

# Correlation Heatmap
st.subheader("🔥 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(filtered_df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.success("✅ EDA Complete! Use sidebar filters to explore data interactively.")
