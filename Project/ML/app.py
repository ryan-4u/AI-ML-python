import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Page Config ---
st.set_page_config(page_title="Pokémon Data Explorer", layout="wide")

st.title("📊 Pokémon Data Cleaning, Modeling & Visualization")

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df['Type 2'] = df['Type 2'].fillna("None")
    df['Name'] = df['Name'].str.encode('ascii', 'ignore').str.decode('ascii')
    df.drop_duplicates(inplace=True)
    return df

df = load_data()

# --- Data Cleaning ---
label_encoders = {}
for col in ['Type 1', 'Type 2', 'Legendary']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# --- Sidebar: Data Filters ---
st.sidebar.header("Filter Pokémon")
type1 = st.sidebar.multiselect("Type 1", options=df['Type 1'].unique())
type2 = st.sidebar.multiselect("Type 2", options=df['Type 2'].unique())
generation = st.sidebar.multiselect("Generation", options=df['Generation'].unique())

filtered_df = df.copy()
if type1:
    filtered_df = filtered_df[filtered_df['Type 1'].isin(type1)]
if type2:
    filtered_df = filtered_df[filtered_df['Type 2'].isin(type2)]
if generation:
    filtered_df = filtered_df[filtered_df['Generation'].isin(generation)]

st.subheader("Filtered Dataset")
st.dataframe(filtered_df)

# --- Train/Test Split ---
X = df[['Type 1', 'Type 2', 'Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']]
y = df['Legendary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.metric("Model Accuracy", f"{acc*100:.2f}%")

# --- Classification Report ---
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# --- Advanced Visualizations ---
st.subheader("Interactive Visualizations")

tab1, tab2, tab3, tab4 = st.tabs(["Scatter Plot", "Stat Distribution", "Feature Importances", "Heatmap"])

with tab1:
    fig = px.scatter(filtered_df, x="Attack", y="Defense", color="Legendary",
                     hover_name="Name", size="Total", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.histogram(filtered_df, x="Total", nbins=30, color="Legendary", marginal="box", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values("Importance", ascending=False)
    fig = px.bar(importance_df, x="Importance", y="Feature", orientation='h', title="Feature Importances", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    corr = filtered_df[['Total','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']].corr()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# --- Save Cleaned Data ---
if st.button("Save Cleaned Data"):
    df.to_csv("cleaned_data.csv", index=False)
    st.success("Cleaned data saved as cleaned_data.csv")
