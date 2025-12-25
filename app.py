import streamlit as st
import pickle
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Training History Viewer",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("📊 Model Training History")

# ---------------- LOAD PKL FILE ----------------
@st.cache_resource
def load_history():
    with open("training_history.pkl", "rb") as file:
        history = pickle.load(file)
    return history

history = load_history()

# ---------------- CHECK CONTENT ----------------
st.subheader("Available Metrics")
st.write(list(history.keys()))

# ---------------- PLOT FUNCTION ----------------
def plot_metric(metric, val_metric):
    plt.figure()
    plt.plot(history[metric], label=metric)
    plt.plot(history[val_metric], label=val_metric)
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    st.pyplot(plt)

# ---------------- DISPLAY GRAPHS ----------------
if "accuracy" in history and "val_accuracy" in history:
    st.subheader("📈 Accuracy")
    plot_metric("accuracy", "val_accuracy")

if "loss" in history and "val_loss" in history:
    st.subheader("📉 Loss")
    plot_metric("loss", "val_loss")

st.markdown("---")
st.caption("Training history loaded from training_history.pkl")
