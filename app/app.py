"""
Streamlit app to load a saved model and run prediction interactively.
Run: streamlit run app/app.py -- --model-path outputs/gb_run/model.pkl
"""
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import argparse, sys

def load_model(path):
    import joblib
    return joblib.load(path)

def main(model_path):
    st.title("Credit Default Predictor — Demo")
    st.markdown("Upload a CSV row or use sample values to predict default (1) / no default (0).")

    st.sidebar.header("Model")
    st.sidebar.write(f"Model path: {model_path}")
    model = load_model(model_path)

    uploaded = st.file_uploader("Upload CSV with a single row (features only)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        X = df.values
    else:
        st.write("Or use manual inputs (sample):")
        # Simple sample inputs for 23 features - you can adapt to your column ordering
        sample = {}
        cols = ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE"] + [f"PAY_{i}" for i in range(6)] + [f"BILL_AMT{i}" for i in range(1,7)] + [f"PAY_AMT{i}" for i in range(1,7)]
        for c in cols[:8]:
            sample[c] = st.number_input(c, value=0)
        # if full features not provided, we will pad zeros for simplicity
        X = np.array([list(sample.values()) + [0]*(23 - len(sample))])
    if st.button("Predict"):
        preds = model.predict(X)
        st.write("Prediction:", int(preds[0]))
        st.write("0 = No default, 1 = Default")

if __name__ == "__main__":
    # allow passing arg via Streamlit flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="outputs/gb_run/model.pkl")
    args, unknown = parser.parse_known_args()
    main(args.model_path)
