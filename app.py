import streamlit as st
import pandas as pd
import numpy as np
import sklearn.externals
import joblib
from PIL import Image

def main():
    st.sidebar.markdown('Model Tahminleri')
    image = Image.open("Diabetes2.png")
    st.image(image, caption="", width=750, use_colomn_width=False, output_format="PNG")

    yas = st.slider("Yaş", 20, 80)
    bmi = st.slider("BMI", 15, 40)
    bel = st.slider("Bel Çevresi", 70, 125)
    apg_aks = st.slider("APG / AKŞ", 0, 200)
    hba1c = st.slider("HBA1C", 0.0, 10.0)
    ogtt75g = st.slider("75 gr OGTT", 0, 1)
    ogtt = st.slider("OGTT", 0, 250)

    df = pd.DataFrame(data = {'yas': [yas],
    'bmi': [bmi],
    'bel': [bel],
    'apg_aks': [apg_aks],
    'hba1c': [hba1c],
    'ogtt75g': [ogtt75g],
    'ogtt': [ogtt]})

    tahmin = ""

    list = []

    if tahmin == "":
        gaussian_naive_bayes = joblib.load('GaussianNB.pkl')
        pred_nb = gaussian_naive_bayes.predict(df)
        list.append(pred_nb)
        logistic_regression = joblib.load('LogisticRegression.pkl')
        pred_lr = logistic_regression.predict(df)
        list.append(pred_lr)
        knn = joblib.load('KNN.pkl')
        pred_knn = knn.predict(df)
        list.append(pred_knn)
        decision_tree = joblib.load('DecisionTree.pkl')
        pred_dt = decision_tree.predict(df)
        list.append(pred_dt)
        rf = joblib.load('RandomForest.pkl')
        pred_rf = rf.predict(df)
        list.append(pred_rf)

    model = ['Gaussian NB', 'Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest']

    for i in range(5):
        if list[i][0] == 0:
            tanı = 'Değerleri kontrol ediniz'
            st.sidebar.success(model[i] + ": " + tanı)
        elif list[i][0] == 1:
            tanı = 'Normal'
            st.sidebar.success(model[i] + ": " + tanı)
        elif list[i][0] == 2:
            tanı = 'Prediyabet / Bozulmuş Glukoz Toleransı'
            st.sidebar.success(model[i] + ": " + tanı)
        elif list[i][0] == 3:
            tanı = 'Prediyabet / Bozulmuş Açlık Glukozu'
            st.sidebar.success(model[i] + ": " + tanı)
        elif list[i][0] == 4:
            tanı = 'Prediyabet / Bozulmuş glukoz Toleransı + Bozulmuş Açlık Glukozu'
            st.sidebar.success(model[i] + ": " + tanı)
        elif list[i][0] == 5:
            tanı = 'Diyabet'
            st.sidebar.success(model[i] + ": " + tanı)

if __name__ == '__main__':
    main()