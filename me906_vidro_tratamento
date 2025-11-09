import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# ----------------------------
# Funções utilitárias
# ----------------------------
def sens_esp(y_true, y_pred, labels):
    """
    Retorna (sensibilidade_macro, especificidade_macro)
    labels: array-like com a ordem das classes usada na matriz de confusão
    """
    sens = recall_score(y_true, y_pred, labels=labels, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    especificidades = []
    for i in range(len(labels)):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)
        esp = TN / (TN + FP) if (TN + FP) > 0 else np.nan
        especificidades.append(esp)

    return sens, np.mean(especificidades)


def plot_confusion(cm, labels, title):
    """Plotly heatmap para matriz de confusão"""
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=[str(l) for l in labels],
        y=[str(l) for l in labels],
        colorscale="Blues",
        showscale=True
    )
    fig.update_layout(title=title, xaxis_title="Predito", yaxis_title="Real")
    return fig


def cross_validate_model(model, X_data, y_series, cv):
    """Retorna listas (accs, sens, esp) por fold para um dado modelo e dados."""
    accs, sens_list, esp_list = [], [], []
    for train_idx, val_idx in cv.split(X_data, y_series):
        X_tr, X_val = X_data[train_idx], X_data[val_idx]
        y_tr, y_val = y_series.iloc[train_idx], y_series.iloc[val_idx]

        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)

        accs.append(accuracy_score(y_val, pred))
        s, e = sens_esp(y_val, pred, labels_sorted)
        sens_list.append(s)
        esp_list.append(e)

    return accs, sens_list, esp_list

# ----------------------------
# Carregar dados
# ----------------------------
st.title("Modelos de Classificação: LDA, QDA, Naive Bayes Multinomial e KNN")


caminho = "https://raw.githubusercontent.com/abibernardo/repositorio/main/dataset_vidro_me906.csv"
df = pd.read_csv(caminho)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]


# ----------------------------
# Pré-processamento
# ----------------------------
X = df.drop(columns=["glass_group"])
y = df["glass_group"].astype(str)  # garantir string
labels_sorted = np.sort(np.unique(y))

# Escalas:
scaler_std = StandardScaler()   # para LDA/QDA/KNN
X_std = scaler_std.fit_transform(X)

scaler_minmax = MinMaxScaler()  # para MultinomialNB (não-negativo)
X_nb = scaler_minmax.fit_transform(X)

# Hold-out final (mantendo estratificação)
X_train_std, X_test_std, y_train, y_test = train_test_split(
    X_std, y, test_size=0.2, stratify=y, random_state=42
)
# para NB: mesma partição por semente (usamos MinMax data)
X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(
    X_nb, y, test_size=0.2, stratify=y, random_state=42
)

# Cross-validation object (usado nas validações cruzadas)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ----------------------------
# Pré-compute CV para LDA, QDA, NB (KNN será computado dinamicamente na aba)
# ----------------------------
with st.spinner("Executando validação cruzada para LDA, QDA e Naive Bayes (pode levar alguns segundos)..."):
    # LDA
    lda_model = LinearDiscriminantAnalysis()
    lda_acc, lda_sens, lda_esp = cross_validate_model(lda_model, X_train_std, y_train, cv)

    # QDA
    qda_model = QuadraticDiscriminantAnalysis()
    qda_acc, qda_sens, qda_esp = cross_validate_model(qda_model, X_train_std, y_train, cv)

    # Multinomial NB (usa X_train_nb)
    nb_model = MultinomialNB()
    nb_acc, nb_sens, nb_esp = cross_validate_model(nb_model, X_train_nb, y_train_nb, cv)

# ----------------------------
# Abas por modelo
# ----------------------------
tab_df, tab_lda, tab_qda, tab_nb, tab_knn, tab_comp = st.tabs([
    "Apresentação do Dataset", "LDA", "QDA", "Naive Bayes Multinomial", "KNN", "Comparação"
])

# ----------------------------
# ABA APRESENTAÇÃO
# ----------------------------
with tab_df:
    st.write("O objetivo deste estudo é comparar diferentes modelos para a classificação de tipos de vidro. São 10 variáveis explicativas contínuas, e uma variável resposta com 3 categorias.")
    st.divider()
    st.dataframe(df)
    fig = px.bar(df, x='glass_group')
    st.plotly_chart(fig)
# ----------------------------
# ABA LDA
# ----------------------------
with tab_lda:
    st.header("Linear Discriminant Analysis (LDA)")

    st.subheader("Validação Cruzada (5 folds) — LDA")
    lda_df = pd.DataFrame({
        "Fold": list(range(1, len(lda_acc) + 1)),
        "Accuracy": lda_acc,
        "Sensitivity (macro)": lda_sens,
        "Specificity (macro)": lda_esp
    })
    st.dataframe(lda_df)
    st.subheader("Média dos Folds")
    st.metric(label="Acurácia média (treinamento)", value=f"{np.mean(lda_acc):.3f}")
    st.metric(label="Sensibilidade", value=f"{np.mean(lda_sens):.3f}")
    st.metric(label="Especificidade", value=f"{np.mean(lda_esp):.3f}")

    # Ajuste final no treino completo e avaliação no teste
    lda_final = LinearDiscriminantAnalysis()
    lda_final.fit(X_train_std, y_train)
    y_pred_lda_test = lda_final.predict(X_test_std)

    st.subheader("Avaliação no conjunto de teste — LDA")
    sens_t, esp_t = sens_esp(y_test, y_pred_lda_test, labels_sorted)
    st.metric(label="Acurácia", value=f"{accuracy_score(y_test, y_pred_lda_test):.3f}")
    st.metric(label="Sensibilidade", value=f"{np.mean(sens_t):.3f}")
    st.metric(label="Especificidade", value=f"{np.mean(esp_t):.3f}")

    # Confusion matrix (test)
    cm_lda = confusion_matrix(y_test, y_pred_lda_test, labels=labels_sorted)
    st.plotly_chart(plot_confusion(cm_lda, labels_sorted, "Confusion Matrix — LDA (test)"))

    # Projeção LD1 x LD2 (toda a base)
    st.subheader("Projeção LDA — LD1 × LD2 (toda a base)")
    lda_proj = lda_final.transform(X_std)
    # proteger caso não haja 2 componentes
    ld2 = lda_proj[:, 1] if lda_proj.shape[1] > 1 else np.zeros(lda_proj.shape[0])
    proj_df = pd.DataFrame({"LD1": lda_proj[:, 0], "LD2": ld2, "glass_group": y.values})
    fig = px.scatter(proj_df, x="LD1", y="LD2", color="glass_group", title="LDA Projection (LD1 × LD2)")
    st.plotly_chart(fig)

# ----------------------------
# ABA QDA
# ----------------------------
with tab_qda:
    st.header("Quadratic Discriminant Analysis (QDA)")

    st.subheader("Validação Cruzada (5 folds) — QDA")
    qda_df = pd.DataFrame({
        "Fold": list(range(1, len(qda_acc) + 1)),
        "Accuracy": qda_acc,
        "Sensitivity (macro)": qda_sens,
        "Specificity (macro)": qda_esp
    })
    st.dataframe(qda_df)
    st.subheader("Média dos Folds")
    st.metric(label="Acurácia média (treinamento)", value=f"{np.mean(qda_acc):.3f}")
    st.metric(label="Sensibilidade", value=f"{np.mean(qda_sens):.3f}")
    st.metric(label="Especificidade", value=f"{np.mean(qda_esp):.3f}")

    # Ajuste final e avaliação no teste
    qda_final = QuadraticDiscriminantAnalysis()
    qda_final.fit(X_train_std, y_train)
    y_pred_qda_test = qda_final.predict(X_test_std)

    st.subheader("Avaliação no conjunto de teste — QDA")
    sens_t, esp_t = sens_esp(y_test, y_pred_qda_test, labels_sorted)
    st.metric(label="Acurácia", value=f"{accuracy_score(y_test, y_pred_qda_test):.3f}")
    st.metric(label="Sensibilidade", value=f"{np.mean(sens_t):.3f}")
    st.metric(label="Especificidade", value=f"{np.mean(esp_t):.3f}")

    # Confusion matrix (test)
    cm_qda = confusion_matrix(y_test, y_pred_qda_test, labels=labels_sorted)
    st.plotly_chart(plot_confusion(cm_qda, labels_sorted, "Confusion Matrix — QDA (test)"))



# ----------------------------
# ABA Naive Bayes Multinomial
# ----------------------------
with tab_nb:
    st.header("Naive Bayes Multinomial")

    st.subheader("Validação Cruzada (5 folds) — Multinomial NB")
    nb_df = pd.DataFrame({
        "Fold": list(range(1, len(nb_acc) + 1)),
        "Accuracy": nb_acc,
        "Sensitivity (macro)": nb_sens,
        "Specificity (macro)": nb_esp
    })
    st.dataframe(nb_df)
    st.subheader("Média dos Folds")
    st.metric(label="Acurácia média (treinamento)", value=f"{np.mean(nb_acc):.3f}")
    st.metric(label="Sensibilidade", value=f"{np.mean(nb_sens):.3f}")
    st.metric(label="Especificidade", value=f"{np.mean(nb_esp):.3f}")

    # Ajuste final e avaliação no teste (NB usa X_nb)
    nb_final = MultinomialNB()
    nb_final.fit(X_train_nb, y_train_nb)
    y_pred_nb_test = nb_final.predict(X_test_nb)

    st.subheader("Avaliação no conjunto de teste — Multinomial NB")
    sens_t, esp_t = sens_esp(y_test_nb, y_pred_nb_test, labels_sorted)
    st.metric(label="Acurácia", value=f"{accuracy_score(y_test_nb, y_pred_nb_test):.3f}")
    st.metric(label="Sensibilidade", value=f"{np.mean(sens_t):.3f}")
    st.metric(label="Especificidade", value=f"{np.mean(esp_t):.3f}")

    cm_nb = confusion_matrix(y_test_nb, y_pred_nb_test, labels=labels_sorted)
    st.plotly_chart(plot_confusion(cm_nb, labels_sorted, "Confusion Matrix — Multinomial NB (test)"))

# ----------------------------
# ABA KNN
# ----------------------------
with tab_knn:
    st.header("K-Nearest Neighbors (KNN)")

    st.subheader("Escolha do parâmetro k")
    k = st.slider("Número de vizinhos (k)", min_value=1, max_value=20, value=5, step=1)

    st.subheader("Validação Cruzada (5 folds) — KNN")
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_acc, knn_sens, knn_esp = cross_validate_model(knn_model, X_train_std, y_train, cv)

    knn_df = pd.DataFrame({
        "Fold": list(range(1, len(knn_acc) + 1)),
        "Accuracy": knn_acc,
        "Sensitivity (macro)": knn_sens,
        "Specificity (macro)": knn_esp
    })
    st.dataframe(knn_df)
    st.subheader("Média dos Folds")
    st.metric(label="Acurácia média (treinamento)", value=f"{np.mean(knn_acc):.3f}")
    st.metric(label="Sensibilidade", value=f"{np.mean(knn_sens):.3f}")
    st.metric(label="Especificidade", value=f"{np.mean(knn_esp):.3f}")

    # Ajuste final KNN e avaliação no teste
    knn_final = KNeighborsClassifier(n_neighbors=k)
    knn_final.fit(X_train_std, y_train)
    y_pred_knn_test = knn_final.predict(X_test_std)

    st.subheader("Avaliação no conjunto de teste — KNN")
    sens_t, esp_t = sens_esp(y_test, y_pred_knn_test, labels_sorted)
    st.metric(label="Acurácia", value=f"{accuracy_score(y_test, y_pred_knn_test):.3f}")
    st.metric(label="Sensibilidade", value=f"{np.mean(sens_t):.3f}")
    st.metric(label="Especificidade", value=f"{np.mean(esp_t):.3f}")


    cm_knn = confusion_matrix(y_test, y_pred_knn_test, labels=labels_sorted)
    st.plotly_chart(plot_confusion(cm_knn, labels_sorted, f"Confusion Matrix — KNN (k={k})"))

    # Gráfico: acurácia média (CV) para vários k (1..20)
    st.subheader("Varredura de k (CV) — acurácia média por k")
    ks = list(range(1, 21))
    accs_for_ks = []
    with st.spinner("Executando CV para cada k (1..20)..."):
        for kk in ks:
            model_k = KNeighborsClassifier(n_neighbors=kk)
            scores = cross_val_score(model_k, X_train_std, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
            accs_for_ks.append(scores.mean())

    df_ks = pd.DataFrame({"k": ks, "CV_accuracy": accs_for_ks})
    fig_ks = px.line(df_ks, x="k", y="CV_accuracy", title="Acurácia média (CV) por k")
    fig_ks.update_traces(mode="markers+lines")
    st.plotly_chart(fig_ks)

    best_k = int(df_ks.loc[df_ks["CV_accuracy"].idxmax(), "k"])
    st.write(f"Melhor k (CV 1..20): {best_k} — acurácia média: {df_ks['CV_accuracy'].max():.3f}")

# ----------------------------
# ABA Comparação
# ----------------------------
with tab_comp:
    st.header("Comparação Geral")

    comp_df = pd.DataFrame({
        "Modelo": ["LDA", "QDA", "KNN (selected k)", "Naive Bayes"],
        "Acurácia (CV mean)": [
            np.mean(lda_acc),
            np.mean(qda_acc),
            np.mean(knn_acc),
            np.mean(nb_acc)
        ],
        "Sensibilidade (CV mean)": [
            np.mean(lda_sens),
            np.mean(qda_sens),
            np.mean(knn_sens),
            np.mean(nb_sens)
        ],
        "Especificidade (CV mean)": [
            np.mean(lda_esp),
            np.mean(qda_esp),
            np.mean(knn_esp),
            np.mean(nb_esp)
        ]
    })
    st.dataframe(comp_df)

    st.subheader("Boxplot comparativo (todas as métricas por fold)")
    # reorganizar cv_df para plot (usar os arrays já calculados)
    cv_comp_df = pd.DataFrame({
        "Fold": list(range(1, len(lda_acc) + 1)) * 4,
        "Modelo": ["LDA"] * len(lda_acc) + ["QDA"] * len(qda_acc) + ["KNN"] * len(knn_acc) + ["NB"] * len(nb_acc),
        "Accuracy": lda_acc + qda_acc + knn_acc + nb_acc
    })
    fig_box = px.box(cv_comp_df, x="Modelo", y="Accuracy", title="Distribuição das Acurácias (CV) — Modelos")
    st.plotly_chart(fig_box)




