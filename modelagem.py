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
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
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
    """Retorna listas (accs, sens, esp, auc_roc) por fold para um dado modelo e dados."""
    accs, sens_list, esp_list, auc_roc_list = [], [], [], []
    for train_idx, val_idx in cv.split(X_data, y_series):
        X_tr, X_val = X_data[train_idx], X_data[val_idx]
        y_tr, y_val = y_series.iloc[train_idx], y_series.iloc[val_idx]

        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)

        accs.append(accuracy_score(y_val, pred))
        s, e = sens_esp(y_val, pred, labels_sorted)

        # Try to calculate AUC-ROC, handle NaN/infinite values gracefully
        try:
            proba = model.predict_proba(X_val)
            auc_roc = roc_auc_score(y_val, proba, multi_class='ovr', average='macro')
        except (ValueError, RuntimeWarning):
            auc_roc = np.nan  # Can't calculate AUC-ROC for this fold

        sens_list.append(s)
        esp_list.append(e)
        auc_roc_list.append(auc_roc)

    return accs, sens_list, esp_list, auc_roc_list

# ----------------------------
# Carregar dados
# ----------------------------
st.title("Modelos de Classificação: LDA, QDA, Naive Bayes Multinomial, KNN e LightGBM")


caminho = "https://raw.githubusercontent.com/abibernardo/repositorio/main/dataset_vidro_me906.csv"
df = pd.read_csv(caminho)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]


# ----------------------------
# Pré-processamento
# ----------------------------
X = df.drop(columns=["glass_group", "glass_type"])
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
    lda_acc, lda_sens, lda_esp, lda_auc = cross_validate_model(lda_model, X_train_std, y_train, cv)

    # QDA
    qda_model = QuadraticDiscriminantAnalysis()
    qda_acc, qda_sens, qda_esp, qda_auc = cross_validate_model(qda_model, X_train_std, y_train, cv)

    # Multinomial NB (usa X_train_nb)
    nb_model = MultinomialNB()
    nb_acc, nb_sens, nb_esp, nb_auc = cross_validate_model(nb_model, X_train_nb, y_train_nb, cv)

# LightGBM with best parameters from previous RandomizedSearchCV
with st.spinner("Executando validação cruzada para LightGBM (pode levar alguns segundos)..."):
    # Best parameters from previous RandomizedSearchCV
    lgbm_model = LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        subsample_freq=1,
        subsample=0.7,
        reg_lambda=2,
        reg_alpha=0.1,
        num_leaves=12,
        min_split_gain=0.1,
        min_child_samples=10,
        max_depth=3,
        max_bin=255,
        learning_rate=0.1,
        colsample_bytree=0.8
    )
    lgbm_acc, lgbm_sens, lgbm_esp, lgbm_auc = cross_validate_model(lgbm_model, X_train_std, y_train, cv)

# ----------------------------
# Abas por modelo
# ----------------------------
tab_df, tab_lda, tab_qda, tab_nb, tab_knn, tab_lgbm, tab_comp = st.tabs([
    "Apresentação do Dataset", "LDA", "QDA", "Naive Bayes Multinomial", "KNN", "LightGBM", "Comparação"
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
        "Specificity (macro)": lda_esp,
        "AUC-ROC (macro)": lda_auc
    })
    st.dataframe(lda_df)
    st.subheader("Média dos Folds")
    st.metric(label="Acurácia média (treinamento)", value=f"{np.mean(lda_acc):.3f}")
    st.metric(label="Sensibilidade", value=f"{np.mean(lda_sens):.3f}")
    st.metric(label="Especificidade", value=f"{np.mean(lda_esp):.3f}")
    st.metric(label="AUC-ROC", value=f"{np.nanmean(lda_auc):.3f}")

    # Ajuste final no treino completo e avaliação no teste
    lda_final = LinearDiscriminantAnalysis()
    lda_final.fit(X_train_std, y_train)
    y_pred_lda_test = lda_final.predict(X_test_std)

    st.subheader("Avaliação no conjunto de teste — LDA")
    sens_t, esp_t = sens_esp(y_test, y_pred_lda_test, labels_sorted)

    # Try to calculate AUC-ROC for test set
    try:
        proba_lda_test = lda_final.predict_proba(X_test_std)
        auc_lda_test = roc_auc_score(y_test, proba_lda_test, multi_class='ovr', average='macro')
    except (ValueError, RuntimeWarning):
        auc_lda_test = np.nan

    st.metric(label="Acurácia", value=f"{accuracy_score(y_test, y_pred_lda_test):.3f}")
    st.metric(label="Sensibilidade", value=f"{np.mean(sens_t):.3f}")
    st.metric(label="Especificidade", value=f"{np.mean(esp_t):.3f}")
    st.metric(label="AUC-ROC", value=f"{auc_lda_test:.3f}" if not np.isnan(auc_lda_test) else "N/A")

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

    # -------------------------------------------
    # Redução de posto para classificação (LD1 × LD2)
    # -------------------------------------------
    st.subheader("Classificação por Redução de Posto — LD1 × LD2")

    # Transformação LDA (rank máximo = C−1 = 2, pois temos 3 classes)
    lda_proj_full = lda_final.transform(X_train_std)

    # Sempre teremos apenas 2 dimensões neste caso
    proj_train_df = pd.DataFrame({
        "LD1": lda_proj_full[:, 0],
        "LD2": lda_proj_full[:, 1],
        "glass_group": y_train.values
    })



    # Classificação do conjunto de teste projetado
    lda_proj_test = lda_final.transform(X_test_std)
    proj_test_df = pd.DataFrame({
        "LD1": lda_proj_test[:, 0],
        "LD2": lda_proj_test[:, 1],
        "Predito": y_pred_lda_test,
        "Real": y_test.values
    })

    fig_test_rank = px.scatter(
        proj_test_df,
        x="LD1",
        y="LD2",
        color="Predito",
        symbol="Real",
        title="Classificação do Conjunto de Teste no Espaço LD1 × LD2"
    )
    st.plotly_chart(fig_test_rank)

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
        "Specificity (macro)": qda_esp,
        "AUC-ROC (macro)": qda_auc
    })
    st.dataframe(qda_df)
    st.subheader("Média dos Folds")
    st.metric(label="Acurácia média (treinamento)", value=f"{np.mean(qda_acc):.3f}")
    st.metric(label="Sensibilidade", value=f"{np.mean(qda_sens):.3f}")
    st.metric(label="Especificidade", value=f"{np.mean(qda_esp):.3f}")
    st.metric(label="AUC-ROC", value=f"{np.nanmean(qda_auc):.3f}")

    # Ajuste final e avaliação no teste
    qda_final = QuadraticDiscriminantAnalysis()
    qda_final.fit(X_train_std, y_train)
    y_pred_qda_test = qda_final.predict(X_test_std)

    st.subheader("Avaliação no conjunto de teste — QDA")
    sens_t, esp_t = sens_esp(y_test, y_pred_qda_test, labels_sorted)

    # Try to calculate AUC-ROC for test set
    try:
        proba_qda_test = qda_final.predict_proba(X_test_std)
        auc_qda_test = roc_auc_score(y_test, proba_qda_test, multi_class='ovr', average='macro')
    except (ValueError, RuntimeWarning):
        auc_qda_test = np.nan

    st.metric(label="Acurácia", value=f"{accuracy_score(y_test, y_pred_qda_test):.3f}")
    st.metric(label="Sensibilidade", value=f"{np.mean(sens_t):.3f}")
    st.metric(label="Especificidade", value=f"{np.mean(esp_t):.3f}")
    st.metric(label="AUC-ROC", value=f"{auc_qda_test:.3f}" if not np.isnan(auc_qda_test) else "N/A")

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
        "Specificity (macro)": nb_esp,
        "AUC-ROC (macro)": nb_auc
    })
    st.dataframe(nb_df)
    st.subheader("Média dos Folds")
    st.metric(label="Acurácia média (treinamento)", value=f"{np.mean(nb_acc):.3f}")
    st.metric(label="Sensibilidade", value=f"{np.mean(nb_sens):.3f}")
    st.metric(label="Especificidade", value=f"{np.mean(nb_esp):.3f}")
    st.metric(label="AUC-ROC", value=f"{np.nanmean(nb_auc):.3f}")

    # Ajuste final e avaliação no teste (NB usa X_nb)
    nb_final = MultinomialNB()
    nb_final.fit(X_train_nb, y_train_nb)
    y_pred_nb_test = nb_final.predict(X_test_nb)

    st.subheader("Avaliação no conjunto de teste — Multinomial NB")
    sens_t, esp_t = sens_esp(y_test_nb, y_pred_nb_test, labels_sorted)

    # Try to calculate AUC-ROC for test set
    try:
        proba_nb_test = nb_final.predict_proba(X_test_nb)
        auc_nb_test = roc_auc_score(y_test_nb, proba_nb_test, multi_class='ovr', average='macro')
    except (ValueError, RuntimeWarning):
        auc_nb_test = np.nan

    st.metric(label="Acurácia", value=f"{accuracy_score(y_test_nb, y_pred_nb_test):.3f}")
    st.metric(label="Sensibilidade", value=f"{np.mean(sens_t):.3f}")
    st.metric(label="Especificidade", value=f"{np.mean(esp_t):.3f}")
    st.metric(label="AUC-ROC", value=f"{auc_nb_test:.3f}" if not np.isnan(auc_nb_test) else "N/A")

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
    knn_acc, knn_sens, knn_esp, knn_auc = cross_validate_model(knn_model, X_train_std, y_train, cv)

    knn_df = pd.DataFrame({
        "Fold": list(range(1, len(knn_acc) + 1)),
        "Accuracy": knn_acc,
        "Sensitivity (macro)": knn_sens,
        "Specificity (macro)": knn_esp,
        "AUC-ROC (macro)": knn_auc
    })
    st.dataframe(knn_df)
    st.subheader("Média dos Folds")
    st.metric(label="Acurácia média (treinamento)", value=f"{np.mean(knn_acc):.3f}")
    st.metric(label="Sensibilidade", value=f"{np.mean(knn_sens):.3f}")
    st.metric(label="Especificidade", value=f"{np.mean(knn_esp):.3f}")
    st.metric(label="AUC-ROC", value=f"{np.nanmean(knn_auc):.3f}")

    # Ajuste final KNN e avaliação no teste
    knn_final = KNeighborsClassifier(n_neighbors=k)
    knn_final.fit(X_train_std, y_train)
    y_pred_knn_test = knn_final.predict(X_test_std)

    st.subheader("Avaliação no conjunto de teste — KNN")
    sens_t, esp_t = sens_esp(y_test, y_pred_knn_test, labels_sorted)

    # Try to calculate AUC-ROC for test set
    try:
        proba_knn_test = knn_final.predict_proba(X_test_std)
        auc_knn_test = roc_auc_score(y_test, proba_knn_test, multi_class='ovr', average='macro')
    except (ValueError, RuntimeWarning):
        auc_knn_test = np.nan

    st.metric(label="Acurácia", value=f"{accuracy_score(y_test, y_pred_knn_test):.3f}")
    st.metric(label="Sensibilidade", value=f"{np.mean(sens_t):.3f}")
    st.metric(label="Especificidade", value=f"{np.mean(esp_t):.3f}")
    st.metric(label="AUC-ROC", value=f"{auc_knn_test:.3f}" if not np.isnan(auc_knn_test) else "N/A")


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
# ABA LightGBM
# ----------------------------
with tab_lgbm:
    st.header("LightGBM (Gradient Boosting)")

    st.subheader("Validação Cruzada (5 folds) — LightGBM")
    lgbm_df = pd.DataFrame({
        "Fold": list(range(1, len(lgbm_acc) + 1)),
        "Accuracy": lgbm_acc,
        "Sensitivity (macro)": lgbm_sens,
        "Specificity (macro)": lgbm_esp,
        "AUC-ROC (macro)": lgbm_auc
    })
    st.dataframe(lgbm_df)
    st.subheader("Média dos Folds")
    st.metric(label="Acurácia média (treinamento)", value=f"{np.mean(lgbm_acc):.3f}")
    st.metric(label="Sensibilidade", value=f"{np.mean(lgbm_sens):.3f}")
    st.metric(label="Especificidade", value=f"{np.mean(lgbm_esp):.3f}")
    st.metric(label="AUC-ROC", value=f"{np.nanmean(lgbm_auc):.3f}")

    # Ajuste final e avaliação no teste
    lgbm_final = LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        subsample_freq=1,
        subsample=0.7,
        reg_lambda=2,
        reg_alpha=0.1,
        num_leaves=12,
        min_split_gain=0.1,
        min_child_samples=10,
        max_depth=3,
        max_bin=255,
        learning_rate=0.1,
        colsample_bytree=0.8
    )
    lgbm_final.fit(X_train_std, y_train)
    y_pred_lgbm_test = lgbm_final.predict(X_test_std)

    st.subheader("Avaliação no conjunto de teste — LightGBM")
    sens_t, esp_t = sens_esp(y_test, y_pred_lgbm_test, labels_sorted)

    # Try to calculate AUC-ROC for test set
    try:
        proba_lgbm_test = lgbm_final.predict_proba(X_test_std)
        auc_lgbm_test = roc_auc_score(y_test, proba_lgbm_test, multi_class='ovr', average='macro')
    except (ValueError, RuntimeWarning):
        auc_lgbm_test = np.nan

    st.metric(label="Acurácia", value=f"{accuracy_score(y_test, y_pred_lgbm_test):.3f}")
    st.metric(label="Sensibilidade", value=f"{np.mean(sens_t):.3f}")
    st.metric(label="Especificidade", value=f"{np.mean(esp_t):.3f}")
    st.metric(label="AUC-ROC", value=f"{auc_lgbm_test:.3f}" if not np.isnan(auc_lgbm_test) else "N/A")

    # Confusion matrix (test)
    cm_lgbm = confusion_matrix(y_test, y_pred_lgbm_test, labels=labels_sorted)
    st.plotly_chart(plot_confusion(cm_lgbm, labels_sorted, "Confusion Matrix — LightGBM (test)"))

# ----------------------------
# ABA Comparação
# ----------------------------
with tab_comp:
    st.header("Comparação Geral")

    # Calculate test set performance for all models
    # LDA
    lda_final = LinearDiscriminantAnalysis()
    lda_final.fit(X_train_std, y_train)
    y_pred_lda = lda_final.predict(X_test_std)
    lda_test_acc = accuracy_score(y_test, y_pred_lda)
    lda_test_sens, lda_test_esp = sens_esp(y_test, y_pred_lda, labels_sorted)
    try:
        lda_test_auc = roc_auc_score(y_test, lda_final.predict_proba(X_test_std), multi_class='ovr', average='macro')
    except (ValueError, RuntimeWarning):
        lda_test_auc = np.nan

    # QDA
    qda_final = QuadraticDiscriminantAnalysis()
    qda_final.fit(X_train_std, y_train)
    y_pred_qda = qda_final.predict(X_test_std)
    qda_test_acc = accuracy_score(y_test, y_pred_qda)
    qda_test_sens, qda_test_esp = sens_esp(y_test, y_pred_qda, labels_sorted)
    try:
        qda_test_auc = roc_auc_score(y_test, qda_final.predict_proba(X_test_std), multi_class='ovr', average='macro')
    except (ValueError, RuntimeWarning):
        qda_test_auc = np.nan

    # KNN (using current k from slider)
    knn_final = KNeighborsClassifier(n_neighbors=k)
    knn_final.fit(X_train_std, y_train)
    y_pred_knn = knn_final.predict(X_test_std)
    knn_test_acc = accuracy_score(y_test, y_pred_knn)
 
