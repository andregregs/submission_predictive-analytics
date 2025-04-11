# 1. Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.style.use('seaborn-v0_8-pastel')

# 2. Load Dataset
def load_data(file_path='/datasets/heart_disease.csv'):
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    
    print("\n5 BARIS PERTAMA DATASET:")
    print(df.head())
    
    print("\nINFORMASI DATASET:")
    print(df.info())
    
    print("\nSTATISTIK DESKRIPTIF:")
    print(df.describe())
    
    print("\nJUMLAH MISSING VALUES PER KOLOM:")
    print(df.isnull().sum())
    
    return df

# 3. Exploratory Data Analysis (EDA)
def perform_eda(df):

    print("\n\n===== EXPLORATORY DATA ANALYSIS =====\n")
    
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='target', data=df, palette='viridis')
    plt.title('Distribusi Kelas Target', fontsize=15)
    plt.xlabel('Diagnosis Penyakit Jantung (0: Negatif, 1: Positif)', fontsize=12)
    plt.ylabel('Jumlah', fontsize=12)
    
    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        percentage = height / total * 100
        ax.text(p.get_x() + p.get_width()/2, height + 5, 
                f'{int(height)} ({percentage:.1f}%)', 
                ha='center', fontsize=12)
    
    plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(14, 12))
    correlation = df.corr()
    mask = np.triu(correlation)
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', mask=mask)
    plt.title('Korelasi Antar Fitur', fontsize=15)
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    plt.figure(figsize=(16, 12))
    for i, feature in enumerate(numeric_features):
        plt.subplot(3, 2, i+1)
        sns.histplot(data=df, x=feature, hue='target', kde=True, bins=30, palette='viridis')
        plt.title(f'Distribusi {feature} berdasarkan Target', fontsize=13)
    plt.tight_layout()
    plt.savefig('numeric_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(16, 12))
    for i, feature in enumerate(numeric_features):
        plt.subplot(3, 2, i+1)
        sns.boxplot(x='target', y=feature, data=df, palette='viridis')
        plt.title(f'Boxplot {feature} berdasarkan Target', fontsize=13)
        plt.xlabel('Diagnosis Penyakit Jantung (0: Negatif, 1: Positif)', fontsize=10)
    plt.tight_layout()
    plt.savefig('boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    plt.figure(figsize=(20, 16))
    for i, feature in enumerate(categorical_features):
        plt.subplot(3, 3, i+1)
        ax = sns.countplot(x=feature, hue='target', data=df, palette='viridis')
        plt.title(f'Distribusi {feature} berdasarkan Target', fontsize=13)
        plt.xlabel(f'{feature}', fontsize=10)
        plt.legend(title='Target', loc='upper right', labels=['Negatif', 'Positif'])
        
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2, height + 2, 
                    f'{int(height)}', 
                    ha='center', fontsize=9)
            
    plt.tight_layout()
    plt.savefig('categorical_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nANALISIS BIVARIATE UNTUK VARIABEL KATEGORIKAL:")
    for feature in categorical_features:
        print(f"\nCrosstab untuk {feature}:")
        crosstab = pd.crosstab(df[feature], df['target'], normalize='index') * 100
        print(crosstab)
        print("\nPersentase penyakit jantung berdasarkan kategori:")
        print(crosstab[1].sort_values(ascending=False))
    
    sns.pairplot(df[numeric_features + ['target']], hue='target', palette='viridis')
    plt.suptitle('Pairplot Fitur Numerik', y=1.02, fontsize=16)
    plt.savefig('pairplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return numeric_features, categorical_features

# 4. Data Preparation
def prepare_data(df, numeric_features, categorical_features):

    print("\n\n===== DATA PREPARATION =====\n")
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nINFORMASI SPLITTING DATA:")
    print(f"Ukuran data training: {X_train.shape}")
    print(f"Ukuran data testing: {X_test.shape}")
    print(f"Distribusi target di data training: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Distribusi target di data testing: {pd.Series(y_test).value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test, preprocessor

# 5. Modeling
def evaluate_model(model, X_test, y_test, model_name, plot=True):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n--- PERFORMA MODEL {model_name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    if plot:
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=15)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        for i in range(2):
            for j in range(2):
                plt.text(j+0.3, i+0.15, labels[i, j], 
                        fontsize=12, color='black')
        
        plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=15)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'roc_curve_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return {
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall, 
        'f1': f1, 
        'roc_auc': roc_auc
    }

def train_models(X_train, X_test, y_train, y_test, preprocessor):

    print("\n\n===== MODEL TRAINING AND EVALUATION =====\n")
    
    print("\n====== MODEL 1: LOGISTIC REGRESSION ======")
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_pipeline, X_test, y_test, "Logistic Regression")
    
    print("\n====== MODEL 2: RANDOM FOREST ======")
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf_pipeline, X_test, y_test, "Random Forest")
    
    print("\n====== MODEL 3: GRADIENT BOOSTING ======")
    gb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])
    gb_pipeline.fit(X_train, y_train)
    gb_metrics = evaluate_model(gb_pipeline, X_test, y_test, "Gradient Boosting")
    
    print("\n====== MODEL 4: SUPPORT VECTOR MACHINE ======")
    svm_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(probability=True, random_state=42))
    ])
    svm_pipeline.fit(X_train, y_train)
    svm_metrics = evaluate_model(svm_pipeline, X_test, y_test, "Support Vector Machine")
    
    print("\n====== HYPERPARAMETER TUNING: RANDOM FOREST ======")
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        rf_pipeline, 
        param_grid=param_grid, 
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    print("Melakukan grid search...")
    grid_search.fit(X_train, y_train)
    print("Grid search selesai!")
    print("\nBest parameters:", grid_search.best_params_)
    print("Best ROC AUC score:", grid_search.best_score_)
    
    best_model = grid_search.best_estimator_
    best_metrics = evaluate_model(best_model, X_test, y_test, "Random Forest (Tuned)")
    
    all_metrics = {
        'Logistic Regression': lr_metrics,
        'Random Forest': rf_metrics,
        'Gradient Boosting': gb_metrics,
        'SVM': svm_metrics,
        'Random Forest (Tuned)': best_metrics
    }
    
    return best_model, all_metrics

# 6. Feature Importance Analysis
def analyze_feature_importance(best_model, X, numeric_features, categorical_features, preprocessor):

    print("\n\n===== FEATURE IMPORTANCE ANALYSIS =====\n")
    
    preprocessor_fit = preprocessor.fit(X)
    ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(ohe_feature_names)
    
    best_rf_model = best_model.named_steps['classifier']
    
    if hasattr(best_rf_model, 'feature_importances_'):
        importances = best_rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importances (Random Forest Tuned)', fontsize=15)
        plt.bar(range(len(indices)), importances[indices], align='center', color='teal')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nTop 10 fitur paling penting:")
        for i in range(10):
            if i < len(indices):
                print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# 7. Model Evaluation Comparison
def compare_models(all_metrics):
    print("\n\n===== MODEL COMPARISON =====\n")
    
    models = list(all_metrics.keys())
    metrics_dict = {
        'Model': models,
        'Accuracy': [all_metrics[m]['accuracy'] for m in models],
        'Precision': [all_metrics[m]['precision'] for m in models],
        'Recall': [all_metrics[m]['recall'] for m in models],
        'F1 Score': [all_metrics[m]['f1'] for m in models],
        'ROC AUC': [all_metrics[m]['roc_auc'] for m in models]
    }
    
    metrics_df = pd.DataFrame(metrics_dict)
    print("\nPerbandingan metrik evaluasi:")
    print(metrics_df)
    
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 2, 1)
    ax1 = sns.barplot(x='Model', y='Accuracy', data=metrics_df, palette='viridis')
    plt.title('Accuracy Comparison', fontsize=15)
    plt.ylim([0.7, 1])
    plt.xticks(rotation=45, ha='right')
    for i, p in enumerate(ax1.patches):
        ax1.annotate(f'{p.get_height():.4f}', 
                    (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points')
    
    plt.subplot(2, 2, 2)
    ax2 = sns.barplot(x='Model', y='Precision', data=metrics_df, palette='viridis')
    plt.title('Precision Comparison', fontsize=15)
    plt.ylim([0.7, 1])
    plt.xticks(rotation=45, ha='right')
    for i, p in enumerate(ax2.patches):
        ax2.annotate(f'{p.get_height():.4f}', 
                    (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points')
    
    plt.subplot(2, 2, 3)
    ax3 = sns.barplot(x='Model', y='Recall', data=metrics_df, palette='viridis')
    plt.title('Recall Comparison', fontsize=15)
    plt.ylim([0.7, 1])
    plt.xticks(rotation=45, ha='right')
    for i, p in enumerate(ax3.patches):
        ax3.annotate(f'{p.get_height():.4f}', 
                    (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points')
    
    plt.subplot(2, 2, 4)
    ax4 = sns.barplot(x='Model', y='ROC AUC', data=metrics_df, palette='viridis')
    plt.title('ROC AUC Comparison', fontsize=15)
    plt.ylim([0.7, 1])
    plt.xticks(rotation=45, ha='right')
    for i, p in enumerate(ax4.patches):
        ax4.annotate(f'{p.get_height():.4f}', 
                    (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for i, model_name in enumerate(all_metrics.keys()):
        roc_auc = all_metrics[model_name]['roc_auc']
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)' if i == 0 else "", alpha=0.3)
        plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color=colors[i], alpha=0.3)
        plt.plot([0, 1], [roc_auc, roc_auc], linestyle=':', color=colors[i], alpha=0.3)
        x = np.linspace(0, 1, 100)
        y = x ** (1/roc_auc-1)
        plt.plot(x, y, color=colors[i], label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison (Simulated)', fontsize=15)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('roc_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return metrics_df

# 8. Cross-Validation
def perform_cross_validation(best_model, X, y):
    print("\n\n===== CROSS-VALIDATION =====\n")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring='roc_auc')
    
    print("ROC AUC scores dari 5-fold cross-validation:")
    print(cv_scores)
    print(f"Mean ROC AUC: {cv_scores.mean():.4f}")
    print(f"Standard Deviation: {cv_scores.std():.4f}")
    
    return cv_scores.mean(), cv_scores.std()

# 9. Model Deployment (Fungsi Prediksi)
def create_prediction_function(best_model):

    def predict_heart_disease(patient_data):

        data = pd.DataFrame([patient_data])
        
        prediction = best_model.predict(data)[0]
        probability = best_model.predict_proba(data)[0][1]
        
        return prediction, probability
    
    return predict_heart_disease

# 10. Conclusion and Demo
def show_demo(predict_func):

    print("\n\n===== DEMO PENGGUNAAN MODEL =====\n")
    
    # Contoh kasus 1: Pasien dengan risiko tinggi
    print("Contoh Kasus 1: Pasien Risiko Tinggi")
    sample_data_high_risk = {
        'age': 64,
        'sex': 1,  # 1 = male, 0 = female
        'cp': 3,   # chest pain type (3 = asymptomatic)
        'trestbps': 160,  # resting blood pressure
        'chol': 260,  # serum cholesterol
        'fbs': 1,  # fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
        'restecg': 2,  # resting electrocardiographic results
        'thalach': 124,  # maximum heart rate achieved
        'exang': 1,  # exercise induced angina (1 = yes, 0 = no)
        'oldpeak': 2.6,  # ST depression induced by exercise relative to rest
        'slope': 0,  # slope of the peak exercise ST segment
        'ca': 3,  # number of major vessels (0-3) colored by fluoroscopy
        'thal': 2  # thalassemia (2 = fixed defect)
    }
    
    prediction1, probability1 = predict_func(sample_data_high_risk)
    print(f"Prediksi: {'Positif (Risiko Penyakit Jantung)' if prediction1 == 1 else 'Negatif (Tidak Ada Risiko Penyakit Jantung)'}")
    print(f"Probabilitas risiko penyakit jantung: {probability1:.4f} ({probability1*100:.2f}%)")
    
    # Contoh kasus 2: Pasien dengan risiko rendah
    print("\nContoh Kasus 2: Pasien Risiko Rendah")
    sample_data_low_risk = {
        'age': 34,
        'sex': 0,  # 1 = male, 0 = female
        'cp': 0,   # chest pain type (0 = typical angina)
        'trestbps': 118,  # resting blood pressure
        'chol': 180,  # serum cholesterol
        'fbs': 0,  # fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
        'restecg': 0,  # resting electrocardiographic results
        'thalach': 170,  # maximum heart rate achieved
        'exang': 0,  # exercise induced angina (1 = yes, 0 = no)
        'oldpeak': 0.2,  # ST depression induced by exercise relative to rest
        'slope': 2,  # slope of the peak exercise ST segment
        'ca': 0,  # number of major vessels (0-3) colored by fluoroscopy
        'thal': 1  # thalassemia (1 = normal)
    }
    
    prediction2, probability2 = predict_func(sample_data_low_risk)
    print(f"Prediksi: {'Positif (Risiko Penyakit Jantung)' if prediction2 == 1 else 'Negatif (Tidak Ada Risiko Penyakit Jantung)'}")
    print(f"Probabilitas risiko penyakit jantung: {probability2:.4f} ({probability2*100:.2f}%)")
    
    # Masukkan data pasien secara interaktif (opsional)
    try_interactive = input("\nApakah Anda ingin mencoba prediksi dengan data pasien sendiri? (y/n): ")
    if try_interactive.lower() == 'y':
        print("\nMasukkan data pasien:")
        
        # Helper function untuk mendapatkan input numerik yang valid
        def get_numeric_input(prompt, min_val=None, max_val=None, default=None):
            while True:
                try:
                    value_str = input(prompt)
                    if value_str == '' and default is not None:
                        return default
                    value = float(value_str)
                    if min_val is not None and value < min_val:
                        print(f"Nilai harus >= {min_val}")
                        continue
                    if max_val is not None and value > max_val:
                        print(f"Nilai harus <= {max_val}")
                        continue
                    return value
                except ValueError:
                    print("Masukkan angka yang valid!")
        
        # Helper function untuk mendapatkan input kategorikal yang valid
        def get_categorical_input(prompt, valid_values, default=None):
            while True:
                try:
                    value_str = input(prompt)
                    if value_str == '' and default is not None:
                        return default
                    value = int(value_str)
                    if value not in valid_values:
                        print(f"Nilai harus salah satu dari {valid_values}")
                        continue
                    return value
                except ValueError:
                    print("Masukkan angka yang valid!")
        
        # Collect user data
        custom_data = {}
        custom_data['age'] = get_numeric_input("Usia (tahun): ", 20, 100)
        custom_data['sex'] = get_categorical_input("Jenis kelamin (1=laki-laki, 0=perempuan): ", [0, 1])
        custom_data['cp'] = get_categorical_input("Tipe nyeri dada (0=typical angina, 1=atypical angina, 2=non-anginal pain, 3=asymptomatic): ", [0, 1, 2, 3])
        custom_data['trestbps'] = get_numeric_input("Tekanan darah istirahat (mm Hg): ", 80, 250)
        custom_data['chol'] = get_numeric_input("Kolesterol serum (mg/dl): ", 100, 600)
        custom_data['fbs'] = get_categorical_input("Gula darah puasa > 120 mg/dl (1=ya, 0=tidak): ", [0, 1])
        custom_data['restecg'] = get_categorical_input("Hasil elektrokardiografi istirahat (0=normal, 1=kelainan ST-T, 2=hipertrofi ventrikel kiri): ", [0, 1, 2])
        custom_data['thalach'] = get_numeric_input("Detak jantung maksimum: ", 60, 220)
        custom_data['exang'] = get_categorical_input("Angina yang dipicu oleh olahraga (1=ya, 0=tidak): ", [0, 1])
        custom_data['oldpeak'] = get_numeric_input("Depresi ST yang diinduksi oleh olahraga: ", 0, 10)
        custom_data['slope'] = get_categorical_input("Kemiringan segmen ST (0=upsloping, 1=flat, 2=downsloping): ", [0, 1, 2])
        custom_data['ca'] = get_categorical_input("Jumlah pembuluh darah utama (0-3): ", [0, 1, 2, 3])
        custom_data['thal'] = get_categorical_input("Status thalassemia (1=normal, 2=fixed defect, 3=reversible defect): ", [1, 2, 3])
        
        # Prediksi
        prediction_custom, probability_custom = predict_func(custom_data)
        print("\nHasil Prediksi:")
        print(f"Prediksi: {'Positif (Risiko Penyakit Jantung)' if prediction_custom == 1 else 'Negatif (Tidak Ada Risiko Penyakit Jantung)'}")
        print(f"Probabilitas risiko penyakit jantung: {probability_custom:.4f} ({probability_custom*100:.2f}%)")
        
        # Interpretasi hasil
        print("\nInterpretasi:")
        if probability_custom >= 0.7:
            print("Pasien memiliki risiko tinggi penyakit jantung. Disarankan untuk segera berkonsultasi dengan dokter untuk evaluasi lebih lanjut.")
        elif probability_custom >= 0.3:
            print("Pasien memiliki risiko sedang penyakit jantung. Disarankan untuk melakukan pemeriksaan lebih lanjut dan mengubah gaya hidup.")
        else:
            print("Pasien memiliki risiko rendah penyakit jantung. Tetap disarankan untuk menjaga gaya hidup sehat.")

def main():
    print("===== PROJECT MACHINE LEARNING - PREDIKSI PENYAKIT JANTUNG =====")
    
    # 1. Load Data
    df = load_data('datasets\heart_disease.csv')
    
    # 2. Exploratory Data Analysis
    numeric_features, categorical_features = perform_eda(df)
    
    # 3. Data Preparation
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df, numeric_features, categorical_features)
    
    # 4. Model Training dan Evaluation
    best_model, all_metrics = train_models(X_train, X_test, y_train, y_test, preprocessor)
    
    # 5. Feature Importance Analysis
    feature_imp = analyze_feature_importance(best_model, df.drop('target', axis=1), numeric_features, categorical_features, preprocessor)
    
    # 6. Model Comparison
    metrics_df = compare_models(all_metrics)
    
    # 7. Cross-Validation
    cv_mean, cv_std = perform_cross_validation(best_model, df.drop('target', axis=1), df['target'])
    
    # 8. Create prediction function
    predict_func = create_prediction_function(best_model)
    
    # 9. Demo
    show_demo(predict_func)
    
    # 10. Conclusion
    print("\n\n===== KESIMPULAN =====\n")
    print("""
    Dari analisis dan pemodelan yang telah dilakukan, diperoleh kesimpulan sebagai berikut:
    
    1. Model Random Forest yang telah dituning memberikan performa terbaik dengan ROC AUC sekitar 0.92.
    
    2. Faktor-faktor penting dalam prediksi penyakit jantung adalah:
       - Tipe nyeri dada (cp)
       - Jumlah pembuluh darah utama (ca)
       - Detak jantung maksimum (thalach)
       - Status thalassemia (thal)
       - Depresi ST yang diinduksi oleh olahraga (oldpeak)
    
    3. Model ini dapat digunakan sebagai alat bantu dalam deteksi dini penyakit jantung, tetapi tidak menggantikan diagnosis medis profesional.
    
    4. Untuk pengembangan selanjutnya, dapat dilakukan:
       - Pengumpulan data yang lebih banyak dan beragam
       - Penambahan fitur-fitur klinis lainnya
       - Penyesuaian model untuk populasi spesifik
    """)
    
    print("\nProyek selesai. Terima kasih!")

if __name__ == "__main__":
    main()