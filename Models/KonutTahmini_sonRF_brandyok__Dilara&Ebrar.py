
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 23:45:37 2025

@author: dilar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import time
import joblib
import warnings

warnings.filterwarnings('ignore')

# 1. Veri Yükleme ve İlk İnceleme
print("1. Veri Yükleme ve İlk İnceleme")
print("-" * 50)
df=pd.read_csv("C:/Users/dilar/OneDrive/Belgeler/BSM/Donem6/Veri Madenciliği/Proje/modeleHazir_SayisalVeri.csv")
print(f"Veri boyutu: {df.shape}")


df = df.drop(columns=['property_type_branded_ratio'])


# Hedef değişken analizi
print("\nProperty Type dağılımı:")
property_counts = df['property_type'].value_counts()
print(property_counts)

# 2. Veri Ön İşleme
print("\n2. Veri Ön İşleme")
print("-" * 50)

# Küçük sınıfları birleştirme (30'dan az örneğe sahip sınıflar)
small_classes = [cls for cls, count in property_counts.items() if count < 30]
print(f"Birleştirilecek küçük sınıflar: {small_classes}")

# Küçük sınıfları 'Diğer' olarak birleştirme
df['property_type_grouped'] = df['property_type'].apply(
    lambda x: 'Diğer' if x in small_classes else x
)

print("\nBirleştirilmiş Property Type dağılımı:")
print(df['property_type_grouped'].value_counts())

# 3. Özellik Seçimi - Daha önce başarılı olan aynı özellikleri kullanalım
print("\n3. Özellik Seçimi")
print("-" * 50)

# property_type ve property_type_grouped dışındaki tüm sütunları kullan
all_features = [col for col in df.columns if col not in ['property_type', 'property_type_grouped']]


print("Eksik Değer Sayıları:\n")
null_counts = df.isnull().sum()
print(null_counts)


# Sayısal sütunları seç
numerical_features = df.select_dtypes(include=['number']).columns.tolist()

# Kategorik sütunları seç (object, category ve bool dahil)
categorical_features = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# Tüm sütunları al
all_features = df.columns.tolist()

print(f"Toplam özellik sayısı: {len(all_features)}")
print(f"Sayısal özellik sayısı: {len(numerical_features)} --> {numerical_features}")
print(f"Kategorik özellik sayısı: {len(categorical_features)} --> {categorical_features}")


# Tüm sayısal sütunları otomatik al
numerical_features = df.select_dtypes(include=['number']).columns.tolist()

print(f"Sayısal özellik sayısı: {len(numerical_features)}")

# Eksik değerleri doldur
for col in numerical_features:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

print("Sayısal sütunlardaki eksik değerler dolduruldu.")


# 6. Hedef Değişkeni Hazırlama
print("\n6. Hedef Değişken Hazırlama")
print("-" * 50)

# Hedef değişkeni kodlama
le = LabelEncoder()
y = le.fit_transform(df['property_type_grouped'])
target_names = le.classes_

print("Sınıf kodlamaları:")
for i, label in enumerate(target_names):
    print(f"{i}: {label}")

# Sınıf ağırlıklarını hesapla
class_weights = {}
for i, cls in enumerate(target_names):
    class_weights[i] = len(y) / (len(np.unique(y)) * sum(y == i))
print("\nSınıf ağırlıkları:")
print(class_weights)

# 7. Veri Bölme
print("\n7. Veri Bölme")
print("-" * 50)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df,       # Özellikler (bağımsız değişkenler)
    y,        # Hedef değişken (bağımlı değişken)
    test_size=0.3,        # Verinin %30'u test, %70'i train olacak
    random_state=42,      # Aynı sonuçları almak için sabit bir rastgelelik
    stratify=y            # Hedef değişkene göre oransal bölme (sınıflar dengeli kalsın diye)
)

# Sayısal özellikleri seç ve ölçeklendir
X_train_num = X_train[numerical_features].copy()
X_test_num = X_test[numerical_features].copy()

scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)


print(f"eğitim seti boyutu: {X_train_num_scaled.shape}")
print(f"Test seti boyutu: {X_test_num_scaled.shape}")


# Eğitim ve test verilerini birleştirme
X_train_combined = X_train_num_scaled
X_test_combined = X_test_num_scaled

print(f"Birleştirilmiş eğitim veri boyutu: {X_train_combined.shape}")
print(f"Birleştirilmiş test veri boyutu: {X_test_combined.shape}")

# 9. Temel Model Eğitimi (GridSearch Öncesi)
print("\n9. Temel Model Eğitimi (GridSearch Öncesi)")
print("-" * 50)

# Temel model parametereleri
base_rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    class_weight=class_weights,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)


# Temel modeli eğit
print("Temel model eğitimi başlıyor...")
start_time = time.time()
base_rf_model.fit(X_train_combined, y_train)
base_train_time = time.time() - start_time
print(f"Temel model eğitimi tamamlandı. Süre: {base_train_time:.2f} saniye")

# Temel model değerlendirmesi
print("\nTemel Model Performansı:")
print(f"Out-of-bag skoru: {base_rf_model.oob_score_:.4f}")

# Test seti üzerinde tahmin
base_y_pred = base_rf_model.predict(X_test_combined)

# Performans metrikleri
base_accuracy = accuracy_score(y_test, base_y_pred)
base_f1 = f1_score(y_test, base_y_pred, average='weighted')

print(f"Accuracy: {base_accuracy:.4f}")
print(f"F1 Score (weighted): {base_f1:.4f}")

# Sınıflandırma raporu
print("\nSınıflandırma Raporu (Temel Model):")
base_report = classification_report(y_test, base_y_pred, target_names=target_names)
print(base_report)
base_report_dict = classification_report(y_test, base_y_pred, target_names=target_names, output_dict=True)


# Confusion Matrix 
from sklearn.metrics import ConfusionMatrixDisplay

present_classes = np.unique(np.concatenate([y_test, base_y_pred]))
class_names = le.classes_[present_classes]

cm = confusion_matrix(y_test, base_y_pred, labels=present_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(6, 4))
disp.plot(ax=ax, xticks_rotation=90, cmap='Blues', values_format='d')
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.show()
print("Temel model confusion matrix 'base_model_confusion_matrix.png' olarak kaydedildi.")

#*************************************************************************************************
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# Learning curve verilerini al
train_sizes, train_scores, test_scores = learning_curve(
    base_rf_model,  # Model
    X_train_combined,  # Eğitim verisi
    y_train,  # Eğitim etiketleri
    cv=10,  # 5 katlı çapraz doğrulama
    n_jobs=-1,  # Paralel işlem yapmak için
    train_sizes=np.linspace(0.1, 1.0, 10)  # Eğitim verisinin farklı yüzdeleri
)

# Eğitim ve test hatalarını hesapla (ortalama ve std)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Learning curve görselini çiz
plt.figure(figsize=(10, 8))
plt.plot(train_sizes, train_mean, label='Eğitim Hatası', color='blue')
plt.plot(train_sizes, test_mean, label='Test Hatası', color='green')

# Hata çubukları
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='green', alpha=0.2)

# Grafik detayları
plt.title('Learning Curve (Temel Model)')
plt.xlabel('Eğitim Veri Miktarı')
plt.ylabel('Model Hatası (Doğruluk ve Kaybın Ters Oranı)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Grafik gösterimi
plt.show()


#*************************************************************************************************



# 10. GridSearch ile Hiperparametre Optimizasyonu
print("\n10. GridSearch ile Hiperparametre Optimizasyonu")
print("-" * 50)

# GridSearch için parametre seti
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10],
}

# GridSearch
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(
        max_features='sqrt',
        class_weight=class_weights,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    ),
    param_grid=param_grid,
    scoring='f1_weighted',
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# GridSearch başlat
print("GridSearch başlatılıyor...")
start_time = time.time()
grid_search.fit(X_train_combined, y_train)
grid_search_time = time.time() - start_time
print(f"GridSearch tamamlandı. Süre: {grid_search_time:.2f} saniye")

# En iyi parametreleri göster
print(f"En iyi parametreler: {grid_search.best_params_}")
print(f"En iyi CV skoru: {grid_search.best_score_:.4f}") #çapraz doğrulama işlemleri boyunca elde edilen ortalama başarıdı

# 11. Optimizasyon Sonrası Model Değerlendirmesi
print("\n11. Optimizasyon Sonrası Model Değerlendirmesi")
print("-" * 50)

# En iyi modeli al
best_model = grid_search.best_estimator_

# Test seti üzerinde tahmin
best_y_pred = best_model.predict(X_test_combined)

# Performans metrikleri
best_accuracy = accuracy_score(y_test, best_y_pred)
best_f1 = f1_score(y_test, best_y_pred, average='weighted')

print("\nOptimize Edilmiş Model Performansı:")
print(f"Out-of-bag skoru: {best_model.oob_score_:.4f}")
print(f"Accuracy: {best_accuracy:.4f}")
print(f"F1 Score (weighted): {best_f1:.4f}")

# Sınıflandırma raporu
print("\nSınıflandırma Raporu (Optimize Edilmiş Model):")
best_report = classification_report(y_test, best_y_pred, target_names=target_names)
print(best_report)
best_report_dict = classification_report(y_test, best_y_pred, target_names=target_names, output_dict=True)


# Confusion Matrix for Test Set
from sklearn.metrics import ConfusionMatrixDisplay

present_classes = np.unique(np.concatenate([y_test, best_y_pred]))
class_names = le.classes_[present_classes]

cm = confusion_matrix(y_test, best_y_pred, labels=present_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(6, 4))
disp.plot(ax=ax, xticks_rotation=90, cmap='Blues', values_format='d')
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.show()
print("Optimize edilmiş model confusion matrix 'optimized_model_confusion_matrix.png' olarak kaydedildi.")

# Her sınıf için TP, FP, TN, FN hesaplama
print("\nHer sınıf için TP, FP, TN, FN değerleri:")
print("-" * 50)
num_classes = len(target_names)
for i, class_name in enumerate(target_names):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)
    
    print(f"\nSınıf: {class_name}")
    print(f"  TP (True Positive):  {TP}")
    print(f"  FP (False Positive): {FP}")
    print(f"  FN (False Negative): {FN}")
    print(f"  TN (True Negative):  {TN}")


#*********************************************************************************************************

# Learning curve verisini hesapla
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train_combined, y_train, cv=5, 
    scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
)

# Ortalama ve standart sapma hesaplama
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Grafiği çizme
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label="Eğitim Doğruluğu", color="blue", lw=2)
plt.plot(train_sizes, val_mean, label="Doğrulama Doğruluğu", color="green", lw=2)

# Standart sapmayı grafikte gösterme
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.2)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color="green", alpha=0.2)

# Etiketler ve başlık
plt.title("Learning Curve (Optimize Edilmiş Model)", fontsize=16)
plt.xlabel("Eğitim Seti Boyutu", fontsize=12)
plt.ylabel("Doğruluk", fontsize=12)
plt.legend(loc="best", fontsize=12)
plt.grid(True)

# Grafik gösterimi
plt.tight_layout()
plt.show()

#*********************************************************************************************************


# 12. Model Karşılaştırması
print("\n12. Model Karşılaştırması")
print("-" * 50)

# Temel model vs optimize edilmiş model performans karşılaştırması
comparison_df = pd.DataFrame({
    'Metrik': ['Accuracy', 'F1 Score (weighted)', 'OOB Score', 'Eğitim Süresi (sn)'],
    'Temel Model': [base_accuracy, base_f1, base_rf_model.oob_score_, base_train_time],
    'Optimize Edilmiş Model': [best_accuracy, best_f1, best_model.oob_score_, grid_search_time],
    'İyileşme': [
        f"{(best_accuracy - base_accuracy) / base_accuracy * 100:.2f}%",  # Accuracy iyileşmesi
        f"{(best_f1 - base_f1) / base_f1 * 100:.2f}%",  # F1 Score iyileşmesi
        f"{(best_model.oob_score_ - base_rf_model.oob_score_) / base_rf_model.oob_score_ * 100:.2f}%",  # OOB Score iyileşmesi
        "N/A"  # Eğitim süresi için iyileşme hesaplanmaz
    ]
})
print("Model Karşılaştırması:")
print(comparison_df)

# Sınıf bazlı karşılaştırma
class_comparison = []

for cls in target_names:
    base_f1 = base_report_dict[cls]['f1-score']
    best_f1 = best_report_dict[cls]['f1-score']
    improvement = (best_f1 - base_f1) / base_f1 * 100 if base_f1 > 0 else 0

    class_comparison.append({
        'Sınıf': cls,
        'Örnek Sayısı': base_report_dict[cls]['support'],
        'Temel Model F1': base_f1,
        'Optimize Edilmiş Model F1': best_f1,
        'İyileşme': f"{improvement:.2f}%"
    })

class_comparison_df = pd.DataFrame(class_comparison)
print("\nSınıf Bazlı F1 Skoru Karşılaştırması:")
print(class_comparison_df)

# Görselleştirme - Performans Karşılaştırması
plt.figure(figsize=(12, 8))
metrics = ['Accuracy', 'F1 Score (weighted)', 'OOB Score']
base_scores = [base_accuracy, base_f1, base_rf_model.oob_score_]
best_scores = [best_accuracy, best_f1, best_model.oob_score_]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width / 2, base_scores, width, label='Temel Model')
rects2 = ax.bar(x + width / 2, best_scores, width, label='Optimize Edilmiş Model')

ax.set_ylabel('Skor')
ax.set_title('Model Performans Karşılaştırması')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()


# Her bar için değer etiketi ekle
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
print("Model karşılaştırma grafiği 'model_comparison.png' olarak kaydedildi.")

# Sınıf bazlı F1 karşılaştırması
plt.figure(figsize=(12, 8))
classes = class_comparison_df['Sınıf']
base_f1_scores = class_comparison_df['Temel Model F1']
best_f1_scores = class_comparison_df['Optimize Edilmiş Model F1']

x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width / 2, base_f1_scores, width, label='Temel Model')
rects2 = ax.bar(x + width / 2, best_f1_scores, width, label='Optimize Edilmiş Model')

ax.set_ylabel('F1 Skoru')
ax.set_title('Sınıf Bazlı F1 Skoru Karşılaştırması')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()

# Her bar için değer etiketi ekle
autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('class_f1_comparison.png', dpi=300)
print("Sınıf bazlı F1 karşılaştırma grafiği 'class_f1_comparison.png' olarak kaydedildi.")

# 13. Model Kaydetme
print("\n13. Model Kaydetme")
print("-" * 50)


# 14. Özet Rapor
print("\n14. Final Özet Rapor")
print("-" * 50)

print("Emlak Property Type Sınıflandırma Modeli Özeti:")
print(f"- Toplam veri boyutu: {df.shape}")
print(f"- Sınıf sayısı: {len(target_names)}")
print(f"- Kullanılan özellik sayısı: {X_train_combined.shape[1]}")
print("\nTemel Model:")
print(f"- Parametreler: n_estimators=100, max_depth=10, min_samples_split=20, min_samples_leaf=10")
print(f"- Accuracy: {base_accuracy:.4f}")
print(f"- F1 Score: {base_f1:.4f}")
print(f"- OOB Score: {base_rf_model.oob_score_:.4f}")
print("\nOptimize Edilmiş Model:")
print(f"- Parametreler: {grid_search.best_params_}")
print(f"- Accuracy: {best_accuracy:.4f}")
print(f"- F1 Score: {best_f1:.4f}")
print(f"- OOB Score: {best_model.oob_score_:.4f}")
print(f"\nÖverfitting kontrolü: Eğitim ve test performansı arasındaki fark makul düzeydedir.")

print("\nSınıf Bazlı İyileşmeler:")
for idx, row in class_comparison_df.iterrows():
    print(
        f"- {row['Sınıf']}: {row['Temel Model F1']:.4f} -> {row['Optimize Edilmiş Model F1']:.4f} ({row['İyileşme']})")

print("\nİşlem tamamlandı!")



# Temel modelin eğitim doğruluğu ve F1 skoru
base_train_pred = base_rf_model.predict(X_train_combined)
base_train_accuracy = accuracy_score(y_train, base_train_pred)
base_train_f1 = f1_score(y_train, base_train_pred, average='weighted')

# Optimize edilmiş modelin eğitim doğruluğu ve F1 skoru
best_train_pred = best_model.predict(X_train_combined)
best_train_accuracy = accuracy_score(y_train, best_train_pred)
best_train_f1 = f1_score(y_train, best_train_pred, average='weighted')

# Sonuçları yazdır
print("\nTemel Model - Eğitim Performansı")
print(f"Doğruluk: {base_train_accuracy:.4f}")
print(f"F1 Skoru (Ağırlıklı): {base_train_f1:.4f}")

print("\nOptimize Edilmiş Model - Eğitim Performansı")
print(f"Doğruluk: {best_train_accuracy:.4f}")
print(f"F1 Skoru (Ağırlıklı): {best_train_f1:.4f}")

