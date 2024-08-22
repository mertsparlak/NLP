import numpy as np
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

nltk.download("stopwords")

yorumlar = pd.read_csv(r"C:\Users\merts\.spyder-py3\NLP\Restaurant_Reviews.csv", error_bad_lines=False)

# Veri işleme ve temizleme
ps = PorterStemmer()
derlem = []
for i in range(len(yorumlar)):
    yorum = re.sub("[^a-zA-Z]", " ", yorumlar["Review"][i])  # Sadece harfler kalsın
    yorum = yorum.lower()  # Tüm harfleri küçük yap
    yorum = yorum.split()  # Kelimeleri ayır
    yorum = [ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]  # Stopwordleri çıkar ve stem uygula
    yorum = " ".join(yorum)  # Kelimeleri tekrar birleştir
    derlem.append(yorum)

# Bag of Words
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(derlem).toarray()  # Bağımsız değişkenler

# Bağımlı değişken
y = yorumlar.iloc[:, 1].values  # Bağımlı değişken (beğenilip beğenilmediği)

# NaN değerlerini kontrol edip doldurma
y = np.nan_to_num(y, nan=0.0)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
# X trainden Y traini eğiticez, X testtekileri tahmin ettirip Y testtekilerle karşılaştırıcaz

# Model oluşturma ve eğitme
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Tahmin
y_pred = gnb.predict(X_test)

# Confusion matrix oluşturma
cm = confusion_matrix(y_test, y_pred)
print(cm)
# %70 accuary oranı var