# Workshop 4: Text Representation & Feature Selection (Spam Classification)

## Course Workshop 4  
**Topic:** TF-IDF, Feature Selection, and Classification  
**Dataset:** spam.csv  
**Submission:** Jupyter Notebook (.ipynb)  
**Student:** Thanachai ID:662115020

---

##  Objective
This workshop aims to apply advanced text representation and feature selection techniques on a spam classification task. The goal is to transform text data into numerical features using TF-IDF, reduce irrelevant features using variance thresholding, and evaluate model performance using proper classification metrics.

---

##  Dataset
- **File:** `spam.csv`
- **Description:**  
  The dataset contains SMS messages labeled as:
  - `ham` → normal message  
  - `spam` → spam message  

Only the first two columns are used:
- `label`
- `text`

---

## 🛠️ Methods & Steps

### 1. Text Preprocessing
The following preprocessing steps were applied:
- Convert text to lowercase
- Remove non-alphabetic characters
- Tokenize text
- Remove English stopwords

This produces a cleaned text column used for feature extraction.

---

### 2. TF-IDF Representation
- **Technique:** Term Frequency–Inverse Document Frequency (TF-IDF)
- **Tool:** `TfidfVectorizer`
- **Purpose:**  
  Convert text data into numerical vectors that reflect the importance of words in documents relative to the entire corpus.

---

### 3. Feature Selection (Variance Threshold)
- **Method:** VarianceThreshold
- **Threshold:** `0.1`
- **Purpose:**  
  Remove low-variance features that contribute little to classification performance.

The number of removed features is reported to show the effect of feature reduction.

---

### 4. Train-Test Split (Stratified Hold-Out)
- **Split Ratio:** 70% Train / 30% Test
- **Strategy:** Stratified hold-out
- **Shuffle:** False
- **Random State:** 1234

Stratification ensures that the class distribution (spam vs ham) is preserved in both sets.

---

### 5. Feature Matrix Shape Reporting
The shapes of training and testing feature matrices are reported to verify correct data splitting and feature selection.

---

### 6. Data Inspection
- Top 10 rows of the dataset
- Bottom 10 rows of the dataset

This step helps verify data integrity and preprocessing correctness.

---

### 7. Classification Model
- **Model:** K-Nearest Neighbors (KNN)
- **Purpose:**  
  Perform spam classification based on the selected TF-IDF features.

---

### 8. Evaluation Metrics
The following evaluation metrics are reported:
- **Accuracy**
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-score)

These metrics provide a complete view of model performance.

---

##  Results
- Accuracy score is printed
- Confusion matrix shows correct vs incorrect classifications
- Classification report provides precision, recall, and F1-score for each class

---

##  Files Included
- `Workshop4.ipynb` → Jupyter Notebook (main submission)
- `README.md` → Project explanation and documentation
- `spam.csv` → Dataset used (if allowed to submit)

---

##  Submission Notes
- The notebook is executed end-to-end without errors
- All workshop requirements are satisfied
- Submission format: **.ipynb only (no zip file)**
- Ready for upload to Moodle

---

##  Conclusion
This workshop demonstrates the complete pipeline of text preprocessing, TF-IDF representation, feature selection, and classification. The applied techniques improve model efficiency while maintaining strong classification performance.

