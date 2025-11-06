# ================================================================
# 💬 CUSTOMER SATISFACTION PREDICTION DASHBOARD
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ================================================================
# 🎨 STREAMLIT PAGE SETTINGS
# ================================================================
st.set_page_config(page_title="Customer Satisfaction Prediction", layout="wide")
st.title("💬 Customer Satisfaction Prediction App")
st.write("Predict customer satisfaction from support ticket data 📊")

# ================================================================
# 📂 UPLOAD CSV DATA
# ================================================================
uploaded_file = st.file_uploader("📁 Upload your CSV dataset", type=["csv"])

if uploaded_file is None:
    st.info("⬆️ Please upload your dataset to begin.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.success("✅ Dataset loaded successfully!")
st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
st.dataframe(df.head())

# ================================================================
# 🎯 DETECT TARGET COLUMN (Satisfaction)
# ================================================================
target_col = None
for col in df.columns:
    if "satisf" in col.lower():
        target_col = col
        break

if not target_col:
    st.error("❌ Could not find a satisfaction column (e.g. 'Customer Satisfaction Rating').")
    st.stop()

st.success(f"🎯 Target column detected: **{target_col}**")

# ================================================================
# 🧹 CLEAN DATA
# ================================================================
df = df.dropna(subset=[target_col])
df[target_col] = df[target_col].astype(str)

# Detect numeric, categorical, and text features
numeric_candidates = ['Customer Age', 'resp_delay_hours', 'resolve_delay_hours',
                      'response_efficiency', 'ticket_count', 'ticket_sentiment']
categorical_candidates = ['Customer Gender', 'Product Purchased', 'Ticket Type',
                          'Ticket Status', 'Ticket Priority', 'Ticket Channel']
text_candidates = ['Ticket Description', 'Ticket Subject']

numeric_features = [c for c in numeric_candidates if c in df.columns]
categorical_features = [c for c in categorical_candidates if c in df.columns]
text_feature = next((c for c in text_candidates if c in df.columns), None)

st.write("🔍 **Detected Features:**")
st.write(f"- Numeric: {numeric_features}")
st.write(f"- Categorical: {categorical_features}")
st.write(f"- Text: {text_feature}")

# ================================================================
# 📊 DATA VISUALIZATION
# ================================================================
st.subheader("📊 Data Visualization")

col1, col2 = st.columns(2)
with col1:
    st.write("**Distribution of Satisfaction Ratings**")
    fig, ax = plt.subplots()
    sns.countplot(x=df[target_col], palette="viridis", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col2:
    if text_feature:
        st.write(f"**WordCloud for {text_feature}**")
        text_data = " ".join(str(x) for x in df[text_feature].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

# ================================================================
# ⚙️ PREPROCESSING PIPELINE
# ================================================================
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
num_transformer = Pipeline([('scaler', StandardScaler())])
cat_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])

transformers = []
if numeric_features:
    transformers.append(('num', num_transformer, numeric_features))
if categorical_features:
    transformers.append(('cat', cat_transformer, categorical_features))
if text_feature:
    transformers.append(('text', tfidf, text_feature))

preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0.3)

# ================================================================
# 🧠 MODEL (Random Forest + SMOTE)
# ================================================================
pipeline = ImbPipeline([
    ('pre', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
])

# ================================================================
# 🏋️ TRAIN MODEL
# ================================================================
X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.info("🚀 Training the model... please wait")
pipeline.fit(X_train, y_train)
st.success("✅ Model trained successfully!")

# ================================================================
# 📈 EVALUATION
# ================================================================
y_pred = pipeline.predict(X_test)
acc = round(accuracy_score(y_test, y_pred) * 100, 2)

st.subheader("📈 Model Performance Summary")
st.metric(label="Accuracy", value=f"{acc}%")

report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Confusion Matrix
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ================================================================
# 🌟 FEATURE IMPORTANCE (Fixed)
# ================================================================
st.subheader("🌟 Top Feature Importance")

model = pipeline.named_steps['clf']

try:
    preprocessor = pipeline.named_steps['pre']
    feature_names = []

    # Numeric
    if 'num' in preprocessor.named_transformers_:
        feature_names += numeric_features

    # Categorical
    if 'cat' in preprocessor.named_transformers_:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
        feature_names += cat_names

    # Text
    if 'text' in preprocessor.named_transformers_:
        text_encoder = preprocessor.named_transformers_['text']
        text_names = [f"tfidf_{i}" for i in range(len(text_encoder.get_feature_names_out()))]
        feature_names += text_names

    n_importances = len(model.feature_importances_)
    imp = pd.DataFrame({
        "Feature": feature_names[:n_importances],
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=imp, palette="mako", ax=ax)
    st.pyplot(fig)

except Exception as e:
    st.warning(f"⚠️ Could not extract feature importances: {e}")

# ================================================================
# 💬 LIVE PREDICTION
# ================================================================
st.subheader("🎯 Predict Customer Satisfaction")

input_data = {}
for col in numeric_features + categorical_features:
    if col in numeric_features:
        input_data[col] = st.number_input(f"Enter {col}", value=0.0)
    else:
        options = df[col].dropna().unique().tolist()
        input_data[col] = st.selectbox(f"Select {col}", options)

if text_feature:
    input_data[text_feature] = st.text_area(f"Enter {text_feature}", height=100)

if st.button("🔮 Predict Satisfaction"):
    user_df = pd.DataFrame([input_data])
    pred = pipeline.predict(user_df)[0]
    st.success(f"✅ Predicted Satisfaction: **{pred}**")

    pred_low = str(pred).lower()
    if any(x in pred_low for x in ["5", "excellent", "very satisfied", "satisfied"]):
        st.markdown("🟢 **Customer is Highly Satisfied!**")
    elif any(x in pred_low for x in ["4", "good", "neutral", "okay"]):
        st.markdown("🟡 **Customer is Moderately Satisfied.**")
    else:
        st.markdown("🔴 **Customer is Unsatisfied.**")
