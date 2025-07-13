import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# df = pd.read_csv("dataset/survey_results_public.csv")
df = pd.read_csv("")

df = df[["Country","EdLevel","YearsCodePro","ConvertedCompYearly"]]

df = df.dropna(subset=["Country", "EdLevel", "YearsCodePro", "ConvertedCompYearly"])

# Preprocessing for cleaning experience

def clear_exp(x):
    if x == "Less than 1 year":
        return 0.5
    elif x == "More than 50 years":
        return 50
    return float(x)


df["YearsCodePro"] = df["YearsCodePro"].apply(clear_exp)


# Now for EdLevel

df["EdLevel"] = df["EdLevel"].apply(
    lambda x: (
        "Bachelor’s degree" if isinstance(x, str) and "Bachelor" in x
        else "Master’s degree" if isinstance(x, str) and "Master" in x
        else "Other"
    )
)


# Encode
le_country = LabelEncoder()
le_education = LabelEncoder()

df["Country"] = le_country.fit_transform(df["Country"])
df["EdLevel"] = le_education.fit_transform(df["EdLevel"])

# Train
X = df.drop("ConvertedCompYearly",axis=1)
y = df["ConvertedCompYearly"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = LinearRegression()
model.fit(X_train,y_train)

joblib.dump(model,"model/model.pkl")
joblib.dump(le_country, "model/le_country.pkl")
joblib.dump(le_education, "model/le_education.pkl")


