# Diabetes Prediction Project Feature Engineering Part

# Business Problem

# It is desired to develop a machine learning model that can predict whether people have diabetes when their 
# characteristics are specified. You are expected to perform the necessary data analysis and feature engineering 
# steps before developing the model.

# Dataset Story

# The dataset is part of the large dataset held at the National Institutes of Diabetes-Digestive-Kidney Diseases 
# in the USA. Data used for diabetes research on Pima Indian women aged 21 and over living in Phoenix, the 5th 
# largest city of the State of Arizona in the USA.
# The target variable is specified as "outcome"; 1 indicates positive diabetes test result, 0 indicates negative.

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Step 1: Examine the overall picture.
df = pd.read_csv("datasets/diabetes.csv")
df.head()
df.describe().T

def check_df(df, head=5):
    print("##################### Shape #####################")
    print(df.shape)

    print("##################### Types #####################")
    print(df.dtypes)

    print("##################### Head #####################")
    print(df.head(head))

    print("##################### Tail #####################")
    print(df.tail(head))

    print("##################### is null? #####################")
    print(df.isnull().sum())

    print("##################### Quantiles #####################")
    print(df.quantile([0, 0.25, 0.50, 0.75, 0.99, 1]).T)
    print(df.describe().T)


check_df(df)


# Step 2: Capture the numeric and categorical variables.
def grab_col_names(df, cat_th=10, car_th=20):
    """

        Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
        Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

        Parameters
        ------
            df: dataframe
                    Değişken isimleri alınmak istenilen dataframe
            cat_th: int, optional
                    numerik fakat kategorik olan değişkenler için sınıf eşik değeri
            car_th: int, optinal
                    kategorik fakat kardinal değişkenler için sınıf eşik değeri

        Returns
        ------
            cat_cols: list
                    Kategorik değişken listesi
            num_cols: list
                    Numerik değişken listesi
            cat_but_car: list
                    Kategorik görünümlü kardinal değişken listesi

        Examples
        ------
            import seaborn as sns
            df = sns.load_dataset("iris")
            print(grab_col_names(df))


        Notes
        ------
            cat_cols + num_cols + cat_but_car = toplam değişken sayısı
            num_but_cat cat_cols'un içerisinde.
            Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

        """

 # cat_cols, cat_but_car
    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes != "O"]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and df[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in df.columns if df[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Numerical Columns
num_cols

# Categorical Columns
cat_cols

# Categoric but Cardinal
cat_but_car

# Step 3: Analyze the numerical and categorical variables.

# Categorical variable analysis
def cat_summary(df, col_name, plot=False):
    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                        "Ratio": 100 * df[col_name].value_counts() / len(df)}))
    print("##########################################")
    if plot:
        sns.countplot(x=df[col_name], data=df)
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot=True)

# Numerical variable analysis
def num_summary(df, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(df[numerical_col].describe(quantiles).T)

    if plot:
        df[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

# Step 4: Perform target variable analysis. (The mean of the target variable according to the categorical variables,
# mean of numeric variables)

#Analysis of categorical variables according to target variable

def target_summary_with_cat(df, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"Target_Mean": df.groupby(cat_cols)[target].mean(),
                        "Count": df[categorical_col].value_counts(),
                        "Ratio": 100 * df[categorical_col].value_counts() / len(df)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Outcome", col)

# Analysis of numerical variables according to target variable

def target_summary_with_num(df, target, numerical_col):
    print(df.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Outcome", col)


# Step 5: Analyze outliers.
# Let's catch outliers
q1 = df["Outcome"].quantile(0.05)
q3 = df["Outcome"].quantile(0.95)
iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

df[(df["Outcome"] < low) | (df["Outcome"] > up)]

df[(df["Outcome"] < low) | (df["Outcome"] > up)].index

# is there an outlier or not?
df[(df["Outcome"] < low) | (df["Outcome"] > up)].any(axis=None)
df[(df["Outcome"] < low)].any(axis=None)

def outlier_thresholds(df, col_name, q1=0.05, q3=0.95):
    quartile1 = df[col_name].quantile(q1)
    quartile3 = df[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(df, col_name):
    low_limit, up_limit = outlier_thresholds(df, col_name)
    if df[(df[col_name] > up_limit) | (df[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Outcome")

def replace_with_thresholds(df, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(df, variable, q1=0.05, q3=0.95)
    df.loc[(df[variable] < low_limit), variable] = low_limit
    df.loc[(df[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))

# Step 6: Missing observation analysis
df.isnull().sum()

# Step 7: Perform correlation analysis.
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="RdPu")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# Model
y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17, shuffle=True, stratify=y)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# By establishing the model in this way, we determined that the model prediction system we set up with the Random Forest method had a success rate of 77%.

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)

# Task 2: Feature Engineering
# Step 1: Take necessary actions for missing and outlier values. There are no missing observations in the data set, but Glucose, Insulin etc.
# observation units containing 0 in variables may represent missing values. For example; a person's glucose or insulin value is 0
# will not be. Considering this situation, assigning the zero values to NaN in the relevant values and then missing the missing values.
# You can apply operations to values.

def maybe_missing(df, col_name):
    variables = df[df[col_name] == 0].shape[0]
    return variables

for col in num_cols:
    print(col, maybe_missing(df, col))


na_cols = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
na_cols

for col in na_cols:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

for col in num_cols:
    print(col, maybe_missing(df, col))

df.isnull().sum()

def missing_values_table(df, na_name=False):
    na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (df[na_columns].isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)

# Examining the Relationship of Missing Values with the Dependent Variable

def missing_vs_target(df, target, na_columns):
    temp_df = df.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Outcome", na_columns)

# In this output, there is seem that there is missing values of each variable. It should be done that application of different methods for fill na values.

# Filling in missing values
for col in na_columns:
     df.loc[df[col].isnull(), col] = df[col].median()

df.isnull().sum()

# Outlier Suppression
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in df.columns:
    print(col, check_outlier(df, col))

# Outliers are still in there in the name of Insulin adn SkinThickness variable

dff = pd.get_dummies(df[["Insulin","SkinThickness"]], drop_first=True)
dff.isnull().sum()

# Standardization of variables
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

# Implement the KNN method
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)

dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff.head()

df["Insulin"] = dff["Insulin"]
df["SkinThickness"]= dff["SkinThickness"]
df.isnull().sum()

# Step 2: Create new variables.
df.head()

# Category of Age
df["Age_Cat"] = pd.cut(
    df["Age"], bins=[0, 15, 25, 64, 82],
    labels=["Child", "Young", "Adult", "Senior"])
df["Age_Cat"].head()

# BMI Group
df.loc[(df['BMI'] < 18.5), 'BMI_Group'] = 'Underweight'
df.loc[((df['BMI'] >= 18.5) & (df['BMI'] < 25)), 'BMI_Group'] = 'Normal'
df.loc[((df['BMI'] >= 25) & (df['BMI'] < 30)), 'BMI_Group'] = 'Overweight'
df.loc[(df['BMI'] >= 30), 'BMI_Group'] = 'Obese'

# Glucose level
def glucose_level(dataframe, col_name="Glucose"):
    if 16 <= dataframe[col_name] <= 140:
        return "Normal"
    else:
        return "Abnormal"
df["Glucose_Level"] = df.apply(glucose_level, axis=1)
df["Glucose_Level"].head()

# Insulin Level
def insulin_level(dataframe):
    if dataframe["Insulin"] <= 100:
        return "Normal"
    if dataframe["Insulin"] > 100 and dataframe["Insulin"] <= 126:
        return "Prediabetes"
    elif dataframe["Insulin"] > 126:
        return "Diabetes"
df["Insulin_Level"] = df.apply(insulin_level, axis=1)

# Blood Pressure Level
def bloodpressure_level(dataframe):
    if dataframe["BloodPressure"] <= 79:
        return "Normal"
    if dataframe["BloodPressure"] > 79 and dataframe["BloodPressure"] <= 89:
        return "Prehypertension"
    elif dataframe["BloodPressure"] > 89:
        return "Hypertension"
df["Bloodpressure_Level"] = df.apply(bloodpressure_level, axis=1)

# Sugar level according to body mass index
df["glucose per bmi"] = df["Glucose"] / df["BMI"]

# Insulin level by age
df["insulin_per_age"] = df["Insulin"] / df["Age"]

df.head()

# Step 3: Perform the encoding operations.
# Label Encoding:
le = LabelEncoder()
le.fit_transform(df["Outcome"])[0:5]
le.inverse_transform([0, 1])

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2 and col not in ["OUTCOME"]]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

# Since we have both the glucose level and our target variable in our cat_cols variables, we choose the variables that do not have them and ohe

# One-Hot Encoding:
def one_hot_encoder(df, categorical_cols, drop_first=True):
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
    return df

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols, drop_first=True)
ohe_cols

df.head()

# Step 4: Standardize for numeric variables.

ss = StandardScaler()
df[num_cols] = ss.fit_transform(df[num_cols])
df.head()

# Step 5: Create the model.
y = df["Outcome"]
X = df.drop(["Outcome",'BMI','Insulin','Glucose','BloodPressure','Age'], axis=1)
X_train, X_text, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_text)
accuracy_score(y_pred, y_test)

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)

