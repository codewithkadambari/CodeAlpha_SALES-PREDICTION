import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =========================================
# CODEALPHA DATA SCIENCE INTERNSHIP
# TASK 4 : SALES PREDICTION
# =========================================

def main():

    print("\n======================================")
    print(" SALES PREDICTION PROJECT ")
    print("======================================\n")

    # -----------------------------------
    # FILE PATH
    # -----------------------------------

    file_path = "C:\\Users\\HP\\Downloads\\Advertising.csv"

    # -----------------------------------
    # CHECK FILE EXISTS
    # -----------------------------------

    if not os.path.exists(file_path):

        print(" ERROR : Dataset file not found")
        print("\nCheck your file path.")
        return

    # -----------------------------------
    # LOAD DATASET
    # -----------------------------------

    df = pd.read_csv(file_path)

    print(" Dataset Loaded Successfully\n")

    # -----------------------------------
    # DISPLAY DATASET
    # -----------------------------------

    print("First 5 Rows:\n")

    print(df.head())

    print("\n-----------------------------------")
    print("Dataset Information")
    print("-----------------------------------\n")

    print(df.info())

    print("\n-----------------------------------")
    print("Missing Values")
    print("-----------------------------------\n")

    print(df.isnull().sum())

    # -----------------------------------
    # REMOVE MISSING VALUES
    # -----------------------------------

    df.dropna(inplace=True)

    # -----------------------------------
    # FEATURES & TARGET
    # -----------------------------------

    X = df[['TV', 'Radio', 'Newspaper']]

    y = df['Sales']

    # -----------------------------------
    # TRAIN TEST SPLIT
    # -----------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------------
    # MODEL TRAINING
    # -----------------------------------

    model = LinearRegression()

    model.fit(X_train, y_train)

    # -----------------------------------
    # PREDICTION
    # -----------------------------------

    y_pred = model.predict(X_test)

    # -----------------------------------
    # EVALUATION
    # -----------------------------------

    mse = mean_squared_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)

    print("\n-----------------------------------")
    print("Model Performance")
    print("-----------------------------------\n")

    print("Mean Squared Error :", mse)

    print("R2 Score :", r2)

    # -----------------------------------
    # VISUALIZATION
    # -----------------------------------

    sns.set_style("whitegrid")

    # HISTOGRAM

    plt.figure(figsize=(8,6))

    sns.histplot(df["Sales"], kde=True)

    plt.title("Sales Distribution")

    plt.savefig("sales_histogram.png")

    plt.show()

    # SCATTER PLOT

    plt.figure(figsize=(8,6))

    plt.scatter(y_test, y_pred)

    plt.xlabel("Actual Sales")

    plt.ylabel("Predicted Sales")

    plt.title("Actual vs Predicted Sales")

    plt.savefig("sales_prediction.png")

    plt.show()

    # HEATMAP

    plt.figure(figsize=(10,6))

    correlation = df.corr(numeric_only=True)

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm"
    )

    plt.title("Correlation Heatmap")

    plt.savefig("sales_heatmap.png")

    plt.show()

    # BAR GRAPH

    plt.figure(figsize=(10,6))

    df["Sales"].head(10).plot(kind='bar')

    plt.title("Top 10 Sales Records")

    plt.savefig("sales_bargraph.png")

    plt.show()

    # -----------------------------------
    # FINAL MESSAGE
    # -----------------------------------

    print("\n======================================")
    print(" PROJECT COMPLETED SUCCESSFULLY ")
    print("======================================\n")

    print("Generated Graph Files:\n")

    print("1. sales_histogram.png")
    print("2. sales_prediction.png")
    print("3. sales_heatmap.png")
    print("4. sales_bargraph.png")


# =========================================
# MAIN FUNCTION
# =========================================

if __name__ == "__main__":
    main()