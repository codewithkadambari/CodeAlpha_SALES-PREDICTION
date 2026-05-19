# Sales Prediction using Python

## 📌 Project Overview
This machine learning project focuses on predicting future sales based on advertising expenditures across various channels, including TV, Radio, and Newspaper. The analysis utilizes a Linear Regression algorithm to understand the impact of these marketing investments on total sales output.

## 🎯 Objectives
- **Data Exploration**: Investigate the dataset to understand the distribution of advertising budgets and sales figures.
- **Model Training**: Construct a Linear Regression model using `Scikit-Learn` to predict sales figures.
- **Model Evaluation**: Validate the model's accuracy and performance using statistical metrics such as Mean Squared Error (MSE) and R-Squared (R2 Score).
- **Visualization**: Generate insightful graphs to visually interpret the relationships between features and the model's predictive capability.

## 🛠️ Technologies & Libraries Used
- **Python 3.x**
- **Pandas**: Used for data ingestion, cleaning, and manipulation.
- **Scikit-Learn**: For dataset splitting, model fitting, predictions, and accuracy metrics.
- **Matplotlib & Seaborn**: For creating highly visual and interpretable analytical charts.

## 📊 Visualizations Generated
The script automatically produces four analytical charts, saving them directly as PNG files:
1.  **`sales_histogram.png`**: Displays the frequency distribution of total sales.
2.  **`sales_prediction.png`**: A scatter plot comparing the Actual Sales vs. Predicted Sales, visually representing model accuracy.
3.  **`sales_heatmap.png`**: A correlation heatmap showing the statistical relationship between different advertising channels and sales.
4.  **`sales_bargraph.png`**: A snapshot highlighting the top 10 sales records from the dataset.

## 🚀 How to Run the Project
1.  **Prerequisites**: Install the necessary data science libraries via pip:
    ```bash
    pip install pandas matplotlib seaborn scikit-learn
    ```
2.  **Dataset Location**: Ensure your `Advertising.csv` file is located at `C:\Users\HP\Downloads\Advertising.csv`. If it is saved elsewhere, please update the `file_path` in the script.
3.  **Execution**: Run the Python script from your terminal:
    ```bash
    python sales_prediction.py
    ```
4.  **Results**: The script will print the model's R2 Score and MSE to the console, and it will save the 4 data visualizations directly into your folder.

## 📝 License
This project is open-source and available under the [MIT License](LICENSE).
