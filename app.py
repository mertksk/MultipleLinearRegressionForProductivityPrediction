import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive use

import os
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from scipy import stats  # Import the stats module for Q-Q plot

app = Flask(__name__)

# Define global variables (initialization)
X_train, X_test, y_train, y_test = None, None, None, None
model, mse_train, r2_train = None, None, None
prediction = None  # Define prediction globally

def load_data():
    df = pd.read_csv('datagarment.csv')  # replace with your data file
    X = df[['department', 'quarter', 'no_of_workers', 'defects_day']].copy()

    production_speed = {'Gloves': 3, 'T-Shirt': 2, 'Sweatshirt': 1}
    X['production_speed'] = X['department'].map(production_speed)

    y = df['Total_Produced']
    return train_test_split(X, y, test_size=0.2, random_state=15, shuffle=True)

def train_model(X_train, y_train):
    categorical_features = ['department', 'quarter']
    numeric_features = ['no_of_workers', 'defects_day']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    model = make_pipeline(preprocessor, poly_features, LinearRegression())

    param_grid = {'polynomialfeatures__degree': [1, 2, 3]}
    grid_search = GridSearchCV(model, param_grid, cv=5)

    X_train_values = X_train
    y_train_values = y_train.values

    grid_search.fit(X_train_values, y_train_values)

    best_model = grid_search.best_estimator_

    y_pred_train = best_model.predict(X_train_values)
    mse_train = mean_squared_error(y_train_values, y_pred_train)
    r2_train = r2_score(y_train_values, y_pred_train)
    mse_train = mse_train / (y_train.max() - y_train.min())  # Normalize MSE

    print(f'R-squared (Train): {r2_train}')

    return best_model, mse_train, r2_train

# Load data and train model when the script runs
X_train, X_test, y_train, y_test = load_data()
model, mse_train, r2_train = train_model(X_train, y_train)


@app.route('/', methods=['GET', 'POST'])
def index():
    global prediction, mse_train, r2_train  # Include mse_train and r2_train here

    show_results = False
    department = None  # Initialize department outside the if block

    if request.method == 'POST':
        data = request.form
        print(f"Form data: {data}")
        department = data['department']  # Assign value inside the block
        prediction = predict(data)
        print(f"Prediction: {prediction}")

        # Retrain the model and update mse_train and r2_train
        model, mse_train, r2_train = train_model(X_train, y_train)

        # Generate and save graphs
        generate_and_save_graphs(X_test, y_test)

        # Set show_results to True when you want to display the results page
        show_results = True

    return render_template('index.html', prediction=prediction, mse_train=mse_train, r2_train=r2_train, show_results=show_results ,department=department)

@app.route('/Result', methods=['GET'])
def result():
    global prediction  # Use the global prediction variable

    if 'show_results' in request.args and request.args['show_results'] == 'true':
        return render_template('Result.html', prediction=prediction, mse_train=mse_train, r2_train=r2_train)
    else:
        # If show_results is False or not provided, redirect to the index page
        return redirect(url_for('index'))
def generate_and_save_graphs(X_test, y_test):
    static_folder = os.path.join(app.root_path, 'static')
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)

    # Residuals Distribution Plot
    y_pred_test = model.predict(X_test)
    residuals = y_test - y_pred_test

    # R-squared Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_test, color='green', alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
    plt.title('R-squared Plot')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    r_squared_plot_path = os.path.join(static_folder, 'r_squared_plot.png')
    plt.savefig(r_squared_plot_path)
    plt.close()

    # Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_test, color='purple', alpha=0.7)
    plt.title('Scatter Plot')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    scatter_plot_path = os.path.join(static_folder, 'scatter_plot.png')
    plt.savefig(scatter_plot_path)
    plt.close()

    # Residuals vs. Fitted Values Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred_test, residuals, color='orange', alpha=0.7)
    plt.title('Residuals vs. Fitted Values Plot')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.grid(True)
    residuals_vs_fitted_plot_path = os.path.join(static_folder, 'residuals_vs_fitted_plot.png')
    plt.savefig(residuals_vs_fitted_plot_path)
    plt.close()

    # Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=50, color='green', alpha=0.7)
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)
    histogram_path = os.path.join(static_folder, 'histogram.png')
    plt.savefig(histogram_path)
    plt.close()

    # Q–Q plot
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q–Q Plot')
    plt.grid(True)
    qq_plot_path = os.path.join(static_folder, 'qq_plot.png')
    plt.savefig(qq_plot_path)
    plt.close()

def predict(data):
    input_data = pd.DataFrame(data, index=[0])
    production_speed = {'Gloves': 3, 'T-Shirt': 2, 'Sweatshirt': 1}
    input_data['production_speed'] = input_data['department'].map(production_speed)
    input_data_transformed = model.named_steps['columntransformer'].transform(input_data)

    prediction = model.predict(input_data)
    rounded_prediction = int(round(prediction[0]))  # Round to the nearest integer
    return rounded_prediction

if __name__ == '__main__':
    app.run(debug=True)
