"""Import necessary libraries."""
from sklearn.metrics import r2_score
import random
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sdv.metadata import SingleTableMetadata
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, cross_val_predict
import pandas as pd
from joblib import dump, load
from sdv.single_table import GaussianCopulaSynthesizer
import json
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from io import StringIO
import matplotlib
import xgboost as xgb
import yaml
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


import Maor_Nave_313603391_EX3_models_Final


def drop_multy_cols(df, cols_list):
    """Drop specified columns from a DataFrame."""
    df_new=df.drop(columns = cols_list)
    return df_new

def calculate_vif(df):
    """Calculate Variance Inflation Factor for each feature in a DataFrame."""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data

def save_txt(path, data):
    """Save DataFrame information to a text file."""
    buffer = StringIO()
    data.info(buf=buffer)
    info_str = buffer.getvalue()
    with open(path, 'w') as f:
        f.write(info_str)
    f.close()

def save_json(path, data):
    """Save data to a JSON file."""
    with open(path, "w", encoding='utf-8') as file:
        file.write(json.dumps(data))
    file.close()

def load_yaml(path):
    """Load YAML data from the specified file path using safe loader."""
    with open(path, 'r') as file:
        yaml_data = yaml.load(file, Loader=yaml.SafeLoader)
    file.close()
    return yaml_data


def fill_cat_data_random(df, col_name, test=False):
    """Fill categorical data randomly in a DataFrame."""
    if col_name == 'Exterior_Color':
        if test:
            df[col_name] = df[col_name].replace('–',
                                                random.choice([val for val in df[col_name].unique() if val != '–']))
        else:
            df[col_name] = df[col_name].replace('01', random.choice([val for val in df[col_name].unique() if val != '01']))
            df[col_name] = df[col_name].replace('unknown', random.choice([val for val in df[col_name].unique() if val != 'unknown']))
            df[col_name] = df[col_name].replace('a', random.choice([val for val in df[col_name].unique() if val != 'a']))
            df[col_name] = df[col_name].replace('other', random.choice([val for val in df[col_name].unique() if val != 'other']))
            df[col_name] = df[col_name].replace('wx / wx', random.choice([val for val in df[col_name].unique() if val != 'wx / wx']))
            df[col_name] = df[col_name].replace('wy', random.choice([val for val in df[col_name].unique() if val != 'wy']))
            df[col_name] = df[col_name].replace('–', random.choice([val for val in df[col_name].unique() if val != '–']))
    elif col_name == 'Interior_Color':
        if test:
            df[col_name] = df[col_name].replace('gray/black', 'black / gray')
            df[col_name] = df[col_name].replace('grey', 'gray')
            df[col_name] = df[col_name].replace('biege', 'beige')
            df[col_name] = df[col_name].replace('biege', 'beige')
            df[col_name] = df[col_name].replace('black / red', 'red / black')
            df[col_name] = df[col_name].replace('black/red', 'red / black')
            df[col_name] = df[col_name].replace('.',
                                                random.choice([val for val in df[col_name].unique() if val != '.']))
            df[col_name] = df[col_name].replace('–',
                                                random.choice([val for val in df[col_name].unique() if val != '–']))
        else:
            df[col_name] = df[col_name].replace('black / red', 'red / black')
            df[col_name] = df[col_name].replace('black/red', 'red / black')
            df[col_name] = df[col_name].replace('nighthawk black pearl / grey', 'nighthawk black pearl / gray')
            df[col_name] = df[col_name].replace('gray/blue', 'gray / blue')
            df[col_name] = df[col_name].replace('grey', 'gray')
            df[col_name] = df[col_name].replace('interior', random.choice([val for val in df[col_name].unique() if val != 'interior']))
            df[col_name] = df[col_name].replace('select', random.choice([val for val in df[col_name].unique() if val != 'select']))
            df[col_name] = df[col_name].replace('–', random.choice([val for val in df[col_name].unique() if val != '–']))
    elif col_name == 'Drivetrain':
        if test:
            df[col_name] = df[col_name].replace('All-wheel Drive', 'AWD')
            df[col_name] = df[col_name].replace('Front-wheel Drive', 'FWD')
            df[col_name] = df[col_name].replace('Four-wheel Drive', '4WD')
        else:
            df[col_name] = df[col_name].replace('All-wheel Drive', 'AWD')
            df[col_name] = df[col_name].replace('Front-wheel Drive', 'FWD')
            df[col_name] = df[col_name].replace('Rear-wheel Drive', 'RWD')
            df[col_name] = df[col_name].replace('Four-wheel Drive', '4WD')
            df[col_name] = df[col_name].replace('–', random.choice([val for val in df[col_name].unique() if val != '–']))
    elif col_name == 'Transmission':
        if test:
            df[col_name] = df[col_name].replace('1-speed cvt w/od', '1-speed cvt with overdrive')
            df[col_name] = df[col_name].replace('10-speed a/t', '10-speed automatic')
            df[col_name] = df[col_name].replace('10 speed automatic', '10-speed automatic')
            df[col_name] = df[col_name].replace('6-speed m/t', '6-speed manual')
            df[col_name] = df[col_name].replace('9-speed a/t', '9-speed automatic')
            df[col_name] = df[col_name].replace('automatic, 6-spd', '6-speed automatic')
            df[col_name] = df[col_name].replace('continuously variable', 'cvt')
            df[col_name] = df[col_name].replace('continuously variable (cvt)', 'cvt')
            df[col_name] = df[col_name].replace('continuously variable (m cvt)', 'cvt')
            df[col_name] = df[col_name].replace('continuously variable automatic', 'cvt')
            df[col_name] = df[col_name].replace('continuously variable w/sport mode', 'cvt w/sport mode')
            df[col_name] = df[col_name].replace('continuously variable w/sport mode', 'cvt w/sport mode')
            df[col_name] = df[col_name].replace('ecvt', 'cvt')
            df[col_name] = df[col_name].replace('electric continuously variable', 'cvt')
            df[col_name] = df[col_name].replace('electronic continuously variable', 'cvt')
            df[col_name] = df[col_name].replace('–',
                                                random.choice([val for val in df[col_name].unique() if val != '–']))

        else:
            df[col_name] = df[col_name].replace('10 speed automatic', '10-speed automatic')
            df[col_name] = df[col_name].replace('10-speed a/t', '10-speed automatic')
            df[col_name] = df[col_name].replace('4-speed a/t', '4-speed automatic')
            df[col_name] = df[col_name].replace('5 speed automatic', '5-speed automatic')
            df[col_name] = df[col_name].replace('automatic, 5-spd', '5-speed automatic')
            df[col_name] = df[col_name].replace('5-speed a/t', '5-speed automatic')
            df[col_name] = df[col_name].replace('5-speed m/t', '5-speed manual')
            df[col_name] = df[col_name].replace('6-speed a/t', '6-speed automatic')
            df[col_name] = df[col_name].replace('6 speed automatic', '6-speed automatic')
            df[col_name] = df[col_name].replace('6-speed m/t', '6-speed manual')
            df[col_name] = df[col_name].replace('9-speed a/t', '9-speed automatic')
            df[col_name] = df[col_name].replace('9 speed automatic', '9-speed automatic')
            df[col_name] = df[col_name].replace('9-speed', '9-speed automatic')
            df[col_name] = df[col_name].replace('automatic cvt', 'cvt')
            df[col_name] = df[col_name].replace('continuously variable', 'cvt')
            df[col_name] = df[col_name].replace('continuously variable (cvt)', 'cvt')
            df[col_name] = df[col_name].replace('continuously variable (ll cvt)', 'cvt')
            df[col_name] = df[col_name].replace('continuously variable (m cvt)', 'cvt')
            df[col_name] = df[col_name].replace('continuously variable automatic', 'cvt')
            df[col_name] = df[col_name].replace('continuously variable transmission', 'cvt')
            df[col_name] = df[col_name].replace('cvt transmission', 'cvt')
            df[col_name] = df[col_name].replace('e continuously variable (e cvt)', 'cvt')
            df[col_name] = df[col_name].replace('ecvt', 'cvt')
            df[col_name] = df[col_name].replace('electric continuously variable', 'cvt')
            df[col_name] = df[col_name].replace('electronic continuously variable', 'cvt')
            df[col_name] = df[col_name].replace('electronically controlled automatic', 'automatic')
            df[col_name] = df[col_name].replace('other', random.choice([val for val in df[col_name].unique() if val != 'other']))
            df[col_name] = df[col_name].replace('–', random.choice([val for val in df[col_name].unique() if val != '–']))
    elif col_name == 'Engine':
        if test:
            pass
        else:
            df[col_name] = df[col_name].replace('1.5L 4 Cyl.', '1.5L 4 Cylinder')
            df[col_name] = df[col_name].replace('1.5L I4 DOHC', '1.5L I4 DOHC 16V')
            df[col_name] = df[col_name].replace('2.0L 4 Cyl', '2.0L 4 Cylinder')
            df[col_name] = df[col_name].replace('2.4L DOHC MPFI 16-valve i-VTEC I4 engine', '2.4L DOHC MPFI i-VTEC 16-valve I4 engine')
            df[col_name] = df[col_name].replace('Engine: 3.5L V6 24-Valve SOHC i-VTEC', '3.5L V6 24-Valve SOHC i-VTEC')
            df[col_name] = df[col_name].replace('Gas/Electric I-4 1.5 L/91', 'Gas/Electric I4 1.5L/91')
    elif col_name == 'Seller_Type':
        df[col_name] = df[col_name].fillna('Dealer')
    elif col_name == 'State':
        if test:
            pass
        else:
            df[col_name] = df[col_name].replace('Glens', 'NY')
            df[col_name] = df[col_name].replace('MO-22', 'MO')
            df[col_name] = df[col_name].replace('Route',
                                                random.choice([val for val in df[col_name].unique() if val != 'Route']))
    elif col_name == 'MPG':
        if test:
            df[col_name] = df[col_name].fillna(
                random.choice([val for val in df[col_name].unique() if not pd.isna(val)]))
            df[col_name] = df[col_name].replace('19–0', '0–19')
            df[col_name] = df[col_name].replace('40–34', '34–40')
            df[col_name] = df[col_name].replace('40–35', '35–40')
            df[col_name] = df[col_name].replace('44–41', '41–44')
            df[col_name] = df[col_name].replace('41–46', '46–41')
            df[col_name] = df[col_name].replace('48–47', '47–48')
            df[col_name] = df[col_name].replace('49–47', '47–49')
            df[col_name] = df[col_name].replace('51–45', '45–51')
            df[col_name] = df[col_name].replace('55–49', '49–55')
        else:
            df[col_name] = df[col_name].fillna(random.choice([val for val in df[col_name].unique() if not pd.isna(val)]))
            df[col_name] = df[col_name].replace('20–27.0', '20–27')
            df[col_name] = df[col_name].replace('28–0', '0–28')
            df[col_name] = df[col_name].replace('40–34', '34–40')
            df[col_name] = df[col_name].replace('40–35', '35–40')
            df[col_name] = df[col_name].replace('43–36', '36–43')
            df[col_name] = df[col_name].replace('44–41', '41–44')
            df[col_name] = df[col_name].replace('48–47', '47–48')
            df[col_name] = df[col_name].replace('49–47', '47–49')
            df[col_name] = df[col_name].replace('50–45', '45–50')
            df[col_name] = df[col_name].replace('51–45', '45–51')
            df[col_name] = df[col_name].replace('55–49', '49–55')
            df[col_name] = df[col_name].replace('8–34.6', '8–34')
            df[col_name] = df[col_name].replace('0–0', random.choice([val for val in df[col_name].unique() if val != '0–0']))
    if col_name == 'Model':
        df[col_name] = df[col_name].replace('civic si si', 'civic si')

    return df


def check_unique_vals(df, paths_dict):
    """Check unique values in each column of a DataFrame and save to a JSON file."""
    data_dict_unique = {}
    for col in df.columns:
        data_dict_unique[col] = df[col].unique().tolist()
    os.makedirs(paths_dict['path_output'], exist_ok=True)
    save_json(os.path.join(paths_dict['path_output'], 'unique_data_after_columns_clean_cars_df.json') ,data_dict_unique)


def clean_multy_cat_data(df, test=False):
    """Clean categorical data in a DataFrame."""
    df['Exterior_Color'] = df['Exterior_Color'].str.lower()
    df = fill_cat_data_random(df, 'Exterior_Color', test)
    df['Interior_Color'] = df['Interior_Color'].str.lower()
    df = fill_cat_data_random(df, 'Interior_Color', test)
    df = fill_cat_data_random(df, 'Drivetrain', test)
    df['Transmission'] = df['Transmission'].str.lower()
    df = fill_cat_data_random(df, 'Transmission', test)
    df = fill_cat_data_random(df, 'Engine', test)
    df = fill_cat_data_random(df, 'State', test)
    df = fill_cat_data_random(df, 'Seller_Type', test)
    df = fill_cat_data_random(df, 'MPG', test)
    if test:
        df['Model'] = df['Model'].str.lower()
        df = fill_cat_data_random(df, 'Model', test)

    return df

def clean_multy_num_data(df, test=False):
    """Clean numerical data in a DataFrame."""
    if test:
        col_list = ['Comfort_Rating', 'Interior_Design_Rating', 'Performance_Rating', 'Value_For_Money_Rating',
                'Exterior_Styling_Rating', 'Reliability_Rating']
    else:
        col_list = ['Price', 'Comfort_Rating', 'Interior_Design_Rating', 'Performance_Rating', 'Value_For_Money_Rating',
                'Exterior_Styling_Rating', 'Reliability_Rating']
    for col in col_list:
        # Calculate mean and standard deviation from non-missing values
        mean_value = df[col].mean()
        std_value = df[col].std()
        if col == 'Price':
            # Generate random values from a normal distribution
            random_values = np.random.normal(mean_value, std_value, (df[col]==0).sum())
            df.loc[df[col]==0, col] = random_values.astype(int)
        else:
            # Generate random values from a normal distribution
            random_values = np.clip(np.random.normal(mean_value, std_value, df[col].isnull().sum()), 1, 5)
            formatted_random_values = [float(f'{val:.1f}') for val in random_values]
            # Fill missing values with random values
            df.loc[df[col].isnull(), col] = formatted_random_values

    return df

def convert_data_to_num_cat_vals(df):
    """Convert categorical data to numerical values using LabelEncoder."""
    label_encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column])

    return df


def visualize_corr_and_hist(df, output_folders):
    """Visualize correlation and histograms of features in a DataFrame."""
    #use agg to not poping the figures
    matplotlib.use('agg')
    # Check the data in each attribute
    os.makedirs(os.path.join(output_folders['path_main_vis_hist'], 'df_info'), exist_ok=True)
    os.makedirs(os.path.join(output_folders['path_main_vis_hist'], 'dist'), exist_ok=True)
    os.makedirs(os.path.join(output_folders['path_main_vis_hist'], 'relationship'), exist_ok=True)
    os.makedirs(os.path.join(output_folders['path_main_vis_hist'], 'corr'), exist_ok=True)
    # save the df info
    save_txt(os.path.join(output_folders['path_main_vis_hist'], 'df_info', 'df_data_cars.txt'), df)
    # Visualize DataFrame as a histogram and save the plots
    for column in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[column], kde=True)
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {column}")
        plt.savefig(f"{output_folders['path_main_vis_hist']}\dist\cars_hist_{column}.png")
        plt.close()
    # Plot the relationship between each col to the depoendent col
    for column in df.columns:
        if column == 'Price':
            continue
        plt.figure(figsize=(8, 6))
        sns.countplot(x=column, hue='Price', data=df)
        plt.title(f'Relationship between {column} and Dependent Variable')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.legend(title='Dependent Variable', loc='upper right')
        plt.savefig(f"{output_folders['path_main_vis_hist']}\\relationship\\cars_relCol_{column}.png")
    # Plot the corr between each col to the depoendent col
    for column in df.columns:
        if column == 'Price':
            continue
        if df[column].dtype == 'object' or df[column].dtype == 'bool':
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=df, x='Price', y= column)
            plt.title('Scatter Plot of Feature Column vs. Dependent Variable')
            plt.xlabel('Dependent Variable - Price')
            plt.ylabel(f'Feature Column - {column}')
            plt.savefig(f"{output_folders['path_main_vis_hist']}\corr\cars_corr_{column}.png")
        else:
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=df, x=column, y= 'Price')
            plt.title('Scatter Plot of Feature Column vs. Dependent Variable')
            plt.xlabel(f'Feature Column - {column}')
            plt.ylabel('Dependent Variable - Price')
            plt.savefig(f"{output_folders['path_main_vis_hist']}\corr\cars_corr_{column}.png")


def gen_syn_data(df):
    """Generate synthetic data using Gaussian Copula Synthesizer."""
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df)
    synthetic_data = synthesizer.sample(num_rows=5000)

    full_data =  pd.concat([df,synthetic_data])
    full_data.to_csv('input/cars_clean_with_syn_data.csv')

    return full_data



"""Main code for data preprocessing and modeling."""
def main():
    # load and EDA data
    cars_df = pd.read_csv('input\cars.csv')
    matplotlib.use('agg')
    # load_yaml_file
    config = load_yaml('Maor_Nave_313603391_EX3_Config_Final.yaml')
    output_path_kwargs = config['params']['output_paths_kwargs']

    # check unique values
    if config['EX2']['functions']['check_unique_vals']:
        check_unique_vals(cars_df, output_path_kwargs)
    # Visualize correlation and histograms
    if config['EX2']['functions']['visualize_corr_and_hist']:
        visualize_corr_and_hist(cars_df, output_path_kwargs)
    # Clean and save data
    if config['EX2']['functions']['clean_and_save_data']:
        cars_df_after_drop = drop_multy_cols(cars_df, ['VIN', 'Consumer_Review_#', 'Stock_#'])
        cars_df_after_clean_cat = clean_multy_cat_data(cars_df_after_drop)
        cars_df_after_clean_cat.to_csv('input\cars_clean_cat.csv')
        cars_df_after_clean_cat = pd.read_csv('input\cars_clean_cat.csv', index_col =0)
        cars_df_after_clean_catAndnum = clean_multy_num_data(cars_df_after_clean_cat)
        cars_df_after_clean_catAndnum.to_csv('input\cars_clean_cat_num.csv')
    else:
        cars_df_after_clean_catAndnum = pd.read_csv('input\cars_clean_cat_num.csv', index_col =0)


    # Convert categorical data to numerical values
    cars_df_after_label_encoding = convert_data_to_num_cat_vals(cars_df_after_clean_catAndnum)
    # Generate synthetic data
    if config['EX2']['functions']['create_syn_data']:
        cars_df_after_syn_data = gen_syn_data(cars_df_after_label_encoding)

    if config['EX2']['functions']['load_syn_data']:
        cars_df_after_label_encoding = pd.read_csv('input/cars_clean_with_syn_data.csv', index_col =0)

    # Define models
    models = Maor_Nave_313603391_EX3_models_Final.models_ex2
    # split the data to X and Y
    X = cars_df_after_label_encoding.drop(columns=['Price'])  # Exclude 'price' column
    y = cars_df_after_label_encoding['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Fit and scale data
    scaler = StandardScaler()
    model_accuracies = {}
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_scaled =  pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    # Find best model
    if config['EX2']['functions']['find_best_model']:
        # Loop over each model
        for model_name, (model, param_grid) in models.items():
            print(f"Training {model_name}...")
            # Initialize Feature selection method
            if model_name in ['SVR', 'Gradient_Boosting' , 'KNN_Regressor', 'MLPRegressor'] :
                selector = SelectKBest(score_func=f_regression, k='all')
                X_train_selected = selector.fit_transform(X_train_scaled, y_train)
                selected_features = X_train_scaled.columns[
                    np.logical_and(selector.scores_ >= np.median(selector.scores_), selector.pvalues_ <= 0.05)]

            else:
                rfe = RFE(estimator=model)
                rfe.fit(X_train_scaled, y_train)
                # Get selected features
                selected_features = X_train_scaled.columns[rfe.support_]

            # Hyperparameter tuning using GridSearchCV
            grid_search = GridSearchCV(model, param_grid,  cv=None, scoring= 'neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train_scaled[selected_features], y_train)

            # Get the best model and its parameters
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_model.fit(X_train_scaled[selected_features], y_train)

            dump(best_model,
                 os.path.join(output_path_kwargs['path_main_model'], 'trained_model_best_'+model_name+'.joblib'))

            # Make predictions on train
            y_pred_train = best_model.predict(X_train_scaled[selected_features])

            r2_train = r2_score(y_train, y_pred_train)

            # Make predictions on test
            y_pred_test = best_model.predict(X_test_scaled[selected_features])

            r2_test = r2_score(y_test, y_pred_test)

            # Save model name with parameters and accuracy
            model_accuracies[model_name] = { 'best_params': best_params, 'r2_train': r2_train, 'r2_test': r2_test}

        os.makedirs(output_path_kwargs['path_main_model'], exist_ok=True)

        # Save model r2 report to a file
        report_df = pd.DataFrame(model_accuracies).T
        report_df.to_csv(os.path.join(output_path_kwargs['path_main_model'],'models_r2_finding_the_best_model_v3.csv'))

    # train with kfolds - test r2 score
    if config['EX2']['functions']['train_cv']:
        chosen_model = load(os.path.join(output_path_kwargs['path_main_model'],'trained_model_best_XGBRegressor.joblib'))
        # Scale the data using StandardScaler
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
        rfe = RFE(estimator=chosen_model)
        rfe.fit(X_scaled, y)
        # Get selected features
        selected_features = X_scaled.columns[rfe.support_]
        # Initialize K-Folds Cross Validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        # Perform cross-validated predictions
        cv_pred = cross_val_predict(chosen_model, X_scaled[selected_features], y, cv=kfold)
        # Calculate F1 scores and accuracy
        r2_score = r2_score(y, cv_pred)

        cv_scores_df = pd.DataFrame({'r2_score': r2_score}, index=['r2_score'])
        cv_scores_df.to_csv(os.path.join(output_path_kwargs['path_main_model'], 'best_XGBREG_Kfold.csv'))

    # predict data on cars_evaluate df
    if config['EX2']['functions']['cleandata_test_on_new_data']:
        cars_ev_df = pd.read_csv('input/cars_evaluate.csv')
        # clean and save data correctly:
        cars_ev_df_after_drop = drop_multy_cols(cars_ev_df, ['VIN', 'Consumer_Review_#', 'Stock_#'])
        cars_ev_df_after_clean_cat = clean_multy_cat_data(cars_ev_df_after_drop, test=True)
        cars_ev_df_after_clean_cat.to_csv('input\cars_ev_clean_cat.csv')
        cars_ev_df_after_clean_cat = pd.read_csv('input\cars_ev_clean_cat.csv', index_col=0)
        cars_ev_df_after_clean_catAndnum = clean_multy_num_data(cars_ev_df_after_clean_cat, test=True)
        cars_ev_df_after_clean_catAndnum.to_csv('input\cars_ev_clean_cat_num.csv')

    # Test model on new test data and predict
    if config['EX2']['functions']['test_on_new_data']:
        cars_ev_df_after_clean_catAndnum = pd.read_csv('input\cars_ev_clean_cat_num.csv', index_col=0)
        cars_ev_df_after_label_encoding = convert_data_to_num_cat_vals(cars_ev_df_after_clean_catAndnum)
        chosen_model =  load(os.path.join(output_path_kwargs['path_main_model'],'trained_model_best_XGBRegressor.joblib'))
        # Scale the data using StandardScaler
        X_scaled = pd.DataFrame(scaler.transform(cars_ev_df_after_label_encoding), columns=cars_ev_df_after_label_encoding.columns)
        # Get selected features
        selected_features = ['Year', 'Model', 'MPG', 'Fuel_Type', 'Transmission', 'Engine',
           'Mileage', 'Comfort_Rating', 'Interior_Design_Rating',
           'Exterior_Styling_Rating']
        # Perform cross-validated predictions
        cv_pred = chosen_model.predict(X_scaled[selected_features])
        cars_ev_df_after_label_encoding['Price'] = cv_pred
        cars_ev_df_after_label_encoding.to_csv('input/cars_evaluate_after_price.csv')

    # check r2 scores for evaluation data
    if config['EX2']['functions']['check_new_ev']:
        cars_ev_df_after_label_encoding = pd.read_csv('input\cars_evaluate_after_price.csv', index_col=0)
        X = cars_ev_df_after_label_encoding.drop(columns=['Price'])  # Exclude 'price' column
        y = cars_ev_df_after_label_encoding['Price']
        chosen_model = load(os.path.join(output_path_kwargs['path_main_model'],'trained_model_best_XGBRegressor.joblib'))
        # Scale the data using StandardScaler
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
        # Get selected features
        selected_features =  ['Year', 'Model', 'MPG', 'Fuel_Type', 'Transmission', 'Engine',
           'Mileage', 'Comfort_Rating', 'Interior_Design_Rating',
           'Exterior_Styling_Rating']
        # Initialize K-Folds Cross Validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        # Perform cross-validated predictions
        cv_scores = cross_val_score(chosen_model, X_scaled[selected_features], y,cv=kfold)

        cv_scores_df = pd.DataFrame()
        cv_scores_df['r2_scores'] = cv_scores
        cv_scores_df.to_csv(os.path.join(output_path_kwargs['path_main_model'], 'best_XGBREG_Kfold_cars_evaluate_r2.csv'))

if __name__ == "__main__":
    main()