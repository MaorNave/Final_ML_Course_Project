from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, f1_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import json
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from io import StringIO
import matplotlib
import yaml
import numpy as np
from sklearn.preprocessing import StandardScaler
import Maor_Nave_313603391_EX3_models_Final


def load_yaml(path):
    """Load YAML data from the specified file path using safe loader."""
    with open(path, 'r') as file:
        yaml_data = yaml.load(file, Loader=yaml.SafeLoader)
    file.close()
    return yaml_data

def add_id_col_to_df(df):
    """Add an ID column to the DataFrame based on the total number of IDs."""
    total_ids = len(df) / 3
    df['ID'] = [int((i // 3) % total_ids + 1) for i in range(len(df))]
    return df

def calculate_vif(df):
    """Calculate the Variance Inflation Factor (VIF) for each feature in the DataFrame."""
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
        # Redirect the standard output to the file
        f.write(info_str)
    f.close()

def save_json(path, data):
    """Save data to a JSON file."""
    with open(path, "w", encoding='utf-8') as file:
        file.write(json.dumps(data))
    file.close()

def check_unique_vals(df, paths_dict):
    """Check unique values in each column of the DataFrame and save to a JSON file."""
    # check for all unique data in each column
    data_dict_unique = {}
    for col in df.columns:
        data_dict_unique[col] = df[col].unique().tolist()
    os.makedirs(paths_dict['path_output'], exist_ok=True)
    save_json(os.path.join(paths_dict['path_output'], 'unique_data_after_columns_clean.json') ,data_dict_unique)


def fill_data_random(df):
    """Fill missing data randomly in the DataFrame."""
    for col in df.columns:
        if col == 'Feelings':
            continue
        if 'DATA_EXPIRED' in df[col].tolist():
            unique_list = df[col].unique().tolist()
            unique_list.remove('DATA_EXPIRED')
            for id_num in df['ID'].unique():
                if 'DATA_EXPIRED' in df.loc[df['ID'] == id_num, col].tolist():
                    df.loc[df['ID'] == id_num, col] =  np.random.choice(unique_list)
        else:
            counts = df[col].value_counts(normalize=True)
            rare_values = counts[counts < 3 / df[col].value_counts().sum()].index.tolist() # the value is divided by 3 becuse its corresponded to a case that we have up to 5 rows
            # row from all the data base have the outlier as finger role, if there are less then this value nin the db the we can be sure iots ouplier
            if len(rare_values) > 0:
                df[col] = df[col].replace(rare_values, np.random.choice([val for val in counts.index if val not in rare_values], size=len(rare_values)))
    return df

def convert_data_to_num_cat_vals(df):
    """Convert categorical data to numerical categorical values using LabelEncoder."""
    label_encoder = LabelEncoder()
    for column in df.columns:
        if column.lower() == 'feelings':
            all_values = set([column+'_'+value for row in df[column] for value in eval(row)])
            one_hot_df = pd.DataFrame(0, index=df.index, columns=list(all_values))
            for index, row in df.iterrows():
                emotions = [column +'_'+ val for val in eval(row[column])]
                one_hot_df.loc[index, emotions] = 1
            df_encoded = one_hot_df.iloc[:, :-1]
            df = pd.concat([df, df_encoded], axis=1)
            df.drop(column, axis=1, inplace=True)

        elif df[column].dtype == 'object' or df[column].dtype == 'bool':
            df[column] = label_encoder.fit_transform(df[column])

    decision_col_data = df['Decision']
    df.drop('Decision', axis=1, inplace=True)
    df = pd.concat([df, decision_col_data], axis=1)
    return df


def visualize_corr_and_hist(df, output_folders):
    """Visualize correlation and histograms of the DataFrame features."""
    #use agg to not poping the figures
    matplotlib.use('agg')
    # Check the data in each attribute
    os.makedirs(os.path.join(output_folders['path_main_vis_hist'], 'df_info'), exist_ok=True)
    os.makedirs(os.path.join(output_folders['path_main_vis_hist'], 'dist'), exist_ok=True)
    os.makedirs(os.path.join(output_folders['path_main_vis_hist'], 'relationship'), exist_ok=True)
    os.makedirs(os.path.join(output_folders['path_main_vis_hist'], 'corr'), exist_ok=True)
    # save the df info
    save_txt(os.path.join(output_folders['path_main_vis_hist'], 'df_info', 'df_data.txt'), df)
    # Visualize DataFrame as a histogram and save the plots
    for column in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[column], kde=True)
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {column}")
        plt.savefig(f"{output_folders['path_main_vis_hist']}\dist\hist_{column}.png")
        plt.close()
    # Plot the relationship between each col to the depoendent col
    for column in df.columns:
        if True in df[column]:
            df[column] = df[column].astype(int)
        if column == 'Decision':
            continue
        plt.figure(figsize=(8, 6))
        sns.countplot(x=column, hue='Decision', data=df)
        plt.title(f'Relationship between {column} and Dependent Variable')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.legend(title='Dependent Variable', loc='upper right')
        plt.savefig(f"{output_folders['path_main_vis_hist']}\\relationship\\relCol_{column}.png")

    # Plot the corr between each col to the depoendent col
    for column in df.columns:
        if column == 'Decision':
            continue
        if df[column].dtype == 'object' or df[column].dtype == 'bool':
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=df, x='Decision', y= column)
            plt.title('Scatter Plot of Feature Column vs. Dependent Variable')
            plt.xlabel('Dependent Variable - Decision')
            plt.ylabel(f'Feature Column - {column}')
            plt.savefig(f"{output_folders['path_main_vis_hist']}\corr\corr_{column}.png")
        else:
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=df, x=column, y= 'Decision')
            plt.title('Scatter Plot of Feature Column vs. Dependent Variable')
            plt.xlabel(f'Feature Column - {column}')
            plt.ylabel('Dependent Variable - Decision')
            plt.savefig(f"{output_folders['path_main_vis_hist']}\corr\corr_{column}.png")


def balance_data(ds_df_after_hot_vector):
    """Balance the data using SMOTE."""
    # Extract features and target columns
    X = ds_df_after_hot_vector.iloc[:, :-1]
    y = ds_df_after_hot_vector.iloc[:, -1:]
    # Initialize SMOTE with random state 42
    smote = SMOTE(k_neighbors=10, random_state=42)
    # Apply SMOTE to generate synthetic samples
    X_resampled, y_resampled = smote.fit_resample(X, y.to_numpy())
    y_resampled_df = pd.DataFrame(y_resampled, columns=y.columns)
    ds_df_balanced = pd.concat([X_resampled, y_resampled_df], axis=1)

    return ds_df_balanced


def heat_map_corr(ds_df_balanced, output_path_kwargs):
    """Generate a heatmap of the correlation matrix."""
    df_clean_balanced = ds_df_balanced.iloc[:, :-1]
    df_clean_balanced.drop(columns=df_clean_balanced.filter(regex='Feelings').columns, inplace=True)
    # Calculate correlation matrix
    corr_matrix = df_clean_balanced.corr()
    corr_matrix.to_csv(os.path.join(output_path_kwargs['path_main_vis_corr'], 'corr_mat_values.csv'))
    # Filter correlations based on mean as threshold
    filtered_corr = corr_matrix[np.abs(corr_matrix) < 1]
    treshold_corr = np.mean(np.abs(filtered_corr))

    high_corr_columns = corr_matrix.columns[(corr_matrix.abs().gt(treshold_corr) & corr_matrix.abs().lt(1)).sum() >=
                                            (corr_matrix.abs().gt(treshold_corr) & corr_matrix.abs().lt(1)).sum().mean()]
    # Get the subset DataFrame with high correlation columns
    high_corr_df = df_clean_balanced[high_corr_columns]
    corr_matrix_high_corr = high_corr_df.corr()
    # Plot heatmap with nan_color for NaN cells
    plt.figure(figsize=(50, 46))
    sns.heatmap(corr_matrix_high_corr, cmap='coolwarm', annot=True, fmt=".2f")
    plt.title('Heatmap of higher then mean correlations values')
    plt.savefig(os.path.join(output_path_kwargs['path_main_vis_corr'], 'corr_mat_moreThenMean.png'))
    # Set threshold for VIF
    vif_threshold = 10
    high_vif_columns = []

    while True:
        vif_result = calculate_vif(high_corr_df)
        max_vif_index = vif_result["VIF"].idxmax()  # Get index of column with max VIF
        max_vif_value = vif_result.loc[max_vif_index, "VIF"]
        if max_vif_value >= vif_threshold:
            # Drop column with highest VIF
            high_vif_columns.append(vif_result.loc[max_vif_index, "Feature"])
            high_corr_df = high_corr_df.drop(columns=vif_result.loc[max_vif_index, "Feature"])
        else:
            break

    if 'STNumber' in high_vif_columns:
        high_vif_columns = [val for val in high_vif_columns if val != 'STNumber']
    df_clean_after_corr_test = ds_df_balanced.drop(columns = high_vif_columns)

    return df_clean_after_corr_test

def main():
    # load_config_file
    config = load_yaml('Maor_Nave_313603391_EX3_Config_Final.yaml')
    # load and EDA data
    ds_df = pd.read_csv('input\decision_making_law.csv', index_col=0)
    matplotlib.use('agg') # For non-interactive mode
    # Define the output path settings based on the configuration
    output_path_kwargs =config['params']['output_paths_kwargs']
    #check unique values
    if config['EX1']['functions']['check_unique_vals']:
        check_unique_vals(ds_df, output_path_kwargs)
    # plot all relevant data for each of the features
    if config['EX1']['functions']['visualize_corr_and_hist']:
        visualize_corr_and_hist(ds_df, output_path_kwargs)
    #clean the data
    if config['EX1']['functions']['clean_and_save_data']:
        # fill cat data
        # Add an ID column to the DataFrame
        ds_df = add_id_col_to_df(ds_df)
        # Fill missing data randomly
        ds_df_after_data_fill = fill_data_random(ds_df)
        ds_df_after_data_fill.drop(columns=['ID'], inplace=True)
        # convert categorical data to Label encoding
        ds_df_after_label_encoding = convert_data_to_num_cat_vals(ds_df_after_data_fill)
        # fill unbalanced data
        ds_df_balanced = balance_data(ds_df_after_label_encoding)
        # Remove highly correlated features and save the cleaned data to CSV
        df_clean_after_corr_test = heat_map_corr(ds_df_balanced, output_path_kwargs)
        df_clean_after_corr_test.to_csv('input/decision_making_law_clean.csv')
    else:
        df_clean_after_corr_test = pd.read_csv('input/decision_making_law_clean.csv', index_col=0)

    # Extract model parameters from the custom module
    models = Maor_Nave_313603391_EX3_models_Final.models_ex1
    # split the data to X and Y
    X = df_clean_after_corr_test.iloc[:, :-1]
    y = df_clean_after_corr_test.iloc[:, -1:]
    # split the data for train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    scaler = StandardScaler()
    model_accuracies = {}
    #scale the data
    scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # search for best model and its params
    if config['EX1']['functions']['find_best_model']:
        class_labels = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6']
        # Loop over each model and perform hyperparameter tuning using GridSearchCV
        for model_name, (model, param_grid) in models.items():
            print(f"Training {model_name}...")
            # Initialize Feature selection method
            if model_name == 'SVM' or model_name == 'KNN' or model_name == 'MLPclassifier':
                selector = SelectKBest(score_func=f_classif, k='all')
                X_train_selected = selector.fit_transform(X_train_scaled, y_train)
                selected_features = X_train_scaled.columns[
                    np.logical_and(selector.scores_ >= 3, selector.pvalues_ <= 0.05)]
            else:
                rfe = RFE(estimator=model)
                rfe.fit(X_train_scaled, y_train)
                # Get selected features
                selected_features = X_train_scaled.columns[rfe.support_]

            selected_features = selected_features.tolist()
            if 'STNumber' not in selected_features:
                selected_features.append('STNumber')

            # Hyperparameter tuning using GridSearchCV
            grid_search = GridSearchCV(model, param_grid,  cv=None, scoring= 'f1_weighted')
            grid_search.fit(X_train_scaled[selected_features], y_train)
            # Get the best model and its parameters
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_model.fit(X_train_scaled[selected_features], y_train)
            # Make predictions on train
            y_pred_train = best_model.predict(X_train_scaled[selected_features])
            accuracy_train = accuracy_score(y_train, y_pred_train)
            f1_score_train_w = f1_score(y_train, y_pred_train, average='weighted')
            f1_score_train = f1_score(y_train, y_pred_train, average=None)

            # Make predictions on test
            y_pred_test = best_model.predict(X_test_scaled[selected_features])
            accuracy_test = accuracy_score(y_test, y_pred_test)
            f1_score_test_w = f1_score(y_test, y_pred_test, average='weighted')
            f1_score_test = f1_score(y_test, y_pred_test, average=None)

            # Save model name with parameters and accuracy
            model_accuracies[model_name] = { 'best_params': best_params, 'accuracy_train': accuracy_train,
                                            'accuracy_test': accuracy_test, 'f1_score_train': f1_score_train,
                                           'f1_score_test': f1_score_test  , 'f1_score_train_w': f1_score_train_w,
                                             'f1_score_test_w': f1_score_test_w }

            # Calculate confusion matrix for test data
            cm_test = confusion_matrix(y_test, y_pred_test)
            cm_train = confusion_matrix(y_train, y_pred_train)

            # Convert the confusion matrix array into a DataFrame for better visualization
            cm_test_df = pd.DataFrame(cm_test, index=class_labels,
                                      columns=class_labels)
            cm_train_df = pd.DataFrame(cm_train, index=class_labels,
                                      columns=class_labels)

            # Save the confusion matrix for the current model to a CSV file
            cm_filename_test = f'confusion_matrix_{model_name}_test.csv'
            cm_test_df.to_csv(os.path.join(output_path_kwargs['path_main_model'], cm_filename_test))
            cm_filename_train = f'confusion_matrix_{model_name}_train.csv'
            cm_train_df.to_csv(os.path.join(output_path_kwargs['path_main_model'], cm_filename_train))

        os.makedirs(output_path_kwargs['path_main_model'], exist_ok=True)
        # Save model accuracies report to a file
        report_df = pd.DataFrame(model_accuracies).T
        report_df.to_csv(os.path.join(output_path_kwargs['path_main_model'],'model_accuracies_report_v18_f1_score_searching for best_model_v5.csv'))

    # train with kfolds - test  accuracy and f1 scores
    if config['EX1']['functions']['train_cv']:
        chosen_model = RandomForestClassifier(max_depth=17, min_samples_leaf=1, min_samples_split= 2, n_estimators= 15000,random_state=42)
        # Scale the data using StandardScaler
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
        rfe = RFE(estimator=chosen_model)
        rfe.fit(X_scaled, y)
        # Get selected features
        selected_features = X_scaled.columns[rfe.support_]

        selected_features = selected_features.tolist()
        if 'STNumber' not in selected_features:
            selected_features.append('STNumber')

        # Initialize K-Folds Cross Validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        # Perform cross-validated predictions
        cv_pred = cross_val_predict(chosen_model, X_scaled[selected_features], y, cv=kfold)
        # Calculate F1 scores and accuracy
        f1_scores = f1_score(y, cv_pred, average=None)
        f1_score_test_w = f1_score(y, cv_pred, average='weighted')
        accu = accuracy_score(y, cv_pred)
        cv_scores_df = pd.DataFrame({'f1': f1_scores, 'f1_score_test_w': f1_score_test_w, 'accuracy': accu})
        cv_scores_df.to_csv(os.path.join(output_path_kwargs['path_main_model'], 'best_RandomForestClassifier_Kfold.csv'))

    # train without a ST column - test accuracy and f1 scores
    if config['EX1']['functions']['train_cv_no_ST']:
        chosen_model = RandomForestClassifier(max_depth=17, min_samples_leaf=1, min_samples_split= 2, n_estimators= 15000,random_state=42)
        # Scale the data using StandardScaler
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
        rfe = RFE(estimator=chosen_model)
        rfe.fit(X_scaled, y)
        # Get selected features
        selected_features = X_scaled.columns[rfe.support_]
        selected_features = selected_features.tolist()
        if 'STNumber' in selected_features:
            selected_features.remove('STNumber')
        # Initialize K-Folds Cross Validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        # Perform cross-validated predictions
        cv_pred = cross_val_predict(chosen_model, X_scaled[selected_features], y, cv=kfold)
        f1_scores = f1_score(y, cv_pred, average=None)
        f1_score_test_w = f1_score(y, cv_pred, average='weighted')
        accu = accuracy_score(y, cv_pred)
        cv_scores_df = pd.DataFrame({'f1': f1_scores, 'f1_score_test_w': f1_score_test_w, 'accuracy': accu})
        cv_scores_df.to_csv(os.path.join(output_path_kwargs['path_main_model'], 'best_RandomForestClassifier_Kfold_noSTNumber.csv'))

if __name__ == "__main__":
    main()







