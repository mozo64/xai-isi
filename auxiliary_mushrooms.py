import os
import random
import re
from builtins import ValueError
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

print(os.getcwd())


def process_primary_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, sep=';')

        tuple_columns = ['cap-diameter', 'stem-height', 'stem-width']
        list_columns = [
            'cap-shape', 'Cap-surface', 'cap-color', 'does-bruise-or-bleed',
            'gill-attachment', 'gill-spacing', 'gill-color', 'stem-root',
            'stem-surface', 'stem-color', 'veil-type', 'veil-color',
            'has-ring', 'ring-type', 'Spore-print-color', 'habitat', 'season'
        ]
        string_columns = ['family', 'name', 'class']

        for col in list_columns:
            df[col] = df[col].apply(lambda x: x[1:-1].split(',') if isinstance(x, str) else [])

        for col in tuple_columns:
            df[col] = df[col].apply(
                lambda x: tuple(map(float, x[1:-1].split(','))) if isinstance(x, str) else (np.nan, np.nan))

        for col in string_columns:
            df[col] = df[col].fillna(np.nan)

        return df

    except Exception as e:
        raise ValueError(f"Error processing file '{file_path}': {e}")


def process_secondary_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, sep=';')

        string_columns = [
            'class', 'cap-shape', 'cap-surface', 'cap-color', 'does-bruise-or-bleed',
            'gill-attachment', 'gill-spacing', 'gill-color', 'stem-root', 'stem-surface',
            'stem-color', 'veil-type', 'veil-color', 'has-ring', 'ring-type',
            'spore-print-color', 'habitat', 'season'
        ]
        float_columns = ['cap-diameter', 'stem-height', 'stem-width']

        for col in string_columns:
            df[col] = df[col].fillna(np.nan)

        for col in float_columns:
            df[col] = df[col].astype(float).fillna(np.nan)

        return df

    except Exception as e:
        raise ValueError(f"Error processing file '{file_path}': {e}")


def decode_categorical_values(df: pd.DataFrame, translation: str = "both") -> pd.DataFrame:
    assert translation in ["none", "polish", "both"], "translation should be one of: \"none\", \"polish\", \"both\""

    def take_second_part(value: str):
        return value.split('__')[1]

    def identity(value: str):
        return value

    translate_to_polish: bool = False
    transformation = identity

    if translation != "none":
        translate_to_polish: bool = True
        if translation == "polish":
            transformation = take_second_part

    # Create a copy of the DataFrame
    decoded_df = df.copy()

    # Rename numerical columns to include units
    decoded_df.rename(
        columns={'cap-diameter': 'cap_diameter_cm', 'stem-height': 'stem_height_cm', 'stem-width': 'stem_width_mm'},
        inplace=True)

    # Define dictionaries for each categorical column
    class_dict = {'p': 'poisonous', 'e': 'edibile'}
    cap_shape_dict = {'b': 'bell', 'c': 'conical', 'x': 'convex', 'f': 'flat', 's': 'sunken', 'p': 'spherical',
                      'o': 'others'}
    cap_surface_dict = {'i': 'fibrous', 'g': 'grooves', 'y': 'scaly', 's': 'smooth', 'h': 'shiny', 'l': 'leathery',
                        'k': 'silky', 't': 'sticky', 'w': 'wrinkled', 'e': 'fleshy'}
    cap_color_dict = {'n': 'brown', 'b': 'buff', 'g': 'gray', 'r': 'green', 'p': 'pink', 'u': 'purple', 'e': 'red',
                      'w': 'white', 'y': 'yellow', 'l': 'blue', 'o': 'orange', 'k': 'black'}
    bruises_bleeding_dict = {'t': 'bruises_or_bleeding', 'f': 'no_bruises_or_bleeding'}
    gill_attachment_dict = {'a': 'adnate', 'x': 'adnexed', 'd': 'decurrent', 'e': 'free', 's': 'sinuate', 'p': 'pores',
                            'f': 'none', 'u': 'unknown'}
    gill_spacing_dict = {'c': 'close', 'd': 'distant', 'f': 'none'}
    gill_color_dict = cap_color_dict.copy()
    gill_color_dict['f'] = 'none'
    stem_root_dict = {'b': 'bulbous', 's': 'swollen', 'c': 'club', 'u': 'cup', 'e': 'equal', 'z': 'rhizomorphs',
                      'r': 'rooted'}
    stem_surface_dict = cap_surface_dict.copy()
    stem_surface_dict['f'] = 'none'
    stem_color_dict = cap_color_dict.copy()
    stem_color_dict['f'] = 'none'
    veil_type_dict = {'p': 'partial', 'u': 'universal'}
    veil_color_dict = cap_color_dict.copy()
    veil_color_dict['f'] = 'none'
    has_ring_dict = {'t': 'ring', 'f': 'none'}
    ring_type_dict = {'c': 'cobwebby', 'e': 'evanescent', 'r': 'flaring', 'g': 'grooved', 'l': 'large', 'p': 'pendant',
                      's': 'sheathing', 'z': 'zone', 'y': 'scaly', 'm': 'movable', 'f': 'none', 'u': 'unknown'}
    spore_print_color_dict = cap_color_dict
    habitat_dict = {'g': 'grasses', 'l': 'leaves', 'm': 'meadows', 'p': 'paths', 'h': 'heaths', 'u': 'urban',
                    'w': 'waste', 'd': 'woods'}
    season_dict = {'s': 'spring', 'u': 'summer', 'a': 'autumn', 'w': 'winter'}

    column_translations = {}
    if translate_to_polish:
        # Polish translations for column names
        column_translations = {
            'cap_diameter_cm': 'cap_diameter_cm__średnica_kapelusza_cm',
            'stem_height_cm': 'stem_height_cm__wysokość_trzonu_cm',
            'stem_width_mm': 'stem_width_mm__szerokość_trzonu_mm',
            'class': 'class__klasa',
            'cap-shape': 'cap_shape__kształt_kapelusza',
            'cap-surface': 'cap_surface__powierzchnia_kapelusza',
            'cap-color': 'cap_color__kolor_kapelusza',
            'does-bruise-or-bleed': 'does_bruise_or_bleed__zmienia_kolor_lub_puszcza_mleczko_w_reakcji_na_uszkodzenie',
            'gill-attachment': 'gill_attachment__sposób_przyrastania_blaszek_do_trzonu',
            'gill-spacing': 'gill_spacing__gęstość_blaszek',
            'gill-color': 'gill_color__kolor_blaszek',
            'stem-root': 'stem_root__podstawa_trzonu',
            'stem-surface': 'stem_surface__powierzchnia_trzonu',
            'stem-color': 'stem_color__kolor_trzonu',
            'veil-type': 'veil_type__typ_hymenoforu',
            'veil-color': 'veil_color__kolor_hymenoforu',
            'has-ring': 'has_ring__obecność_pierścienia',
            'ring-type': 'ring_type__typ_pierścienia',
            'spore-print-color': 'spore_print_color__kolor_zarodników',
            'habitat': 'habitat__siedlisko',
            'season': 'season__pora_roku'
        }
        column_translations_bis = {}
        for key, value in column_translations.items():
            value_bis = transformation(value)
            column_translations_bis[key] = value_bis
        column_translations = column_translations_bis

        # Polish translations for category levels
        class_dict_pl = {'poisonous': 'trujący', 'edibile': 'jadalny'}
        cap_shape_dict_pl = {'bell': 'dzwonkowaty', 'conical': 'stożkowaty', 'convex': 'wypukły', 'flat': 'płaski',
                             'sunken': 'wklęsły', 'spherical': 'półkulisty', 'others': 'inne'}
        cap_surface_dict_pl = {'fibrous': 'włóknista', 'grooves': 'bruzdowana', 'scaly': 'łuskowata',
                               'smooth': 'gładka', 'shiny': 'błyszcząca', 'leathery': 'zamszowa',
                               'silky': 'jedwabista', 'sticky': 'lepka', 'wrinkled': 'pomarszczona',
                               'fleshy': 'mięsista'}
        cap_color_dict_pl = {'brown': 'brązowy', 'buff': 'jasnobrązowy', 'gray': 'szary', 'green': 'zielony',
                             'pink': 'różowy', 'purple': 'fioletowy', 'red': 'czerwony', 'white': 'biały',
                             'yellow': 'żółty', 'blue': 'niebieski', 'orange': 'pomarańczowy', 'black': 'czarny'}
        bruises_bleeding_dict_pl = {'bruises_or_bleeding': 'zmienia_kolor_lub_puszcza_mleczko',
                                    'no_bruises_or_bleeding': 'nie_zmienia_koloru_lub_brak_mleczka'}
        gill_attachment_dict_pl = {'adnate': 'przyrośnięte', 'adnexed': 'wykrojone', 'decurrent': 'zbiegające',
                                   'free': 'wolne', 'sinuate': 'faliste', 'pores': 'pory', 'none': 'brak',
                                   'unknown': 'nieznane'}
        gill_spacing_dict_pl = {'close': 'gęste', 'distant': 'rzadkie', 'none': 'brak'}
        gill_color_dict_pl = {**cap_color_dict_pl, 'none': 'brak'}
        stem_root_dict_pl = {'bulbous': 'bulwiasty', 'swollen': 'pękaty', 'club': 'maczugowaty',
                             'cup': 'kubeczkowaty', 'equal': 'prosty', 'rhizomorphs': 'strzępkokształtny',
                             'rooted': 'korzeniasty'}
        stem_surface_dict_pl = {**cap_surface_dict_pl, 'none': 'brak'}
        stem_color_dict_pl = {**cap_color_dict_pl, 'none': 'brak'}
        veil_type_dict_pl = {'partial': 'z_zasnówką', 'universal': 'bez_zasnówki'}
        veil_color_dict_pl = {**cap_color_dict_pl, 'none': 'brak'}
        has_ring_dict_pl = {'ring': 'pierścień', 'none': 'brak'}
        ring_type_dict_pl = {'cobwebby': 'pajęczynowaty', 'evanescent': 'nietrwały', 'flaring': 'wzniesiony',
                             'grooved': 'bruzdowany', 'large': 'duży', 'pendant': 'zwisający', 'sheathing': 'otulający',
                             'zone': 'strefowany', 'scaly': 'łuskowaty', 'movable': 'wolny', 'none': 'brak',
                             'unknown': 'nieznany'}
        spore_print_color_dict_pl = {**cap_color_dict_pl}
        habitat_dict_pl = {'grasses': 'trawa', 'leaves': 'liście', 'meadows': 'łąki', 'paths': 'ścieżki',
                           'heaths': 'wrzosowiska', 'urban': 'miejskie', 'waste': 'odpady', 'woods': 'lasy'}
        season_dict_pl = {'spring': 'wiosna', 'summer': 'lato', 'autumn': 'jesień', 'winter': 'zima'}

        # Apply Polish translations to dictionaries
        class_dict = {k: transformation(f"{v}__{class_dict_pl[v]}") for k, v in class_dict.items()}
        cap_shape_dict = {k: transformation(f"{v}__{cap_shape_dict_pl[v]}") for k, v in cap_shape_dict.items()}
        cap_surface_dict = {k: transformation(f"{v}__{cap_surface_dict_pl[v]}") for k, v in cap_surface_dict.items()}
        cap_color_dict = {k: transformation(f"{v}__{cap_color_dict_pl[v]}") for k, v in cap_color_dict.items()}
        bruises_bleeding_dict = {k: transformation(f"{v}__{bruises_bleeding_dict_pl[v]}") for k, v in
                                 bruises_bleeding_dict.items()}
        gill_attachment_dict = {k: transformation(f"{v}__{gill_attachment_dict_pl[v]}") for k, v in
                                gill_attachment_dict.items()}
        gill_spacing_dict = {k: transformation(f"{v}__{gill_spacing_dict_pl[v]}") for k, v in gill_spacing_dict.items()}
        gill_color_dict = {k: transformation(f"{v}__{gill_color_dict_pl[v]}") for k, v in gill_color_dict.items()}
        stem_root_dict = {k: transformation(f"{v}__{stem_root_dict_pl[v]}") for k, v in stem_root_dict.items()}
        stem_surface_dict = {k: transformation(f"{v}__{stem_surface_dict_pl[v]}") for k, v in stem_surface_dict.items()}
        stem_color_dict = {k: transformation(f"{v}__{stem_color_dict_pl[v]}") for k, v in stem_color_dict.items()}
        veil_type_dict = {k: transformation(f"{v}__{veil_type_dict_pl[v]}") for k, v in veil_type_dict.items()}
        veil_color_dict = {k: transformation(f"{v}__{veil_color_dict_pl[v]}") for k, v in veil_color_dict.items()}
        has_ring_dict = {k: transformation(f"{v}__{has_ring_dict_pl[v]}") for k, v in has_ring_dict.items()}
        ring_type_dict = {k: transformation(f"{v}__{ring_type_dict_pl[v]}") for k, v in ring_type_dict.items()}
        spore_print_color_dict = {k: transformation(f"{v}__{spore_print_color_dict_pl[v]}") for k, v in
                                  spore_print_color_dict.items()}
        habitat_dict = {k: transformation(f"{v}__{habitat_dict_pl[v]}") for k, v in habitat_dict.items()}
        season_dict = {k: transformation(f"{v}__{season_dict_pl[v]}") for k, v in season_dict.items()}

    # Apply the dictionaries to decode each categorical column
    decoded_df['class'] = decoded_df['class'].map(class_dict)
    decoded_df['cap-shape'] = decoded_df['cap-shape'].map(cap_shape_dict)
    decoded_df['cap-surface'] = decoded_df['cap-surface'].map(cap_surface_dict)
    decoded_df['cap-color'] = decoded_df['cap-color'].map(cap_color_dict)
    decoded_df['does-bruise-or-bleed'] = decoded_df['does-bruise-or-bleed'].map(bruises_bleeding_dict)
    decoded_df['gill-attachment'] = decoded_df['gill-attachment'].map(gill_attachment_dict)
    decoded_df['gill-spacing'] = decoded_df['gill-spacing'].map(gill_spacing_dict)
    decoded_df['gill-color'] = decoded_df['gill-color'].map(gill_color_dict)
    decoded_df['stem-root'] = decoded_df['stem-root'].map(stem_root_dict)
    decoded_df['stem-surface'] = decoded_df['stem-surface'].map(stem_surface_dict)
    decoded_df['stem-color'] = decoded_df['stem-color'].map(stem_color_dict)
    decoded_df['veil-type'] = decoded_df['veil-type'].map(veil_type_dict)
    decoded_df['veil-color'] = decoded_df['veil-color'].map(veil_color_dict)
    decoded_df['has-ring'] = decoded_df['has-ring'].map(has_ring_dict)
    decoded_df['ring-type'] = decoded_df['ring-type'].map(ring_type_dict)
    decoded_df['spore-print-color'] = decoded_df['spore-print-color'].map(spore_print_color_dict)
    decoded_df['habitat'] = decoded_df['habitat'].map(habitat_dict)
    decoded_df['season'] = decoded_df['season'].map(season_dict)

    if translate_to_polish:
        # Rename columns with Polish translations
        decoded_df = decoded_df.rename(columns=column_translations)

    return decoded_df


def calculate_global_importances(sample_indices, explainer, X_test_reset, best_model, X_train):
    """
    Funkcja do obliczania globalnego znaczenia cech
    :param sample_indices:
    :return:
    """
    lime_weights = []
    for i in sample_indices:
        # Używamy iloc zamiast loc, aby uniknąć ostrzeżeń
        exp = explainer.explain_instance(X_test_reset.iloc[i], best_model.predict_proba,
                                         num_features=len(X_train.columns))
        lime_weights.append(exp.as_list())

    feature_importances = {}
    for feature_weight in lime_weights:
        for feature, weight in feature_weight:
            if feature in feature_importances:
                feature_importances[feature] += weight
            else:
                feature_importances[feature] = weight

    normalized_importances = {k: v / len(sample_indices) for k, v in feature_importances.items()}
    return normalized_importances


def wrap_title(label):
    return '\n'.join(label.split('__')).replace('_', ' ')


def wrap_labels(label, max_words_per_line=7):
    lines = '\n'.join(label.split('__')).replace('_', '\n')
    words = lines.split('.')
    return '\n'.join([' '.join(words[i:i + max_words_per_line]) for i in range(0, len(words), max_words_per_line)])


def correlation_analysis(df: pd.DataFrame, high_corr_threshold=0.7):
    df_copy = df.copy().dropna(axis=1, how='all')
    df_copy = df_copy.loc[:, df_copy.nunique() > 1]

    new_column_names = {col: wrap_labels(col, max_words_per_line=20) for col in df_copy.columns}
    df_copy.rename(columns=new_column_names, inplace=True)

    numeric_columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()

    corr_matrix = df_copy[numeric_columns].corr()

    # Wyszukanie par zmiennych z wysoką korelacją
    high_corrs = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                  .stack()
                  .reset_index())
    high_corrs.columns = ['Variable 1', 'Variable 2', 'Correlation']
    high_corrs = high_corrs[abs(high_corrs['Correlation']) > high_corr_threshold]

    print("Pary zmiennych z wysoką korelacją:")
    print(high_corrs)

    plt.figure(figsize=(12 * 1, 8 * 1))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.show()


def plot_lime_importances(importances, best_model, label_encoders):
    # Odwracanie mapowania LabelEncoder dla kolumny 'klasa'
    # Uzyskanie etykiet klas zwracanych przez model
    class_labels = best_model.classes_

    # Używamy inverse_transform, aby uzyskać oryginalne etykiety dla klas
    positive_class_label = label_encoders['klasa'].inverse_transform([class_labels[1]])[0]

    # Sortowanie cech według ich znaczenia
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    # Przygotowanie danych do wykresu
    features, scores = zip(*sorted_features)

    # Tworzenie wykresu
    plt.figure(figsize=(10, 12))
    plt.barh(features, scores)
    plt.xlabel('Średnie znaczenie cechy')
    plt.title('Globalne znaczenia cech z LIME')
    plt.gca().invert_yaxis()  # Odwrócenie osi y, aby najważniejsze cechy były na górze

    # Dodawanie adnotacji pod wykresem
    plt.figtext(0.5, 0.05, f"Wartości dodatnie na wykresie kontrybuują do klasy \"{positive_class_label}\"",
                ha="center", fontsize=12)

    plt.show()


def create_synthetic_samples(sample, num_samples=100, noise_level=0.1):
    """
    Create synthetic samples around a given sample by adding noise.
    """
    synthetic_samples = []
    for _ in range(num_samples):
        noise = np.random.normal(0, noise_level, sample.shape)
        synthetic_sample = sample + noise
        synthetic_samples.append(synthetic_sample)
    return np.array(synthetic_samples)


def fit_surrogate_model(X, y, max_depth=3):
    """
    Fit a decision tree surrogate model.
    """
    surrogate = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    surrogate.fit(X, y)
    return surrogate


def visualize_tree(tree_model, feature_names, class_names):
    """
    Visualize the decision tree.
    """
    plt.figure(figsize=(20, 10))
    plot_tree(tree_model, filled=True, feature_names=feature_names, class_names=class_names)
    plt.show()


def plot_feature_importance(model, feature_names):
    """
    Plot feature importance of the model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()


def create_pivot_table(predictions: List[int], label_encoder: LabelEncoder) -> pd.DataFrame:
    """
    Converts numeric predictions to text labels and creates a pivot table.

    Args:
    predictions (List[int]): List of numeric predictions.
    label_encoder (LabelEncoder): The LabelEncoder used for the target variable.

    Returns:
    pd.DataFrame: A pivot table with counts of each class.
    """
    # Convert numeric predictions to text labels
    text_labels = label_encoder.inverse_transform(predictions)

    # Create a pandas series from the text labels
    label_series = pd.Series(text_labels)

    # Generate a pivot table with counts for each class
    pivot_table = label_series.value_counts().to_frame(name='Count')

    return pivot_table


def plot_custom_pdp(model, X, feature_name, num_points=20):
    """
    Plots a custom Partial Dependency Plot for a single numerical or categorical feature.

    Args:
    model: Trained model object.
    X: DataFrame, features used for training or testing the model.
    feature_name: Name of the categorical feature for which PDP is to be plotted.
    """
    is_categorical = X[feature_name].dtype == 'object' or X[feature_name].dtype.name == 'category'

    if is_categorical:
        # Dla zmiennych kategorycznych użyj unikalnych wartości
        feature_values = X[feature_name].unique()
        num_points = len(feature_values)
    else:
        # Dla zmiennych numerycznych generuj równomiernie rozłożone wartości
        feature_values = np.linspace(X[feature_name].min(), X[feature_name].max(), num_points)

    feature_grid = np.array([feature_values, ] * X.shape[0]).transpose()
    X_copy = X.copy()
    pdp_values = []

    for i in range(num_points):
        X_copy[feature_name] = feature_grid[i]
        pdp_values.append(model.predict_proba(X_copy)[:, 1].mean())

    if is_categorical:
        plt.bar(range(num_points), pdp_values)
        plt.xticks(range(num_points), feature_values, rotation=90)  # Ustawienie etykiet osi X
    else:
        plt.plot(feature_values, pdp_values)

    plt.xlabel(feature_name)
    plt.ylabel('Partial Dependence')
    plt.title(f'Partial Dependence Plot for {feature_name}')
    plt.show()


def plot_numerical_distributions(df, observation_index=None):
    """
    Generates histograms and box plots for numerical columns in a DataFrame.
    Optionally adds a vertical line for a specified observation.

    Args:
    df: DataFrame containing the data.
    observation_index: Index of the observation to highlight (optional).
    """
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

    print("\nHistograms and Box Plots for Numerical Columns:")
    for col in numerical_columns:
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))

        # Histogram with KDE
        sns.histplot(df[col], kde=True, ax=axes[0], color='#C39BD3')  # Jasny fioletowy
        axes[0].set_title(f'Histogram dla {col}')
        axes[0].set_ylabel('')
        axes[0].set_xlabel('')

        # Highlighting the observation in histogram
        if observation_index is not None and col in df.columns:
            observation_value = df.iloc[observation_index][col]
            axes[0].axvline(observation_value, color='#FFA07A', linestyle='--')  # Jasny pomarańczowy

        # Box plot
        sns.boxplot(x=df[col], ax=axes[1], color='#C39BD3')  # Jasny fioletowy
        axes[1].set_title(f'Wykres pudełkowy dla {col}')
        axes[1].set_ylabel('')
        axes[1].set_xlabel('')

        # Highlighting the observation in box plot
        if observation_index is not None and col in df.columns:
            axes[1].axvline(observation_value, color='#FFA07A', linestyle='--')  # Jasny pomarańczowy

        plt.show()


def plot_categorical_columns(df: pd.DataFrame, y_df=None, label_encoder=None, target_name=None, observation_index=None,
                             NA: str = "_NA_") -> None:
    """
    Generates bar plots for categorical columns in a DataFrame.
    Optionally highlights the bar corresponding to a specified observation.

    Args:
    df (pd.DataFrame): DataFrame for which the plots are to be generated.
    observation_index (int, optional): Index of the observation to highlight.
    NA (str, optional): Representation for missing data. Defaults to "_NA_".
    """
    df_cp = df.copy()

    if y_df is not None and label_encoder is not None and target_name is not None:
        inverse_label_map = {v: k for v, k in enumerate(label_encoder.classes_)}
        df_cp[target_name] = np.vectorize(inverse_label_map.get)(y_df)
        cols = [target_name] + [col for col in df_cp.columns if col != target_name]
        df_cp = df_cp[cols]

    print("\nBar Plots for Categorical Columns:")
    categorical_columns = df_cp.select_dtypes(include=['object']).columns

    num_vars = len(categorical_columns)
    num_rows = (num_vars + 2) // 3

    fig, axes = plt.subplots(num_rows, 3, figsize=(20, 5 * num_rows))  # Increased figure height

    for i, col in enumerate(categorical_columns):
        row = i // 3
        col_pos = i % 3

        # Adding category for missing data
        temp_series = df_cp[col].fillna(NA)

        # Sorting labels by frequency except for 'Missing data'
        order = temp_series.value_counts().index.tolist()
        if NA in order:
            order.remove(NA)
            order.append(NA)

        # Drawing the bar plot
        sns.countplot(y=temp_series, ax=axes[row, col_pos], order=order)
        axes[row, col_pos].set_title(f'\n{col}')

        # Setting colors and highlighting the bar for the selected observation
        for bar, label in zip(axes[row, col_pos].patches, order):
            if label == NA:
                bar.set_color('#D3D3D3')  # Jasnoszary dla _NA_
            else:
                bar.set_color('#C39BD3')  # Jasny fioletowy dla pozostałych słupków
            if observation_index is not None and col in df_cp.columns:
                observation_value = df_cp.iloc[observation_index][col]
                if pd.isna(observation_value):
                    observation_value = NA
                if label == observation_value:
                    bar.set_edgecolor('#FFA07A')  # Jasny pomarańczowy dla ramki
                    bar.set_linewidth(2)

        axes[row, col_pos].set_ylabel('')  # Removing y-axis label
        axes[row, col_pos].set_xlabel('')  # Removing x-axis label

    # Hiding empty subplots
    for j in range(i + 1, num_rows * 3):
        fig.delaxes(axes[j // 3, j % 3])

    plt.subplots_adjust(hspace=0.6, wspace=0.4)  # Increased spacing between rows and columns
    plt.show()


def plot_shap_waterfall_for_class(model, X, y, explainer, shap_values_for_X, label_encoder, eatable_label, class_label):
    """
    Plots a SHAP waterfall chart for a randomly selected observation from a specified class.

    Args:
    model: The trained model (Pipeline).
    X: Test or train.
    y: y labels for X.
    explainer: SHAP explainer object.
    label_encoder: Operates on y.
    eatable_label: Namme of 0 class: "jadalny" or "eatable".
    class_label: The class label (0 or 1) for which to generate the plot.
    """
    # Selecting a random observation from the specified class
    class_indices = np.where(y == class_label)[0]
    selected_index = random.choice(class_indices)

    # Przewidywanie klasy dla wybranej obserwacji
    predicted_class = model.predict(X.iloc[[selected_index]])[0]
    predicted_class_name = label_encoder.inverse_transform([predicted_class])[0]
    # Getting the class name from label encoder
    class_name = label_encoder.inverse_transform([class_label])[0]

    assert predicted_class == class_label, f"Model prediction {predicted_class}, {predicted_class_name} != true class {class_label}, {class_name}"
    # print(f"Model prediction {predicted_class}, {predicted_class_name} == true class {class_label}, {class_name}")

    # Transforming the data
    transformed_X = model[:-1].transform(X)
    selected_observation = transformed_X[selected_index]

    # Reshaping the selected observation if necessary
    if isinstance(selected_observation, np.ndarray):
        # X_sample_for_shap = transformed_X
        raise ValueError("No transform should be needed")
    else:
        X_sample_for_shap = selected_observation.toarray()

    _numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    _categorical_features = X.select_dtypes(include=['object']).columns

    _feature_names = model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(
        _numeric_features).tolist() + model.named_steps['preprocessor'].transformers_[1][1].named_steps[
                         'onehot'].get_feature_names_out(_categorical_features).tolist()

    # Calculating SHAP values for the selected observation
    shap_explanation = explainer(X_sample_for_shap)
    shap_values_single = shap_explanation[0]

    precalculated_shap_values_for_obs = shap_values_for_X[selected_index]

    assert [a == b for a, b in zip(shap_values_single.values,
                                   precalculated_shap_values_for_obs)], f"shap_explanation= {shap_values_single.values} != precalculated= {precalculated_shap_values_for_obs}"
    # print(f"shap_explanation= {shap_values_single} == precalculated= {precalculated_shap_values}")

    # Creating an Explanation object with feature names
    # background_data = shap.sample(transformed_X, nsamples=100)

    shap_values_single = shap.Explanation(values=shap_values_single.values,
                                          base_values=shap_values_single.base_values,  # TODO ???  background_data
                                          data=shap_values_single.data,
                                          feature_names=_feature_names)

    # Visualizing SHAP values for the selected observation
    # Generowanie wykresu SHAP "waterfall"
    shap.plots.waterfall(shap_values_single, show=False)

    # Zmiana tytułu wykresu
    plt.title(f"Analiza wpływu cech na przewidywanie klasy '{predicted_class_name}' (prawdziwa klasa: '{class_name}')")
    this_class, opposite_class = ("jadalnego", "trującego") if predicted_class == eatable_label else (
        "trującego", "jadalnego")
    # Zmiana opisu osi X
    plt.xlabel('Wkład poszczególnych cech w przewidywanie klasy grzyba przez model. '
               '\nWartości pozytywne (w prawo) wskazują na wzrost prawdopodobieństwa \nklasyfikacji jako "trującego" '
               'wg modelu, wartości negatywne (w lewo) - na zmniejszenie. '
               '\nE[f(X)] to średni wynik modelu, a f(X) to przewidywanie dla tej obserwacji.\n')

    # Przeszukanie i modyfikacja wszystkich obiektów tekstowych w wykresie
    for text_obj in plt.gcf().findobj(lambda obj: isinstance(obj, plt.Text)):
        if text_obj.get_text().startswith('f(x)'):
            original_value = text_obj.get_text().split('=')[1]  # Zachowanie oryginalnej wartości
            text_obj.set_text(f'Końcowy wynik modelu: {original_value.strip()}')

    plt.show()

    return selected_index


def decode_categorical_features(encoded_data, categorical_features, category_encodings):
    # Funkcja do dekodowania danych kategorycznych
    decoded_data = encoded_data.copy()
    for feature in categorical_features:
        inv_map = {v: k for k, v in category_encodings[feature].items()}
        decoded_data[feature] = encoded_data[feature].map(inv_map)
    return decoded_data


def predict_proba_wrapper(model, encoded_data, categorical_features, category_encodings):
    # Funkcja, która dekoduje dane kategoryczne i wywołuje predict_proba
    decoded_data = decode_categorical_features(encoded_data, categorical_features, category_encodings)
    return model.predict_proba(decoded_data)


def calculate_global_importances_decoded(sample_indices, explainer, X_test_encoded, model, feature_names,
                                         categorical_features,
                                         category_encodings):
    lime_weights = []
    for i in sample_indices:
        # Przekształcenie pojedynczego wiersza z powrotem w DataFrame
        single_row_df = X_test_encoded.iloc[i:i + 1]

        exp = explainer.explain_instance(single_row_df.values[0],
                                         lambda x: predict_proba_wrapper(model,
                                                                         pd.DataFrame(x, columns=single_row_df.columns),
                                                                         categorical_features, category_encodings),
                                         num_features=len(feature_names))
        lime_weights.append(exp.as_list())

    feature_importances = {}
    for feature_weight in lime_weights:
        for feature, weight in feature_weight:
            if feature in feature_importances:
                feature_importances[feature] += weight
            else:
                feature_importances[feature] = weight

    normalized_importances = {k: v / len(sample_indices) for k, v in feature_importances.items()}
    return normalized_importances


def encode_categorical_features(raw_data, categorical_features, category_encodings):
    # Funkcja do enkodowania danych kategorycznych
    encoded_data = raw_data.copy()
    for feature in categorical_features:
        # print(feature)
        encoded_data[feature] = raw_data[feature].map(category_encodings[feature])
    return encoded_data


def predict_proba_func(classifier, data):
    # Funkcja, która przetwarza dane i wywołuje predict_proba
    return classifier.predict_proba(data)


def calculate_global_importances(sample_indices, explainer, X_test_transformed, classifier, feature_names_transformed):
    lime_weights = []
    for i in sample_indices:
        exp = explainer.explain_instance(X_test_transformed[i], lambda x: predict_proba_func(classifier, x),
                                         num_features=len(feature_names_transformed))
        lime_weights.append(exp.as_list())

    feature_importances = {}
    for feature_weight in lime_weights:
        for feature, weight in feature_weight:
            if feature in feature_importances:
                feature_importances[feature] += weight
            else:
                feature_importances[feature] = weight

    normalized_importances = {k: v / len(sample_indices) for k, v in feature_importances.items()}
    return normalized_importances


def generate_counterfactuals(index, instance, dice_explainer, feature_names: list, d, total_CFs=2,
                             desired_class="opposite"):
    """
    Generates counterfactual examples for a given observation.

    :param index: Index of the observation.
    :param instance: DataFrame with the observation data.
    :param total_CFs: Total number of counterfactuals to generate.
    :param desired_class: Desired class for the counterfactuals.
    :return: Counterfactual examples.
    """

    # Ensure data is a DataFrame
    if isinstance(instance, np.ndarray):
        instance = pd.DataFrame(instance.reshape(1, -1), columns=feature_names)
    elif isinstance(instance, pd.Series):
        instance = instance.to_frame().transpose()

    # Converting instance to the format required by DiCE
    query_instance = d.prepare_query_instance(query_instance=instance)
    query_instance = query_instance.iloc[0].to_dict()
    print(query_instance)

    # Generating counterfactuals
    counterfactuals = dice_explainer.generate_counterfactuals(instance, total_CFs=total_CFs,
                                                              desired_class=desired_class)

    # Visualizing the counterfactuals
    print(f"Counterfactuals for observation {index} and class {desired_class}:")
    counterfactuals.visualize_as_dataframe(show_only_changes=True)
    return counterfactuals


def train_imputers(X_train, numeric_features, categorical_features):
    """
    Trains imputers on the training data.

    :param X_train: DataFrame of training data.
    :param numeric_features: List of numeric feature names.
    :param categorical_features: List of categorical feature names.
    :return: Tuple of trained numeric and categorical imputers.
    """
    # Imputer for numeric features
    numeric_imputer = SimpleImputer(strategy='mean')
    numeric_imputer.fit(X_train[numeric_features])

    # Imputer for categorical features
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    categorical_imputer.fit(X_train[categorical_features])

    return numeric_imputer, categorical_imputer


def impute_observation(observation, numeric_imputer, categorical_imputer, numeric_features, categorical_features):
    """
    Imputes missing values in a single observation.

    :param observation: DataFrame or Series with a single observation.
    :param numeric_imputer: Trained imputer for numeric features.
    :param categorical_imputer: Trained imputer for categorical features.
    :param numeric_features: List of numeric feature names.
    :param categorical_features: List of categorical feature names.
    :return: DataFrame or Series with imputed values.
    """
    if isinstance(observation, pd.Series):
        observation = observation.to_frame().transpose()

    # Impute numeric features
    observation[numeric_features] = numeric_imputer.transform(observation[numeric_features])

    # Impute categorical features
    observation[categorical_features] = categorical_imputer.transform(observation[categorical_features])

    return observation


def parse_feature_importance(feature_importance_str):
    regex = r"([0-9.]+)?\s*([<>=]+)?\s*([\w_]+(?:[\w_]*[\w_]+)*)\s*([<>=]+)?\s*([0-9.]+)?"
    match = re.match(regex, feature_importance_str)

    if match:
        l_value = float(match.group(1)) if match.group(1) else None
        l_operator = match.group(2) if match.group(2) else None
        feature_name = match.group(3)
        r_operator = match.group(4) if match.group(4) else None
        r_value = float(match.group(5)) if match.group(5) else None

        return feature_name, l_value, l_operator, r_value, r_operator
    else:
        return None, None, None, None, None


def get_decoded_feature(feature, categorical_features, category_encodings):
    feature_name, l_value, l_operator, r_value, r_operator = parse_feature_importance(feature)
    # print(f"feature_name= {feature_name}")
    # print(f"l_operator= {l_operator}")
    # print(f"l_value= {l_value}")
    # print(f"r_operator= {r_operator}")
    # print(f"r_value= {r_value}")

    if feature_name in categorical_features:
        category_mapping = category_encodings[feature_name]
        reverse_mapping = {v: k for k, v in category_mapping.items()}

        # Obsługa przypadku z jednym warunkiem
        if l_value is None or r_value is None:
            if l_value is not None:
                l_index = int(float(l_value))
                if l_operator == '<':
                    value_range = list(range(l_index + 1, len(reverse_mapping)))
                elif l_operator == '<=':
                    value_range = list(range(l_index, len(reverse_mapping)))
                elif l_operator == '>':
                    value_range = list(range(0, l_index))
                elif l_operator == '>=':
                    value_range = list(range(0, l_index + 1))
                else:
                    raise Exception

            if r_value is not None:
                r_index = int(float(r_value))
                if r_operator == '>':
                    value_range = list(range(r_index + 1, len(reverse_mapping)))
                elif r_operator == '>=':
                    value_range = list(range(r_index, len(reverse_mapping)))
                elif r_operator == '<':
                    value_range = list(range(0, r_index))
                elif r_operator == '<=':
                    value_range = list(range(0, r_index + 1))
                else:
                    raise Exception

        # Obsługa przypadku z dwoma warunkami
        else:
            if l_operator in ('<', '<=') and r_operator in ('<', '<='):
                l_index = int(float(l_value)) + (1 if l_operator == '<' else 0)
                r_index = int(float(r_value)) - (1 if r_operator == '<' else 0)
            elif l_operator in ('>', '>=') and r_operator in ('>', '>='):
                r_index = int(float(l_value)) - (1 if l_operator == '>' else 0)
                l_index = int(float(r_value)) + (1 if r_operator == '>' else 0)
            else:
                raise Exception
            value_range = list(range(l_index, r_index + 1))

        # print(f"Przedział dla {feature_name}: {value_range}")

        # Dekodowanie wartości
        decoded_values = []
        for idx in value_range:
            if idx in reverse_mapping:
                value = reverse_mapping[idx]
                if pd.isna(value):  # Sprawdzenie, czy wartość jest NaN
                    decoded_values.append("brak_danych")
                else:
                    decoded_values.append(value)
            else:
                decoded_values.append("brak_danych")
        decoded_feature = f"{feature_name} == {{{', '.join(decoded_values)}}}"
    else:
        decoded_feature = feature

    return decoded_feature


def plot_lime_importances(importances, model, categorical_features, category_encodings, label_encoder):
    class_labels = model.named_steps['classifier'].classes_

    # Dekodowanie wartości
    decoded_importances = {}
    for feature, importance in importances.items():
        decoded_feature = get_decoded_feature(feature, categorical_features, category_encodings)
        decoded_importances[decoded_feature] = importance

    # Filtrowanie cech o niskim znaczeniu
    sorted_features = {feature: score for feature, score in
                       sorted(decoded_importances.items(), key=lambda x: x[1], reverse=True) if abs(score) > 0.01}
    features, scores = zip(*sorted_features.items())

    plt.figure(figsize=(12, 8))
    # bars = plt.barh(features, scores, color='green')  # Zmiana koloru słupków
    bars = plt.barh(features, scores, color=['red' if score > 0 else 'blue' for score in scores])
    # plt.xlabel('Średnie znaczenie cechy (jadalny <- | -> trujący)')
    plt.title('Wpływ cech na predykcję klasyfikacji grzybów na podstawie LIME')

    # Odwrócenie osi y, aby najważniejsze cechy były na górze
    plt.gca().invert_yaxis()

    # Dodanie wartości na końcach słupków
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width if width > 0 else width - 0.006  # Dostosuj wartość, aby etykiety były bliżej słupków
        plt.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', va='center')

    # Odwracanie mapowania LabelEncoder dla kolumny target
    inverse_label_map = {v: k for v, k in enumerate(label_encoder.classes_)}

    # Sprawdzanie, która klasa jest klasą pozytywną
    positive_class_index = np.argmax(model.classes_)
    positive_class_label = inverse_label_map[positive_class_index]

    plt.figtext(0.13, 0.05, f"jadalny", ha="left", fontsize=10)
    plt.figtext(0.9, 0.05, f"{positive_class_label}", ha="right", fontsize=10)
    plt.figtext(0.5, 0.05, f" - większe prawdopodbieństwo klasy -", ha="center", fontsize=10)

    plt.show()

    return sorted_features


def model_predict(model, data: np.ndarray, feature_names: list, categorical_features: list, label_encoders: dict,
                  NA: str = "_NA_") -> np.ndarray:
    """
    Wrapper function for model prediction.

    :param model: Trained sklearn Pipeline.
    :param data: Data to predict on.
    :param feature_names: List of all feature names.
    :param categorical_features: List of categorical feature names.
    :param label_encoders: Dictionary of label encoders for categorical features.
    :return: Model predictions.
    """
    # Ensure data is a 2D array
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Convert the numpy array to DataFrame for easier manipulation
    data_df = pd.DataFrame(data, columns=feature_names)

    # Encode categorical features
    for col in categorical_features:
        if col in data_df.columns:
            # print("Column=", col)
            # Decode using label encoder and replace placeholder with np.nan
            data_df[col] = label_encoders[col].inverse_transform(data_df[col].astype(int))
            data_df[col] = data_df[col].replace(NA, np.nan)

    # Predict using the model
    return model.predict(data_df)


def random_observation_for_class(X, X_str_label_encoded, y, target_class, target, eatable_label, label_encoder,
                                 label_encoders_for_X_str, original_feature_names, original_categorical_features_names,
                                 model, plot=False):
    """
    Losuje obserwację z X_test dla zadanej klasy.

    :param X: DataFrame z danymi testowymi.
    :param y: Series lub lista z etykietami klas dla X_test.
    :param target_class: Klasa, dla której losujemy obserwację.
    :return: Indeks wylosowanej obserwacji.
    """
    indices = np.where(y == target_class)[0]
    chosen_index = np.random.choice(indices)

    instance: pd.DataFrame = X.drop(target, axis=1).iloc[chosen_index]
    instance_str_label_encoded = X_str_label_encoded.drop(target, axis=1).iloc[chosen_index].values

    if plot:
        plot_numerical_distributions(X, observation_index=chosen_index)
        plot_categorical_columns(X, y_df=y, label_encoder=label_encoder, target_name=target,
                                 observation_index=chosen_index)

    model_prediction = \
        model_predict(model, instance_str_label_encoded, original_feature_names,
                      original_categorical_features_names,
                      label_encoders_for_X_str)[0]
    print("Prediction=", "jadalny" if model_prediction == eatable_label else "trujący")

    return chosen_index, instance, instance_str_label_encoded


def generate_anchor_explanation(index, instance, explainer, model, model_predict_func, feature_names,
                                categorical_features,
                                label_encoders, eatable_label, max_attempts=3, NA="_NA_"):
    """
    Generates an explanation for a selected observation.

    :param index: Index of the observation to explain.
    :param instance: Series with observation to be explained.
    :param explainer: AnchorTabularExplainer object.
    :param model: Model to be explained.
    :param model_predict_func: Function for model prediction.
    :param feature_names: List of all feature names.
    :param categorical_features: List of categorical feature names.
    :param label_encoders: Dictionary of LabelEncoders for categorical features.
    :param max_attempts: Maximum number of attempts to generate an explanation.
    :return: Generated explanation.
    """

    best_exp = None
    best_metric_for_exp = -1

    for attempt in range(max_attempts):
        exp = explainer.explain_instance(
            instance,
            lambda x: model_predict_func(model, x, feature_names, categorical_features, label_encoders),
            threshold=0.95
        )

        if exp.coverage() > best_metric_for_exp:
            best_exp = exp
            best_metric_for_exp = exp.coverage() * exp.precision()

    # Replace "_NA_" in best_exp names with the actual feature name
    assert len(best_exp.names()) == len(best_exp.features()), "Length of names and features should match"
    for i, feature_index in enumerate(best_exp.features()):
        if best_exp.names()[i] == NA:
            _feature_name = feature_names[feature_index]
            best_exp.__dict__['exp_map']['names'][i] = "*" + _feature_name + " = " + NA
        else:
            if feature_names[feature_index] not in best_exp.names()[i]:
                raise ValueError("Mismatch between feature names and explanation names.")

    # Check if "_NA_" still exists in exp names
    if "_NA_" in best_exp.names():
        raise ValueError("Unresolved '_NA_' in explanation names.")

    print("For observation#", index, "prediction=",
          "jadalny" if model_predict_func(model, instance, feature_names, categorical_features, label_encoders)[
                           0] == eatable_label else "trujący")
    anchor_names = [name for name in best_exp.names() if name != "_NA_"]
    print('Anchor: %s' % (' AND '.join(anchor_names)))
    print('Precision: %.2f' % best_exp.precision())
    print('Coverage: %.2f' % best_exp.coverage())

    return best_exp


def split_long_name(name: str, max_length: int = 50) -> str:
    """
    Splits a long name into two lines if it exceeds the maximum length.

    :param name: The name to be split.
    :param max_length: Maximum length of a single line.
    :return: The name split into two lines if necessary.
    """
    if len(name) > max_length:
        split_point = max_length // 2
        return name[:split_point] + '_\n   ' + name[split_point:]
    return name


def compare_observations(obs1: pd.DataFrame, obs1_imputed: pd.DataFrame, obs2: pd.DataFrame, feature_names: list,
                         NA="_NA_") -> None:
    """
    Compares a single observation with one or more comparison instances and prints the differences in a tabulated format.
    Includes comparison with an imputed version of the original observation.

    :param obs1: The original observation as a pandas Series or a single-row DataFrame.
    :param obs1_imputed: The imputed version of the original observation.
    :param obs2: DataFrame with one or more rows for comparison.
    :param feature_names: List of feature names.
    :param NA: Placeholder for missing values.
    """
    # Convert obs1 to DataFrame if it is a Series
    if isinstance(obs1, pd.Series):
        obs1 = obs1.to_frame().transpose()

    # Ensure obs1 and obs1_imputed are single-row DataFrames
    assert len(obs1) == 1, "obs1 must be a single-row DataFrame."
    assert obs1_imputed is None or len(obs1_imputed) == 1, "obs1_imputed must be None or a single-row DataFrame."
    assert isinstance(obs2, pd.DataFrame), "obs2 must be a pandas DataFrame."
    assert all(feature in obs2.columns for feature in feature_names), "All feature names must be in obs2 columns."

    # Extract the first row for comparison
    obs1_row = obs1.iloc[0]
    obs1_imputed_row = obs1_imputed.iloc[0] if obs1_imputed is not None else None

    for index, row in obs2.iterrows():
        print(f"\nComparison with instance {index}:")
        print("Feature\t\t\t\t\t\t\t\t\t\tOriginal Value\t\t\t\t\tModified Value")
        print("-" * 100)  # Print a separator line for better readability

        for feature in feature_names:
            feature_display = split_long_name(feature)
            original_value = obs1_row[feature]
            imputed_value = obs1_imputed_row[feature] if obs1_imputed_row is not None else None
            modified_value = row[feature]

            # Replace np.nan with NA and check for changes after imputation
            original_value = NA if pd.isna(original_value) else original_value
            imputed_value = NA if pd.isna(imputed_value) else imputed_value
            modified_value = NA if pd.isna(modified_value) else modified_value

            star_indicator = "**" if imputed_value != modified_value and imputed_value != NA else "* "

            if original_value != modified_value:
                print(
                    f"{star_indicator if original_value == NA else '  '}{feature_display:40}\t{str(original_value):30}\t->\t{str(modified_value):30}")
            else:
                print(f"  {feature_display:40}\t{str(original_value):30}")

# VERSION: 2024/12/17 - 13:57
