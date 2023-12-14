import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


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
        return value.split('--')[1]

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
        columns={'cap-diameter': 'cap-diameter-cm', 'stem-height': 'stem-height-cm', 'stem-width': 'stem-width-mm'},
        inplace=True)

    # Define dictionaries for each categorical column
    class_dict = {'p': 'poisonous', 'e': 'edibile'}
    cap_shape_dict = {'b': 'bell', 'c': 'conical', 'x': 'convex', 'f': 'flat', 's': 'sunken', 'p': 'spherical',
                      'o': 'others'}
    cap_surface_dict = {'i': 'fibrous', 'g': 'grooves', 'y': 'scaly', 's': 'smooth', 'h': 'shiny', 'l': 'leathery',
                        'k': 'silky', 't': 'sticky', 'w': 'wrinkled', 'e': 'fleshy'}
    cap_color_dict = {'n': 'brown', 'b': 'buff', 'g': 'gray', 'r': 'green', 'p': 'pink', 'u': 'purple', 'e': 'red',
                      'w': 'white', 'y': 'yellow', 'l': 'blue', 'o': 'orange', 'k': 'black'}
    bruises_bleeding_dict = {'t': 'bruises-or-bleeding', 'f': 'no-bruises-or-bleeding'}
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
            'cap-diameter-cm': 'cap-diameter-cm--średnica-kapelusza-cm',
            'stem-height-cm': 'stem-height-cm--wysokość-trzonu-cm',
            'stem-width-mm': 'stem-width-mm--szerokość-trzonu-mm',
            'class': 'class--klasa',
            'cap-shape': 'cap-shape--kształt-kapelusza',
            'cap-surface': 'cap-surface--powierzchnia-kapelusza',
            'cap-color': 'cap-color--kolor-kapelusza',
            'does-bruise-or-bleed': 'does-bruise-or-bleed--zmienia-kolor-lub-puszcza-mleczko-w-reakcji-na-uszkodzenie',
            'gill-attachment': 'gill-attachment--sposób-przyrastania-blaszek-do-trzonu',
            'gill-spacing': 'gill-spacing--gęstość-blaszek',
            'gill-color': 'gill-color--kolor-blaszek',
            'stem-root': 'stem-root--podstawa-trzonu',
            'stem-surface': 'stem-surface--powierzchnia-trzonu',
            'stem-color': 'stem-color--kolor-trzonu',
            'veil-type': 'veil-type--typ-hymenoforu',
            'veil-color': 'veil-color--kolor-hymenoforu',
            'has-ring': 'has-ring--obecność-pierścienia',
            'ring-type': 'ring-type--typ-pierścienia',
            'spore-print-color': 'spore-print-color--kolor-zarodników',
            'habitat': 'habitat--siedlisko',
            'season': 'season--pora-roku'
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
        bruises_bleeding_dict_pl = {'bruises-or-bleeding': 'zmienia-kolor-lub-puszcza-mleczko',
                                    'no-bruises-or-bleeding': 'nie-zmienia-koloru-lub-brak-mleczka'}
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
        veil_type_dict_pl = {'partial': 'z-zasnówką', 'universal': 'bez-zasnówki'}
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
        class_dict = {k: transformation(f"{v}--{class_dict_pl[v]}") for k, v in class_dict.items()}
        cap_shape_dict = {k: transformation(f"{v}--{cap_shape_dict_pl[v]}") for k, v in cap_shape_dict.items()}
        cap_surface_dict = {k: transformation(f"{v}--{cap_surface_dict_pl[v]}") for k, v in cap_surface_dict.items()}
        cap_color_dict = {k: transformation(f"{v}--{cap_color_dict_pl[v]}") for k, v in cap_color_dict.items()}
        bruises_bleeding_dict = {k: transformation(f"{v}--{bruises_bleeding_dict_pl[v]}") for k, v in
                                 bruises_bleeding_dict.items()}
        gill_attachment_dict = {k: transformation(f"{v}--{gill_attachment_dict_pl[v]}") for k, v in
                                gill_attachment_dict.items()}
        gill_spacing_dict = {k: transformation(f"{v}--{gill_spacing_dict_pl[v]}") for k, v in gill_spacing_dict.items()}
        gill_color_dict = {k: transformation(f"{v}--{gill_color_dict_pl[v]}") for k, v in gill_color_dict.items()}
        stem_root_dict = {k: transformation(f"{v}--{stem_root_dict_pl[v]}") for k, v in stem_root_dict.items()}
        stem_surface_dict = {k: transformation(f"{v}--{stem_surface_dict_pl[v]}") for k, v in stem_surface_dict.items()}
        stem_color_dict = {k: transformation(f"{v}--{stem_color_dict_pl[v]}") for k, v in stem_color_dict.items()}
        veil_type_dict = {k: transformation(f"{v}--{veil_type_dict_pl[v]}") for k, v in veil_type_dict.items()}
        veil_color_dict = {k: transformation(f"{v}--{veil_color_dict_pl[v]}") for k, v in veil_color_dict.items()}
        has_ring_dict = {k: transformation(f"{v}--{has_ring_dict_pl[v]}") for k, v in has_ring_dict.items()}
        ring_type_dict = {k: transformation(f"{v}--{ring_type_dict_pl[v]}") for k, v in ring_type_dict.items()}
        spore_print_color_dict = {k: transformation(f"{v}--{spore_print_color_dict_pl[v]}") for k, v in
                                  spore_print_color_dict.items()}
        habitat_dict = {k: transformation(f"{v}--{habitat_dict_pl[v]}") for k, v in habitat_dict.items()}
        season_dict = {k: transformation(f"{v}--{season_dict_pl[v]}") for k, v in season_dict.items()}

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
    return '\n'.join(label.split('--')).replace('-', ' ')


def wrap_labels(label, max_words_per_line=7):
    lines = '\n'.join(label.split('--')).replace('-', '\n')
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
                             NA: str = "<NA>") -> None:
    """
    Generates bar plots for categorical columns in a DataFrame.
    Optionally highlights the bar corresponding to a specified observation.

    Args:
    df (pd.DataFrame): DataFrame for which the plots are to be generated.
    observation_index (int, optional): Index of the observation to highlight.
    NA (str, optional): Representation for missing data. Defaults to "<NA>".
    """

    if y_df is not None and label_encoder is not None and target_name is not None:
        inverse_label_map = {v: k for v, k in enumerate(label_encoder.classes_)}
        df[target_name] = np.vectorize(inverse_label_map.get)(y_df)
        cols = [target_name] + [col for col in df.columns if col != target_name]
        df = df[cols]

    print("\nBar Plots for Categorical Columns:")
    categorical_columns = df.select_dtypes(include=['object']).columns

    num_vars = len(categorical_columns)
    num_rows = (num_vars + 2) // 3

    fig, axes = plt.subplots(num_rows, 3, figsize=(20, 5 * num_rows))  # Increased figure height

    for i, col in enumerate(categorical_columns):
        row = i // 3
        col_pos = i % 3

        # Adding category for missing data
        temp_series = df[col].fillna(NA)

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
                bar.set_color('#D3D3D3')  # Jasnoszary dla <NA>
            else:
                bar.set_color('#C39BD3')  # Jasny fioletowy dla pozostałych słupków
            if observation_index is not None and col in df.columns:
                observation_value = df.iloc[observation_index][col]
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


def plot_shap_waterfall_for_class(model, X, y, explainer, label_encoder, class_label):
    """
    Plots a SHAP waterfall chart for a randomly selected observation from a specified class.

    Args:
    model: The trained model (Pipeline).
    X: Test or train.
    y: y labels.
    explainer: SHAP explainer object.
    class_label: The class label (0 or 1) for which to generate the plot.
    feature_names: List of feature names.
    """
    # Transforming the data
    transformed_X = model[:-1].transform(X)

    # Selecting a random observation from the specified class
    class_indices = np.where(y == class_label)[0]
    selected_index = random.choice(class_indices)
    selected_observation = transformed_X[selected_index]

    # Reshaping the selected observation if necessary
    if isinstance(selected_observation, np.ndarray):
        X_sample_for_shap = transformed_X
    else:
        X_sample_for_shap = selected_observation.toarray()

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    a_feature_names = model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(
        numeric_features).tolist() + \
                      model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(
                          categorical_features).tolist()

    # Calculating SHAP values for the selected observation
    shap_explanation = explainer(X_sample_for_shap)
    # Creating an Explanation object with feature names
    shap_values_single = shap_explanation[0]
    shap_values_single = shap.Explanation(values=shap_values_single.values,
                                          base_values=shap_values_single.base_values,
                                          data=shap_values_single.data,
                                          feature_names=a_feature_names)

    # Getting the class name from label encoder
    class_name = label_encoder.inverse_transform([class_label])[0]

    # Visualizing SHAP values for the selected observation
    plt.title(f"SHAP Waterfall Plot for Class '{class_name}'")
    shap.plots.waterfall(shap_values_single)

    return selected_index

# VERSION: 2024/12/14 - 07:02
