import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
            'does-bruise-or-bleed': 'does-bruise-or-bleed--siniaczenie-lub-krwawienie-w-reakcji-na-uszkodzenie',
            'gill-attachment': 'gill-attachment--przyczepność-blaszek',
            'gill-spacing': 'gill-spacing--rozmieszczenie-blaszek',
            'gill-color': 'gill-color--kolor-blaszek',
            'stem-root': 'stem-root--korzeń-trzonu',
            'stem-surface': 'stem-surface--powierzchnia-trzonu',
            'stem-color': 'stem-color--kolor-trzonu',
            'veil-type': 'veil-type--typ-zarodni',
            'veil-color': 'veil-color--kolor-zarodni',
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
                             'sunken': 'wklęsły', 'spherical': 'kulisty', 'others': 'inne'}
        cap_surface_dict_pl = {'fibrous': 'włóknista', 'grooves': 'bruzdowana', 'scaly': 'łuskowata',
                               'smooth': 'gładka', 'shiny': 'błyszcząca', 'leathery': 'skórzasta',
                               'silky': 'jedwabista', 'sticky': 'lepka', 'wrinkled': 'pomarszczona',
                               'fleshy': 'mięsista'}
        cap_color_dict_pl = {'brown': 'brązowy', 'buff': 'jasnobrązowy', 'gray': 'szary', 'green': 'zielony',
                             'pink': 'różowy', 'purple': 'purpurowy', 'red': 'czerwony', 'white': 'biały',
                             'yellow': 'żółty', 'blue': 'niebieski', 'orange': 'pomarańczowy', 'black': 'czarny'}
        bruises_bleeding_dict_pl = {'bruises-or-bleeding': 'występują-siniaki-lub-krwawienie',
                                    'no-bruises-or-bleeding': 'bez-siniaków-lub-krwawienia'}
        gill_attachment_dict_pl = {'adnate': 'przylegające', 'adnexed': 'przyczepione', 'decurrent': 'zbiegające',
                                   'free': 'wolne', 'sinuate': 'faliste', 'pores': 'pory', 'none': 'brak',
                                   'unknown': 'nieznane'}
        gill_spacing_dict_pl = {'close': 'ciasne', 'distant': 'rzadkie', 'none': 'brak'}
        gill_color_dict_pl = {**cap_color_dict_pl, 'none': 'brak'}
        stem_root_dict_pl = {'bulbous': 'bulwiasty', 'swollen': 'spuchnięty', 'club': 'maczugowaty',
                             'cup': 'kubeczkowaty', 'equal': 'równy', 'rhizomorphs': 'strzępkokształtne',
                             'rooted': 'ukorzeniony'}
        stem_surface_dict_pl = {**cap_surface_dict_pl, 'none': 'brak'}
        stem_color_dict_pl = {**cap_color_dict_pl, 'none': 'brak'}
        veil_type_dict_pl = {'partial': 'częściowy', 'universal': 'uniwersalny'}
        veil_color_dict_pl = {**cap_color_dict_pl, 'none': 'brak'}
        has_ring_dict_pl = {'ring': 'pierścień', 'none': 'brak'}
        ring_type_dict_pl = {'cobwebby': 'pajęczynowaty', 'evanescent': 'przemijający', 'flaring': 'rozchylający-się',
                             'grooved': 'bruzdowany', 'large': 'duży', 'pendant': 'wiszący', 'sheathing': 'otulający',
                             'zone': 'strefowy', 'scaly': 'łuskowaty', 'movable': 'ruchomy', 'none': 'brak',
                             'unknown': 'nieznany'}
        spore_print_color_dict_pl = {**cap_color_dict_pl}
        habitat_dict_pl = {'grasses': 'trawy', 'leaves': 'liście', 'meadows': 'łąki', 'paths': 'ścieżki',
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


def wrap_labels(label):
    return '\n'.join(label.split('--')).replace('-', '\n')


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
