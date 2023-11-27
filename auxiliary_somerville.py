from typing import Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

NA = "<NA>"


def translate_and_unify_categories(df: pd.DataFrame) -> pd.DataFrame:
    translation_dict = {
        'What.is.your.gender': {
            'Female': 'Kobieta',
            'Male': 'Mężczyzna',
            'Nonbinary': 'Niebinarny',
            'Multiple people': 'Wypełnia wiele osób'
        },
        'Age': {
            '35-44': '35-44',
            '55-64': '55-64',
            '45-54': '45-54',
            '65-74': '65-74',
            '75+': '75+',
            '25-34': '25-34',
            '18-24': '18-24',
            '61+': '61+',
            '26-30': '26-30',
            '22-25': '22-25',
            '41-50': '41-50',
            '31-40': '31-40',
            '51-60': '51-60',
            '18-21': '18-21',
            '17 or younger': '17 lub młodszy'
        },
        'Marital.status': {
            'Single (never married)': 'Singiel',
            'Divorced/ Separated': 'Rozwiedziony/W separacji',
            'Widowed': 'Wdowiec/Wdowa',
            'Married': 'Żonaty/Zamężna',
            'Living with a partner, but not married': 'Żyjący w związku, niezamężny',
            'Single, Never Married': 'Singiel',
            'Divorced': 'Rozwiedziony'
        },
        'What.is.your.race.or.ethnicity': {
            'Two or more races': 'Dwie lub więcej ras',
            'White': 'Biała',
            'Black or African American': 'Czarna lub Afroamerykanin',
            'Asian or Pacific Islander': 'Azjata lub mieszkaniec wysp Pacyfiku',
            'Hispanic or Latino': 'Hiszpański lub Latynos',
            'Another race': 'Inna rasa'
        },
        'Do.you.have.children.age.18.or.younger.who.live.with.you': {
            'yes': 'tak',
            'no': 'nie'
        },
        'Describe.your.housing.status.in.Somerville': {
            'Rent': 'Wynajmuję',
            'Own': 'Własność',
            'Other': 'Inne'
        },
        'Do.you.plan.to.move.away.from.Somerville.in.the.next.two.years': {
            'no': 'nie',
            'yes': 'tak'
        },
        'How.long.have.you.lived.here': {
            '11-15 years': '11-15 lat',
            '21 years or more': '21 lat lub więcej',
            '1-3 years': '1-3 lata',
            '8-10 years': '8-10 lat',
            '16-20 years': '16-20 lat',
            '4-7 years': '4-7 lat',
            'Less than a year': 'Mniej niż rok'
        },
        'What.is.your.annual.household.income': {
            '$60,000 to $79,999': '60,000 do 79,999 dolarów',
            '$20,000 to $39,999': '20,000 do 39,999 dolarów',
            '$40,000 to $59,999': '40,000 do 59,999 dolarów',
            'Less than $20,000': 'Mniej niż 20,000 dolarów',
            '$80,000 to $99,999': '80,000 do 99,999 dolarów',
            '$160,000 to $179,999': '160,000 do 179,999 dolarów',
            '$100,000 to $119,999': '100,000 do 119,999 dolarów',
            '$200,000 or more': '200,000 dolarów lub więcej',
            '$140,000 to $159,999': '140,000 do 159,999 dolarów',
            '$120,000 to $139,999': '120,000 do 139,999 dolarów',
            '$180,000 to $199,999': '180,000 do 199,999 dolarów',
            'Less than $10,000': 'Mniej niż 10,000 dolarów',
            '$25,000 to $49,999': '25,000 do 49,999 dolarów',
            '$10,000 to $19,999': '10,000 do 19,999 dolarów',
            '$10,000 to $24,999': '10,000 do 24,999 dolarów',
            '$50,000 to $74,999': '50,000 do 74,999 dolarów',
            '$75,000 to $99,999': '75,000 do 99,999 dolarów',
            '$100,000 to $149,999': '100,000 do 149,999 dolarów',
            '$100,000 or more': '100,000 dolarów lub więcej',
            '$150,000 to $199,999': '150,000 do 199,999 dolarów',
            '$150,000 or more': '150,000 dolarów lub więcej'
        },
        'Are.you.a.student': {
            'yes': 'tak',
            'no': 'nie'
        },
        'Precinct': {
            '1/2/2023 0:00': '2 stycznia 2023',
            '2/3/2023 0:00': '3 lutego 2023',
            '1/4/2023 0:00': '4 stycznia 2023',
            '2/2/2023 0:00': '2 lutego 2023',
            '1/3/2023 0:00': '3 stycznia 2023',
            '1/1/2023 0:00': '1 stycznia 2023',
            '2-1A': '2-1A'  # Ten wpis wydaje się być specyficznym kodem, który nie wymaga tłumaczenia
        },
        'Do.you.feel.the.City.is.headed.in.the.right.direction.or.is.it.on.the.wrong.track': {
            'Right direction': 'Dobry kierunek',
            'Wrong track': 'Zły kierunek',
            'Not sure': 'Nie jestem pewny/pewna'
        },
        'What.is.your.primary.mode.of.transportation': {
            'Car': 'Samochód',
            'Walk, Car': 'Spacer, Samochód',
            'Walk': 'Spacer',
            'Public transit': 'Transport publiczny',
            'Bike, Car': 'Rower, Samochód',
            'Walk, Public transit': 'Spacer, Transport publiczny',
            'Bike': 'Rower',
            'Bike, Public transit': 'Rower, Transport publiczny',
            'Walk, Bike, Public transit': 'Spacer, Rower, Transport publiczny',
            'Walk, Bike, Car': 'Spacer, Rower, Samochód',
            'Public transit, Car': 'Transport publiczny, Samochód',
            'Walk, Bike, Public transit, Car': 'Spacer, Rower, Transport publiczny, Samochód',
            'Walk, Bike': 'Spacer, Rower',
            'Walk, Public transit, Car': 'Spacer, Transport publiczny, Samochód',
            'Bike, Public transit, Car': 'Rower, Transport publiczny, Samochód'
        },
        'Which.of.the.following.have.you.used.in.the.past.month.to.get.around': {
            'Car': 'Samochód',
            'Walk, Car': 'Spacer, Samochód',
            'Walk, Public transit, Car': 'Spacer, Transport publiczny, Samochód',
            'Public transit': 'Transport publiczny',
            'Public transit, Car': 'Transport publiczny, Samochód',
            'Walk, Bike, Public transit, Car': 'Spacer, Rower, Transport publiczny, Samochód',
            'Walk, Public transit': 'Spacer, Transport publiczny',
            'Walk, Bike, Public transit': 'Spacer, Rower, Transport publiczny',
            'Walk, Bike': 'Spacer, Rower',
            'Walk': 'Spacer',
            'Walk, Bike, Car': 'Spacer, Rower, Samochód',
            'Bike, Car': 'Rower, Samochód',
            'Bike, Public transit': 'Rower, Transport publiczny',
            'Bike': 'Rower'
        },
        'Language': {
            'English': 'Angielski',
            'Spanish': 'Hiszpański',
            'Portuguese': 'Portugalski',
            'Haitian Creole': 'Kreolski haitański',
            'Nepali': 'Nepalski'
        },
        'survey_method': {
            'Phone': 'Telefon',
            'Email': 'E-mail',
            'Facebook 18-24 year olds': 'Facebook (18-24 lat)',
            'Mail': 'Poczta'
        },
        'language_spoken_category': {
            'English Only Speaker': 'Tylko angielski',
            'English+ Speaker': 'Angielski i inne',
            'Non-English Speaker': 'Nie mówi po angielsku',
            'Non-English Language Speaker (English Unknown)': 'Mówi w innym języku (angielski nieznany)'
        },
        'disability_yn': {
            'yes': 'tak',
            'no': 'nie'
        },
        'employment_status': {
            'Employed': 'Zatrudniony',
            'Unemployed': 'Bezrobotny',
            'Retired': 'Na emeryturze',
            'Disabled': 'Niepełnosprawny',
            'Student': 'Student',
            'Homemaker': 'Gospodarz domowy',
            'Self-employed': 'Samozatrudniony'
        },
        'in_the_past_year_have_you_used_311_via_phone_online_etc': {
            'Yes, more than once': 'Tak, więcej niż raz',
            'No': 'Nie',
            'Yes, once': 'Tak, raz'
        },
        'in_the_past_year_did_you_attend_a_city_led_meeting': {
            'No': 'Nie',
            'Yes, more than once': 'Tak, więcej niż raz',
            'Yes, once': 'Tak, raz'
        },
    }

    for column, mapping in translation_dict.items():
        df[column] = df.copy()[column].map(mapping).fillna(df[column])

    return df


def recode_one_hot_columns(df: pd.DataFrame, prefix: str, translations: dict = None, NA="<NA>") -> pd.DataFrame:
    """
    Funkcja przekształca kolumny one-hot w DataFrame na jedną kolumnę z wartościami kategorycznymi.

    Args:
    df (pd.DataFrame): DataFrame do przetworzenia.
    prefix (str): Prefiks kolumn one-hot.
    translations (dict, optional): Słownik tłumaczeń nazw kategorii.
    NA (str, optional): Reprezentacja brakujących danych. Domyślnie "<NA>".

    Returns:
    pd.DataFrame: DataFrame z dodaną kolumną recoded.
    """
    # Znajdź kolumny, które pasują do wzorca i są typu float64
    one_hot_columns = [col for col in df.columns if col.startswith(prefix) and df[col].dtype == 'float64']

    # Tworzenie nowej kolumny z połączonych wartości
    def recode_row(row):
        # Lista kategorii dla danego wiersza
        categories = [col.replace(prefix, '') for col in one_hot_columns if row[col] == 1.0]
        if translations:
            try:
                translated_categories = [translations[category] for category in categories]
            except KeyError as e:
                raise KeyError(f"Brak tłumaczenia dla kategorii: {e.args[0]}") from None
            return ', '.join(translated_categories) if translated_categories else NA
        else:
            return ', '.join(categories) if categories else NA

    # Dodaj nową kolumnę do DataFrame
    df[f'{prefix}recoded'] = df.copy().apply(recode_row, axis=1)
    return df


def convert_to_categorical(df: pd.DataFrame, columns: list, NA="<NA>") -> pd.DataFrame:
    new_df = df.copy()

    for col in columns:
        assert col in new_df.columns, f"Column '{col}' not found!"
        if new_df[col].dtype in ['float64', 'int64']:
            # Sprawdź, czy kolumna jest one-hot (tylko wartości 0 i 1)
            if set(new_df[col].unique()).issubset({0, 1, pd.NA}):
                new_df[col] = new_df[col].apply(lambda x: 'tak' if x == 1 else ('nie' if x == 0 else NA))
            else:
                # Dla innych kolumn numerycznych, konwertuj na string
                new_df[col] = new_df[col].apply(lambda x: str(int(x)) if pd.notnull(x) else NA)

    return new_df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    translations: Dict[str, str] = {
        "Combined_ID": "combined.id",
        "Year": "rok",
        "How.happy.do.you.feel.right.now": "jak.szczęśliwy.jesteś.teraz",
        "How.satisfied.are.you.with.your.life.in.general": "jak.zadowolony.jesteś.z.życia.ogólnie",
        "How.satisfied.are.you.with.Somerville.as.a.place.to.live": "jak.zadowolony.jesteś.z.Somerville.jako.miejsca.do.życia",
        "In.general..how.similar.are.you.to.other.people.you.know": "jak.podobny.jesteś.do.innych.ludzi.których.znasz",
        "When.making.decisions..are.you.more.likely.to.seek.advice.or.decide.for.yourself": "czy.podejmując.decyzje.szukasz.rady.czy.sam.decydujesz",
        "How.satisfied.are.you.with.your.neighborhood": "jak.zadowolony.jesteś.z.twojej.okolicy",
        "How.proud.are.you.to.be.a.Somerville.resident": "jak.dumny.jesteś.z.bycia.mieszkańcem.Somerville",
        "How.would.you.rate.the.following..The.availability.of.information.about.city.services": "jak.oceniasz.następujące..dostępność.informacji.o.usługach.miejskich",
        "How.would.you.rate.the.following..The.availability.of.affordable.housing": "jak.oceniasz.następujące..dostępność.przystępnych.cenowo.mieszkań",
        "How.would.you.rate.the.following..The.cost.of.housing": "jak.oceniasz.następujące..koszt.mieszkań",
        "How.would.you.rate.the.following..The.overall.quality.of.public.schools": "jak.oceniasz.następujące..ogólną.jakość.szkoł.publicznych",
        "How.would.you.rate.the.following..The.beauty.or.physical.setting.of.Somerville": "jak.oceniasz.następujące..piękno.lub.otoczenie.fizyczne.Somerville",
        "How.would.you.rate.the.following..The.effectiveness.of.the.local.police": "jak.oceniasz.następujące..skuteczność.policji.lokalnej",
        "How.would.you.rate.the.following..Your.trust.in.the.local.police": "jak.oceniasz.następujące..zaufanie.do.policji.lokalnej",
        "How.would.you.rate.the.following..The.maintenance.of.streets..sidewalks..and..squares": "jak.oceniasz.następujące..utrzymanie.ulic..chodników..i.placów",
        "How.would.you.rate.the.following..The.maintenance.of.streets.and.sidewalks": "jak.oceniasz.następujące..utrzymanie.ulic.i.chodników",
        "How.would.you.rate.the.following..The.availability.of.social.community.events": "jak.oceniasz.następujące..dostępność.wydarzeń.społecznościowych",
        "How.safe.do.you.feel.walking.in.your.neighborhood.at.night": "jak.bezpiecznie.czujesz.się.chodząc.po.twojej.okolicy.nocą",
        "How.satisfied.are.you.with.the.beauty.or.physical.setting.of.your.neighborhood": "jak.zadowolony.jesteś.z.piękna.lub.otoczenia.fizycznego.twojej.okolicy",
        "How.satisfied.are.you.with.the.appearance.of.parks.in.your.neighborhood": "jak.zadowolony.jesteś.z.wyglądu.parków.w.twojej.okolicy",
        "How.satisfied.are.you.with.the.appearance.of.parks.and.squares.in.your.neighborhood": "jak.zadowolony.jesteś.z.wyglądu.parków.i.placów.w.twojej.okolicy",
        "What.is.your.gender": "jakiej.jesteś.płci",
        "Age": "wiek",
        "Marital.status": "stan.cywilny",
        "Are.you.of.Hispanic..Latino..or.Spanish.origin": "czy.jesteś.pochodzenia.hiszpańskiego.lub.latynoskiego",
        "What.is.your.race.or.ethnicity": "jakiej.jesteś.rasy.lub.etniczności",
        "Do.you.have.children.age.18.or.younger.who.live.with.you": "czy.masz.dzieci.w.wieku.do.18.lat.mieszkające.z.tobą",
        "Describe.your.housing.status.in.Somerville": "opisz.swój.status.mieszkaniowy.w.Somerville",
        "Do.you.plan.to.move.away.from.Somerville.in.the.next.two.years": "czy.planujesz.przeprowadzkę.z.Somerville.w.najbliższych.dwóch.latach",
        "How.long.have.you.lived.here": "jak.długo.tu.mieszkasz",
        "What.is.your.annual.household.income": "jaki.jest.roczny.dochód.twojego.gospodarstwa.domowego",
        "Are.you.a.student": "czy.jesteś.studentem",
        "Ward": "dzielnica",
        "Precinct": "obwód.wyborczy",
        "How.anxious.did.you.feel.yesterday": "jak.zaniepokojony.czułeś.się.wczoraj",
        "How.satisfied.are.you.with.the.quality.and.number.of.transportation.options.available.to.you": "jak.zadowolony.jesteś.z.jakości.i.ilości.dostępnych.dla.ciebie.opcji.transportowych",
        "Do.you.feel.the.City.is.headed.in.the.right.direction.or.is.it.on.the.wrong.track": "czy.uważasz.że.miasto.idzie.we.właściwym.kierunku.czy.jest.na.złej.drodze",
        "How.safe.do.you.feel.crossing.a.busy.street.in.Somerville": "jak.bezpiecznie.czujesz.się.przechodząc.przez.zatłoczoną.ulicę.w.Somerville",
        "How.convenient.is.it.for.you.to.get.where.you.want.to.go": "jak.wygodnie.jest.ci.dostać.się.tam.gdzie.chcesz",
        "How.satisfied.are.you.with.the.condition.of.your.housing": "jak.zadowolony.jesteś.z.stanu.twojego.mieszkania",
        "What.is.your.primary.mode.of.transportation": "jaki.jest.twój.główny.środek.transportu",
        "Which.of.the.following.have.you.used.in.the.past.month.to.get.around": "którego.z.następujących.użyłeś.w.ostatnim.miesiącu.do.poruszania.się",
        "Language": "język",
        "survey_method": "metoda.analizy",
        "language.spoken.english": "język.mówiony.angielski",
        "language.spoken.spanish": "język.mówiony.hiszpański",
        "language.spoken.portuguese": "język.mówiony.portugalski",
        "language.spoken.chinese": "język.mówiony.chiński",
        "language.spoken.haitian.creole": "język.mówiony.haitański.kreolski",
        "language.spoken.nepali": "język.mówiony.nepalski",
        "language.spoken.other": "język.mówiony.inny",
        "language.spoken.category": "kategoria.języka.mówionego",
        "race.ethnicity.asian.pacific.islander": "rasa.etniczność.azjatycka.wyspiarska",
        "race.ethnicity.black": "rasa.etniczność.czarna",
        "race.ethnicity.white": "rasa.etniczność.biała",
        "race.ethnicity.other": "rasa.etniczność.inna",
        "race.ethnicity.prefernottosa": "rasa.etniczność.woli.nie.podawać",
        "age.mid": "średni.wiek",
        "tenure.mid": "średni.okres.zamieszkania",
        "household.income.mid": "średni.dochód.gospodarstwa.domowego",
        "somerville.median.income": "mediana.dochodu.w.Somerville",
        "inflation.adjustment": "dostosowanie.do.inflacji",
        "disability.yn": "niepełnosprawność.tak.nie",
        "employment.status": "status.zatrudnienia",
        "zipcode": "kod.pocztowy",
        "in.the.past.year.have.you.used.311.via.phone.online.etc": "w.ostatnim.roku.czy.korzystałeś.z.311.przez.telefon.internet.itp",
        "in.the.past.year.did.you.attend.a.city.led.meeting": "w.ostatnim.roku.czy.brałeś.udział.w.spotkaniu.zorganizowanym.przez.miasto",
        "in.the.past.year.how.satisfied.were.you.with.your.ability.to.access.city.services": "w.ostatnim.roku.jak.zadowolony.byłeś.z.możliwości.korzystania.z.usług.miejskich",
        "comments.survey.complaints.political": "komentarze.analiza.skargi.polityczne",
        "comments.transportation.roads.locations": "komentarze.transport.drogi.lokalizacje",
        "comments.living.prices": "komentarze.ceny.za.życie",
        "comments.natural.physical.beauty": "komentarze.naturalne.piękno.fizyczne",
        "comments.noise.youth.complaints": "komentarze.hałas.skargi.młodzieży",
        "comments.city.events.structures": "komentarze.wydarzenia.miejskie.struktury",
        "comments.health.concerns": "komentarze.zagrożenia.dla.zdrowia",
        "comments.public.safety": "komentarze.bezpieczeństwo.publiczne",
        "comments.elaboration.general": "komentarze.elaboracja.ogólna",
        "comments.misc": "komentarze.różne",
        "language_spoken_english": "język.mówiony.angielski",
        "language_spoken_spanish": "język.mówiony.hiszpański",
        "language_spoken_portuguese": "język.mówiony.portugalski",
        "language_spoken_chinese": "język.mówiony.chiński",
        "language_spoken_haitian_creole": "język.mówiony.haitański.kreolski",
        "language_spoken_nepali": "język.mówiony.nepalski",
        "language_spoken_other": "język.mówiony.inny",
        "language_spoken_recoded": "język.mówiony.przekodowane",
        "language_spoken_category": "kategoria.języka.mówionego",
        "race_ethnicity_asian_pacific_islander": "rasa.etniczność.azjatycka.wyspiarska",
        "race_ethnicity_black": "rasa.etniczność.czarna",
        "race_ethnicity_white": "rasa.etniczność.biała",
        "race_ethnicity_other": "rasa.etniczność.inna",
        "race_ethnicity_prefernottosa": "rasa.etniczność.woli.nie.podawać",
        "race_ethnicity_recoded": "rasa.etniczność.przekodowane",
        "age_mid": "średni.wiek",
        "tenure_mid": "średni.okres.zamieszkania",
        "household_income_mid": "średni.dochód.gospodarstwa.domowego",
        "somerville_median_income": "mediana.dochodu.w.Somerville",
        "inflation_adjustment": "dostosowanie.do.inflacji",
        "disability_yn": "niepełnosprawność.tak.nie",
        "employment_status": "status.zatrudnienia",
        "in_the_past_year_have_you_used_311_via_phone_online_etc": "w.ostatnim.roku.czy.korzystałeś.z.311.przez.telefon.internet.itp",
        "in_the_past_year_did_you_attend_a_city_led_meeting": "w.ostatnim.roku.czy.brałeś.udział.w.spotkaniu.zorganizowanym.przez.miasto",
        "in_the_past_year_how_satisfied_were_you_with_your_ability_to_access_city_services": "w.ostatnim.roku.jak.zadowolony.byłeś.z.możliwości.korzystania.z.usług.miejskich",
        "comments_survey_complaints_political": "komentarze.analiza.skargi.polityczne",
        "comments_transportation_roads_locations": "komentarze.transport.drogi.lokalizacje",
        "comments_living_prices": "komentarze.ceny.za.życie",
        "comments_natural_physical_beauty": "komentarze.naturalne.piękno.fizyczne",
        "comments_noise_youth_complaints": "komentarze.hałas.skargi.młodzieży",
        "comments_city_events_structures": "komentarze.wydarzenia.miejskie.struktury",
        "comments_health_concerns": "komentarze.zagrożenia.dla.zdrowia",
        "comments_public_safety": "komentarze.bezpieczeństwo.publiczne",
        "comments_elaboration_general": "komentarze.ogólne.uwagi",
        "comments_misc": "komentarze.różne"
    }
    df_renamed = df.rename(columns=translations)

    unmapped_columns = [col for col in df.columns if col not in translations]
    if unmapped_columns:
        print("Niezmienione kolumny:", unmapped_columns)

    return df_renamed


def wrap_labels(label, max_words_per_line=7):
    words = label.split('.')
    return '\n'.join([' '.join(words[i:i + max_words_per_line]) for i in range(0, len(words), max_words_per_line)])


def wrap_title(label):
    return '\n'.join(label.split('--')).replace('.', ' ')


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


def plot_histograms_and_boxplots(df: pd.DataFrame) -> None:
    """
    Funkcja wyświetla histogramy i wykresy pudełkowe dla kolumn numerycznych w DataFrame.

    Args:
    df (pd.DataFrame): DataFrame, dla którego mają być wygenerowane wykresy.
    """
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

    print("\nHistogramy i wykresy pudełkowe dla kolumn numerycznych:")
    for i, col in enumerate(numerical_columns):
        if df[col].notna().any():
            print(f"Plot {i} out of {len(numerical_columns) - 1}, col: '{col}'")
            fig, axes = plt.subplots(1, 2, figsize=(16, 4))

            unique_values = df[col].dropna().unique()
            unique_values_rounded = np.round(unique_values)
            if len(unique_values) <= 10 and np.array_equal(unique_values, unique_values_rounded):
                bins = len(np.unique(unique_values_rounded))
            else:
                bins = 'auto'

            sns.histplot(df[col], kde=False, bins=bins, ax=axes[0])
            axes[0].set_title(f'{col}')  # Histogram
            axes[0].set_ylabel('')
            axes[0].set_xlabel('')

            # Ustawienie formatowania osi X tylko wtedy, gdy liczba binów jest ograniczona
            if bins != 'auto' and max(unique_values_rounded) - min(unique_values_rounded) <= 10:
                axes[0].xaxis.set_major_locator(ticker.MultipleLocator(1))
                axes[0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.f'))

            sns.boxplot(x=df[col], ax=axes[1])
            axes[1].set_title(f'{col}')  # Boxplot
            axes[1].set_ylabel('')
            axes[1].set_xlabel('')

            plt.show()
        else:
            print(f"Kolumna '{col}' jest pusta lub zawiera tylko wartości NaN.")




def plot_categorical_columns(df: pd.DataFrame, NA: str = "<NA>") -> None:
    """
    Funkcja generuje wykresy słupkowe dla kolumn kategorycznych w DataFrame.

    Args:
    df (pd.DataFrame): DataFrame, dla którego mają być wygenerowane wykresy.
    NA (str, optional): Reprezentacja brakujących danych. Domyślnie "<NA>".
    """
    print("\nWykresy słupkowe dla kolumn kategorycznych:")
    categorical_columns = df.select_dtypes(include=['object']).columns

    num_vars = len(categorical_columns)
    num_rows = (num_vars + 2) // 3

    fig, axes = plt.subplots(num_rows, 3, figsize=(20, 5 * num_rows))  # Zwiększona wysokość figury

    for i, col in enumerate(categorical_columns):
        row = i // 3
        col_pos = i % 3

        # Dodanie kategorii dla brakujących danych
        temp_series = df[col].fillna(NA)

        # Sortowanie etykiet według częstotliwości z wyjątkiem 'Brak danych'
        order = temp_series.value_counts().index.tolist()
        if NA in order:
            order.remove(NA)
            order.append(NA)

        # Rysowanie wykresu słupkowego
        sns.countplot(y=temp_series, ax=axes[row, col_pos], order=order)
        wrapped_title = wrap_title(col)
        axes[row, col_pos].set_title(f'\n{wrapped_title}')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            axes[row, col_pos].set_yticklabels(
                [wrap_labels(label.get_text()) for label in axes[row, col_pos].get_yticklabels()])

        axes[row, col_pos].set_ylabel('')  # Usunięcie nazwy osi y
        axes[row, col_pos].set_xlabel('')  # Usunięcie nazwy osi x

    # Ukrywanie pustych subplotów
    for j in range(i + 1, num_rows * 3):
        fig.delaxes(axes[j // 3, j % 3])

    plt.subplots_adjust(hspace=0.6, wspace=0.4)  # Zwiększony odstęp między wierszami i kolumnami
    plt.show()


# VERSION: 2024/11/27 - 21:02
