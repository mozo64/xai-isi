
# def distribution_analysis(df: pd.DataFrame):
#     # numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
#     categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
#
#     # Histogramy dla zmiennych numerycznych
#     # n_numeric_columns = len(numeric_columns)
#     # ncols_numeric = 5
#     # nrows_numeric = int(np.ceil(n_numeric_columns / ncols_numeric))
#
#     # print("Rozkład zmiennych numerycznych:")
#     # fig, axes = plt.subplots(nrows=nrows_numeric, ncols=ncols_numeric, figsize=(15, nrows_numeric * 3))
#     # for i, col in enumerate(numeric_columns):
#     #     ax = axes[i // ncols_numeric, i % ncols_numeric]
#     #     df[col].hist(bins=15, ax=ax)
#     #     ax.set_title(wrap_labels(col))
#     #
#     # # Ukrywanie pustych wykresów, jeśli istnieją
#     # for j in range(i + 1, nrows_numeric * ncols_numeric):
#     #     axes[j // ncols_numeric, j % ncols_numeric].set_visible(False)
#     #
#     # plt.tight_layout()
#     # plt.show()
#
#     # Wykresy dla zmiennych kategorycznych
#     n_categorical_columns = len(categorical_columns)
#     ncols_categorical = 5
#     nrows_categorical = int(np.ceil(n_categorical_columns / ncols_categorical))
#
#     print("Rozkład zmiennych kategorycznych:")
#     fig, axes = plt.subplots(nrows=nrows_categorical, ncols=ncols_categorical, figsize=(15, nrows_categorical * 4))
#     for i, col in enumerate(categorical_columns):
#         ax = axes[i // ncols_categorical, i % ncols_categorical]
#         sns.countplot(x=col, data=df, ax=ax)
#         ax.set_xticklabels([wrap_labels(label.get_text()) for label in ax.get_xticklabels()], rotation=90)
#
#     # Ukrywanie pustych wykresów, jeśli istnieją
#     for j in range(i + 1, nrows_categorical * ncols_categorical):
#         axes[j // ncols_categorical, j % ncols_categorical].set_visible(False)
#
#     plt.tight_layout()
#     plt.show()
#
# distribution_analysis(df)
