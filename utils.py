def transform_char_df(character_df):
    character_df = character_df.set_index(['Consonant', 'Glyph']).unstack()['Character'].fillna('*')

    character_split = {}
    glyphs = character_df.columns
    for row_idx, row in character_df.iterrows():
        for col_idx, value in enumerate(row.values):
            character_split[value] = {'consonant': row_idx, 'glyph': glyphs[col_idx]}

    return character_df, character_split


def remove_duplicates(text):
    if len(text) > 1:
        letters = [text[0]] + [letter for idx, letter in enumerate(text[1:], start=1) if text[idx] != text[idx-1]]
    elif len(text) == 1:
        letters = [text[0]]
    else:
        return ""
    return "".join(letters)


def correct_prediction(word):
    parts = word.split("-")
    parts = [remove_duplicates(part) for part in parts]
    corrected_word = "".join(parts)
    return corrected_word


def levenshtein_distance(row):
    s1 = row['actual']
    s2 = row['prediction_corrected']
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]
