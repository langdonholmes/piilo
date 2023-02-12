import pandas as pd

from names_database import NameDatabase

names_db = NameDatabase

def describe_name(first_names, last_names):
    gender = names_db.get_gender() if first_names else None
    country = names_db.get_country() if last_names else None
    return gender, country

def split_name(all_names):
    '''Splits name into parts.
    If one token, assume it is a first name.
    If two tokens, first and last name.
    If three tokens, one first name and two last names.
    If four tokens, two first names and two last names.'''
    match all_names.split():
        case [first]:
            return first, None
        case [first, last]:
            return first, last
        case [first, last_1, last_2]:
            return first, ' '.join((last_1, last_2))
        case [first_1, first_2, last_1, last_2]:
            return ' '.join((first_1, first_2)), ' '.join((last_1, last_2))
        case _:
            return None, None

def match_name(original_name):
    # FIXME: take too LONG time to run (large df used multi-times), how to improve
    # FIXME: here we only keep the first name for now
    # TODO: how to match both first and last? -- first name match gender, last name match country?
    # gender is not applied to last name
    # the name distinguished by first and last?
    # FIXME: since it is completely random, the same original name may be diff after replacing. How to know whether the two names is the same person?
    first_name = original_name.split()[0]
    global fb_df
    names = fb_df[fb_df['first']==first_name]
    if not names.empty:
        name_df = names.sample(n=1)
        # prevent for same name - deleting same name from df
        new_df = fb_df[fb_df['first'] != first_name]
        new_name = replace_name(name_df, new_df)
        return new_name
    else:
        return 'Jane Doe'

def replace_name(name_df, new_df):
    """
    :param name_df: df that match the original first name -> data frame
    :param new_df: df that does not repeat with original name
    :return: whole name: that match country & gender -> str
    """
    gender = name_df['gender'].to_string(index=False)
    country = name_df['country'].to_string(index=False)

    # match country, then match gender
    country_df = new_df[new_df['country'] == country]
    country_g_df = country_df[country_df['gender'] == gender]

    first = country_g_df['first'].sample(n=1).to_string(index=False)
    last = country_g_df['last'].sample(n=1).to_string(index=False)
    return first+' '+last



def match_name_2(original_name):
    """
    Work by match gender from first name, match country from the last name
    :param original_name:
    :return:
    """
    global fb_df
    fb_df = pd.read_parquet('ascii_fb_names_small.parquet')
    # FIXME: work when get a full name, may need branch to only first or last name....
    gender = name_match_gender(original_name.split()[0])
    print(original_name.split()[1])
    country = name_match_country(original_name.split()[-1])
    return replace_name_2(gender, country)


def name_match_country(last_name):
    names = fb_df[fb_df['last'] == last_name]
    if not names.empty:
        country = names['country'].sample(n=1).to_string(index=False)
        return country
    else:
        return 'US'

def name_match_gender(first_name):
    names = fb_df[fb_df['first'] == first_name]
    gender = names['gender'].sample(n=1).to_string(index=False)
    return gender

def replace_name_2(gender, country):
    # TODO: prevent same name
    country_df = fb_df[fb_df['country'] == country]
    country_g_df = country_df[country_df['gender'] == gender]

    first = country_g_df['first'].sample(n=1).to_string(index=False)
    last = country_g_df['last'].sample(n=1).to_string(index=False)
    full_name = first +' ' + last
    return full_name

def replace_text(str_list):
    surrogate_text = ''
    for i in str_list:
        if isinstance(i, tuple):
            i = match_entity(i[0], i[1])
        surrogate_text += i
    return surrogate_text

if __name__ == "__main__":
    fb_df = pd.read_parquet('ascii_fb_names_small.parquet')
    # print(matching("PH", 'female', 'first', 'Momo', fb_df))
    print(match_entity('Nora Wang', 'STUDENT'))
