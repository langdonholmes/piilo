import pandas as pd

def replace_name_old(country_code, gender, f_l, original_name, fb_df):
    """
    Receiving country, gender, first_last name, and the original name.
    Match with a name that matches gender and country, and is randomly retrieved from the
    facebook dataset.
    Compare the surrogate name with the original name to make sure they are different.
    Return the surrogate name in a form of string.
    f_l: F or L for first or last name -> str
    """
    # prioritizing GENDER over country?
    # it is a very large dataset so can take long, how to improve the speed?
    # Q: If want to get a whole name at a time? (just combining)
    # Q: If only get initials? (change to other letters which should be easy)
    # translating gender code
    ###### randomly find a match in the data set! And a return a similar one
    # if gender == 'male':
    #     gender = 'M'
    # elif gender == 'female':
    #     gender = 'F'
    # else:
    #     gender = None

    surrogate_name = original_name
    # checking whether the surrogate name and the original name is the same
    # using the while loop
    # TODO: [Old version] the order of gender and country need to be changed
    while(surrogate_name == original_name):
        # situation when gender can be matched
        if not gender:
            gender_df = fb_df[fb_df["gender"] == gender]
            gender_c_df = gender_df[gender_df["country"] == country_code]
            # situations: whether country code can be matched
            if gender_c_df.shape[0] > 0:
                surrogate_name = gender_c_df[f_l].sample(n=1).to_string()
            # if gender match, country not match: randomly return from gender df
            else:
                surrogate_name = gender_df[f_l].sample(n=1).to_string()
        else:
            # situation when gender cannot be match: gender is None
            country_df = fb_df[fb_df["country"] == country_code]
            # situation when country can be matched
            if country_df.shape[0] > 0:
                surrogate_name = country_df[f_l].sample(n=1).to_string()
            # situation when neither gender nor country can be matched
            # randomly return one name from the whole dataset
            else:
                surrogate_name = fb_df[f_l].sample(n=1).to_string()

    return surrogate_name

def match_entity(original_info, entity):
    # TODO: need refinement for each kind of entity
    if entity == 'STUDENT':
    # TODO: here, change between 1 and 2
        return match_name_2(original_info)
    elif entity == 'EMAIL_ADDRESS':
        return 'JaneDoe@mail.com'
    elif entity == 'PHONE_NUMBER':
        #TODO: specific form of number will be returned for consistency
        return '000-000-0000'
    elif entity == 'URL':
        return 'google.com'
    else:
        pass

def match_name(original_name):
    # FIXME: take too LONG time to run (large df used multi-times), how to improve
    # FIXME: here we only keep the first name for now
    # TODO: how to match both first and last? -- first name match gender, last name match country?
    # gender is not applied to last name
    # the name distinguished by first and last?
    # FIXME: since it is completely random, the same original name may be diff after replacing. How to know whether the two names is the same person?
    first_name = original_name.split()[0]
    global fb_df
    fb_df = pd.read_parquet('ascii_fb_names_small.parquet')
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
