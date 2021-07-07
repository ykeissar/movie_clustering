import numpy as np
import pandas as pd
import pickle

def main():
    ratings,users,movies = load_tables('ml-1m')

    n_users = users.shape[0]
    n_movies = movies[-1]

    print(f'Process Data: {n_users} users, {n_movies} movies loaded')
    # create users x movies tables
    users_movies = np.zeros((n_users+1,n_movies+1))

    for rate in ratings:
        users_movies[rate[0]][rate[1]] = 1

    print(f'Process Data: saving (users x movies) table in "users_movies.csv"')
    np.savetxt("users_movies.csv", users_movies, delimiter=",")

    calc_indicator()
# Load raiting
def load_tables(path):
    ratings = np.genfromtxt(path+'/ratings.dat',
                         skip_header=0,
                         skip_footer=0,
                         names=None,
                         dtype=None,
                         delimiter='::',
                         usecols = (0,1))
    print(f'Process Data: {path}/ratings.dat loaded')

    users = np.genfromtxt(path+'/users.dat',
                     skip_header=0,
                     skip_footer=0,
                     names=None,
                     dtype=None,
                     delimiter='::',
                     encoding=None,
                     usecols = (0))
    print(f'Process Data: {path}/users.dat loaded')

    movies = np.genfromtxt(path+'/movies.dat',
                     skip_header=0,
                     skip_footer=0,
                     names=None,
                     dtype=None,
                     delimiter='::',
                     encoding='latin-1',
                     usecols = (0))
    print(f'Process Data: {path}/movies.dat loaded')

    return ratings,users,movies

def calc_indicator():
    df_rating = pd.read_csv('ml-1m/ratings.dat',
                        delimiter='::',
                        names=['u_id','m_id','rate'],
                        usecols=(0,1,2))

    df_users = pd.read_csv('ml-1m/users.dat',
                        delimiter='::',
                        names=['u_id','gender','age','occupation'],
                        usecols=(0,1,2,3))

    movies = np.genfromtxt('ml-1m/movies.dat',
                 skip_header=0,
                 skip_footer=1,
                 names=None,
                 dtype=None,
                 delimiter='::',
                 encoding='latin-1',
                 invalid_raise = False,
                 usecols = (0,2))
    
    
    df_joined = df_rating.join(df_users.set_index('u_id'),on='u_id')

    df_age = df_joined.groupby(by=['m_id'])['age'].apply(list)
    df_gender = df_joined.groupby(by=['m_id'])['gender'].apply(list)
    df_occupation = df_joined.groupby(by=['m_id'])['occupation'].apply(list)

    age_counter = {idx:Counter(df_age[idx]) for idx in df_age.keys()}
    gender_counter = {idx:Counter(df_gender[idx]) for idx in df_gender.keys()}
    occupation_counter = {idx:Counter(df_occupation[idx]) for idx in df_occupation.keys()}

    age_in = {i:get_higher_age(age_counter[i],0.25) for i in age_counter.keys()}
    gender_in = {i:get_higher_gender(gender_counter[i],0.75) for i in gender_counter.keys()}
    occupation_in = {i:get_higher_occupatin(occupation_counter[i],0.25) for i in occupation_counter.keys()}
    genre_in = {m[0]:m[1].split('|') for m in movies}

    pickle.dump(age_in,open("age_in",'wb'))
    pickle.dump(gender_in,open("gender_in",'wb'))
    pickle.dump(occupation_in,open("occupation_in",'wb'))
    pickle.dump(genre_in,open("genre_in",'wb'))


def get_higher_gender(gci,threshold):
    if gci['M']/(gci['M']+gci['F']) > threshold:
        return 'M'
    if gci['M']/(gci['M']+gci['F']) < 1-threshold:
        return 'F'
    return None

def get_higher_occupatin(oci,threshold):
    total = sum(oci.values())
    oct = []
    for k,v in oci.items():
        if v/total >threshold:
            oct.append(k)
    return oct

def get_higher_age(aci,threshold):
    total = sum(aci.values())
    ages = []
    for k,v in aci.items():
        if v/total >threshold:
            ages.append(k)
    return ages

    
if __name__ == '__main__':
    main()