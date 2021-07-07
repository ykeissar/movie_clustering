import numpy as np
import random
from main import debug
import pickle

def movies_match(movie_i, movie_j, age_in, gender_in, occupation_in,genre_in):
    matches = 0
    
    age_match = [value for value in age_in[movie_i] if value in age_in[movie_j]]
    if age_match:
        matches += 0
    
    occupation_match = [value for value in occupation_in[movie_i] if value in occupation_in[movie_j]]
    if occupation_match:
        matches += 0

    if gender_in[movie_i] == gender_in[movie_j]:
        matches += 1

    # Add match if movies genre is the same
    if genre_in[movie_i] == genre_in[movie_j]:
        matches += 0
    
    return matches

def org_CCPivot_improved(movies_ids, probs2, probs1, age_in, gender_in, occupation_in,genre_in):
    c_list = []
    movies_ids_n = movies_ids.copy()    

    while movies_ids_n:
        movie_i = random.choice(movies_ids_n)
        ind_i = movies_ids.index(movie_i)
        movies_ids_n.remove(movie_i)
        C = [movie_i]
        for movie_j in movies_ids_n:
            ind_j = movies_ids.index(movie_j)
            if (probs2[ind_i][ind_j] > probs1[ind_i]*probs1[ind_j] and movies_match(movie_i, movie_j, age_in, gender_in, occupation_in,genre_in) > 0) or movies_match(movie_i, movie_j, age_in, gender_in, occupation_in, genre_in) > 1:
                # if debug > 1:
                #     print(f'Main: placing {i},{j} together cause p({i},{j}):{probs2[i_ind][ind_j]} > p({i}):{probs_one[i_ind]}*p({j}):{probs_one[ind_j]}')
                C.append(movie_j)
                movies_ids_n.remove(movie_j)
        
        c_list.append(C)
    
    return c_list

def CCPivot_improved(movies_ids, probs2, probs1, age_in, gender_in, occupation_in, genre_in, treshold=0.5):
    c_list = []
    movies_ids_n = movies_ids.copy()    

    while movies_ids_n:
        movie_i = random.choice(movies_ids_n)
        ind_i = movies_ids.index(movie_i)
        movies_ids_n.remove(movie_i)
        C = [movie_i]
        for movie_j in movies_ids_n:
            ind_j = movies_ids.index(movie_j)
            match_counter = 0
            for movie_k in C:
                ind_k = movies_ids.index(movie_k)
                if (probs2[ind_k][ind_j] > probs1[ind_k]*probs1[ind_j] and movies_match(movie_k, movie_j, age_in, gender_in, occupation_in, genre_in) > 0) or movies_match(movie_k, movie_j, age_in, gender_in, occupation_in ,genre_in) > 1:
                    match_counter += 1
                # if debug > 1:
                #     print(f'Main: placing {i},{j} together cause p({i},{j}):{probs2[i_ind][ind_j]} > p({i}):{probs_one[i_ind]}*p({j}):{probs_one[ind_j]}')
            if match_counter/len(C) > treshold:
                C.append(movie_j)
                movies_ids_n.remove(movie_j)
        
        c_list.append(C)
    
    return c_list

def CCPivot(movies_ids: list, probs_two, probs_one): 
    c_list = []
    movies_ids_n = movies_ids.copy()    
    
    while movies_ids_n:
        i = random.choice(movies_ids_n)
        i_ind = movies_ids.index(i)
        movies_ids_n.remove(i)
        C = [i]
    
        for j in movies_ids_n:
            ind_j = movies_ids.index(j)
            if probs_two[i_ind][ind_j] > probs_one[i_ind]*probs_one[ind_j]:
                if debug > 1:
                    print(f'Main: placing {i},{j} together cause p({i},{j}):{probs_two[i_ind][ind_j]} > p({i}):{probs_one[i_ind]}*p({j}):{probs_one[ind_j]}')
                C.append(j)
                movies_ids_n.remove(j)
        
        c_list.append(C)
    
    return c_list

def remove_if_below_10(movies_ids, users_movies):
    movies_copy = movies_ids.copy()
    for id in movies_copy:
        curr_sum = np.sum(users_movies[:,id])
        if curr_sum < 10:
            movies_ids.remove(id)
            print(f'Movie {id} ignored because it has only {curr_sum} ratings')

def read_movies_ids(path):
    f = open(path,"r")
    s = f.read()
    ids = list(int(id) for id in s.split('\n') if id)
    if debug > 0:
        print(f'Main: movie ids loaded from {path} - {ids}')
    return ids

# Calculate probabilities of p(m_j,m_t) for all (m_j,m_t) in M' 
def calc_probs2(movies_ids,N,k,ns,matrix):
    if debug > 0:
        print(f'Main: calculating probs2, #movies:{len(movies_ids)} N:{N}, k:{k}, ns:{ns.shape}, matrix:{matrix.shape}')
    m = len(movies_ids)
    probs = np.zeros((m,m))
    
    for t in range(m):
        probs[t] = list(calc_two(movies_ids[t],mj,N,k,ns,matrix) for mj in movies_ids)
    
    return probs
    
# Calculate probabilities of p(m_j) for all m_j in M' 
def calc_probs1(movies_ids,N,k,ns,matrix):
    if debug > 0:
        print(f'Main: calculating probs1, #movies:{len(movies_ids)} N:{N}, k:{k}, ns:{ns.shape}, matrix:{matrix.shape}')
    return list(calc_one(mj,N,k,ns,matrix) for mj in movies_ids)

# Calculate probabilitiy of p(m_j)
def calc_one(j,N,k,ns,matrix):
    prob = (1/(N+1))*(2/k + ((2/ns)@matrix[:,j]))
    if debug > 1:
        print(f'Main: calculating p(m_{j}): {prob}')
    return prob

# Calculate probabilitiy of p(m_j,m_t)
def calc_two(j,t,N,k,ns,matrix):
    nis = ns * (ns-1)
    vjvt = matrix[:,j] * matrix[:,t]
    prob = (1/(N+1))*(2/(k*(k-1)) + ((2/nis)@vjvt))

    if debug > 1:
        print(f'Main: calculating p(m_{j},m_{t}): {prob}')
    return prob


def print_clusters(clusters):
    movies = np.genfromtxt('ml-1m/movies.dat',
                     skip_header=0,
                     skip_footer=0,
                     names=None,
                     dtype=None,
                     delimiter='::',
                     encoding='latin-1',
                     usecols = (0,1))
    movies_ids = list(m[0] for m in  movies)
    for c in clusters:
        s = []
        for id in c:
            idx = movies_ids.index(id)
            s.append(str(id)+' "'+movies[idx][1]+'"')
        print(', '.join(s))
        print('###')

def cost(cluster,N,k,ns,matrix):
    cost = 0
    for c in cluster:
        l = len(c)
        if l == 1:
            cost += np.log(1/calc_one(c[0],N,k,ns,matrix))
        else:
            for c1 in c:
                for c2 in c:
                    if c1 != c2:
                        a = 1/(l-1)
                        a *= np.log(1/calc_two(c1,c2,N,k,ns,matrix))
                        cost += a

    return cost

def load_movies_indicators(path=''):
    age_in = pickle.load(open(path+'age_in','rb'))
    gender_in = pickle.load(open(path+'gender_in','rb'))
    occupation_in = pickle.load(open(path+'occupation_in','rb'))
    genre_in = pickle.load(open(path+'genre_in','rb'))

    return age_in,gender_in,occupation_in,genre_in