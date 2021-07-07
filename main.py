import sys
import numpy as np
import pandas as pd
from utils import * 

debug = 0
def main():
    datasetfolder = sys.argv[1]
    algo = int(sys.argv[2])
    moviesubsetfilepath = sys.argv[3]
    
    users_movies = pd.read_csv('users_movies.csv', delimiter=',').to_numpy()
    if debug > 0:
        print(f'Main: "users_movies.csv" loaded')

    movies_ids = read_movies_ids(moviesubsetfilepath)
    remove_if_below_10(movies_ids, users_movies)

    N = users_movies.shape[0]
    k = users_movies.shape[1]
    ns = np.sum(users_movies,axis=1)

    probs1 = calc_probs1(movies_ids,N,k,ns,users_movies)
    probs2 = calc_probs2(movies_ids,N,k,ns,users_movies)
    
    # Original pivot
    if algo == 1:
        clusters = CCPivot(movies_ids, probs2, probs1)

    # Improved pivot
    if algo == 2:
        age_in, gender_in, ooccupation_in,genre_in = load_movies_indicators()
        clusters = CCPivot_improved(movies_ids, probs2, probs1, age_in, gender_in, ooccupation_in,genre_in)
        
    # Comparing mode
    if algo == 3:
        ITERS = 10

        org_cost = []
        improved_cost = []
        new_improved_cost = []
        age_in, gender_in, ooccupation_in,genre_in = load_movies_indicators()
        for _ in range(ITERS):
            org_cost.append(cost(CCPivot(movies_ids, probs2, probs1),N,k,ns,users_movies))
            improved_cost.append(cost(org_CCPivot_improved(movies_ids, probs2, probs1,age_in, gender_in, ooccupation_in,genre_in),N,k,ns,users_movies))   
            new_improved_cost.append(cost(CCPivot_improved(movies_ids, probs2, probs1,age_in, gender_in, ooccupation_in,genre_in),N,k,ns,users_movies))   
        org_cluster = CCPivot(movies_ids, probs2, probs1)
        improved_cluster = CCPivot_improved(movies_ids, probs2, probs1,age_in, gender_in, ooccupation_in, genre_in)

        avg_org = np.average(org_cost)
        avg_improved = np.average(improved_cost)
        avg_new_improved = np.average(new_improved_cost)

        print(f"Main: over {ITERS} iterations\nAverage cost original algo: {avg_org}\nAverage cost improved algo: {avg_improved}\nAverage cost new improved algo: {avg_new_improved}")
        print(f'Orginal clusters:\n{org_cluster}\nImproved cluster:\n{improved_cluster}')
    if algo < 3:
        print_clusters(clusters)
        if debug > 0:
            print(f'Main: cost: {cost(clusters,N,k,ns,users_movies)}')
        else:
            print(cost(clusters,N,k,ns,users_movies))

if __name__ == '__main__':
    main()
