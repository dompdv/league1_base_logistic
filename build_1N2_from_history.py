import csv
import itertools
from sklearn.linear_model import LogisticRegression
import numpy as np

def write_matrices_to_file(matrices, to_file):
    if to_file is not '':
        with open(to_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=['Aa', 'Ad', 'Ba', 'Bd', 's1', 's2', 'p'])
            writer.writeheader()
            s = [ range(i) for i in matrices.shape]
            for (Aa, Ad, Ba, Bd, s1, s2) in itertools.product(*s):
                w_r = {'Aa': Aa, 'Ad': Ad, 'Ba': Ba, 'Bd': Bd, 's1': s1, 's2': s2, 'p':matrices[Aa, Ad, Ba, Bd, s1, s2]}
                writer.writerow(w_r)

# Load data from
def load_data(from_file, from_year, to_year):
    # Recupere les matches des dernières saisons
    matches = []
    with open(from_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for r in reader:
            season = int(r['Season'])
            if season not in range(from_year, to_year):
                continue
            row = {}
            for f in ['Date', 'HomeTeam', 'AwayTeam', 'Country', 'League']:
                row[f] = r[f].strip()
            skip = False
            for f in ['FTHG', 'FTAG', 'Season']:
                try:
                    row[f] = int(r[f])
                except:
                    skip = True
            if not skip:
                matches.append(row)

    # Trouve toutes les équipes
    teams = set(r['HomeTeam'] for r in matches) | set(r['AwayTeam'] for r in matches)
    # Liste complète des matches, chaque match apparaisant deux fois, une par équipe,
    # Determine le nombre de matches joués par équipe
    match_list = [r['HomeTeam'] for r in matches] + [r['AwayTeam'] for r in matches]
    teams_count = { t: match_list.count(t) for t in teams}
    total_match = len(match_list)
    # Plus petit et plus grand
    max_score = max(max(r['FTHG'], r['FTAG']) for r in matches)
    min_score = min(min(r['FTHG'], r['FTAG']) for r in matches)
    return matches, total_match, teams, teams_count, min_score, max_score

def split_teams_by_seasons_into_groups(matches, n_cat):

    # On crée le meilleur groupe à la main. De niveau n_cat-1
    # Les matchs joués par équipe
    played = {}
    for r in matches:
        season = r['Season']
        ht = "{}_{}".format(r['HomeTeam'], season)
        if ht not in played:
            played[ht] = [r]
        else:
            played[ht].append(r)
        at = "{}_{}".format(r['AwayTeam'], season)
        if at not in played:
            played[at] = [r]
        else:
            played[at].append(r)

    adjusted_teams = list(played.keys())
    # Ranking by team
    points_marked = {t:0 for t in adjusted_teams}
    for r in matches:
        season = r['Season']
        ht = "{}_{}".format(r['HomeTeam'], season)
        at = "{}_{}".format(r['AwayTeam'], season)
        if r['FTHG'] > r['FTAG']:
            points_marked[ht] += 3
        elif r['FTHG'] == r['FTAG']:
            points_marked[ht] += 1
            points_marked[at] += 1
        else:
            points_marked[at] += 3
    points_marked_ordered = sorted(points_marked.items(), key=lambda x:x[1], reverse=False)

    # groupes de même taille
    group_size = len(points_marked_ordered) / n_cat
    force_group = {t: int(i / group_size) for i,(t,_) in enumerate(points_marked_ordered)}

    goal_marked = {t:0 for t in adjusted_teams}
    goal_received = {t:0 for t in adjusted_teams}
    for r in matches:
        season = r['Season']
        ht = "{}_{}".format(r['HomeTeam'], season)
        at = "{}_{}".format(r['AwayTeam'], season)
        goal_marked[ht] += r['FTHG']
        goal_received[ht] += r['FTAG']
        goal_marked[at] += r['FTAG']
        goal_received[at] += r['FTHG']
    # Average using the number of matches playes by a team
    # Nombre de buts moyens données ou recus. Les équipes sont classées par ordre de force croissante
    for t in adjusted_teams:
        goal_marked[t] /= len(played[t])
        goal_received[t] /= len(played[t])

    goal_marked_ordered = sorted(goal_marked.items(), key=lambda x:x[1], reverse=False)
    goal_received_ordered = sorted(goal_received.items(), key=lambda x:x[1], reverse=True)
    attack_group = {t: int(i / group_size) for i,(t,_) in enumerate(goal_marked_ordered)}
    defense_group = {t: int(i / group_size) for i,(t,_) in enumerate(goal_received_ordered)}

    return force_group, attack_group, defense_group

def prepare_logistic_regression(from_year, to_year, n_cat, s_max, from_file):
    matches, total_match, teams, teams_count, min_score, max_score = load_data(from_file, from_year, to_year)

    ## Séparer les équipes en groupes de meilleurs attaquants et défenseur
    force_groups, attack_group, defense_group = split_teams_by_seasons_into_groups(matches, n_cat)

    s_stats = []
    for r in matches:
        season = r['Season']
        ht = "{}_{}".format(r['HomeTeam'], season)
        at = "{}_{}".format(r['AwayTeam'], season)
        Aa = attack_group[ht]
        Ad = defense_group[ht]
        Ba = attack_group[at]
        Bd = defense_group[at]
        s1 = r['FTHG']
        s2 = r['FTAG']
        if s1 >= s_max or s2 >= s_max:
            if s1 == s2:
                s1 = s2 = s_max
            elif s1 > s2:
                s1 = min(s_max, s1)
                s2 = min(s_max, s2)
                if s1 == s2:
                    s2 -= 1
            else:
                s1 = min(s_max, s1)
                s2 = min(s_max, s2)
                if s1 == s2:
                    s1 -= 1
        cat = s2 + s1 * (s_max + 1)
        s_stats.append([Aa, Ad, Ba, Bd, cat])

    data = np.array(s_stats)
    X = data[:, :4]
    y = data[:, 4]

    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)
    matrix = np.zeros((n_cat, n_cat,n_cat, n_cat, s_max + 1, s_max + 1))
    ra = range(n_cat)
    for Aa, Ad, Ba, Bd in itertools.product(ra, repeat=4):
        p = clf.predict_proba([[Aa, Ad, Ba, Bd]])
        for s1, s2 in itertools.product(list(range(s_max + 1)), repeat=2):
            i = s2 + s1 * (s_max + 1)
            matrix[Aa, Ad, Ba, Bd, s1, s2] = p[0][i]

    return matrix

if __name__ == "__main__":
    n_cat = 4
    stats = prepare_logistic_regression(
        1900, 2020,
        n_cat=n_cat,
        s_max=3,
        from_file='paris_sportifs_filtered.csv'
    )
    write_matrices_to_file(stats, 'out/matrices_cat{}.csv'.format(n_cat))
