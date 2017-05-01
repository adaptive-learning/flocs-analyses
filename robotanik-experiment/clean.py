import csv
from robotanik_read import load_problem, parse_roboprogram
from similarity import compute_visited_path

problem = load_problem(problem_id='639')
cleaned_data = []
user = None
game = None
with open('../data/robotanik/task639.txt') as f:
    # skip first two rows
    f.readline()
    f.readline()
    for l in f.readlines():
        if l.startswith('User'):
            user = int(l[5:])
        if l.startswith('Game'):
            game = int(l[5:])
        if ";" in l:
            [step, time, other] = l.split(';')
            step = int(step) + 1
            time = int(time)
            solution = 'solution:' in other
            code = (other[9:] if solution else other).strip()
            cleaned_data.append({
                'user': user,
                'game': game,
                'step': step,
                'time': time,
                'correct': solution,
                'code': code,
                'trace': compute_visited_path(problem, parse_roboprogram(code))
            })

with open('../data/robotanik/task639.csv', 'w') as outfile:
    fieldnames = ['user', 'game', 'step', 'time', 'correct', 'code', 'trace']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in cleaned_data:
        writer.writerow(row)
