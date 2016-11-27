
def parse_problems(filename = "../data/robotanik/tasks-desription.txt"):
    f = open(filename)
    s = f.read().split("id:") #each problem definition begins with id:
    f.close()
    s = list(map(lambda r : r.splitlines(), s)) #split lines in each definition
    s = s[1:] #ignore initial text (before first id)
    for i in range(len(s)):
        s[i][0] = "id:"+s[i][0].split(';')[0][1:] #add id: back and remove problem name from the end of the line
        for j in range(len(s[i])):
            s[i][j] = s[i][j].rstrip(", ").replace('"','').split(":") #ignore trailing characters and quotes, split the property name and property value
        s[i] = filter(lambda x: len(x)>1, s[i]) #ignore empty lines

    problems = {}
    for p in map(lambda x: dict(x), s):
        id = p['id']
        problems[id] = {}
        for n in ["robotRow","robotCol","robotDir","allowedCommands"]:
            problems[id][n] = int(p[n])
        problems[id]["subs"] = list(map(int, p["subs"].strip("[]").split(',')))
        problems[id]["board"] = p["board"]
    return problems

def process_roboprogram(command_string):
    if "undefined" in command_string: return None
    s = command_string.split("|")
    toTuple = lambda r: [(r[i],r[i+1]) for i in range(0,len(r),2)] #change commands to tuples
    return list(map(toTuple,s))

def get_attempts(filename, count = 100):
    with open(filename) as f:
        # skip first two rows
        f.readline()
        f.readline()
        output = []
        for l in f.readlines():
            if ";" in l:
                tmp = l.split(';')[2]
                if "solution:" in tmp:
                    tmp = tmp[9:]
                c = process_roboprogram(tmp)
                if c != None:
                    output.append(c)
            if len(output) >= count: break
        return output
