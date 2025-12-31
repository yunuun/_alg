import sys

objs = ["人", "狼", "羊", "菜"]
state = [0, 0, 0, 0]

visitedmap = {}
path = []

def neighbors(s):
    side = s[0]
    next = []
    checkadd(next, move(s, 0))
    for i in range(1, len(s)):
        if s[i] == side:
            checkadd(next, move(s, i))           
    return next

def checkadd(next, s):
    if not isdead(s):
        next.append(s)

def isdead(s):
    if s[1] == s[2] and s[1] != s[0]: return True 
    if s[2] == s[3] and s[2] != s[0]: return True 
    return False

def move(s, obj):
    news = s[:]  
    side = s[0]
    anotherside = 1 if side == 0 else 0
    news[0] = anotherside
    news[obj] = anotherside
    return news 

def visited(s):
    key = "".join(map(str, s))
    return key in visitedmap

def issuccess(s):
    for i in range(len(s)):
        if s[i] == 0: return False
    return True

def state2str(s):
    str = ""
    for i in range(len(s)):
        str += f"{objs[i]}{s[i]} "
    return str

def printpath(path):
    for p in path:
        print(state2str(p))

def dfs(s):
    if visited(s): return

    path.append(s)
    
    if issuccess(s):
        print("success!")
        printpath(path)
        return

    visitedmap["".join(map(str, s))] = True
    neighbors_list = neighbors(s)

    for next_state in neighbors_list:
        dfs(next_state)

    path.pop()

if __name__ == "__main__":
    dfs(state)