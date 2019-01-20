from Session import Session

def parseYandexLog(location):
    f = open(location, "r")
    sessions = []

    last_session_id = -1
    for line in f:
        line = line.split()
        if line[2] == "Q":
            session_id = line[0]
            last_session_id = session_id
            query_id = line[3]
            results = line[5:]
            sessions.append(Session(session_id, query_id, results))
        if line[2] == "C":
            document_id = line[3]
            i = -1
            while sessions[i].session_id == last_session_id and not sessions[i].add_click(document_id):
                i -= 1
    return sessions