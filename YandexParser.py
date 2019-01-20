from Session import Session

def parseYandexLog(location):
    sessions = []
    with open(location, 'r') as fp:
        last_session_id = -1
        for line in fp:
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
                while sessions[i].session_id == last_session_id and document_id not in sessions[i].results:
                    i -= 1
                sessions[i].click(document_id)
    return sessions
