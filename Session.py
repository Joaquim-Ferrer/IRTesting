class Session:
    #Session is unique for the pair session id and query id
    def __init__(self, session_id, query_id, results):
        self.session_id = session_id
        self.query_id = query_id
        self.results = results #List of document ids
        self.clicks_id = []
        self.clicks_rank = [False for i in range(len(results))]

    #The document_id maps to a result of self.results
    def add_click(self, document_id) -> str:
        self.clicks_id.append(document_id)
        try:
            self.clicks_rank[self.results.index(document_id)] = True
        except ValueError:
            return False
        return True