from opensearchpy import OpenSearch
from dotenv import dotenv_values


class OpenSearchRetriever():
    # constructor
    def __init__(self, path_to_env = ".env"):
        self.ENV = dotenv_values(path_to_env)
        self.host = self.ENV['OPENSEARCH_URL']
        self.port = self.ENV['OPENSEARCH_PORT']
        self.client = OpenSearch(
            hosts=[{'host': self.host, 'port': self.port}],
            http_compress = True, # enables gzip compression for request bodies
            http_auth = (self.ENV['OPENSEARCH_USER'],  self.ENV['OPENSEARCH_PW']),
            use_ssl = True,
            verify_certs = True,
            ssl_assert_hostname = False,
            ssl_show_warn = False
        )
    
    def get_n_documents_from_index_with_scroll_id(self, idx, n=100, s_id=None):
        """
        Params:
            self: OpenSearchRetriever object
            idx: index to query from
            n: number of documents to fetch
            s_id: scroll id (leave None if you don't have one yet)

        Returns:
            tuple (documents, scroll_id)
        """
        os = self.client
        # check if scroll id is given
        if s_id == None:
            # if not search
            data = os.search(index=idx, size=n, scroll='2m', body={"query":{"match_all":{}}})
        else:
            # else just fetch from scroll
            data = os.scroll(scroll_id=s_id, scroll='2m')

        # get scroll id
        s_id = data['_scroll_id']
        # get results
        results = data['hits']['hits']

        return (results, s_id)

