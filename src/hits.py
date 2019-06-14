import numpy as np
import scipy.sparse as sparse
import time
import pickle
from igraph import *
# from dataset_fetcher import ListToMatrixConverter
# from .dataset_fetcher import ListToMatrixConverter
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import psycopg2
import psycopg2.extras
import json
import time
from collections import namedtuple
debug = False


class HITS():
    """An instance of HITS is used to model the idea of hubs and authorities
    and execute the corresponding algorithm
    """

    def __init__(self, link_matrix, users, index_id_map, is_sparse=False):
        """
        Initializes an instance of HITS

        Args:
            link_matrix: The link matrix
            users: Details of all users
            index_id_map: Dictionary representing a map from link matrix index
            to user id
            is_sparse: True if the links matrix is a sparse matrix
        """
        self.__is_sparse = is_sparse
        self.__link_matrix = link_matrix
        self.__link_matrix_tr = link_matrix.transpose()
        self.__n = self.__link_matrix.shape[0]
        self.__hubs = np.ones(self.__n)
        self.__auths = np.ones(self.__n)
        self.__size = 30
        self.__names = [users[index_id_map[i]]['screen_name'] for i in range(0, self.__size)]
        self.__index_id_map = index_id_map
        self.__users = users
        self.all_hubs = []
        self.all_auths = []

    def calc_scores(self, epsilon=1e-4):
        """Calculates hubbiness and authority
        """
        epsilon_matrix = epsilon * np.ones(self.__n)
        if self.__is_sparse:
            while True:
                hubs_old = self.__hubs
                auths_old = self.__auths

                self.__auths = self.__link_matrix_tr * hubs_old
                max_score = self.__auths.max(axis=0)
                if max_score != 0:
                    self.__auths = self.__auths / max_score
                self.all_auths.append(self.__auths)

                self.__hubs = self.__link_matrix * self.__auths
                max_score = self.__hubs.max(axis=0)
                if max_score != 0:
                    self.__hubs = self.__hubs / max_score
                self.all_hubs.append(self.__hubs)

                if (((abs(self.__hubs - hubs_old)) < epsilon_matrix).all()) and (
                        ((abs(self.__auths - auths_old)) < epsilon_matrix).all()):
                    break

        else:
            while True:
                hubs_old = self.__hubs
                auths_old = self.__auths

                self.__auths = np.dot(self.__link_matrix_tr, hubs_old)
                max_score = self.__auths.max(axis=0)
                if max_score != 0:
                    self.__auths = self.__auths / max_score
                self.all_auths.append(self.__auths)

                self.__hubs = np.dot(self.__link_matrix, self.__auths)
                max_score = self.__hubs.max(axis=0)
                if max_score != 0:
                    self.__hubs = self.__hubs / max_score
                self.all_hubs.append(self.__hubs)

                if (((abs(self.__hubs - hubs_old)) < epsilon_matrix).all()) and (
                        ((abs(self.__auths - auths_old)) < epsilon_matrix).all()):
                    break

    def get_all_hubs(self):
        """Returns the hubbiness score for each user for each iteration
        """
        return self.all_hubs

    def get_all_auths(self):
        """Returns the authority score for each user for each iteration
        """
        return self.all_auths

    def get_hubs(self):
        """Returns the hubbiness for each node (user)
        """
        return self.__hubs

    def get_auths(self):
        """Returns the authority for each node (user)
        """
        return self.__auths

    def get_names(self):
        """Returns the screen name of each user
        """
        return self.__names

    def plot_graph(self, x, names, c):
        """Plots the graph
        """
        if self.__is_sparse:
            g = Graph.Adjacency((self.__link_matrix[0:self.__size, 0:self.__size]).toarray().tolist())
        else:
            g = Graph.Adjacency((self.__link_matrix[0:self.__size, 0:self.__size]).tolist())
        g.vs["name"] = names
        g.vs["attr"] = ["%.3f" % k for k in x]

        array_min = 0
        if x.min(axis=0) < 0.001:
            array_min = 0.001
        else:
            array_min = x.min(axis=0)

        ###layout###
        layout = g.layout("kk")
        visual_style = {}
        visual_style["vertex_size"] = [(x[i] / array_min) * 0.3 if x[i] >= 0.001 else 10 for i in
                                       range(0, min(self.__size, len(x)))]
        visual_style["vertex_label"] = [(g.vs["name"][i], float(g.vs["attr"][i])) for i in
                                        range(0, min(self.__size, len(x)))]
        color_dict = {"0": "red", "1": "yellow"}
        g.vs["color"] = color_dict[str(c)]
        visual_style["edge_arrow_size"] = 2
        visual_style["vertex_label_size"] = 35
        visual_style["layout"] = layout
        visual_style["bbox"] = (3200, 2200)
        visual_style["margin"] = 250
        visual_style["edge_width"] = 4
        plot(g, **visual_style)

    def plot_stats(self):
        screen_name_index_map = {}
        for key in self.__index_id_map:
            screen_name_index_map[self.__users[self.__index_id_map[key]]['screen_name']] = key

        cands = ['austinnotduncan', 'str_mape', 'LeoDiCaprio', 'aidanf123', 'MKBHD']
        colors = ['green', 'cyan', 'magenta', 'blue', 'brown']
        all_hubs = np.array(self.all_hubs)
        all_auths = np.array(self.all_auths)

        plt.figure(1, figsize=(12, 7))
        ax = plt.gca()
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Hubbiness Score")
        legend_handles = []
        for i in range(len(cands)):
            legend_handles.append(mp.Patch(label=cands[i], color=colors[i]))
            ax.plot(np.arange(1, all_hubs.shape[0] + 1), all_hubs[:, screen_name_index_map[cands[i]]], color=colors[i])
        ax.legend(handles=legend_handles)
        ax.set_title("Change in hubbiness score with increasing iterations")
        plt.show()

        plt.figure(2, figsize=(12, 7))
        ax = plt.gca()
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Authority Score")
        legend_handles = []
        for i in range(len(cands)):
            legend_handles.append(mp.Patch(label=cands[i], color=colors[i]))
            ax.plot(np.arange(1, all_auths.shape[0] + 1), all_auths[:, screen_name_index_map[cands[i]]],
                    color=colors[i])
        ax.legend(handles=legend_handles)
        ax.set_title("Change in authority score with increasing iterations")
        plt.show()


class DatasetReader():
    """An instance of DatasetReader is used to read different files from the
    dataset
    """

    def __init__(self):
        """Initializes an instance of DatasetReader
        """
        pass

    def read_users(self, users_path):
        """Returns the dictionary (stored in a file) containing details of
        all users

        Args:
            users_path: Path to the file where info of all users is stored
        """
        with open(users_path, mode='rb') as f:
            users = pickle.load(f)
        return users

    def read_map(self, map_path):
        """Returns the dictionary (stored in a file) that represents a map
        from the link matrix index to user id

        Args:
            map_path: Path to the file where the map is stored
        """
        with open(map_path, mode='rb') as f:
            index_id_map = pickle.load(f)
        return index_id_map

    def read_link_matrix(self, link_matrix_path, is_sparse=False):
        """Returns the array (stored in a file) that represents the link matrix

        Args:
            link_matrix_path: Path to the file where the link matrix is stored
            is_sparse: True if the link matrix is stored as a sparse matrix
        """
        with open(link_matrix_path, mode='rb') as f:
            if is_sparse:
                link_matrix = sparse.load_npz(link_matrix_path)
            else:
                link_matrix = np.load(f)
        return link_matrix


class DBConfig:
    hostname = 'localhost'
    username = 'postgres'
    password = '@dm1n'
    database = 'dbtwitter'


def create_record(obj, fields):
    ''' given obj from db returns named tuple with fields mapped to values '''
    Record = namedtuple("Record", fields)
    mappings = dict(zip(fields, obj))
    return Record(**mappings)


class Database:
    result = []
    conn = None

    def __init__(self):
        self.conn = psycopg2.connect(
            host=DBConfig.hostname,
            user=DBConfig.username,
            password=DBConfig.password,
            dbname=DBConfig.database)

    def execute(self, sql):
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(sql)

        headers = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        cur.close()
        for row in rows:
            self.result.append(create_record(row, headers))

    def close(self):
        self.conn.close()

    def write_to_file(self, filename, data):
        with open(filename, mode='wb') as f:
            pickle.dump(data, f)

    def read_file(self, filename):
        with open(filename, mode='rb') as f:
            data = pickle.load(f)
            print(data)

    def write_user(self, filename, data):
        data_key = {}
        for row in data:
            user_name = row.user_name
            screen_name = row.screen_name
            source_user_id_str = row.source_user_id_str
            data_key[source_user_id_str] = {
                "user_name": user_name,
                "screen_name": screen_name
            }
        self.write_to_file(filename, data_key)

    def write_adj_list(self, filename):
        data_key = {}
        self.execute(
            # --get in degree
            "select tweets_user_id_str as source_user_id_str, retweet_user_id_str as indegree, '' as outdegree from twitter_retweets " +
            "where hashtag_id > 230 " +
            "union " +
            "select in_reply_to_user_id_str as source_user_id_str, reply_to_user_id_str as indegree, '' as outdegree from twitter_replies " +
            "where hashtag_id > 230 " +
            "and in_reply_to_user_id_str <> '' " +
            "union " +
            "select mention_user_id_str as source_user_id_str, tweet_user_id_str as indegree, '' as outdegree from twitter_mentions " +
            "where hashtag_id > 230 " +
            # --get out degree
            "union " +
            "select retweet_user_id_str as source_user_id_str, '' as indegree, tweets_user_id_str as outdegree from twitter_retweets " +
            "where hashtag_id > 230 " +
            "union " +
            "select reply_to_user_id_str as source_user_id_str, '' as indegree, in_reply_to_user_id_str as outdegree from twitter_replies " +
            "where hashtag_id > 230 " +
            "and in_reply_to_user_id_str <> '' " +
            "union " +
            "select tweet_user_id_str as source_user_id_str, '' as indegree, mention_user_id_str as outdegree from twitter_mentions " +
            "where hashtag_id > 230 "
        )
        self.close()
        for row in self.result:
            in_degree = row.indegree
            out_degree = row.outdegree
            source_user_id_str = row.source_user_id_str
            if source_user_id_str not in data_key:
                data_key.update({source_user_id_str: {
                    "indegree": [],
                    "outdegree": []
                }})
            if in_degree:
                data_key[source_user_id_str]["indegree"].append(in_degree)
            if out_degree:
                data_key[source_user_id_str]["outdegree"].append(out_degree)
            self.write_to_file(filename, data_key)

    def map_user(self, source, destination):
        with open(source, mode='rb') as f:
            data = pickle.load(f)
            # index = 0
            res = {}
            for idx, key in enumerate(data):
                res[idx] = key
            self.write_to_file(destination, res)

    def convert_dict_to_json(self, file_path):
        with open(file_path, 'rb') as fpkl, open('%s.json' % file_path, 'w') as fjson:
            data = pickle.load(fpkl)
            json.dump(data, fjson, ensure_ascii=False, sort_keys=True, indent=4)

    def get_user(self, filename):
        self.execute(
            "select q.source_user_id_str, tu.user_name, tu.screen_name from ( " +
            "select source_user_id_str from twitter_indegree " +
            "union " +
            "select source_user_id_str from twitter_outdegree )q " +
            "left join twitter_users tu on q.source_user_id_str = tu.user_id_str " +
            "order by source_user_id_str"
        )
        self.close()
        self.write_user(filename, self.result)
        # data_user = {}
        # for row in self.result:
        #     user_name = row.user_name
        #     screen_name = row.screen_name
        #     source_user_id_str = row.source_user_id_str
        #     data_user[source_user_id_str] = {
        #         "user_name": user_name,
        #         "screen_name": screen_name
        #     }
        # self.write_to_file(filename, data_user)


def main():
    sparse = True
    epsilon = 1e-10
    show_iters = False
    # db = Database()
    users_path = "../data/result/user.pickle"
    map_path = "../data/result/map.pickle"
    adj_list_path = "../data/result/adj_list"

    # db.get_user(users_path)
    # db.map_user(users_path, map_path)
    # db.write_adj_list(adj_list_path)
    # users_path = '../data/users'
    # users_path = ''

    # map_path = '../data/map'
    # map_path = db.map_user(users_path)

    # users_path = '../data/result/user.pickle'
    # map_path = '../data/result/map.pickle'
    sparse_link_matrix_path = '../data/result/sparse_link_matrix'
    dense_link_matrix_path = '../data/result/dense_link_matrix'
    if sparse:
        link_matrix_path = sparse_link_matrix_path
    else:
        link_matrix_path = dense_link_matrix_path

    # Load the stored data into objects
    r = DatasetReader()
    users = r.read_users(users_path)
    index_id_map = r.read_map(map_path)
    link_matrix = r.read_link_matrix(link_matrix_path, is_sparse=sparse)

    # Run the algorithm
    h = HITS(link_matrix, users, index_id_map, is_sparse=sparse)
    h.calc_scores(epsilon=epsilon)

    if show_iters:
        x = h.get_all_hubs()
        for i in x:
            h.plot_graph(i, h.get_names(), 0)

        y = h.get_all_auths()
        for i in y:
            h.plot_graph(i, h.get_names(), 1)
    else:
        # h.plot_graph(h.get_hubs(), h.get_names(), 0)
        # h.plot_graph(h.get_auths(), h.get_names(), 1)
        fhub = open("../data/result/hubResult.txt", "w+", encoding='UTF-8')
        for i, k in enumerate(h.get_hubs()):
            print("{}. {} : {}".format(i, k, h.get_names()[i]))
            fhub.write("{}|{}\n".format(k, h.get_names()[i]))
        fhub.close

        # faut = open("../data/result/autResult.txt", "w+", encoding='UTF-8')
        # for i, k in enumerate(h.get_auths()):
        #     # print("{}. {} : {}".format(i, k, h.get_names()[i]))
        #     faut.write("{}|{}\n".format(k, h.get_names()[i]))
        # faut.close

    # Print graphs
    # h.plot_stats()


if __name__ == '__main__':
    main()
