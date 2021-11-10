import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import math


def precision_recall_ndcg_at_k(k, rankedlist, test_matrix):
    idcg_k = 0
    dcg_k = 0
    n_k = k if len(test_matrix) > k else len(test_matrix)
    for i in range(n_k):
        idcg_k += 1 / math.log(i + 2, 2)

    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)

    for c in range(count):
        dcg_k += 1 / math.log(hits[c][0] + 2, 2)

    return float(count / k), float(count / len(test_matrix)), float(dcg_k / idcg_k)


def map_mrr_ndcg(rankedlist, test_matrix):
    ap = 0
    map = 0
    dcg = 0
    idcg = 0
    mrr = 0
    for i in range(len(test_matrix)):
        idcg += 1 / math.log(i + 2, 2)

    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)

    for c in range(count):
        ap += (c + 1) / (hits[c][0] + 1)
        dcg += 1 / math.log(hits[c][0] + 2, 2)

    if count != 0:
        mrr = 1 / (hits[0][0] + 1)

    if count != 0:
        map = ap / count

    return map, mrr, float(dcg / idcg)


def evaluate(self):
    pred_ratings_10 = {}
    pred_ratings_5 = {}
    pred_ratings = {}
    ranked_list = {}
    p_at_5 = []
    p_at_10 = []
    r_at_5 = []
    r_at_10 = []
    map = []
    mrr = []
    ndcg = []
    ndcg_at_5 = []
    ndcg_at_10 = []
    for u in self.test_users:
        user_ids = []
        user_neg_items = self.neg_items[u]
        item_ids = []
        # scores = []
        for j in user_neg_items:
            item_ids.append(j)
            user_ids.append(u)

        scores = self.predict(user_ids, item_ids)
        # print(type(scores))
        # print(scores)
        # print(np.shape(scores))
        # print(ratings)
        neg_item_index = list(zip(item_ids, scores))

        ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
        pred_ratings[u] = [r[0] for r in ranked_list[u]]
        pred_ratings_5[u] = pred_ratings[u][:5]
        pred_ratings_10[u] = pred_ratings[u][:10]

        p_5, r_5, ndcg_5 = precision_recall_ndcg_at_k(5, pred_ratings_5[u], self.test_data[u])
        p_at_5.append(p_5)
        r_at_5.append(r_5)
        ndcg_at_5.append(ndcg_5)
        p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(10, pred_ratings_10[u], self.test_data[u])
        p_at_10.append(p_10)
        r_at_10.append(r_10)
        ndcg_at_10.append(ndcg_10)
        map_u, mrr_u, ndcg_u = map_mrr_ndcg(pred_ratings[u], self.test_data[u])
        map.append(map_u)
        mrr.append(mrr_u)
        ndcg.append(ndcg_u)

    print("------------------------")
    print("precision@10:" + str(np.mean(p_at_10)))
    print("recall@10:" + str(np.mean(r_at_10)))
    print("precision@5:" + str(np.mean(p_at_5)))
    print("recall@5:" + str(np.mean(r_at_5)))
    print("map:" + str(np.mean(map)))
    print("mrr:" + str(np.mean(mrr)))
    print("ndcg:" + str(np.mean(ndcg)))
    print("ndcg@5:" + str(np.mean(ndcg_at_5)))
    print("ndcg@10:" + str(np.mean(ndcg_at_10)))


def load_data_neg(path="movielens_100k.dat", header=['user_id', 'item_id', 'rating', 'category'],
                  test_size=0.2, sep="\t"):
    df = pd.read_csv(path, sep=sep, names=header, engine='python')

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row = []
    train_col = []
    train_rating = []

    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        train_row.append(u)
        train_col.append(i)
        train_rating.append(1)
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    # all_items = set(np.arange(n_items))
    # neg_items = {}
    # for u in range(n_users):
    #     neg_items[u] = list(all_items - set(train_matrix.getrow(u).nonzero()[1]))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(1)
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

    test_dict = {}
    for u in range(n_users):
        test_dict[u] = test_matrix.getrow(u).nonzero()[1]

    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_matrix.todok(), test_dict, n_users, n_items


# train_mat, test, users, items = load_data_all(path="1m_ratings.dat", sep="::")
train_mat, test, users, items = load_data_neg()

# print(test)
print(users)
print(items)
# print(train_mat)

import tensorflow as tf
import time
import random


class MLP():
    def __init__(self, sess, num_user, num_item, learning_rate=0.5, reg_rate=0.001, epoch=500, batch_size=256,
                 verbose=False, T=5, display_step=1000):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.verbose = verbose
        self.T = T
        self.display_step = display_step
        print("You are running MLP.")

    def build_network(self, num_factor_mlp=10, hidden_dimension=10, num_neg_sample=2):
        self.num_neg_sample = num_neg_sample
        self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')

        self.mlp_P = tf.Variable(tf.random_normal([self.num_user, num_factor_mlp]), dtype=tf.float32)
        self.mlp_Q = tf.Variable(tf.random_normal([self.num_item, num_factor_mlp]), dtype=tf.float32)

        mlp_user_latent_factor = tf.nn.embedding_lookup(self.mlp_P, self.user_id)
        mlp_item_latent_factor = tf.nn.embedding_lookup(self.mlp_Q, self.item_id)

        layer_1 = tf.layers.dense(inputs=tf.concat([mlp_item_latent_factor, mlp_user_latent_factor], axis=1),
                                  units=num_factor_mlp * 2, kernel_initializer=tf.random_normal_initializer,
                                  activation=tf.nn.relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
        layer_2 = tf.layers.dense(inputs=layer_1, units=hidden_dimension * 2, activation=tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
        MLP = tf.layers.dense(inputs=layer_2, units=hidden_dimension, activation=tf.nn.relu,
                              kernel_initializer=tf.random_normal_initializer,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))

        self.pred_y = tf.nn.sigmoid(tf.reduce_sum(MLP, axis=1))

        # self.pred_y = tf.layers.dense(inputs=MLP, units=1, activation=tf.sigmoid)

        self.loss = - tf.reduce_sum(
            self.y * tf.log(self.pred_y + 1e-10) + (1 - self.y) * tf.log(1 - self.pred_y + 1e-10)) \
                    + tf.losses.get_regularization_loss() + self.reg_rate * (
        tf.nn.l2_loss(self.mlp_P) + tf.nn.l2_loss(self.mlp_Q))

        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

        return self

    def prepare_data(self, train_data, test_data):

        t = train_data.tocoo()
        self.user = list(t.row.reshape(-1))
        self.item = list(t.col.reshape(-1))
        self.label = list(t.data)
        self.test_data = test_data

        self.neg_items = self._get_neg_items(train_data.tocsr())
        self.test_users = set([u for u in self.test_data.keys() if len(self.test_data[u]) > 0])

        print("data preparation finished.")
        return self

    def train(self):

        item_temp = self.item[:]
        user_temp = self.user[:]
        labels_temp = self.label[:]

        user_append = []
        item_append = []
        values_append = []
        for u in self.user:
            list_of_random_items = random.sample(self.neg_items[u], self.num_neg_sample)
            user_append += [u] * self.num_neg_sample
            item_append += list_of_random_items
            values_append += [0] * self.num_neg_sample

        item_temp += item_append
        user_temp += user_append
        labels_temp += values_append

        self.num_training = len(item_temp)
        self.total_batch = int(self.num_training / self.batch_size)
        # print(self.total_batch)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        user_random = list(np.array(user_temp)[idxs])
        item_random = list(np.array(item_temp)[idxs])
        labels_random = list(np.array(labels_temp)[idxs])

        # train
        for i in range(self.total_batch):
            start_time = time.time()
            batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_label = labels_random[i * self.batch_size:(i + 1) * self.batch_size]

            _, loss = self.sess.run((self.optimizer, self.loss),
                                    feed_dict={self.user_id: batch_user, self.item_id: batch_item, self.y: batch_label})

            if i % self.display_step == 0:
                if self.verbose:
                    print("Index: %04d; cost= %.9f" % (i + 1, np.mean(loss)))
                    print("one iteration: %s seconds." % (time.time() - start_time))

    def test(self):
        evaluate(self)

    def execute(self, train_data, test_data):

        self.prepare_data(train_data, test_data)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        for epoch in range(self.epochs):
            self.train()
            if (epoch) % self.T == 0:
                print("Epoch: %04d; " % (epoch), end='')
                self.test()

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        return self.sess.run([self.pred_y], feed_dict={self.user_id: user_id, self.item_id: item_id})[0]

    def _get_neg_items(self, data):
        all_items = set(np.arange(self.num_item))
        neg_items = {}
        for u in range(self.num_user):
            neg_items[u] = list(all_items - set(data.getrow(u).nonzero()[1]))

        return neg_items


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    model = MLP(sess, users, items)
    model.build_network()
    model.execute(train_mat, test)
