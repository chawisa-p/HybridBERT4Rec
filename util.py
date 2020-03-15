from __future__ import print_function
import sys
import copy
import random
import numpy as np
from collections import defaultdict


def data_partition(fname, mode):
    if mode == "item-sequence":
        usernum = 0
        itemnum = 0
        User = defaultdict(list)
        user_train = {}
        user_valid = {}
        user_test = {}

        user_num = []
        item_num = []

        # assume user/item index starting from 1
        f = open(fname, 'r')
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

            user_num.append(u)
            item_num.append(i)

        user_num = list(dict.fromkeys(user_num))
        item_num = list(dict.fromkeys(item_num))

        for user in User:
            nfeedback = len(User[user])
            if nfeedback < 3:
                user_train[user] = User[user]
                user_valid[user] = []
                user_test[user] = []
            else:
                user_train[user] = User[user][:-2]
                user_valid[user] = []
                user_valid[user].append(User[user][-2])
                user_test[user] = []
                user_test[user].append(User[user][-1])
        return [user_train, user_valid, user_test, len(user_num), len(item_num)]

    if mode == "user-sequence":
        usernum = 0
        itemnum = 0
        Item = defaultdict(list)
        item_train = {}
        item_valid = {}
        item_test = {}

        user_num = []
        item_num = []

        # assume user/item index starting from 1
        f = open(fname, 'r')
        for line in f:
            i, u = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            Item[i].append(u)

            user_num.append(u)
            item_num.append(i)

        user_num = list(dict.fromkeys(user_num))
        item_num = list(dict.fromkeys(item_num))

        for item in Item:
            nfeedback = len(Item[item])
            if nfeedback < 3:
                item_train[item] = Item[item]
                item_valid[item] = []
                item_test[item] = []
            else:
                item_train[item] = Item[item][:-2]
                item_valid[item] = []
                item_valid[item].append(Item[item][-2])
                item_test[item] = []
                item_test[item].append(Item[item][-1])
        return [item_train, item_valid, item_test, len(user_num), len(item_num)]