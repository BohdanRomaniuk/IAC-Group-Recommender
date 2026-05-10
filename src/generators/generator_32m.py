"""
Processing Datasets
"""
import os
import shutil

import numpy as np
import scipy.sparse as sp
from collections import Counter


class DictMatrix:
    """Lightweight sparse matrix backed by a nested dict {row: {col: value}}.

    Provides the subset of the scipy dok_matrix interface used by this script:
      - mat[row, col]           → value (0 if missing)
      - mat[row, :]             → a _DictRow proxy
      - _DictRow.nonzero()      → (None, list_of_col_ids)  ← mirrors dok_matrix
    """

    def __init__(self, data):
        # data: dict-like {row_id: {col_id: value}}
        self._data = data

    def __getitem__(self, key):
        row, col = key
        if col == slice(None):          # mat[row, :]  — return row proxy
            return _DictRow(self._data.get(row, {}))
        return self._data.get(row, {}).get(col, 0)


class _DictRow:
    """Proxy for a single row of DictMatrix, mimicking the sparse row API."""

    def __init__(self, row_dict):
        self._row = row_dict            # {col_id: value}

    def nonzero(self):
        # dok_matrix.nonzero() returns (row_array, col_array); callers only
        # use the second element, so we return (None, list_of_keys).
        return None, list(self._row.keys())

    def __getitem__(self, col):
        return self._row.get(col, 0)


class GroupGenerator(object):
    """
    Group Data Generator
    """
    def __init__(self, data_path, output_path, rating_threshold, num_groups,
                 group_sizes, min_num_ratings, train_ratio, val_ratio,
                 negative_sample_size, verbose=False):
        self.rating_threshold = rating_threshold
        self.negative_sample_size = negative_sample_size

        # ── detect dataset format (ml-1m uses .dat, ml-32m uses .csv) ───────
        users_dat_path   = os.path.join(data_path, 'users.dat')
        items_dat_path   = os.path.join(data_path, 'movies.dat')
        ratings_dat_path = os.path.join(data_path, 'ratings.dat')
        items_csv_path   = os.path.join(data_path, 'movies.csv')
        ratings_csv_path = os.path.join(data_path, 'ratings.csv')

        is_csv = os.path.exists(ratings_csv_path) and not os.path.exists(ratings_dat_path)

        if is_csv:
            items   = self.load_items_csv(items_csv_path)
            users   = self.load_users_from_ratings(ratings_csv_path)
            rating_mat, timestamp_mat = self.load_ratings_csv(ratings_csv_path)
        else:
            users   = self.load_users_file(users_dat_path)
            items   = self.load_items_file(items_dat_path)
            rating_mat, timestamp_mat = self.load_ratings_file(
                ratings_dat_path, max(users), max(items))

        groups, group_ratings, groups_rated_items_dict, groups_rated_items_set = \
            self.generate_group_ratings(users, rating_mat, timestamp_mat,
                                        num_groups=num_groups,
                                        group_sizes=group_sizes,
                                        min_num_ratings=min_num_ratings)
        members, group_ratings_train, group_ratings_val, group_ratings_test, \
            group_negative_items_val, group_negative_items_test, \
            user_ratings_train, user_ratings_val, user_ratings_test, \
            user_negative_items_val, user_negative_items_test = \
            self.split_ratings(group_ratings, rating_mat, timestamp_mat,
                               groups, groups_rated_items_dict, groups_rated_items_set,
                               train_ratio=train_ratio, val_ratio=val_ratio)

        groups_path = os.path.join(output_path, 'groupMember.dat')
        group_ratings_train_path = os.path.join(output_path, 'groupRatingTrain.dat')
        group_ratings_val_path = os.path.join(output_path, 'groupRatingVal.dat')
        group_ratings_test_path = os.path.join(output_path, 'groupRatingTest.dat')
        group_negative_items_val_path = os.path.join(output_path, 'groupRatingValNegative.dat')
        group_negative_items_test_path = os.path.join(output_path, 'groupRatingTestNegative.dat')
        user_ratings_train_path = os.path.join(output_path, 'userRatingTrain.dat')
        user_ratings_val_path = os.path.join(output_path, 'userRatingVal.dat')
        user_ratings_test_path = os.path.join(output_path, 'userRatingTest.dat')
        user_negative_items_val_path = os.path.join(output_path, 'userRatingValNegative.dat')
        user_negative_items_test_path = os.path.join(output_path, 'userRatingTestNegative.dat')

        self.save_groups(groups_path, groups)
        self.save_ratings(group_ratings_train, group_ratings_train_path)
        self.save_ratings(group_ratings_val, group_ratings_val_path)
        self.save_ratings(group_ratings_test, group_ratings_test_path)
        self.save_negative_samples(group_negative_items_val, group_negative_items_val_path)
        self.save_negative_samples(group_negative_items_test, group_negative_items_test_path)
        self.save_ratings(user_ratings_train, user_ratings_train_path)
        self.save_ratings(user_ratings_val, user_ratings_val_path)
        self.save_ratings(user_ratings_test, user_ratings_test_path)
        self.save_negative_samples(user_negative_items_val, user_negative_items_val_path)
        self.save_negative_samples(user_negative_items_test, user_negative_items_test_path)

        if is_csv:
            # Generate movies.dat (movieId::title::genres) from movies.csv
            out_movies_dat = os.path.join(output_path, 'movies.dat')
            with open(items_csv_path, 'r', encoding='utf-8') as src, \
                 open(out_movies_dat, 'w', encoding='utf-8') as dst:
                next(src)  # skip header
                for line in src:
                    parts = line.rstrip('\n').split(',', 2)
                    dst.write('::'.join(parts) + '\n')

            # Generate users.dat (userId::placeholder) from the user list
            out_users_dat = os.path.join(output_path, 'users.dat')
            with open(out_users_dat, 'w') as dst:
                for uid in users:
                    dst.write('{}::M::25::1::00000\n'.format(uid))
        else:
            shutil.copyfile(src=os.path.join(data_path, 'movies.dat'),
                            dst=os.path.join(output_path, 'movies.dat'))
            shutil.copyfile(src=os.path.join(data_path, 'users.dat'),
                            dst=os.path.join(output_path, 'users.dat'))

        if verbose:
            num_group_ratings = len(group_ratings)
            num_user_ratings = len(user_ratings_train) + len(user_ratings_val) + len(user_ratings_test)
            num_rated_items = len(groups_rated_items_set)

            print('Save data: ' + output_path)
            print('# Users: ' + str(len(members)))
            print('# Items: ' + str(num_rated_items))
            print('# Groups: ' + str(len(groups)))
            print('# U-I ratings: ' + str(num_user_ratings))
            print('# G-I ratings: ' + str(num_group_ratings))
            print('Avg. # ratings / user: {:.2f}'.format(num_user_ratings / len(members)))
            print('Avg. # ratings / group: {:.2f}'.format(num_group_ratings / len(groups)))
            print('Avg. group size: {:.2f}'.format(np.mean(list(map(len, groups)))))

            # ── per-group-size statistics table ──────────────────────────────
            # Build a mapping: group_id (1-based) -> group tuple
            group_id_to_group = {i + 1: g for i, g in enumerate(groups)}

            from collections import defaultdict
            size_pos   = defaultdict(int)   # total positive group ratings
            size_neg   = defaultdict(int)   # total negative group ratings
            size_items = defaultdict(set)   # unique items rated by groups of this size

            for gid, item, label, _ in group_ratings:
                group = group_id_to_group[gid]
                sz = len(group)
                size_items[sz].add(item)
                if label == 1:
                    size_pos[sz] += 1
                else:
                    size_neg[sz] += 1

            all_sizes = sorted(set(len(g) for g in groups))

            col = [14, 10, 10, 10, 10, 12, 12, 16]
            headers = ['Group Size', '# Groups', '# Users', '# Items',
                       '# Ratings', '% Positive', '% Negative', 'Avg Ratings/Group']
            sep = '+' + '+'.join('-' * w for w in col) + '+'
            hdr = '|' + '|'.join(h.center(w) for h, w in zip(headers, col)) + '|'

            print()
            print('── Group-Size Distribution & Rating Statistics ──')
            print(sep)
            print(hdr)
            print(sep)

            for sz in all_sizes:
                n_groups  = sum(1 for g in groups if len(g) == sz)
                n_users   = n_groups * sz
                n_items   = len(size_items[sz])
                n_pos     = size_pos[sz]
                n_neg     = size_neg[sz]
                n_total   = n_pos + n_neg
                pct_pos   = 100.0 * n_pos / n_total if n_total else 0.0
                pct_neg   = 100.0 * n_neg / n_total if n_total else 0.0
                avg_per_g = n_total / n_groups if n_groups else 0.0
                row_vals  = [str(sz), str(n_groups), str(n_users), str(n_items),
                             str(n_total),
                             '{:.1f}%'.format(pct_pos),
                             '{:.1f}%'.format(pct_neg),
                             '{:.2f}'.format(avg_per_g)]
                print('|' + '|'.join(v.center(w) for v, w in zip(row_vals, col)) + '|')

            print(sep)

            tot_groups  = len(groups)
            tot_users   = sum(
                sum(1 for g in groups if len(g) == sz) * sz for sz in all_sizes)
            tot_items   = num_rated_items
            tot_pos     = sum(size_pos[s] for s in all_sizes)
            tot_neg     = sum(size_neg[s] for s in all_sizes)
            tot_total   = tot_pos + tot_neg
            tot_pct_pos = 100.0 * tot_pos / tot_total if tot_total else 0.0
            tot_pct_neg = 100.0 * tot_neg / tot_total if tot_total else 0.0
            tot_avg     = tot_total / tot_groups if tot_groups else 0.0
            tot_vals    = ['TOTAL', str(tot_groups), str(tot_users), str(tot_items),
                           str(tot_total),
                           '{:.1f}%'.format(tot_pct_pos),
                           '{:.1f}%'.format(tot_pct_neg),
                           '{:.2f}'.format(tot_avg)]
            print('|' + '|'.join(v.center(w) for v, w in zip(tot_vals, col)) + '|')
            print(sep)
            print()

    def load_users_file(self, users_path):
        """Not used for ml-32m — users are derived from ratings."""
        users = []
        with open(users_path, 'r') as file:
            for line in file.readlines():
                users.append(int(line.split('::')[0]))
        return users

    def load_users_from_ratings(self, ratings_path):
        """Derive the sorted list of unique user IDs directly from ratings.csv."""
        user_ids = set()
        with open(ratings_path, 'r') as file:
            next(file)  # skip header
            for line in file:
                user_ids.add(int(line.split(',', 1)[0]))
        return sorted(user_ids)

    def load_items_file(self, items_path):
        items = []

        with open(items_path, 'r', encoding='iso-8859-1') as file:
            for line in file.readlines():
                items.append(int(line.split('::')[0]))

        return items

    def load_items_csv(self, items_path):
        """Load item IDs from a CSV file with header: movieId,title,genres"""
        items = []
        with open(items_path, 'r', encoding='utf-8') as file:
            next(file)  # skip header
            for line in file:
                items.append(int(line.split(',', 1)[0]))
        return items

    def load_ratings_file(self, ratings_path, max_num_users, max_num_items):
        rating_mat = sp.dok_matrix((max_num_users + 1, max_num_items + 1),
                                   dtype=int)
        timestamp_mat = rating_mat.copy()

        with open(ratings_path, 'r') as file:
            for line in file.readlines():
                arr = line.replace('\n', '').split('::')
                user, item, rating, timestamp = \
                    int(arr[0]), int(arr[1]), int(arr[2]), int(arr[3])
                rating_mat[user, item] = rating
                timestamp_mat[user, item] = timestamp

        return rating_mat, timestamp_mat

    def load_ratings_csv(self, ratings_path):
        """Load ratings from a CSV file with header: userId,movieId,rating,timestamp.
        Returns a pair of DictMatrix objects (rating, timestamp) backed by
        nested dicts for O(1) row access — far faster than dok_matrix at scale.
        Ratings are floats (e.g. 4.0) and are rounded to the nearest integer."""
        from collections import defaultdict
        rating_data    = defaultdict(dict)   # {user: {item: rating}}
        timestamp_data = defaultdict(dict)   # {user: {item: timestamp}}

        with open(ratings_path, 'r') as file:
            next(file)  # skip header
            for line in file:
                arr = line.split(',')
                user      = int(arr[0])
                item      = int(arr[1])
                rating    = round(float(arr[2]))
                timestamp = int(arr[3])
                rating_data[user][item]    = rating
                timestamp_data[user][item] = timestamp

        return DictMatrix(rating_data), DictMatrix(timestamp_data)

    def generate_group_ratings(self, users, rating_mat, timestamp_mat,
                               num_groups, group_sizes, min_num_ratings):
        np.random.seed(0)

        # Sort users by most recent rating (descending) so groups are formed
        # from recently active users rather than the oldest activity.
        if hasattr(timestamp_mat, '_data'):
            user_max_ts = {u: (max(d.values()) if d else 0)
                           for u, d in timestamp_mat._data.items()}
        else:
            user_max_ts = Counter()
            for (u, _i), ts in timestamp_mat.items():
                if ts > user_max_ts[u]:
                    user_max_ts[u] = ts
        users = sorted(users, key=lambda u: user_max_ts.get(u, 0), reverse=True)

        groups = set()
        groups_ratings = []
        groups_rated_items_dict = {}
        groups_rated_items_set = set()

        while len(groups) < num_groups:
            group_id = len(groups) + 1

            while True:
                group = tuple(np.sort(
                    np.random.choice(users, np.random.choice(group_sizes),
                                     replace=False)))
                if group not in groups:
                    break

            pos_group_rating_counter = Counter()
            neg_group_rating_counter = Counter()
            group_rating_list = []
            group_rated_items = set()

            for member in group:
                _, items = rating_mat[member, :].nonzero()
                pos_items = [item for item in items
                             if rating_mat[member, item] >= self.rating_threshold]
                neg_items = [item for item in items
                             if rating_mat[member, item] < self.rating_threshold]
                pos_group_rating_counter.update(pos_items)
                neg_group_rating_counter.update(neg_items)

            for item, num_ratings in pos_group_rating_counter.items():
                if num_ratings == len(group):
                    timestamp = max([timestamp_mat[member, item]
                                     for member in group])
                    group_rated_items.add(item)
                    group_rating_list.append((group_id, item, 1, timestamp))

            for item, num_ratings in neg_group_rating_counter.items():
                if (num_ratings == len(group)) \
                        or (num_ratings + pos_group_rating_counter[item] == len(group)):
                    timestamp = max([timestamp_mat[member, item]
                                     for member in group])
                    group_rated_items.add(item)
                    group_rating_list.append((group_id, item, 0, timestamp))

            if len(group_rating_list) >= min_num_ratings:
                groups.add(group)
                groups_rated_items_dict[group_id] = group_rated_items
                groups_rated_items_set.update(group_rated_items)
                for group_rating in group_rating_list:
                    groups_ratings.append(group_rating)

        return list(groups), groups_ratings, groups_rated_items_dict, groups_rated_items_set

    def split_ratings(self, group_ratings, rating_mat, timestamp_mat,
                      groups, groups_rated_items_dict, groups_rated_items_set, train_ratio, val_ratio):
        num_group_ratings = len(group_ratings)
        num_train = int(num_group_ratings * train_ratio)
        num_test = int(num_group_ratings * (1 - train_ratio - val_ratio))

        group_ratings = \
            sorted(group_ratings, key=lambda group_rating: group_rating[-1])
        group_ratings_train = group_ratings[:num_train]
        group_ratings_val = group_ratings[num_train:-num_test]
        group_ratings_test = group_ratings[-num_test:]

        timestamp_split_train = group_ratings_train[-1][-1]
        timestamp_split_val = group_ratings_val[-1][-1]

        user_ratings_train = []
        user_ratings_val = []
        user_ratings_test = []

        members = set()
        users_rated_items_dict = {}

        for group in groups:
            for member in group:
                if member in members:
                    continue
                members.add(member)
                user_rated_items = set()
                _, items = rating_mat[member, :].nonzero()
                for item in items:
                    if item not in groups_rated_items_set:
                        continue
                    user_rated_items.add(item)
                    if rating_mat[member, item] >= self.rating_threshold:
                        rating_tuple = (member, item, 1,
                                        timestamp_mat[member, item])
                    else:
                        rating_tuple = (member, item, 0,
                                        timestamp_mat[member, item])
                    if timestamp_mat[member, item] <= timestamp_split_train:
                        user_ratings_train.append(rating_tuple)
                    elif timestamp_split_train < timestamp_mat[member, item] <= timestamp_split_val:
                        user_ratings_val.append(rating_tuple)
                    else:
                        user_ratings_test.append(rating_tuple)

                users_rated_items_dict[member] = user_rated_items

        np.random.seed(0)

        user_negative_items_val = self.get_negative_samples(
            user_ratings_val, groups_rated_items_set, users_rated_items_dict)
        user_negative_items_test = self.get_negative_samples(
            user_ratings_test, groups_rated_items_set, users_rated_items_dict)
        group_negative_items_val = self.get_negative_samples(
            group_ratings_val, groups_rated_items_set, groups_rated_items_dict)
        group_negative_items_test = self.get_negative_samples(
            group_ratings_test, groups_rated_items_set, groups_rated_items_dict)

        return members, group_ratings_train, group_ratings_val, group_ratings_test, \
            group_negative_items_val, group_negative_items_test, \
            user_ratings_train, user_ratings_val, user_ratings_test, \
            user_negative_items_val, user_negative_items_test

    def get_negative_samples(self, ratings, groups_rated_items_set, rated_items_dict):
        negative_items_list = []
        for sample in ratings:
            sample_id, item, _, _ = sample
            missed_items = groups_rated_items_set - rated_items_dict[sample_id]
            negative_items = \
                np.random.choice(list(missed_items), self.negative_sample_size,
                                 replace=(len(missed_items) < self.negative_sample_size))
            negative_items_list.append((sample_id, item, negative_items))
        return negative_items_list

    def save_groups(self, groups_path, groups):
        with open(groups_path, 'w') as file:
            for i, group in enumerate(groups):
                file.write(str(i + 1) + ' '
                           + ','.join(map(str, list(group))) + '\n')

    def save_ratings(self, ratings, ratings_path):
        with open(ratings_path, 'w') as file:
            for rating in ratings:
                file.write(' '.join(map(str, list(rating))) + '\n')

    def save_negative_samples(self, negative_items, negative_items_path):
        with open(negative_items_path, 'w') as file:
            for samples in negative_items:
                user, item, negative_items = samples
                file.write('({},{}) '.format(user, item)
                           + ' '.join(map(str, list(negative_items))) + '\n')


if __name__ == '__main__':
    data_folder_path = os.path.join('./', 'data')
    data_path = os.path.join(data_folder_path, 'ml-32m')
    output_path = os.path.join(data_folder_path, 'MovieLens-32m')

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    group_generator = GroupGenerator(data_path, output_path,
                                     rating_threshold=4,
                                     num_groups=1000,
                                     group_sizes=[2, 3, 4],
                                     min_num_ratings=10,
                                     train_ratio=0.7,
                                     val_ratio=0.1,
                                     negative_sample_size=100,
                                     verbose=True)
