from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import torch
from scipy import stats

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except:
    pass

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from baco.param.space import Space
from baco.param.transformations import preprocess_parameters_array


class RFRegressionModel(RandomForestRegressor):
    """
    Implementation of our adapted RF model. We extend scikit-learns RF implementation to
    implement the adapted RF model proposed by Hutter et al.: https://arxiv.org/abs/1211.0906
    """

    def __init__(self, **kwargs):
        RandomForestRegressor.__init__(self, **kwargs)
        self.leaf_means = []
        self.leaf_variances = []

    def set_leaf_means(
            self,
            X_leaves: np.array,
            y: np.array,
    ):
        """
        Compute the mean value for each leaf in the forest.

        Input:
            - X_leaves: matrix with dimensions (number_of_samples, number_of_trees). Stores the leaf each sample fell into for each tree.
            - y: values of each sample used to build the forest.

        Returns:
            - list of number_of_trees dictionaries. Each dictionary contains the means for each leaf in a tree.
        """
        number_of_samples, number_of_trees = X_leaves.shape
        for tree_idx in range(number_of_trees):
            leaf_means = defaultdict(float)
            leaf_sample_count = defaultdict(int)
            for sample_idx in range(number_of_samples):
                leaf = X_leaves[sample_idx, tree_idx]
                leaf_sample_count[leaf] += 1
                leaf_means[leaf] += y[sample_idx]
            for leaf in leaf_sample_count.keys():
                leaf_means[leaf] = leaf_means[leaf] / leaf_sample_count[leaf]
            self.leaf_means.append(leaf_means)

    def set_leaf_variance(
            self,
            X_leaves: np.array,
            y: np.array,
    ):
        """
        Compute the variance for each leaf in the forest.

        Input:
            - X_leaves: matrix with dimensions (number_of_samples, number_of_trees). Stores the leaf each sample fell into for each tree.
            - y: values of each sample used to build the forest.

        Returns:
            - list of number_of_trees dictionaries. Each dictionary contains the variance for each leaf in a tree.
        """
        number_of_samples, number_of_trees = X_leaves.shape
        for tree_idx in range(number_of_trees):
            leaf_values = defaultdict(list)
            for sample_idx in range(number_of_samples):
                leaf = X_leaves[sample_idx, tree_idx]
                leaf_values[leaf].append(y[sample_idx])

            leaf_vars = {}
            for leaf in leaf_values.keys():
                if len(leaf_values[leaf]) > 1:
                    leaf_vars[leaf] = np.var(leaf_values[leaf], ddof=1)
                else:
                    leaf_vars[leaf] = 0
                # leaf_vars[leaf] = max(leaf_vars[leaf], 0.01) # This makes BaCO exploit too much. We will revisit this.
            self.leaf_variances.append(leaf_vars)

    @staticmethod
    def get_node_visits(tree: Any, x_leaves: np.ndarray):
        """
        Compute which samples passed through each node in a tree.

        Input:
            - tree: sklearn regression tree.
            - x_leaves: matrix with dimensions number_of_samples. Stores the leaf each sample fell into for each tree.

        Returns:
            - list of lists. Each internal list contains which samples went through the node represented by the index in the outer list.
        """
        node_count = tree.tree_.node_count
        node_visits = [[] for i in range(node_count)]
        for sample_idx in range(len(x_leaves)):
            leaf = x_leaves[sample_idx]
            node_visits[leaf].append(sample_idx)

        parents = [-1] * node_count
        left_children = tree.tree_.children_left
        right_children = tree.tree_.children_right
        for node_idx in range(node_count):
            if left_children[node_idx] != -1:
                parents[left_children[node_idx]] = node_idx
            if right_children[node_idx] != -1:
                parents[right_children[node_idx]] = node_idx

        for node_idx in range(node_count - 1, -1, -1):
            parent = parents[node_idx]
            if parent != -1:
                node_visits[parent] += node_visits[node_idx]
        return node_visits

    @staticmethod
    def get_node_bounds(samples: List[int], data_array: np.ndarray, threshold: float):
        """
        Compute the lower and upper bounds used to make a splitting decision at a tree node.

        Input:
            - samples: list containing the indices of all samples that went through the node.
            - data_array: list containing the values of one parameter for all the samples from the data.
            - threshold: original threshold used to split the node.

        Returns:
            - lower and upper bound that were used to compute the split.
        """
        lower_bound = float("-inf")
        upper_bound = float("inf")
        for sample in samples:
            sample_value = data_array[sample]
            if sample_value <= threshold:
                lower_bound = max(lower_bound, data_array[sample])
            else:
                upper_bound = min(upper_bound, data_array[sample])

        return lower_bound, upper_bound

    def get_configurations_leaves(self, X: np.ndarray):
        """
        Compute in which leaf each sample falls into for each tree.

        Input:
            - X: np array containing the samples.
        Returns:
            - array containing the corresponding leaf of each tree for each sample.
        """
        X_leaves = self.apply(X)
        return X_leaves

    def fit_rf(
            self,
            X: torch.Tensor,
            y: torch.Tensor,
    ):
        """
        Fit the adapted RF model.

        Input:
            - X: the training data for the RF model.
            - y: the training data labels for the RF model.
            - data_array: a dictionary containing previously explored points and their function values.
        """
        X, y = X.numpy(), y.numpy()
        self.fit(X, y)

        X_leaves = self.get_configurations_leaves(X)  # (n_samples, n_trees)
        self.set_leaf_means(X_leaves, y)
        self.set_leaf_variance(X_leaves, y)

        for tree_idx, tree in enumerate(self):
            node_visits = self.get_node_visits(tree, X_leaves[:, tree_idx])

            left_children = tree.tree_.children_left
            right_children = tree.tree_.children_right
            for node_idx in range(tree.tree_.node_count):
                if (
                        left_children[node_idx] == right_children[node_idx]
                ):  # If both children are equal, this is a leaf in the tree
                    continue
                # feature = parametrization_names[tree.tree_.feature[node_idx]]
                threshold = tree.tree_.threshold[node_idx]
                lower_bound, upper_bound = self.get_node_bounds(
                    node_visits[node_idx], X[:, tree.tree_.feature[node_idx]], threshold
                )
                new_split = stats.uniform.rvs(
                    loc=lower_bound, scale=upper_bound - lower_bound
                )
                tree.tree_.threshold[node_idx] = new_split

    def compute_rf_prediction(self, X_leaves: np.ndarray):
        """
        Compute the forest prediction for a list of samples based on the mean of the leaves in each tree.

        Input:
            - X_leaves: matrix with dimensions (number_of_samples, number_of_trees). Stores the leaf each sample fell into for each tree.
        Returns:
            - list containing the mean of each sample.
        """
        number_of_samples, number_of_trees = X_leaves.shape
        sample_means = np.zeros(number_of_samples)
        for tree_idx in range(number_of_trees):
            for sample_idx in range(number_of_samples):
                sample_means[sample_idx] += (self.leaf_means[tree_idx][X_leaves[sample_idx, tree_idx]] / number_of_trees)
        return sample_means

    def compute_rf_prediction_variance(self, X_leaves: np.ndarray, sample_means: np.ndarray):
        """
        Compute the forest prediction variance for a list of samples based on the mean and variances of the leaves in each tree.
        The variance is computed as proposed by Hutter et al. in https://arxiv.org/pdf/1211.0906.pdf.

        Input:
            - X_leaves: matrix with dimensions (number_of_samples, number_of_trees). Stores the leaf each sample fell into for each tree.
            - sample_means: list containing the mean of each sample.

        Returns:
            - list containing the variance of each sample.
        """
        number_of_samples, number_of_trees = X_leaves.shape
        mean_of_the_vars = np.zeros(number_of_samples)
        var_of_the_means = np.zeros(number_of_samples)
        sample_vars = np.zeros(number_of_samples)
        for sample_idx in range(number_of_samples):
            for tree_idx in range(number_of_trees):
                sample_leaf = X_leaves[sample_idx, tree_idx]
                mean_of_the_vars[sample_idx] += (self.leaf_variances[tree_idx][sample_leaf] / number_of_trees)
                var_of_the_means[sample_idx] += (self.leaf_means[tree_idx][sample_leaf] ** 2) / number_of_trees

            var_of_the_means[sample_idx] = abs(
                var_of_the_means[sample_idx] - sample_means[sample_idx] ** 2
            )
            sample_vars[sample_idx] = (
                    mean_of_the_vars[sample_idx] + var_of_the_means[sample_idx]
            )
            if sample_vars[sample_idx] == 0:
                sample_vars[sample_idx] = 0.00001

        return sample_vars

    def get_mean_and_std(self, X: torch.Tensor, _, use_var=False):
        """
        Compute the predicted mean and uncertainty (either standard deviation or variance) for a number of points with an RF model.

        Input:
            - X: torch tensor array containing points to predict.
            - _: argument for GPs
            - var: whether to compute variance or standard deviation.

        Returns:
            - the predicted mean and uncertainty for each point.
        """

        X_leaves = self.get_configurations_leaves(X.numpy())
        mean = self.compute_rf_prediction(X_leaves)
        var = self.compute_rf_prediction_variance(X_leaves, mean)
        if use_var:
            uncertainty = var
        else:
            uncertainty = np.sqrt(var)

        return torch.tensor(mean), torch.tensor(uncertainty)


class RFClassificationModel(RandomForestClassifier):

    def __init__(
            self,
            settings: Dict,
            param_space: Space,
            X: torch.Tensor,
            y: torch.Tensor,
    ):
        """
        Fit a Random Forest classification model.

        Input:
            - settings:
            - param_space: parameter space object for the current application.
            - X: input training data
            - y: boolean feasibility training values
        """
        self.param_space = param_space
        self.settings = settings
        self.X = X
        self.y = y
        n_estimators = 50
        max_features = 0.75

        class_weight = {True: 0.8, False: 0.2}
        RandomForestClassifier.__init__(
            self,
            class_weight=class_weight,
            n_estimators=n_estimators,
            max_features=max_features,
            bootstrap=False,
        )
        self.fit(self.X.numpy(), self.y.numpy())

    def feas_probability(self, data):
        """
        Compute the predictions of a model over a data array.

        Input:
            - data: data array with points to be predicted.

        Returns:
            - tensor with model predictions.
        """
        normalized_data, names = preprocess_parameters_array(data, self.param_space)
        return torch.tensor(self.predict_proba(normalized_data.numpy()))[:, list(self.classes_).index(True)]
