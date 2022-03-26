import pandas as pd
import numpy as np


class Rule:
    def __init__(self, class_label):
        self.conditions = []  # list of conditions
        self.class_label = class_label  # rule class

    def add_condition(self, condition):
        self.conditions.append(condition)

    def set_params(self, accuracy, coverage):
        self.accuracy = accuracy
        self.coverage = coverage



    def __repr__(self):
        return "If {} then {}. Coverage:{}, accuracy: {}".format(self.conditions, self.class_label,
                                                                 self.coverage, self.accuracy)


class Condition:
    def __init__(self, attribute, value, true_false = None):
        self.attribute = attribute
        self.value = value
        self.true_false = true_false

    def __repr__(self):
        if self.true_false is None:
            return "{}={}".format(self.attribute, self.value)
        else:
            if self.true_false:
                return "{}>={}".format(self.attribute, self.value)
            else:
                return "{}<{}".format(self.attribute, self.value)


def condition_filter(condition, dataset_name):
    if condition is None:
        return ""
    if condition.true_false is None:  # categorical attribute
        return '('+dataset_name+'["' + condition.attribute + '"]' + "==" + '"' + condition.value + '")'
    if condition.true_false: # >= numeric value
        return '('+dataset_name+'["' + condition.attribute + '"]' + ">=" + str(condition.value) + ")"
    return '('+dataset_name+'["' + condition.attribute + '"]' + "<" + str(condition.value) + ")"


def condition_list_filter(condition_list, dataset_name):
    result = ""

    for cond in condition_list:
        result += condition_filter(cond, dataset_name) + " & "

    result += "True"
    return result

# current_subset remains unchanged - we only perform reading operations of counting
def get_best_condition(columns, current_subset, prev_conditions, class_label, min_coverage=30, prev_best_accuracy=0.0):
    used_attributes = [x.attribute for x in prev_conditions]
    best_accuracy = prev_best_accuracy
    best_coverage = None
    best_col = None
    best_val = None
    best_true_false = None

    # we iterate over all attributes except the class - which is in the last column
    for col in columns[:-1]:
        # we do not use the same column in one rule
        if col in used_attributes:
            continue

        # Extract unique values from the column
        unique_vals = current_subset[col].unique().tolist()

        # Consider each unique value in turn
        # The treatment is different for numeric and categorical attributes
        for val in unique_vals:
            if isinstance(val, int) or isinstance(val, float):
                # Here we construct 2 conditions:
                # if actual value >= val or if actual value < val

                # First if actual value >= val
                # construct a new condition
                new_condition = Condition(col, val, True)

                # create a filtering condition
                filter = condition_filter(new_condition, "current_subset")

                # total covered by current condition
                total_covered = len(current_subset[eval(filter)])
                if total_covered >= min_coverage:
                    # total with this condition and a given class
                    total_correct = len(current_subset[(current_subset[columns[-1]] == class_label) & eval(filter)])

                    acc = total_correct/total_covered
                    if acc > best_accuracy or (acc == best_accuracy and
                                               (best_coverage is None or total_covered > best_coverage)):
                        best_accuracy = acc
                        best_coverage = total_covered
                        best_col = col
                        best_val = val
                        best_true_false = True

                # now repeat the same for the case - if actual value < val
                # construct new condition
                new_condition = Condition(col, val, False)

                # create a filtering condition
                filter = condition_filter(new_condition, "current_subset")

                # total covered by current condition
                total_covered = len(current_subset[eval(filter)])
                if total_covered >= min_coverage:

                    # total with this condition and a given class
                    total_correct = len(current_subset[(current_subset[columns[-1]] == class_label) & eval(filter)])

                    acc = total_correct / total_covered
                    if acc > best_accuracy or (acc == best_accuracy and
                                               (best_coverage is None or total_covered > best_coverage)):
                        best_accuracy = acc
                        best_coverage = total_covered
                        best_col = col
                        best_val = val
                        best_true_false = False

            else: # categorical attribute
                # For categorical attributes - this is just single condition if actual value == val
                new_condition = Condition(col, val)

                # create a filtering condition
                filter = condition_filter(new_condition, "current_subset")

                # total covered by current condition
                total_covered = len(current_subset[eval(filter)])

                if total_covered >= min_coverage:
                    # total with this condition and a given class
                    total_correct = len(current_subset[(current_subset[columns[-1]] == class_label) & eval(filter)])


                    acc = total_correct / total_covered
                    if acc > best_accuracy or (acc == best_accuracy and
                                               (best_coverage is None or total_covered > best_coverage)):
                        best_accuracy = acc
                        best_coverage = total_covered
                        best_col = col
                        best_val = val
                        best_true_false = None

    if best_col is None:
        return None
    return Condition(best_col,best_val, best_true_false)


def learn_one_rule(columns, current_data, class_label,
                   min_coverage=30):

    covered_subset = None
    # start with creating a new Rule with a single best condition
    current_rule = Rule(class_label)
    best_condition = get_best_condition(columns, current_data, [], class_label, min_coverage)

    if best_condition is None:
        return None

    current_rule.add_condition(best_condition)
    # create a filtering condition
    filter = condition_filter(best_condition, "current_data")

    # total covered by current condition
    total_covered = len(current_data[eval(filter)])

    # total with this condition and a given class
    total_correct = len(current_data[(current_data[columns[-1]] == class_label) & eval(filter)])

    current_accuracy = total_correct / total_covered
    current_rule.set_params(current_accuracy, total_covered )

    if total_covered < min_coverage:
        return None

    if current_accuracy == 1.0:
        return current_rule

    # leave only a subset where the best condition holds
    covered_subset = current_data[eval(filter)]

    # repeatedly try to improve Rule's accuracy as long as coverage remains sufficient
    while True:
        best_condition = get_best_condition(columns, covered_subset, current_rule.conditions,
                                            class_label, min_coverage, current_accuracy)

        if best_condition is None:
            return current_rule

        # create an additional filtering condition on the current subset
        filter = condition_filter(best_condition, "covered_subset")

        # total covered by current condition
        total_covered = len(covered_subset[eval(filter)])

        if total_covered < min_coverage:
            return current_rule  # we could not improve previous rule

        # total with this condition and a given class
        total_correct = len(covered_subset[(covered_subset[columns[-1]] == class_label) & eval(filter)])

        new_accuracy = total_correct / total_covered

        current_rule.add_condition(best_condition)
        current_rule.set_params(new_accuracy, total_covered)

        if new_accuracy == 1:
            return current_rule

        # update subset to continue working with
        covered_subset = covered_subset[eval(filter)]
        current_accuracy = new_accuracy


def learn_rules(columns, data, classes=None,
                min_coverage=30, min_accuracy=0.6):
    # List of final rules
    rules = []

    # If list of classes of interest is not provided - it is extracted from the last column of data
    if classes is not None:
        class_labels = classes
    else:
        class_labels = data[columns[-1]].unique().tolist()

    current_data = data.copy()

    # This follows the logic of the original PRISM algorithm
    # It processes each class in turn. Because for high accuracy
    # the rules generated are disjoint with respect to class label
    # this is not a problem when we are just interested in rules themselves - not classification
    # For classification the order in which the rules are discovered matters, and we should
    # process all classes at the same time, as shown in the lecture examples
    for class_label in class_labels:
        done = False
        while len(current_data) >= min_coverage and not done:
            # Learn a rule with a single condition
            rule = learn_one_rule(columns, current_data, class_label, min_coverage)

            # The best rule does not pass the coverage threshold - we are done with this class
            if rule is None:
                break

            # If we get the rule with coverage above threshold
            # We check if it passes accuracy threshold
            if rule.accuracy >= min_accuracy:
                rules.append(rule)

                # remove rows covered by this rule
                # we have to remove the rows where all of the conditions hold
                # create a filtering condition
                filter = condition_list_filter(rule.conditions,"current_data")

                current_data = current_data.drop(current_data[eval(filter)].index)

            else:
                done = True

    return rules

if __name__ == "__main__":
    data_file = "titanic.csv"
    data = pd.read_csv(data_file)

    # take a subset of attributes
    data = data[['Pclass', 'Sex', 'Age', 'Survived']]

    # drop all columns and rows with missing values
    data = data.dropna(how="any")
    print("Total rows", len(data))

    column_list = data.columns.to_numpy().tolist()
    print("Columns:", column_list)

    # we can set different accuracy thresholds
    # here we can reorder class labels - to first learn the rules with class label "survived".
    rules = learn_rules(column_list, data, [1, 0], 30, 0.6)

    from operator import attrgetter
    # sort rules by accuracy descending
    rules.sort(key=attrgetter('accuracy', 'coverage'), reverse=True)
    for rule in rules[:10]:
        print(rule)

    '''
    Total rows 714
Columns: ['Pclass', 'Sex', 'Age', 'Survived']
If [Sex=female, Pclass<2, Age>=26.0] then 1. Coverage:57, accuracy: 0.9824561403508771
If [Sex=male, Pclass>=2, Age>=32.5] then 0. Coverage:42, accuracy: 0.9761904761904762
If [Sex=female, Pclass<3, Age<24.0] then 1. Coverage:37, accuracy: 0.972972972972973
If [Sex=female, Pclass<3, Age>=28.0] then 1. Coverage:41, accuracy: 0.926829268292683
If [Age>=39.0, Pclass>=2, Sex=male] then 0. Coverage:48, accuracy: 0.9166666666666666
If [Age>=54.0, Sex=male, Pclass>=1] then 0. Coverage:37, accuracy: 0.8918918918918919
If [Age<24.0, Sex=male, Pclass>=2] then 0. Coverage:115, accuracy: 0.8782608695652174
If [Sex=male, Pclass>=2, Age<27.0] then 0. Coverage:41, accuracy: 0.8780487804878049
If [Age>=28.0, Pclass>=2, Sex=male] then 0. Coverage:62, accuracy: 0.8064516129032258
If [Age<6.0, Pclass>=2] then 1. Coverage:41, accuracy: 0.7073170731707317'''
