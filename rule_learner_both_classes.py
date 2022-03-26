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

    def to_filter(self):
        result = ""
        for cond in self.conditions:
            result += cond.to_filter() + " & "
        result += "(current_data[columns[-1]] == class_label)"
        return result


    def to_filter_no_class(self):
        result = ""
        for cond in self.conditions:
            result += cond.to_filter() + " & "
        result += "True"
        return result

    def __repr__(self):
        return "If {} then {}. Coverage:{}, accuracy: {}".format(self.conditions, self.class_label,
                                                                 self.coverage, self.accuracy)


class Condition:
    def __init__(self, attribute, value, true_false = None):
        self.attribute = attribute
        self.value = value
        self.true_false = true_false


    def to_filter(self):
        result = ""
        if self is None:
            return result
        if self.true_false is None:
            result += '(current_data["' + self.attribute + '"]' + "==" + '"' + self.value + '")'
        elif self.true_false:
            result += '(current_data["' + self.attribute + '"]' + ">=" + str(self.value) + ")"
        else:
            result += '(current_data["' + self.attribute + '"]' + "<" + str(self.value) + ")"
        return result


    def __repr__(self):
        if self.true_false is None:
            return "{}={}".format(self.attribute, self.value)
        else:
            if self.true_false:
                return "{}>={}".format(self.attribute, self.value)
            else:
                return "{}<{}".format(self.attribute, self.value)

def filter_for_list(condition_list):
    result = ""

    for cond in condition_list:
        result += cond.to_filter() + " & "

    result += "True"
    return result


def get_best_condition(columns, current_data, prev_conditions, class_labels, min_coverage=30, prev_best_accuracy=0):
    used_attributes = [x.attribute for x in prev_conditions]
    best_accuracy = prev_best_accuracy
    best_coverage = None
    best_col = None
    best_val = None
    best_true_false = None
    best_class_label = None

    for class_label in class_labels:
        # we iterate over all attributes except the class - which is in the last column
        for col in columns[:-1]:
            # we do not use the same column in one rule
            if col in used_attributes:
                continue

            # Extract unique values from the column
            unique_vals = current_data[col].unique().tolist()

            # Consider each unique value in turn
            # The treatment is different for numeric and categorical attributes
            for val in unique_vals:
                if isinstance(val, int) or isinstance(val, float):
                    # Here we construct 2 conditions:
                    # if actual value >= val or if actual value < val

                    # First if actual value >= val
                    # construct new set of conditions by adding a new condition
                    new_conditions = prev_conditions.copy()
                    current_cond = Condition(col, val, True)
                    new_conditions.append(current_cond)

                    # create a filtering condition
                    filter = filter_for_list(new_conditions)

                    # total covered by current condition
                    total_covered = len(current_data[eval(filter)])
                    if total_covered >= min_coverage:
                        # total with this condition and a given class
                        total_correct = len(current_data[(current_data[columns[-1]] == class_label) & eval(filter)])

                        acc = total_correct/total_covered
                        if acc > best_accuracy or (acc == best_accuracy and
                                                   (best_coverage is None or total_covered > best_coverage)):
                            best_accuracy = acc
                            best_coverage = total_covered
                            best_col = col
                            best_val = val
                            best_true_false = True
                            best_class_label = class_label

                    # now repeat the same for the case - if actual value < val
                    # construct new set of conditions by adding a new condition
                    new_conditions = prev_conditions.copy()
                    current_cond = Condition(col, val, False)
                    new_conditions.append(current_cond)

                    # create a filtering condition
                    filter = filter_for_list(new_conditions)

                    # total covered by current condition
                    total_covered = len(current_data[eval(filter)])
                    if total_covered >= min_coverage:

                        # total with this condition and a given class
                        total_correct = len(current_data[(current_data[columns[-1]] == class_label) & eval(filter)])

                        acc = total_correct / total_covered
                        if acc > best_accuracy or (acc == best_accuracy and
                                                   (best_coverage is None or total_covered > best_coverage)):
                            best_accuracy = acc
                            best_coverage = total_covered
                            best_col = col
                            best_val = val
                            best_true_false = False
                            best_class_label = class_label

                else: # categorical attribute
                    # For categorical attributes - this is just single condition if actual value == val
                    new_conditions = prev_conditions.copy()
                    current_cond = Condition(col, val)
                    new_conditions.append(current_cond)

                    # create a filtering condition
                    filter = filter_for_list(new_conditions)

                    # total covered by current condition
                    total_covered = len(current_data[eval(filter)])

                    if total_covered >= min_coverage:
                        # total with this condition and a given class
                        total_correct = len(current_data[(current_data[columns[-1]] == class_label) & eval(filter)])


                        acc = total_correct / total_covered
                        if acc > best_accuracy or (acc == best_accuracy and
                                                   (best_coverage is None or total_covered > best_coverage)):
                            best_accuracy = acc
                            best_coverage = total_covered
                            best_col = col
                            best_val = val
                            best_true_false = None
                            best_class_label = class_label

    if best_col is None:
        return None
    return (best_class_label, Condition(best_col,best_val, best_true_false))


def learn_one_rule(columns, current_data, class_labels,
                   min_coverage=30):

    tuple = get_best_condition(columns, current_data, [], class_labels, min_coverage)

    if tuple is None:
        return None
    class_label, best_condition  = tuple



    # start with creating a new Rule with a single best condition
    current_rule = Rule(class_label)

    current_rule.add_condition(best_condition)
    # create a filtering condition
    filter = current_rule.to_filter_no_class()

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

    # repeatedly try to improve Rule's accuracy as long as coverage remains sufficient
    while True:
        tuple = get_best_condition(columns, current_data, current_rule.conditions,
                                            class_labels, min_coverage, current_accuracy)


        if tuple is None:
            return current_rule

        class_label, best_condition = tuple
        new_rule = Rule(class_label)
        for cond in current_rule.conditions:
            new_rule.add_condition(cond)

        new_rule.add_condition(best_condition)

        # create a filtering condition
        filter = new_rule.to_filter_no_class()

        # total covered by current condition
        total_covered = len(current_data[eval(filter)])

        if total_covered < min_coverage:
            return current_rule  # return previous rule

        # total with this condition and a given class
        total_correct = len(current_data[(current_data[columns[-1]] == class_label) & eval(filter)])

        new_accuracy = total_correct / total_covered

        new_rule.set_params(new_accuracy, total_covered)

        if new_accuracy == 1:
            return new_rule

        current_rule = new_rule

    return current_rule


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
    done = False
    while len(current_data) >= min_coverage and not done:
        # Learn a rule with a single condition
        rule = learn_one_rule(columns, current_data, class_labels, min_coverage)

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
            filter = rule.to_filter_no_class()

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
If [Pclass<2, Sex=female, Age>=26.0] then 1. Coverage:38, accuracy: 1.0
If [Age<25.0, Pclass<3, Sex=female] then 1. Coverage:48, accuracy: 0.9791666666666666
If [Sex=male, Pclass>=3, Age>=33.0] then 0. Coverage:59, accuracy: 0.9491525423728814
If [Sex=male, Pclass>=2, Age>=32.5] then 0. Coverage:31, accuracy: 0.9354838709677419
If [Sex=male, Age>=54.0, Pclass>=1] then 0. Coverage:37, accuracy: 0.8918918918918919
If [Sex=male, Pclass>=2, Age<29.0] then 0. Coverage:52, accuracy: 0.8653846153846154
If [Sex=male, Age<25.0, Pclass>=1] then 0. Coverage:33, accuracy: 0.8484848484848485
If [Sex=male, Pclass>=3, Age<25.0] then 0. Coverage:118, accuracy: 0.847457627118644
If [Age<6.0, Pclass>=1] then 1. Coverage:31, accuracy: 0.8387096774193549
If [Age>=48.0, Pclass<3] then 1. Coverage:39, accuracy: 0.8205128205128205'''