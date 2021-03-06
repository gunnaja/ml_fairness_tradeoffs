from aif360.datasets import GermanDataset, CompasDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas, load_preproc_data_german
from sklearn.model_selection import train_test_split


def load_german_dataset():
    """
    Collect the aif360 preprocessed German Credit Data Set.
    Assigns 'age' as the protected attribute with age >= 25 considered privileged.
    Sex-related attributes are removed (the other option for privileged attribute)

    :return: The German Credit Data Set
    """
    dataset = GermanDataset(
        protected_attribute_names=['age'],
        privileged_classes=[lambda x: x >= 25],
        features_to_drop=['personal_status', 'sex']
    )
    dataset_orig_train, dataset_orig_test = dataset.split([0.7], shuffle=True)

    return dataset_orig_train, dataset_orig_test

def load_optimpreproc_german_dataset():
    """
    Collect the Optimized Preprocessed German Data Set.

    :return: The Optimized Preprocessed German Dataset, split into training and test sets
    """
    dataset = load_preproc_data_german()
    train, test = dataset.split([0.7], shuffle=True)
    return train, test

def load_compas_dataset():
    """
    Collect the aif360 preprocessed Compas Data Set.
    Charge descriptions are removed.

    :return: The Compas Dataset, split into training and test sets
    """
    dataset = CompasDataset(
        features_to_drop=['c_charge_desc']  # Drop charge description, as they unnecessarily overloads the dataset
    )
    ind = int(len(dataset.instance_names) * 0.8)
    train, test = dataset.split([ind])
    #train, test = dataset.split([0.8], shuffle=True) #fra gamle koden
    #train, test = train_test_split(dataset, test_size=0.2)
    return train, test


def load_optimpreproc_compas_dataset():
    """
    Collect the Optimized Preprocessed Compas Data Set.

    :return: The Optimized Preprocessed Compas Dataset, split into training and test sets
    """
    dataset = load_preproc_data_compas()
    ind = int(len(dataset.instance_names)*0.8)
    train, test = dataset.split([ind])
    return train, test


def to_dataframe(dataset, favorable_label=None, unfavorable_label=None):
    """
    Convert the ai360 data set into a Pandas Dataframe

    :param dataset: aif360 StructuredDataset type to convert into a dataframe
    :param favorable_label: Favorable label value to add to attributes, in case of binary label. Default: None
    :param unfavorable_label: Unfavorable label value to add to attributes, in case of binary label. Default: None
    :return: Tuple containing:
        - The converted dataframe
        - Dictionary of attributes with the following structure:
          attributes = {
            "feature_names": self.feature_names,
            "label_names": self.label_names,
            "protected_attribute_names": self.protected_attribute_names,
            "instance_names": self.instance_names,
            "instance_weights": self.instance_weights,
            "privileged_protected_attributes": self.privileged_protected_attributes,
            "unprivileged_protected_attributes": self.unprivileged_protected_attributes,
            "favorable_label": The favorable label,
            "unfavorable_label": The unfavorable label
          }
    """
    df, attributes = dataset.convert_to_dataframe()

    attributes["favorable_label"] = favorable_label
    attributes["unfavorable_label"] = unfavorable_label

    return df, attributes
