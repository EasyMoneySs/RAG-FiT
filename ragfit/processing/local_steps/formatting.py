from ..step import LocalStep


class ColumnUpdater(LocalStep):
    """
    Simple class to create new columns from existing columns in a dataset.
    Existing columns are not modified.

    Args:
        keys_mapping (dict): Dictionary with "from:to" mapping.
    """

    def __init__(self, keys_mapping: dict, **kwargs):
        super().__init__(**kwargs)
        self.keys_mapping = keys_mapping

    def process_item(self, item, index, datasets, **kwargs):
        for from_key, to_key in self.keys_mapping.items():
            item[to_key] = item[from_key]
        return item


class FlattenList(LocalStep):
    """
    Class to join a list of strings into a single string.
    """

    def __init__(self, input_key, output_key, string_join=", ", **kwargs):
        """
        Args:
            input_key (str): Key to the list of strings.
            output_key (str): Key to store the joined string.
            string_join (str): String to join the list of strings. Defaults to ", ".
        """
        super().__init__(**kwargs)
        self.input_key = input_key
        self.output_key = output_key
        self.string_join = string_join

    def process_item(self, item, index, datasets, **kwargs):
        item[self.output_key] = self.string_join.join(item[self.input_key])
        return item


class UpdateField(LocalStep):
    """
    Class to update a field in the dataset with a new value.
    """

    def __init__(self, input_key: str, value, **kwargs):
        """
        Args:
            input_key (str): example key to change.
            value: New value to set for the field.
        """
        super().__init__(**kwargs)
        self.input_key = input_key
        self.value = value

    def process_item(self, item, index, datasets, **kwargs):
        item[self.input_key] = self.value
        return item


class CombineFields(LocalStep):
    """
    Class to combine multiple fields into a single string field.
    """

    def __init__(self, input_keys: list, output_key: str, separator=" ", **kwargs):
        """
        Args:
            input_keys (list): List of keys to combine.
            output_key (str): Key to store the combined string.
            separator (str): String to separate the values. Defaults to " ".
        """
        super().__init__(**kwargs)
        self.input_keys = input_keys
        self.output_key = output_key
        self.separator = separator

    def process_item(self, item, index, datasets, **kwargs):
        values = [str(item.get(k, "")) for k in self.input_keys]
        item[self.output_key] = self.separator.join(values)
        return item


class RemoveFields(LocalStep):
    """
    Class to remove fields from the dataset.
    """

    def __init__(self, keys_to_remove: list, **kwargs):
        """
        Args:
            keys_to_remove (list): List of keys to remove.
        """
        super().__init__(**kwargs)
        self.keys_to_remove = keys_to_remove

    def process_item(self, item, index, datasets, **kwargs):
        for key in self.keys_to_remove:
            if key in item:
                del item[key]
        return item
