# -*- coding: utf-8 -*-
"""
BvhNode class

"""

__author__ = "Denis Tome"
__license__ = "Proprietary"
__copyright__ = "Copyright 1998-2020 Epic Games, Inc. All Rights Reserved."
__version__ = "0.1.0"
__author__ = "Denis Tome"
__email__ = "denis.tome@epicgames.com"
__status__ = "Development"

class BvhNode:
    """Class"""

    def __init__(self, value=None, parent=None):
        """Init"""

        if value:
            self.value = value
        else:
            self.value = []

        self.children = []
        self.parent = parent
        self.trsf = None
        self.local_trsf = None
        if self.parent:
            self.parent.add_child(self)

    def add_child(self, item) -> None:
        """Add child

        Arguments:
            item {BvhNode} -- child
        """

        item.parent = self
        self.children.append(item)

    def isLeafJoint(self) -> bool:
        """Check if node is a leaf node

        Returns:
            bool -- True if leaf node
        """

        children_joints = self.filter('JOINT')
        if any(True for _ in children_joints):
            return False

        return True

    def filter(self, key):
        """Filter children

        Arguments:
            key {str} -- value to search

        Yields:
            BvhNode -- node
        """

        for child in self.children:
            if child.value[0] == key:
                yield child

    def get_filter_val(self, key):
        """Filter children and get values

        Arguments:
            key {str} -- value to search

        Yields:
            BvhNode -- node
        """

        for child in self.children:
            if child.value[0] == key:
                if len(child.value) == 2:
                    yield child.value[1]
                else:
                    yield child.value[1:]

    def __iter__(self):
        for child in self.children:
            yield child

    def __getitem__(self, key):
        for child in self.children:
            for index, item in enumerate(child.value):
                if item == key:
                    if index + 1 >= len(child.value):
                        return None
                    else:
                        return child.value[index + 1:]
        raise IndexError('key {} not found'.format(key))

    def __repr__(self):
        return str(' '.join(self.value))

    @property
    def name(self):
        """Get name"""
        return self.value[1]
