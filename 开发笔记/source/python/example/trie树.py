class Node:
    def __init__(self, val, isWord=False):
        self.val = val
        self.children = []
        self.isWord = isWord


class Trie:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = Node(None)

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """

        def add_char(node, word, index):
            if index < len(word) - 1:
                for child in node.children:
                    if child.val == word[index]:
                        return add_char(child, word, index + 1)
                else:
                    node.children.append(Node(word[index]))
                return add_char(node.children[-1], word, index + 1)
            else:
                for child in node.children:
                    if child.val == word[index]:
                        child.isWord = True
                        break
                else:
                    node.children.append(Node(word[index], True))
            return

        add_char(self.root, word, 0)

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """

        def search_word(node, word, index):
            if index < len(word):
                for n in node.children:
                    if n.val == word[index]:
                        return search_word(n, word, index + 1)
                return False
            else:
                if node.val == word[-1] and node.isWord:
                    return True
                else:
                    return False

        return search_word(self.root, word, 0)

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """

        def search_prefix(node, word, index):
            if index < len(word):
                for n in node.children:
                    if n.val == word[index]:
                        return search_prefix(n, word, index + 1)
                return False
            return True

        return search_prefix(self.root, prefix, 0)


obj = Trie()
obj.insert('app')
obj.insert('appl')
obj.insert('apple')
obj.insert('applause')
param_3 = obj.search('apple')
# obj.insert('app')
pass
