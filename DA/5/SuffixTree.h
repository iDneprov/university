
#ifndef SUFFIXTREE_H
#define SUFFIXTREE_H

#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <algorithm>

class SuffixArray;

class Node
{
public:

    Node (std::string::iterator begin, std::string::iterator end);
    ~Node() {};

    std::map<char, Node *> childMap;
    std::string::iterator begin, end;
    Node *suffixLink;

};

class SuffixTree
{
public:

    SuffixTree(std::string str);
    ~SuffixTree();

    friend SuffixArray;

private:

    std::string text;

    Node *root;
    Node *takeSuffLink;
    Node *currentNode;

    int remainder;
    int currentLength;

    std::string::iterator currentEdge;

    void Destroy (Node *node);
    void SuffixLinkAdd (Node *node);
    void DeepFirstSearch (Node *node, std::vector<int> &result, int deep);
    void BildTree (std::string::iterator begin);
    bool HaveDownhill (std::string::iterator position, Node *node);
};

class SuffixArray
{
public:

    SuffixArray (SuffixTree *tree);
    ~SuffixArray () {};

    std::vector<int> Find (std::string &pattern);

private:

    int FindLeft (const std::string &pattern);
    int FindRight (const std::string &pattern);

    std::string text;
    std::vector<int> array;
};


#endif