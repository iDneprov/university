
#include "SuffixTree.h"

SuffixTree::SuffixTree(std::string string):
    text(string), root(new Node(text.end(), text.end())), remainder(0)
{
    currentEdge = text.begin();
    currentNode = root;
    root->suffixLink = root;
    takeSuffLink = root;
    currentLength = 0;

    for (std::string::iterator suffix = text.begin(); suffix != text.end(); ++suffix)
        BildTree(suffix);
}

Node::Node(std::string::iterator begin, std::string::iterator end): 
    begin(begin), end(end), suffixLink(nullptr) {}

void SuffixTree::Destroy(Node *node)
{
    for (std::map<char, Node *>::iterator it = node->childMap.begin(); it != node->childMap.end(); ++it)
        Destroy(it->second);
    delete node;
}

SuffixTree::~SuffixTree()
{
    Destroy(root);
}


void SuffixTree::BildTree(std::string::iterator begin)
{
    takeSuffLink = root;
    remainder++;

    while (remainder) {
        if (!currentLength) currentEdge = begin;
        std::map<char, Node *>::iterator vertex = currentNode->childMap.find(*currentEdge);
        Node *next;
        if (vertex == currentNode->childMap.end()) {

            Node *leaf = new Node(begin, text.end());
            currentNode->childMap[*currentEdge] = leaf;
            SuffixLinkAdd(currentNode);
        } else {
            next = vertex->second;
            if (HaveDownhill(begin, next)) continue;
            if (*(next->begin + currentLength) == *begin) {
                currentLength++;
                SuffixLinkAdd(currentNode);
                break;
            }

            Node *split = new Node(next->begin, next->begin + currentLength);
            Node *leaf = new Node(begin, text.end());
            currentNode->childMap[*currentEdge] = split;
            split->childMap[*begin] = leaf;
            next ->begin += currentLength;
            split->childMap[*next->begin] = next;
            SuffixLinkAdd(split);
        }
        remainder--;
        if (currentNode == root && currentLength) {
            currentLength--;
            currentEdge = begin - remainder + 1;
        } else {
            currentNode = (currentNode->suffixLink) ? currentNode->suffixLink : root;
        }
    }
}

bool SuffixTree::HaveDownhill(std::string::iterator position, Node *node)
{
    int length;
    if (position + 1 < node->end)
        length = position + 1 - node->begin;
    else
        length = node->end - node->begin;
    if (currentLength >= length) {
        currentEdge += length;
        currentLength -= length;
        currentNode = node;
        return true;
    }

    return false;
}

void SuffixTree::SuffixLinkAdd(Node *node)
{
    if (takeSuffLink != root)
        takeSuffLink->suffixLink = node;
    takeSuffLink = node;
}

void SuffixTree::DeepFirstSearch(Node *node, std::vector<int> &result, int deep)
{
    if (node->childMap.empty()) {
        result.push_back(text.size() - deep);
        return;
    }
    for (std::map<char, Node *>::iterator it = node->childMap.begin(); it != node->childMap.end(); ++it) {
        int tmp = deep;
        tmp += it->second->end - it->second->begin;
        DeepFirstSearch(it->second, result, tmp);
    }
}

SuffixArray::SuffixArray(SuffixTree *tree):
    text(tree->text), array()
{
    tree->DeepFirstSearch(tree->root, array, 0);
}


int SuffixArray::FindLeft(const std::string &pattern)
{
    int left = 0;
    int right = text.size() - 1;
    int length = pattern.size();

    while (left <= right) {
        int mid = (left + right) / 2;
        std::string tmp = text.substr(array[mid], length);
        if (pattern > tmp)
            left = mid + 1;
        else
            right = mid - 1;
    }

    return left;
}

int SuffixArray::FindRight(const std::string &pattern)
{
    int left = 0;
    int right = text.size() - 1;
    int length = pattern.size();

    while (left <= right) {
        int mid = (left + right) / 2;
        std::string tmp = text.substr(array[mid], length);
        if (pattern >= tmp)
            left = mid + 1;
        else
            right = mid - 1;
    }
    
    return left;
}

std::vector<int> SuffixArray::Find(std::string &pattern)
{
    int left = FindLeft (pattern);
    int right = FindRight(pattern);

    std::vector<int> result;

    for (int i = left; i < right; ++i)
        result.push_back(array[i]);

    std::sort(result.begin(), result.end());
    return result;
}