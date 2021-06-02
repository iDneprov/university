
#include "SuffixTree.h"
#include <string>
#include <iostream>

int main(void)
{
    std::string text, pattern;
    std::cin >> text;

    SuffixTree tree (text + "$");
    SuffixArray array(&tree);

    for (int test = 1; std::cin >> pattern; test++) {
        std::vector<int> result = array.Find(pattern);
        if (!result.empty()) {
            std::cout << test << ": ";
            for (int i = 0; i < result.size(); ++i) {
                std::cout << result[i] + 1;
                if (i < result.size() -  1)
                    std::cout << ", ";
            }
            std::cout << std::endl;
        }
    }

    return 0;
}