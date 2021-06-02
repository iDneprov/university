#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wordabc.h"

void InitWordABC(WordABC* abc) {
    abc->quantity = 0;
    abc->tree = NULL;
}

bool GetWordIdxInABC(WordABC abc, char* word, ABCIdx* i) {
    
    TPatriciaNode node = TPNodeSearch(abc.tree, word);
    if (node == NULL) {
        *i = UNKNOWN_WORD_IDX;
        return false;
    } else {
        *i = node->value;
        return true;
    }
}

bool AddWordToABC(WordABC* abc, char* word, ABCIdx* i) {
    *i = abc->quantity;
    if (TPKeyInsert(&(abc->tree), word, i)) {
        abc->quantity++;
        return true;
    } else
        return false;
}
