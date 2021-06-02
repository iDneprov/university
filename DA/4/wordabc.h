#ifndef wordabc_h
#define wordabc_h

#include <stdbool.h>
#include "common.h"
#include "TPatricia.h"

typedef struct {
    ABCIdx quantity;
    TPatriciaNode tree;
} WordABC;

void InitWordABC(WordABC* abc);
bool GetWordIdxInABC(WordABC abc, char* word, ABCIdx* i);
bool AddWordToABC(WordABC* abc, char* word, ABCIdx* i);

#endif
