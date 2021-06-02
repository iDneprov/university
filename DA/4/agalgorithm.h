#ifndef agalgorithm_h
#define agalgorithm_h

#include <stdio.h>
#include "wordabc.h"

// Размер буфера текста (измеряется кратно длинам Pattern)
#define TEXT_BUFFER_SIZE_BY_PATTERN 10

typedef long textIdx;

typedef struct {
    textIdx quantity;
    textIdx arraySize;
    ABCIdx* wordIdx;
} Pattern;

typedef struct {
    ABCIdx wordIdx;
    unsigned int line;
    unsigned int pos;
} TextElement;

typedef struct {
    textIdx quantity;
    TextElement* text;
} Text;

typedef struct {
    textIdx quantity;
    textIdx arraySize;
    textIdx* r;
} BMRElement;

void InitBMR (WordABC abc, Pattern p, BMRElement* bmr);
textIdx GetBMR(BMRElement* bmr, ABCIdx ai, textIdx pi);
void InitBMN (Pattern p, textIdx* bmn);
void InitBML (textIdx* bmn, textIdx n, textIdx* bmL);
void InitBMl (textIdx* bmn, textIdx n, textIdx* bml);
void BMPreprocess(WordABC abc, Pattern p, BMRElement* bmr, textIdx* bmn, textIdx* bmL, textIdx* bml);
void AGSearchAndPrint(FILE* f, WordABC abc, Pattern p, BMRElement* bmr, textIdx* bmn, textIdx* bmL, textIdx* bml);

#endif /* agalgorithm_h */
