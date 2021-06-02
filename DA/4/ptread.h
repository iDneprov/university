#ifndef ptread_h
#define ptread_h

#include "agalgorithm.h"

bool ReadPattern(FILE* f, WordABC* abc, Pattern* p);
bool ReadText(FILE* f, unsigned int* line, unsigned int* pos, WordABC abc, Text* t, textIdx tSize, textIdx pSize);

#endif /* ptread_h */
