#ifndef common_h
#define common_h

#include <limits.h>
#include <stdio.h>

typedef long ABCIdx;

#define MIN_WORDABC_SIZE 10
#define MIN_PATTERN_SIZE 10

#define MAX_WORD_LEN 16
#define UNKNOWN_WORD_IDX -1

#define EXIT_NO_MEMORY -1
#define EXIT_BAD_PATTERN -2
#define EXIT_BAD_TEXT -3

void ExitNoMemory(void);
void ExitBadPattern(void);
void ExitBadText(void);

#endif /* common_h */
