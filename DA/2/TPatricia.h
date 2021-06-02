#ifndef TPatriciaNode_H
#define TPatriciaNode_H
#define EXIT_NO_MEMORY -1

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

typedef struct patriciaNode *TPatriciaNode;

struct patriciaNode {
    char* subKey;
    uint64_t value;
    TPatriciaNode sibling;
    TPatriciaNode child;
    bool isEoW; // Является ли нода концом слова.
};

TPatriciaNode TPNodeSearch(TPatriciaNode tree, char* key);
bool TPKeyInsert(TPatriciaNode* tree, char* key, uint64_t value);
bool TPNodeDelete(TPatriciaNode* tree, char* key);
TPatriciaNode TPCINodeSearch(TPatriciaNode tree, char* key);
bool TPCIKeyInsert(TPatriciaNode* tree, char* key, uint64_t value);
bool TPCINodeDelete(TPatriciaNode* tree, char* key);
void TPDestroy(TPatriciaNode* tree);
bool TPWrite(FILE* f, TPatriciaNode tree);
bool TPRead(FILE* f, TPatriciaNode* tree, char* lastMark);

#endif
