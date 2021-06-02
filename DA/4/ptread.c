#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "ptread.h"

bool ReadPattern(FILE* f, WordABC* abc, Pattern* p) {
    int c;
    ABCIdx idx;
    char word[MAX_WORD_LEN + 1];
    
    // Пропускаем разделители в начале
    while ((c = fgetc(f)) != EOF && isspace(c))
        ;
    
    p->quantity = 0;
    if (c == EOF) {
        p->arraySize = 0;
        p->wordIdx = NULL;
        return false;
    }
    
    // Инициируем Pattern
    p->arraySize = MIN_PATTERN_SIZE;
    p->wordIdx = (ABCIdx*) malloc(sizeof(ABCIdx) * p->arraySize);
    if (p->wordIdx == NULL)
        ExitNoMemory();
    
    while (c != EOF && c != '\n') {
        int i = 0; // Считываем слово
        do {
            word[i] = tolower(c);
            i++;
        } while (i < MAX_WORD_LEN && (c = fgetc(f)) != EOF && !isspace(c));
        word[i] = '\0';
        // Добавляем слово в алфавит
        AddWordToABC(abc, word, &idx);
        // Расширяем массив индексов Pattern при достижении конца
        if (p->quantity >= p->arraySize) {
            p->arraySize += MIN_PATTERN_SIZE;
            p->wordIdx = (ABCIdx*) realloc(p->wordIdx, sizeof(ABCIdx) * p->arraySize);
            if (p->wordIdx == NULL)
                ExitNoMemory();
        }
        // Добавляем элемент алфавита в Pattern
        p->wordIdx[p->quantity++] = idx;
        if (c != EOF && c != '\n') {
            // Проверяем не превышает ли слово максимальную длину
            if (i == MAX_WORD_LEN) {
                c = fgetc(f);
                if (!isspace(c)) // Слово диннее максимума
                    ExitBadPattern();
            }
            // Пропускаем разделители до следующего слова
            while (c != EOF && isspace(c) && c != '\n')
                c = fgetc(f);
        }
    }
    //p->quantity -= 1;
    return c != EOF;
}

bool ReadText(FILE* f, unsigned int* line, unsigned int* pos, WordABC abc, Text* t, textIdx tSize, textIdx pSize) {
    int c;
    textIdx count = 0;
    ABCIdx idx;
    char word[MAX_WORD_LEN + 1];

    // Пропускаем разделители в начале
    while ((c = fgetc(f)) != EOF && isspace(c)) {
        if (c == '\n') {
            (*line)++;
            (*pos) = 1;
        }
    }
    
    t->quantity = 0;
    if (c == EOF) {
        t->text = NULL;
        return false;
    }
    
    // Инициируем Text
    if (t->text == NULL) {
        t->text = (TextElement*) malloc(sizeof(TextElement) * TEXT_BUFFER_SIZE_BY_PATTERN * pSize);
        t->quantity = 0;
    } else
        t->quantity = TEXT_BUFFER_SIZE_BY_PATTERN * pSize - tSize;
    if (t->text == NULL)
        ExitNoMemory();
    
    // Читаем текст
    while (c != EOF && count < tSize) {
        int i = 0; // Считываем слово
        do {
            word[i] = tolower(c);
            i++;
        } while (i < MAX_WORD_LEN && (c = fgetc(f)) != EOF && !isspace(c));
        word[i] = '\0';
        // Ищем слово в алфавите
        GetWordIdxInABC(abc, word, &idx);
        
        // Добавляем элемент алфавита в Text
        t->text[t->quantity].wordIdx = idx;
        t->text[t->quantity].line = *line;
        t->text[t->quantity].pos = *pos;
        t->quantity++;
        count++;
        if (c == '\n') {
            *pos = 1;
            (*line)++;
        } else
            (*pos)++;
        
        if (c != EOF && count < tSize) {
            // Проверяем не превышает ли слово максимальную длину
            if (i == MAX_WORD_LEN) {
                c = fgetc(f);
                if (c == '\n') {
                    (*line)++;
                    *pos = 1;
                }
                if (!isspace(c)) // Слово диннее максимума
                    ExitBadText();
            }

            // Пропускаем разделители до следующего слова
            while (c != EOF && isspace(c)) {
                c = fgetc(f);
                if (c == '\n') {
                    (*line)++;
                    *pos = 1;
                }
            }
        }
    }
    return true;
}

