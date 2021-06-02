#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <ctype.h>
#include "common.h"
#include "ptread.h"


int main() {
    Pattern p;
    WordABC abc;
    InitWordABC(&abc);
    //FILE* f = fopen("/Users/vanyadneprov/Desktop/da4/input.txt", "r");
    
    // Получаем Pattern и Text
    if (!ReadPattern(stdin, &abc, &p))
        ExitBadPattern();

    // Выполняем препроцессинг для алгоритма Бойера-Мура
    BMRElement* bmr = (BMRElement*) malloc (sizeof(BMRElement) * abc.quantity);
    textIdx* bmn = (textIdx*) malloc (sizeof(textIdx) * p.quantity);
    textIdx* bmL = (textIdx*) malloc (sizeof(textIdx) * p.quantity);
    textIdx* bml = (textIdx*) malloc (sizeof(textIdx) * p.quantity);
    BMPreprocess (abc, p, bmr, bmn, bmL, bml);
    
    // Запускаем поиск
    AGSearchAndPrint(stdin, abc, p, bmr, bmn, bmL, bml);

    // Освобождаем память
    free(p.wordIdx);
    TPDestroy(&(abc.tree));
    free(bmr);
    free(bmn);
    free(bmL);
    free(bml);

    return 0;
}
