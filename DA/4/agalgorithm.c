#include <stdlib.h>
#include "agalgorithm.h"
#include "ptread.h"

void InitBMR (WordABC abc, Pattern p, BMRElement* bmr) {

    for (ABCIdx i = 0; i < abc.quantity; i++) {
        bmr[i].arraySize = 0;
        bmr[i].quantity = 0;
        bmr[i].r = NULL;
    }
    
    BMRElement* bmrChar;
    for (textIdx i = p.quantity - 1; i >= 0; i--) {
        bmrChar = bmr + p.wordIdx[i];
        bmrChar->quantity++;
        if (bmrChar->quantity > bmrChar->arraySize) {
            bmrChar->arraySize += MIN_PATTERN_SIZE;
            if (bmrChar->r == NULL)
                bmrChar->r = (textIdx*) malloc(sizeof(textIdx) * bmrChar->arraySize);
            else
                bmrChar->r = (textIdx*) realloc(bmrChar->r, sizeof(textIdx) * bmrChar->arraySize);
        }
        bmrChar->r[bmrChar->quantity - 1] = i;
    }
}

// R-функция - Расширенное правило плохого символа Бойера-Мура. Возвращает позицию ближайшего слева от позиции ti символа с алфавитным индексом ai. Возвращаемые позиции начинаются с 0. Если символа нет - то возвращается -1.

textIdx GetBMR(BMRElement* bmr, ABCIdx ai, textIdx pi) {
    if (ai == UNKNOWN_WORD_IDX || pi <= 0)
       return -1;
    if (bmr[ai].quantity == 0)
        return -1;
    for (textIdx i = 0; i < bmr[ai].quantity; i++)
        if (bmr[ai].r[i] < pi)
            return bmr[ai].r[i];
    return -1;
}

void InitBMN (Pattern p, textIdx* bmn) {
    textIdx n = p.quantity, k, l, r, size;
    for (k = 0; k < n; k++) // Обнуляем массив значений ункции Ni
        bmn[k] = 0;
    for (k = n - 2, l = n - 1 , r = n - 1; k >= 0; k--) {
        // Если не внутри Z-блока - находим совпадение
        if (k <= l) {
            // При совпадении меняем l и r
            if (p.wordIdx[k] == p.wordIdx[n - 1]) {
                r = k;
                l = k;
                while (l > 0 && p.wordIdx[l - 1] == p.wordIdx[n - 1 - (r - l + 1)])
                    l--;
                bmn[k] = r - l + 1;
            }
        } else {
            size = bmn[n - 1 - (r - k)];
            if (size > 0) { // Внутри Z-блока ранее посчитанная функция ненулевая
                if (k - size + 1 > l) // Блок внутри блока
                    bmn[k] = size;
                else { // Блок за границами блока. Обновляем l и r
                    r = k;
                    if (l > 0 && p.wordIdx[l - 1] == p.wordIdx[n - 1 - (r - l + 1)]) { // Символы за границами блока совпадают. Ищем новую левую позицию
                        l = l - 1;
                        while (l > 0 && p.wordIdx[l - 1] == p.wordIdx[n - 2 - (r - l)])
                            l--;
                        bmn[k] = r - l + 1;
                    } else // Символы за границами блока не совпадают
                        bmn[k] = k - l + 1;
                }
            }
        }
    }
}

void InitBML (textIdx* bmn, textIdx n, textIdx* bmL) {
    textIdx i;
    for (i = 0; i < n; i++)
        bmL[i] = -1;
    for (textIdx j = 0; j < n - 1; j++) {
        if (bmn[j] > 0) {
            i = n - bmn[j];
            bmL[i] = j;
        }
    }
}

void InitBMl (textIdx* bmn, textIdx n, textIdx* bml) {
    textIdx j = 0;
    bml[0] = j;
    
    for (textIdx i = n - 1; i > 0; i--) {
        if (bmn[n - 1 - i] == n - i)
            j = n - i;
        bml[i] = j;
    }
}

void BMPreprocess (WordABC abc, Pattern p, BMRElement* bmr, textIdx* bmn, textIdx* bmL, textIdx* bml) {
    InitBMR(abc, p, bmr);
    InitBMN(p, bmn);
    InitBML(bmn, p.quantity, bmL);
    InitBMl(bmn, p.quantity, bml);
    /*
    printf("ABC quantity %d\n", abc.quantity);
    for (textIdx i = 0; i < p.quantity; i++)
        printf("N <%d - %d>\n", i, bmn[i]);
    for (textIdx i = 0; i < p.quantity; i++)
        printf("L\' <%d - %d>\n", i, bmL[i]);
    for (textIdx i = 0; i < p.quantity; i++)
        printf("l\' <%d - %d>\n", i, bml[i]);
    */
}

void AGSearchAndPrint(FILE* f, WordABC abc, Pattern p, BMRElement* bmr, textIdx* bmn, textIdx* bmL, textIdx* bml) {
    
    Text t;
    t.text = NULL;
    textIdx n = p.quantity, m, k, i, h, bmShift, bmCharShift, bmSuffixShift, tSize = 0, tBeginCopy, tCopySize;
    unsigned int line = 1, pos = 1;
    
    if (!ReadText(f, &line, &pos, abc, &t, TEXT_BUFFER_SIZE_BY_PATTERN * n, n))
        ExitBadText();
    m = t.quantity;
    k = n - 1;

    textIdx* agM = (textIdx*) malloc(sizeof(textIdx) * m);
    for (textIdx i = 0; i < m; i++)
        agM[i] = 0;
    
    // Алгоритм Бойера-Мура
    while (k < m) {
        // Зачитываем текст в буфер
        if (tSize > 0) {
            if (!ReadText(f, &line, &pos, abc, &t, tSize, n))
                break;
            m = t.quantity;
            if (m == n)
                break;
            tSize = 0;
        }

        i = n - 1;
        h = k;
        while (i >= 0) {
            while (agM[h] > 1 && bmn[i] > 1) {
                if (agM[h] < bmn[i]) {
                    i -= agM[h];
                    h -= agM[h];
                } else if (agM[h] >= bmn[i] && bmn[i] == (i + 1)) {
                    i = -1;
                    break;
                }
                else if (agM[h] > bmn[i] && bmn[i] < (i + 1)) {
                    h -= bmn[i];
                    i -= bmn[i];
                    break;
                } else if (agM[h] == bmn[i] && bmn[i] < (i + 1)) {
                    i -= agM[h];
                    h -= agM[h];
                }
            }
            if (i >= 0) {
                if (p.wordIdx[i] != t.text[h].wordIdx)
                    break;
                else {
                    i--;
                    h--;
                }
            }
        }
        if (i < n - 2)
            agM[k] = n - 1 - i;
        if (i < 0) { // Строка полностью совпала
            printf("%u, %u\n", t.text[k - n + 1].line, t.text[k - n + 1].pos);
            bmShift = (n > 1 ? n - bml[1] : n);
        } else { // Выполняем сдвиг по правилу плохого символа и хорошего суффикса
            bmCharShift = i - GetBMR(bmr, t.text[h].wordIdx, i);
            if (i < n - 1) {
                if (bmL[i + 1] >= 0)
                    bmSuffixShift = n - 1 - bmL[i + 1];
                else
                    bmSuffixShift = n - bml[i + 1];
                bmShift = (bmCharShift > bmSuffixShift ? bmCharShift : bmSuffixShift);
            } else
                bmShift = bmCharShift;
        }
        if (k + bmShift > m - 1) { // вылезаем за пределы буфера
            tBeginCopy = k - n + 1;
            tCopySize = m - tBeginCopy;
            tSize = TEXT_BUFFER_SIZE_BY_PATTERN * n - tCopySize;
            // Переносим остаток текста и M-функцию в начало буфера
            for (textIdx j = 0; j < tCopySize; j++) {
                t.text[j] = t.text[j + tBeginCopy];
                agM[j] = agM[j + tBeginCopy];
            }
            for (textIdx j = tCopySize; j < m; j++)
                agM[j] = 0;
            k = n - 1 + bmShift;
        } else
            k += bmShift;
    }
    free (agM);
}
