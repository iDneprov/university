#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include "TPatricia.h"


enum commandTypeTag {FINAL, INSERT, REMOVE, SEARCH, SAVE, LOAD, ERROR};

typedef enum commandTypeTag commandType; 


commandType ReadLine(FILE *in, char * key, uint64_t *value)
{
    // fgetc поочередно при каждом вызове возвращает символ указанного файла
    // с -- текущий обрабатываемый символ
    int c = fgetc(in);
    //зачитываем до пробела или окончания файла
    while (isspace(c) && (c != EOF))
        c = fgetc(in);
    //если мы достигли конца файла, возвращаем FINAL
    if (c == EOF)
        return FINAL;
    // Проверяем, является ли символ цифрой или буквой (isalnum(c) = 0 если символ цифра или буква)
    if (isalnum(c))
        //возвращаем последний считанный символ в in
        ungetc(c, in);
    // Если не по 3 лучилось считать ключ возвращаем ошибкуы
    if (fscanf(in, "%s", key) != 1)
        return ERROR;
    // Сохранение и загрузка словаря в файл
    if (c == '!') {
        if (strcmp(key, "Save") == 0) { // Сохранение
            if (fscanf(in, "%s", key) == 1) // если после ! нет Save или Load, возвращаем ошибку
                return SAVE;
        } else if (strcmp(key, "Load") == 0) { // Удаление
            if (fscanf(in, "%s", key) == 1) // если после ! нет Save или Load, возвращаем ошибку
                return LOAD;
        }
        return ERROR;
    // Добавление элемента в словарь
    } else if (c == '+') {
        if (fscanf(in, "%lu", value) == 1)
            return INSERT;
        else
            return ERROR;
    // Удаление элемента из словаря
    } else if (c == '-')
        return REMOVE;

    // По умолчанию производится поиск элемента словаря
    return SEARCH;
}

int main(void)
{   // Инициализируем словарь
    TPatriciaNode tree = NULL, node;
    // Инициализируем строку ключа символом конца строки 
    char key[257] = { '\0' };
    // Инициализируем нулем переменную значения
    uint64_t value = 0;
    commandType command;
    FILE* f;

    while ((command = ReadLine(stdin, key, &value)) != FINAL) {
        
        // Обработка возврата ReadLine
        switch (command) {
            case INSERT: // Вставка
                if (TPCIKeyInsert(&tree, key, value))
                    printf("OK\n");
                else 
                    printf("Exist\n");
                break;

           case REMOVE: // Удаление
                if(TPCINodeDelete(&tree, key))
                    printf("OK\n");
                else 
                    printf("NoSuchWord\n");
                break;

            case SAVE: // Сохранение дерева в текстовый файл
                f = fopen(key, "wb");
                if (f == NULL) {
                    printf("ERROR: unable open file\n");
                    break;
                }

                if (TPWrite(f, tree))
                    printf("OK\n");
                else
                    printf("ERROR: unable to write a tree\n");
                fclose(f);
                break;

            case LOAD: // Загрузка дерева из текстового файла
                TPDestroy(&tree);
                char mark;
                if ((f = fopen(key, "rb")) == NULL) {
                    printf("ERROR: unable open file\n");
                    break;
                }
                if (TPRead(f, &tree, &mark))
                    printf("OK\n");
                else 
                    printf("ERROR: unable to read a tree\n");
                fclose(f);
                break;

            case SEARCH: // Поиск элемента в дереве
                node = TPCINodeSearch(tree, key);
                if (node != NULL)
                    printf("OK: %lu\n", node->value);
                else
                    printf("NoSuchWord\n");
                break;
            default:
                printf("ERROR: parse ERROR\n");
            break;
        }
    }
    // Удаление дерева
    TPDestroy(&tree);
    return 0;
}
