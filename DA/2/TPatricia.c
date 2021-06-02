#include "TPatricia.h"

// Маркеры для сериализации
#define NODE_MARK 1 // Начало записи для элемента
#define NO_CHILD_MARK 2 // У элемента нет потомков
#define NO_SIBLING_MARK 3 // У элемента нет братьев

short StrBeginsWith(char* str, char* beginStr) {
    if (beginStr == NULL)
        return str == NULL;
    if (str == NULL)
        return -1;
    for (int i = 0; ; i++) {
        if (str [i] == '\0') {
            if (beginStr[i] == '\0') // дошли до конца обеих строк, строки совпали
                return 1;
            else if (beginStr[i] != '\0') // закончилась первая строка
                return -1;
        }
        if (beginStr[i] == '\0') // начало совпало
            return 0;
        else if (str[i] != beginStr[i]) // вхождение не найдено
            return -1;
    }
}

int CommonSubstrLen(char* str1, char* str2) {
    int i;
    if (str1 == NULL || str2 == NULL)
        return 0;
    for (i = 0; str1[i] != '\0' && str2[i] != '\0' && str1[i] == str2[i]; i++)
        ;
    return i;
}

void StrLower (char* dest, char* src) {
    unsigned long srcLen = strlen(src);
    for (int i = 0; i < srcLen; i++)
        dest[i] = tolower(src[i]);
    dest[srcLen] = 0;
}

TPatriciaNode CreateNewTPNode (char* key, uint64_t value, TPatriciaNode sibling, TPatriciaNode child, bool isEoW) {
    
    TPatriciaNode node = (TPatriciaNode) malloc(sizeof(struct patriciaNode));
    
    if (node == NULL) // Нехватило памяти
        exit(EXIT_NO_MEMORY);
    node->subKey = key;
    node->value = value;
    node->sibling = sibling;
    node->child = child;
    node->isEoW = isEoW;
    
    return node;
}

void FreeTPNode (TPatriciaNode node) {
    
    free(node->subKey);
    free(node);
}

bool WriteSubKey (FILE* f, char* str) {
    int len, c;
    if (str == NULL || !(len = strlen(str)))
        return false;
    c = putc(len, f);
    if (c == EOF)
        return false;
    for (int i = 0; i < len; i++) {
        c = putc(str[i], f);
        if (c == EOF)
            return false;
    }
    return true;
}

bool ReadSubKey (FILE* f, char** str){
    int len = getc(f), c;
    if (len == EOF)
        return false;
    
    *str = malloc(sizeof(char) * (len + 1));
    for (int i = 0; i < len; i++) {
        c = getc(f);
        if (c == EOF) {
            free(str);
            return false;
        }
        (*str)[i] = c;
    }
    (*str)[len] = '\0';
    return true;
}

bool WriteValue (FILE* f, uint64_t value) {
    char* binValue = (char *) &value;
    int c;
    for (int i = 0; i < sizeof(uint64_t); i++) {
        c = putc(binValue[i], f);
        if (c == EOF)
            return false;
    }
    return true;
}

bool ReadValue (FILE* f, uint64_t* value) {
    char* binValue = (char *) value;
    int c;
    for (int i = 0; i < sizeof(uint64_t); i++) {
        c = getc(f);
        if (c == EOF)
            return false;
        binValue[i] = c;
    }
    return true;
}

bool WriteIsEoW (FILE* f, bool isEoW) {
    int c;
    c = putc(isEoW, f);
    if (c == EOF)
        return false;
    return true;
}

bool ReadIsEoW (FILE* f, bool* isEoW) {
    int c;
    c = getc(f);
    if (c == EOF || (c != 0 && c != 1))
        return false;
    *isEoW = c;
    return true;
}

bool WriteMark (FILE* f, char mark) {
    int c;
    c = putc(mark, f);
    if (c == EOF)
        return false;
    return true;
}

bool ReadMark (FILE* f, char* mark) {
    int c;
    c = getc(f);
    if (c == EOF || (c != NODE_MARK && c != NO_CHILD_MARK && c != NO_SIBLING_MARK))
        return false;
    *mark = c;
    return true;
}

bool WriteRecord (FILE* f, char mark, TPatriciaNode node) {
    if (node == NULL && mark == NODE_MARK)
        return false;
    if (mark == NODE_MARK) {
        if (WriteMark(f, mark)) {
            if (!WriteSubKey(f, node->subKey) || !WriteIsEoW(f, node->isEoW))
                return false;
            if (node->isEoW)
                return WriteValue(f, node->value);
        } else
            return false;
    } else if (mark == NO_CHILD_MARK || mark == NO_SIBLING_MARK) {
        return WriteMark(f, mark);
    } else
        return false;
    
    return true;
}


bool ReadRecord (FILE* f, char* mark, TPatriciaNode* node) {
    if (!ReadMark(f, mark))
        return false;
    if (*mark == NODE_MARK) {
        char* subKey;
        uint64_t value;
        bool isEoW;
        
        if (!ReadSubKey(f, &subKey) || !ReadIsEoW(f, &isEoW))
            return false;
        // Значение считываем, если есть признак конца слова
        if (isEoW && !ReadValue(f, &value))
            return false;
        *node = CreateNewTPNode (subKey, value, NULL, NULL, isEoW);
    }
    return true;
}

/*
 Поиск элемента в дереве с возвратом значений
 node - найденный элемент в дереве tree по ключу key. Если не найден - то NULL.
 parent - родительский элемент найденного элемента. Если элемент на первом уровне - то NULL.
 firstSibling - первый элемент в списке уровня найденного элемента
 */
void ExtendedTPNodeSearch(TPatriciaNode tree, char* key, TPatriciaNode *node, TPatriciaNode *parent, TPatriciaNode *firstSibling) {
    if (tree == NULL || key == NULL || strcmp(key, "") == 0) {
        *node = NULL;
        return;
    }
    for (TPatriciaNode iNode = tree; iNode != NULL; iNode = iNode->sibling) {
        switch (StrBeginsWith(key, iNode->subKey)) {
            case 1:
                if (iNode->isEoW) {
                    *node = iNode;
                    *firstSibling = tree;
                    return;
                }
                else {
                    *node = NULL;
                    return;
                }
            case 0:
                *parent = iNode;
                ExtendedTPNodeSearch (iNode->child, key + strlen(iNode->subKey), node, parent, firstSibling);
                return;
        }
    }
}

TPatriciaNode TPNodeSearch(TPatriciaNode tree, char* key) {
    if (tree == NULL || key == NULL || strcmp(key, "") == 0)
        return NULL;
    for (TPatriciaNode node = tree; node != NULL; node = node->sibling) {
        switch (StrBeginsWith(key, node->subKey)) {
            case 1:
                if (node->isEoW)
                    return node;
                else
                    return NULL;
            case 0:
                return TPNodeSearch (node->child, key + strlen(node->subKey));
        }
    }
    return NULL;
}

bool TPKeyInsert(TPatriciaNode* tree, char* key, uint64_t value) {
    if (key == NULL || strcmp(key, "") == 0)
        return false;
    
    char* keyCopy;
    unsigned long keyLen = strlen(key), subKeyLen, commonLen;
    TPatriciaNode newNode;

    if (*tree == NULL) { // Создаем дерево, если оно пустое
        keyCopy  = (char*) malloc(sizeof(char) * (keyLen + 1));
        if (keyCopy == NULL) // Нехватило памяти
            exit(EXIT_NO_MEMORY);
        strcpy(keyCopy, key);
        *tree = CreateNewTPNode (keyCopy, value, NULL, NULL, true);
        return true;
    }
    
    
    for (TPatriciaNode node = *tree; node != NULL; node = node->sibling) {
        commonLen = CommonSubstrLen(node->subKey, key);
        if ( commonLen > 0) { // Нашли пересечение двух ключей
            subKeyLen = strlen(node->subKey);
            if (keyLen == subKeyLen && keyLen == commonLen) { // Ключи полностью совпали
                if (node->isEoW) // Ключ уже был в словаре
                    return false;
                else { // Ключа не было. Добавляем признак
                    node->value = value;
                    node->isEoW = true;
                    return true;
                }
            } else if (keyLen == commonLen && keyLen < subKeyLen) { //Ключ - подстрока подключа
                keyCopy = (char*) malloc(sizeof(char) * (subKeyLen - commonLen + 1));
                if (keyCopy == NULL) // Нехватило памяти
                    exit(EXIT_NO_MEMORY);
                strcpy(keyCopy, node->subKey + commonLen);
                newNode = CreateNewTPNode(keyCopy, node->value, NULL, node->child, node->isEoW);
                free(node->subKey);
                node->subKey = (char*) malloc(sizeof(char) * (strlen(key) + 1));
                if (node->subKey == NULL) // Нехватило памяти
                    exit(EXIT_NO_MEMORY);
                strcpy(node->subKey, key);
                node->value = value;
                node->child = newNode;
                node->isEoW = true;
                return true;
            } else if (subKeyLen == commonLen && subKeyLen < keyLen) { //Подключ - подстрока нового ключа
                return TPKeyInsert (&(node->child), key + subKeyLen, value);
            } else { // Ключ и новый ключ имеют в начале общие символы, при этом, не являсь подстроками друг друга.
                keyCopy = (char*) malloc(sizeof(char) * (subKeyLen - commonLen + 1));
                if (keyCopy == NULL) // Нехватило памяти
                    exit(EXIT_NO_MEMORY);
                strcpy(keyCopy, node->subKey + commonLen);
                newNode = CreateNewTPNode(keyCopy, node->value, NULL, node->child, node->isEoW); // Разбиваем элемент на два
                free(node->subKey);
                node->subKey = (char*) malloc(sizeof(char) * (commonLen + 1));
                if (node->subKey == NULL) // Нехватило памяти
                    exit(EXIT_NO_MEMORY);
                strncpy(node->subKey, key, commonLen); // Инициализируем новый узел с подключом
                node->subKey[commonLen] = '\0';
                node->value = 0;
                node->child = newNode;
                node->isEoW = false;
                TPKeyInsert(&newNode, key + commonLen, value); // Добавляем новый ключ
                return true;
            }
        } else if (node->sibling == NULL) { // Не нашли пересечений на текущем уровне. Добавляем брата.
            keyCopy = (char*) malloc(sizeof(char) * (keyLen + 1));
            if (keyCopy == NULL) // Нехватило памяти
                exit(EXIT_NO_MEMORY);
            strcpy(keyCopy, key);
            node->sibling = CreateNewTPNode (keyCopy, value, NULL, NULL, true);
            return true;
        }
    }
    return false;
}

bool TPNodeDelete(TPatriciaNode* tree, char* key) {
    if (*tree == NULL || key == NULL || strcmp (key, "") == 0)
        return false;
    
    TPatriciaNode node = NULL, parent = NULL, firstSibling = NULL, tmpNode;
    ExtendedTPNodeSearch(*tree, key, &node, &parent, &firstSibling);
    
    if (node == NULL)
        return false;
    
    if (node->child == NULL) {// Нет потомков. Удаляем элемент из словаря
        if (node == firstSibling && node->sibling == NULL) { // Единственный элемент на уровне. Удаляем полностью уровень.
            if (parent == NULL) // Последний элемент в словаре
                *tree = NULL;
            else
                parent->child = NULL;
            FreeTPNode(node);
            return true;
        } else { // Есть несколько элементов на уровне
            if (node == firstSibling) { // Элемент первый в списке
                if (parent == NULL) // Элемент на первом уровне
                    *tree = node->sibling;
                else
                    parent->child = node->sibling;
            } else { // Элемент не первый в списке. Находим предыдущего
                for (tmpNode = firstSibling; tmpNode->sibling != node; tmpNode = tmpNode->sibling)
                    ;
                tmpNode->sibling = node->sibling;
            }
            FreeTPNode(node);
            // Проверяем не остался ли на уровне элемента единственный элемент у которого предок не является ключом и объединяем
            if (parent != NULL && !parent->isEoW && parent->child->sibling == NULL) {
                tmpNode = parent->child;
                parent->subKey = realloc(parent->subKey, strlen(parent->subKey) + strlen(tmpNode->subKey) + 1);
                strcat(parent->subKey, tmpNode->subKey);
                parent->value = tmpNode->value;
                parent->child = tmpNode->child;
                parent->isEoW = tmpNode->isEoW;
                FreeTPNode(tmpNode);
            }
            return true;
        }
    } else { // У элемента есть потомки
        if (node->child->sibling == NULL) { // Потомок только один. Объединяем потомка с родителем
            tmpNode = node->child;
            node->subKey = realloc(node->subKey, strlen(node->subKey) + strlen(tmpNode->subKey) + 1);
            strcat(node->subKey, tmpNode->subKey);
            node->value = tmpNode->value;
            node->child = tmpNode->child;
            node->isEoW = tmpNode->isEoW;
            FreeTPNode(tmpNode);
        } else { // Несколько потомков. Удаляем признак
            node->value = 0;
            node->isEoW = false;
        }
        return true;
    }
    return false;
}

TPatriciaNode TPCINodeSearch(TPatriciaNode tree, char* key) {
    char* lowerKey = (char*) malloc (sizeof(char) * (strlen(key) + 1));
    StrLower (lowerKey, key);
    
    TPatriciaNode node = TPNodeSearch(tree, lowerKey);
    free(lowerKey);
    return node;
}

bool TPCIKeyInsert(TPatriciaNode* tree, char* key, uint64_t value) {
    char* lowerKey = (char*) malloc (sizeof(char) * (strlen(key) + 1));
    StrLower (lowerKey, key);
    
    bool result = TPKeyInsert(tree, lowerKey, value);
    free(lowerKey);
    return result;
}

bool TPCINodeDelete(TPatriciaNode* tree, char* key) {
    char* lowerKey = (char*) malloc (sizeof(char) * (strlen(key) + 1));
    StrLower (lowerKey, key);
    
    bool result = TPNodeDelete(tree, lowerKey);
    free(lowerKey);
    return result;
}

void TPDestroy(TPatriciaNode* tree) {
    if (tree == NULL || *tree == NULL)
        return;
    TPatriciaNode node = *tree;
    TPatriciaNode siblingNode = node->sibling;
    while (node != NULL) {
        if (node->child != NULL)
            TPDestroy(&(node->child));
        FreeTPNode(node);
        node = siblingNode;
        if (node != NULL)
            siblingNode = node->sibling;
    }
    *tree = NULL;
}

/*
 
 Запись дерева в бинарный файл с помощью рекурсивного обхода словаря
 f - дескриптор файла для чтения
 tree - корневой узел дерева словаря.
 
 В файл последовательно записываются элементы. Сначала все потомки, затем братья. Запись каждого элемента дерева начинается с маркера NODE_MARK и затем записывается бинарный блок данных. Когда потомки заканчиваются, записывается маркер NO_CHILD_MARK и начинается запись братьев. При завершении цикла записи братьев записывется маркер NO_SIBLING_MARK. Для формирования записей используется вспомогательная функция WriteRecord.
 
 Функция возвращает:
 - true - если чтение прошло успешно.
 - false - при возникновении любых проблем.
 
 */

bool TPWrite(FILE* f, TPatriciaNode tree) {
    if (tree == NULL)
        return WriteMark(f, NO_CHILD_MARK) && WriteMark(f, NO_SIBLING_MARK);
    for (TPatriciaNode node = tree; node != NULL; node = node->sibling) {
        if (!WriteRecord(f, NODE_MARK, node))
            return false;
        if (node->child != NULL) {
            if (!TPWrite(f, node->child))
                return false;
        } else {
            if (!WriteRecord(f, NO_CHILD_MARK, NULL))
                return false;
        }
    }
    if (!WriteRecord(f, NO_SIBLING_MARK, NULL))
        return false;
    return true;
}

/*
 
 Чтение дерева из файла после записи TPWrite
 f - дескриптор файла для чтения
 curNode - указатель на текущий узел начиная с которого записывается дерево.
 lastMark - последний считанный маркер при рекурсивном вызове. Передается в вызывающую функцию через указатель
 
 Функция возвращает:
 - true - если чтение прошло успешно.
 - false - при возникновении любых проблем. При этом память выделенная под динамические переменные не освобождается.
 */

bool TPRead(FILE* f, TPatriciaNode* curNode, char* lastMark) {
    char mark, nextMark;
    TPatriciaNode readNode, *nextNode;
    
    if (!ReadRecord(f, &mark, &readNode))
        return false;
    
    *lastMark = mark; // С помощью маркера управляем дальнейшим ходом чтения
    
    if (mark == NODE_MARK) { // Прочитали элемент дерева
        *curNode = readNode;
        nextNode = &((*curNode)->child);
        if (!TPRead(f, nextNode, &nextMark)) // Сразу пытаемся рекурсивно прочитать следующую запись в потомка текущего элемента
            return false;
        if (nextMark != NO_SIBLING_MARK) { // Элементы читаем рекурсивно в братьев пока не встретим маркер
            nextNode = &((*curNode)->sibling);
            return TPRead(f, nextNode, &nextMark);
        } else
            return true;
    } else if (mark == NO_CHILD_MARK || mark == NO_SIBLING_MARK) // Когда встречаем маркер, то просто выходим из функции передавая маркер в вызывающую функцию через переменную *lastMark
        return true;
    return false;
}
