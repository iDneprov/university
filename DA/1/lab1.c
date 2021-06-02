#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#define MAX_VAL_SIZE 2050

//елемент списка типа: число, строка и ссылка на следующий элемент
typedef struct node{
    unsigned short ID;
    char* value;
    struct node* next;
} KeyPair;
//элемент массива типа: первый элемент и последний известный элемент списка с даннымм ID
typedef struct{
    KeyPair* first;
    KeyPair* last;
} IndexedElement;

void counting_sort(KeyPair *list, unsigned int idRange, unsigned short minID, IndexedElement *sortedArray)
{
    unsigned short idx;

    for (unsigned int i = 0; i < idRange; i += 1)
        sortedArray[i].first = NULL;   

    while (list != NULL){
        idx = list->ID - minID;
        if (sortedArray[idx].first == NULL) {
            sortedArray[idx].first = list;
            sortedArray[idx].last = list;
        } else {
            sortedArray[idx].last->next = list;
            sortedArray[idx].last = list;
        }
        list = list->next;
        sortedArray[idx].last->next = NULL;
    }
    return;
}

void print_sorted (IndexedElement *sortedArray, unsigned int idRange) {
    //выводит отсортированные данные
    KeyPair* current;
    for (unsigned int i = 0; i < idRange; i += 1){
        if (sortedArray[i].first != NULL) {
            current = sortedArray[i].first;
            do {
                printf("%hu%s", current->ID, current->value);
                if (current->value[strlen(current->value)-1] != '\n')
                    printf("\n");
                current = current->next;
            } while (current != NULL);
        }
    }
}

int main(void)
{
    unsigned int idRange;
    long currentID;
    unsigned short minID = USHRT_MAX, maxID = 0;
    KeyPair *current, *list, *last;
    int scannedKey;
    unsigned char isFirst = 1;


    // Считываем пары ключ-значение в динамический список list
    while((scannedKey = scanf("%li", &currentID)) != EOF) {
        if (scannedKey == 0) {
            printf("ERROR: В начале строки отсутствует ключ.\n");
            return 0;
        }
        if (currentID < 0 || currentID > USHRT_MAX) {
            printf("ERROR: Ключ за пределами допустимого диапазона значений.\n");
            return 0;
        }

        if (currentID > maxID) maxID = currentID;
        if (currentID < minID) minID = currentID;
        
        current = (KeyPair*)malloc(sizeof(KeyPair));
        current->ID = currentID;
        current->value = (char *) malloc(MAX_VAL_SIZE + 1);
        fgets (current->value, MAX_VAL_SIZE, stdin);
        current->value = realloc (current->value, strlen(current->value) + 1); 

        if (isFirst) {
                list = current;
                isFirst = 0;
        } else
            last->next = current;
        
        last = current;
    }
    
    // Если считаны элементы - производим сортировку и печать
    if (!isFirst) {
        last->next = NULL;

        idRange = maxID - minID + 1;
        IndexedElement countingArray[idRange];

        counting_sort(list, idRange, minID, (IndexedElement*) &countingArray);
        print_sorted((IndexedElement*) &countingArray, idRange);
    }
    
    return 0;
}
