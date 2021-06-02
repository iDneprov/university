#include <iostream>
#include <vector>
#include <stack>

int main()
{
    // Объявляем переменные
    uint32_t width, length;
    std::string string;
    int squareMax = 0, squareCurrent, k, currentIndex;
    // Задаем поле, вспомогательный массив и стек
    std::cin >> width >> length;
    std::vector<std::vector<char> > field(width);
    std::vector<int> countingArray(length);
    std::stack<int> stackOfIndexes;
    // Считываем поле
    for (int i = 0; i < width; ++i) {
        field[i].resize(length);
        std::cin >> string;
        for (int j = 0; j < length; ++j) {
            field[i][j] = string[j];
        }
    }

    // Обрабатываем поле
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < length; ++j) {
            // Изменяем массив-счётчик countingArray для текущей строки
            if (field[i][j] == '0') {
                // Увеличиваем его в соответствующей ячейке, если встретили ноль
                ++countingArray[j];
            } else {
                // Либо обнуляем соответствующую ячейку, если ей соответствует другой символ
                countingArray[j] = 0;
            }
        }
        // Обнуляем индекс countingArray
        k = 0;
        // Здесь мы и находим максимальную площадь, обрабатывая массив-счётчик countingArray для каждой строки
        while (k < length || !stackOfIndexes.empty()) { // пока не превысили длинну и не очистили стек
            if (stackOfIndexes.empty() || (countingArray[stackOfIndexes.top()] <= countingArray[k] && k < length)) {
                stackOfIndexes.push(k); // Если по индексу в стеке хренится значение, не превышающее значение по K, добавляем К в стек и увеличиваем значение
                ++k;
            } else { // В противном случае получаем текущий индекс из стека
                currentIndex = stackOfIndexes.top();
                stackOfIndexes.pop();
                // И вычисляем текущую площадь
                if (stackOfIndexes.empty()) {
                    squareCurrent = countingArray[currentIndex] * k;
                } else {
                    squareCurrent = countingArray[currentIndex] * (k - stackOfIndexes.top() - 1);
                }
                // И если она превышает максимум, сохраняем её значение
                if (squareCurrent > squareMax) squareMax = squareCurrent;
            }
        }
    }

    std::cout << squareMax << std::endl;
    return 0;
}