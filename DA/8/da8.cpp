#include <iostream>
#include <vector>

#define TYPES 3

void Swap(std::vector<int> &vector, int a, int b) {
    int tmp = vector[a];
    vector[a] = vector[b];
    vector[b] = tmp;
}

int SortWithSwapsCounting(std::vector<int> &numbers, int quantity) {
    if (quantity <= 1) return 0;
    int swapCount = 0;
    std::vector<int> count(TYPES); // задаем вектор для подстёта количества чисел каждого типа (сколько троек, двоек и едениц)
    for (int i = 0; i < quantity; ++i) ++count[numbers[i] - 1];
    // производим операции над count, чтобы каждой в соответсвующей ячейке хранилось количество чисел меньше данного
    count[2] = count[1] + count[0];
    count[1] = count[0];
    count[0] = 0;

    // начинаем сортировку
    for (int current = 0; current < quantity; ++current) { // проходим по всем значениям numbers
        if (current < count[1]) { // это число стоит на месте для единицы
            if (numbers[current] == 2) { // если текущее число -- два, нужно тоже свапнуть её с ближайшей единицей, правее текущей цифры
                for (int k = count[1]; k < quantity; ++k) { // находим единицу, которая стоит на месте дойки и делаем свап
                    if (numbers[k] == 1) {
                        Swap(numbers, current, k);
                        ++swapCount;
                        break;
                    }
                }
            } else if (numbers[current] == 3) { // текущее число -- три и свапнуть его нужно с еденицей
                for (int k = quantity - 1; k >= count[1]; --k) { // находим единицу, которая стоит на месте тройки и делаем свап
                    if (numbers[k] == 1) {
                        Swap(numbers, current, k);
                        ++swapCount;
                        break;
                    }
                }
            }

        } else if (current < count[2]) { // это число стоит на месте двойки
            if (numbers[current] == 3){ // текущее число -- три и свапнуть его нужно с двойкой
                for (int k = count[2]; k < quantity; ++k) { // находим двойку, которая стоит на месте тройки и делаем свап
                    if (numbers[k] == 2) {
                        Swap(numbers, current, k);
                        ++swapCount;
                        break;
                    }
                }
            }   
        }
    }
    return swapCount;
}

int main(int argc, char const *argv[]) {
    int quantity;
    std::cin >> quantity;
    std::vector<int> numbers(quantity); // задаем вектор, в котором храним числа для сотритовки
    for (int i = 0; i < quantity; ++i) std::cin >> numbers[i]; // и зачитываем в него значения
    std::cout << SortWithSwapsCounting(numbers, quantity) << std::endl;
    return 0;
}
