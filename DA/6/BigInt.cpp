#include "BigInt.h"

BigInt::BigInt(int n)
{
    if (n < BASE)
        BigIntData.push_back(n);
    else {
        for(; n; n /= BASE)
            BigIntData.push_back(n % BASE);
    }
}

BigInt::BigInt(std::string &input)
{
    BigIntData.clear(); // удаляем предыдущее число

    std::stringstream cellStream;
    int i;

    for (i = 0 ;input[i] == '0'; ++i); // Проходим ведущие нули
    input = (i == (int)input.size()) ? "0" : input.substr(i); // удаляем их из строки (обрезаем строку)

    for (int i = input.size() - 1; i >= 0; i -= DIGITS_NUMBER) { // обрабатываем куски длинной DIGITS_NUMBER
        int start = i - DIGITS_NUMBER + 1; 
        start = (start < 0) ? 0 : start;
        int end = i - start + 1;

        cellStream << input.substr(start, end); // отрубаем куски от start до end
        int cell = 0;
        cellStream >> cell; // и преобразовывая к числу, пишем в интовую переменную
        BigIntData.push_back(cell); // дописываем к числу в спец предсьавлении
        cellStream.clear(); 
    }
}

BigInt BigInt::operator+(const BigInt &secondNumber)
{
    BigInt result;
    int transferOverflow = 0; // это перенос в следующий разряд при переполнении
    size_t sizeFirst = BigIntData.size(),
        sizeSecond = secondNumber.BigIntData.size(),
        size = std::max(sizeFirst, sizeSecond);

    for (size_t i = 0; i < size || transferOverflow; ++i) { // пока мы не прошли длинну наибольшего числа и не выполнили все обязательства по переносу переполнений
        int firstTerm = (i < sizeFirst) ? BigIntData[i] : 0; // записываем число из текущего разряда или ноль в случае выхода за границу
        int secondTerm = (i < sizeSecond) ? secondNumber.BigIntData[i] : 0; // записываем число из текущего разряда или ноль в случае выхода за границу
        result.BigIntData.push_back(firstTerm + secondTerm + transferOverflow); // записываем сумму и переполнение с прошлой операции
        // вычленяем переполнение
        transferOverflow = result.BigIntData.back() >= BASE;
        if (transferOverflow)
            result.BigIntData.back() -= BASE;
    }
    return result;
}

BigInt BigInt::operator-(const BigInt &secondNumber)
{
    BigInt result;
    int transferOverflow = 0; // это перенос в следующий разряд
    size_t sizeFirst = BigIntData.size(),
        sizeSecond = secondNumber.BigIntData.size();

    for (size_t i = 0; i < sizeFirst || transferOverflow; ++i) { // пока мы не прошли длинну наибольшего числа и не выполнили все обязательства по переносу переполнений
        int firstTerm = (i < sizeFirst) ? BigIntData[i] : 0; // записываем число из текущего разряда или ноль в случае выхода за границу
        int secondTerm = (i < sizeSecond) ? secondNumber.BigIntData[i] : 0; // записываем число из текущего разряда или ноль в случае выхода за границу
        result.BigIntData.push_back(firstTerm - transferOverflow - secondTerm); // записываем разность и переполнение с прошлой операции
        // вычленяем переполнение
        transferOverflow = result.BigIntData.back() < 0;
        if (transferOverflow)
            result.BigIntData.back() += BASE;
    }
    result.DeleteLeadingZeros();
    return result;
}

BigInt BigInt::operator*(const BigInt &secondNumber) const // Наивное умножение
{
    BigInt result;
    size_t sizeFirst = BigIntData.size(),
        sizeSecond = secondNumber.BigIntData.size();
    result.BigIntData.resize(sizeFirst + sizeSecond); // задаем  размер числа такой, чтобы в него точно влез результат

    for (size_t i = 0; i < sizeFirst; ++i) {
        int transferOverflow = 0; // это перенос в следующий разряд при переполнении
        for (size_t j = 0; j < sizeSecond || transferOverflow; ++j) {
            int secondTerm = (j < sizeSecond) ? secondNumber.BigIntData[j] : 0; // разряд второго числа (следим за тем, чтобы не вылезти за границы вектора)
            result.BigIntData[i + j] += BigIntData[i] * secondTerm + transferOverflow; // само умножение
            // вычленяем переполнение
            transferOverflow = result.BigIntData[i + j] / BASE;
            result.BigIntData[i + j] -= transferOverflow * BASE;
        }
    }
    result.DeleteLeadingZeros(); // уменьшаем длинну числа при необходимости
    return result;
}

BigInt BigInt::operator/(const BigInt &secondNumber)
{
    BigInt result,
        currentDischarge = BigInt(0); // Текущий разряд делимого или два разряда, если разряд делителя больше
    result.BigIntData.resize(BigIntData.size());
    for (int i = (int) BigIntData.size() - 1; i >= 0; --i) { // пробегаем по всем разрядам делимого числа, начиная со старшего
        currentDischarge.BigIntData.insert(currentDischarge.BigIntData.begin(), BigIntData[i]); // вставляем текущий разряд
        if (!currentDischarge.BigIntData.back()) currentDischarge.BigIntData.pop_back(); // если старшый разряд пустой, удаляем его
        int resultDischarge = 0,
            left = 0,
            right = BASE;
        while (left <= right) { // методом половинного деления подбираем число
            int middle = (left + right) / 2;
            BigInt current = secondNumber * BigInt(middle);
            if (current < currentDischarge || current == currentDischarge) {
                resultDischarge = middle;
                left = middle + 1;
            } else {
                right = middle - 1;
            }
        }
        result.BigIntData[i] = resultDischarge; // записываем полученный результат в i-тый разряд ответа
        currentDischarge = currentDischarge - secondNumber * BigInt(resultDischarge); // и уменьшаем число, отвечающее за текущий разряд
    }
    result.DeleteLeadingZeros();
    return result;
}

BigInt BigInt::operator^(long long n) // переопределил xor, за то получилось красиво
{
    BigInt result(1);
    for (; n; n /= 2) {
        if (n & 1) // нечетная степень
            result = result * (*this); // тогда домножаем на себя же
        (*this) = (*this) * (*this); // возводим число в квадрат
    }
    return result;
}

bool BigInt::operator==(const BigInt &secondNumber) const
{
    return this->BigIntData == secondNumber.BigIntData; // просто сравниваем два вектора
}

bool BigInt::operator<(const BigInt &secondNumber) const
{
    // на вход подаются числа без ведущих нулей
    // если длины не одинаковы, сравниваем длины, в противном случае запускаем поэлементное сравнение для векторов  при помощи lexicographical_compare
    size_t sizeFirst = BigIntData.size(),
        sizeSecond = secondNumber.BigIntData.size();

    if (sizeFirst != sizeSecond)
        return sizeFirst < sizeSecond;
    return std::lexicographical_compare(BigIntData.rbegin(), BigIntData.rend(), secondNumber.BigIntData.rbegin(), secondNumber.BigIntData.rend());
}

bool BigInt::operator>(const BigInt &secondNumber) const
{
    // на вход подаются числа без ведущих нулей
    // если длины не одинаковы, сравниваем длины, в противном случае запускаем поэлементное сравнение для векторов при помощи lexicographical_compare
    size_t sizeFirst = BigIntData.size(),
        sizeSecond = secondNumber.BigIntData.size();

    if (sizeFirst != sizeSecond)
        return sizeFirst > sizeSecond;
    return std::lexicographical_compare(secondNumber.BigIntData.rbegin(), secondNumber.BigIntData.rend(), BigIntData.rbegin(), BigIntData.rend());
}

void BigInt::DeleteLeadingZeros()
{
    while (BigIntData.size() > 1 && !BigIntData.back())
        BigIntData.pop_back(); // разряды хранятся в обратном порядке
}

std::ostream &operator<<(std::ostream &stream, const BigInt &num)
{
    int digit = num.BigIntData.size();
    if (!digit) // Если число пустое, поток без изменений
        return stream;
    stream << num.BigIntData[digit - 1]; // выводим первый разряд
    for (int i = digit - 2; i >= 0; --i) // пока разряды не закончились
        stream << std::setfill('0') << std::setw(DIGITS_NUMBER) << num.BigIntData[i]; // дописываем ведущие нули (setfill наполняет поле размера DIGITS_NUMBER, заданного через setw) и еще один разряд
    
    return stream;
}
