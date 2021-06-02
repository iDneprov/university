#include "BigInt.h"

int main(void)
{
    std::string firstStringNumber,
        secondStringNumber;
    char operation;
    while (std::cin >> firstStringNumber >> secondStringNumber >> operation) {
        BigInt firstNumber(firstStringNumber);
        BigInt secondNumber(secondStringNumber);
        switch(operation) {
            case '+':
                std::cout << firstNumber + secondNumber << "\n";
                break;
                
            case '-':
                if (firstNumber < secondNumber)
                    std::cout << "Error\n";
                else
                    std::cout << firstNumber - secondNumber << "\n";
                break;
                
            case '*':
                std::cout << firstNumber * secondNumber << "\n";
                break;
                
            case '/':
                if (secondNumber == BigInt(0))
                    std::cout << "Error\n";
                else
                    std::cout << firstNumber / secondNumber << "\n";
                break;
                
            case '^':
                if (firstNumber == BigInt(0)) {
                    if (secondNumber == BigInt(0))
                        std::cout << "Error\n";
                    else
                        std::cout << "0\n";
                } else if (firstNumber == BigInt(1)) {
                    std::cout << "1\n";
                } else if (secondNumber == BigInt(0)) {
                    std::cout << "1\n";
                } else if (secondNumber == BigInt(1)) {
                    std::cout << firstNumber;
                } else
                    std::cout << (firstNumber ^ (std::stoll(secondStringNumber))) << "\n"; // stoll преобразует строку к long long
                break;
                
            case '<':
                if (firstNumber < secondNumber) {
                    std::cout << "true\n";
                } else
                    (std::cout << "false\n");
                break;
                
            case '>':
                if (firstNumber > secondNumber) {
                    std::cout << "true\n";
                } else
                    (std::cout << "false\n");
                break;
                
            case '=':
                if (firstNumber == secondNumber) {
                    std::cout << "true\n";
                } else
                    (std::cout << "false\n");
                break;
        }
    }
    return 0;
}
