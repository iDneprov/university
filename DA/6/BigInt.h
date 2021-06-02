#ifndef _BigInt_H_
#define _BigInt_H_

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

const int BASE = 10000;
const int DIGITS_NUMBER = 4;

class BigInt
{
public:
    BigInt() {};
    BigInt(std::string&);
    BigInt(int n);
    ~BigInt() {};
    
    BigInt operator+(const BigInt&);
    BigInt operator-(const BigInt&);
    BigInt operator*(const BigInt&) const;
    BigInt operator/(const BigInt&);
    BigInt operator^(long long n); // Да, я переопределил xor как возведение в степень, но слоблазн был так велик
    bool operator==(const BigInt&) const;
    bool operator<(const BigInt&) const;
    bool operator>(const BigInt&) const;
    friend std::ostream& operator<<(std::ostream&, const BigInt&);
    
private:
    void DeleteLeadingZeros();
    std::vector<int> BigIntData;
};

#endif
