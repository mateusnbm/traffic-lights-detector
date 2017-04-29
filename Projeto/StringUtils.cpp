//
//  StringUtils.cpp
//  Projeto
//
//  Created by Mateus Nunes de B Magalhaes on 6/21/16.
//  Copyright © 2016 MateusNBM. All rights reserved.
//

#include "StringUtils.hpp"

void reverseString(char *str, int lenght) {
    int i;
    char exch;
    for (i = 0; i < (lenght/2); i++) {
        exch = str[lenght-i-1];
        str[lenght-i-1] = str[i];
        str[i] = exch;
    }
}

char *integerToChars(int n) {
    int length = 1;
    char *str = (char *)calloc(1, sizeof(char));
    while (n) {
        length++;
        str = (char *)realloc(str, length*sizeof(char));
        str[length-1] = '0' + (n%10);
        n /= 10;
    }
    reverseString(str, length);
    return str;
}