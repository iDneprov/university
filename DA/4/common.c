#include <stdlib.h>
#include "common.h"

void ExitNoMemory() {
    printf("ERROR: no memory.");
    exit(EXIT_NO_MEMORY);
}

void ExitBadPattern() {
    printf("ERROR: bad pattern.");
    exit(EXIT_BAD_PATTERN);
}

void ExitBadText() {
    exit(0);
}

