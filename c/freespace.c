/* 20250620 how much free space on the present volume?
*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/statvfs.h>

int main(void) {
    struct statvfs stat;

    if (statvfs(".", &stat) != 0) {
        perror("statvfs");
        return EXIT_FAILURE;
    }

    unsigned long long free_space = stat.f_bsize * stat.f_bavail;

    printf("Free space: %llu bytes (%.2f MB, %.2f GB)\n",
           free_space,
           free_space / (1024.0 * 1024.0),
           free_space / (1024.0 * 1024.0 * 1024.0));

    return EXIT_SUCCESS;
}

