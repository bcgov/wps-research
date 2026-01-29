/* 20260128 run python command to fix ENVI headers */
#include <unistd.h>
#include <vector>
#include <string>

int main(int argc, char* argv[]) {
    std::vector<char*> args;

    // Program to execute
    args.push_back(const_cast<char*>("fh.py"));

    // Forward all arguments (if any)
    for (int i = 1; i < argc; ++i) {
        args.push_back(argv[i]);
    }

    args.push_back(nullptr); // execvp requires null-terminated argv

    execvp("fh.py", args.data());

    // Only reached if execvp fails
    return 1;
}
