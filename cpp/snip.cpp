#include"misc.h"
/* 20190708 snip.cpp: remove trailing whitespace from a file without loading it into memory */

// think fseek/ ftell is 0-indexed: have to go to end and back one to get last char
int main(int argc, char** argv){
  if(argc < 2){
    err("snip [filename] #remove whitespace from end of file: optional additional argument: if provided, don't create backup! By default, filename.bak is created");
  }
  str fn(argv[1]);
  if(str(".dat") == fn.substr(fn.size()-4, fn.size())){
    err("probably don't want to remove whitespace from a fixed-width file");
  }

  FILE * f = fopen(argv[1], "rb");
  if(!f) err("failed to open input file");
  fseek(f, 0, SEEK_END);
  size_t fp = ftell(f);
  size_t fs = fp; // record the file size

  // start at the beginning of the end of the file:
  cout << "fs: " << fp << endl;
  fseek(f, fp - 1, SEEK_SET);
  cout << "fp: " << ftell(f) << endl;

  fp = ftell(f);
  // count how many whitespace chars there are at the end:
  int n_white = 0;
  char c;
  c = fgetc(f);
  while(1){
    if(!isspace(c)) break;
    n_white ++;
    cout << "fp before read: " << ftell(f) - 1 << ": " << " [" << (unsigned int)(unsigned char)c << "]" << (isspace(c)?"IS_SPACE":"NOT_SPACE") << endl;
    fseek(f, fp - 1, SEEK_SET);
    fp = ftell(f);
    c = fgetc(f);
  }
  cout << "n_white: " << n_white << endl;
  cout << "fp: " << ftell(f) << endl;

  if(n_white == 0){
    cout << "\nNo whitespace at end of file.\ndone" << endl;
    return 0;
  }

  // back up the file, if default not over-ridden
  if(argc == 2){
    str cmd(str("cp ") + str(argv[1]) + str(" ") + str(argv[1]) + str(".bak"));
    cout << "Creating file backup:\n\t" << cmd << endl;
    system(cmd.c_str());
  }

  // run the command to truncate the file (removing extra whitespace)
  str cmd(str("truncate -s-") + str(std::to_string(n_white)) + str(" ") + str(argv[1]));
  cout << "Truncating file to remove whitespace: \n\t" << cmd << endl;
  system(cmd.c_str());
  cout << "done" << endl;

  return 0;
}
