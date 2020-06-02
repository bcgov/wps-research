/* cat in-place: append files onto another file: e.g., use case: if wanted to use cat to combine files, but insuff. space! */
#include"misc.h"

int main(int argc, char ** argv){
  if(argc < 3) err("cat_append [file: cat onto] [file] .. [another file?] ..");

  int i;
  for0(i, argc){
    if(i > 0){
      if(!exists(argv[i])){
        err(str("file does not exist: ") + str(argv[i]));
      }
      else{
        cout << argv[i] << endl;
      }
    }
  }

  str name;
  cout << "concatenate other files onto: " << argv[1] << endl;
  cout << "press RETURN or ctrl-c to exit";
  getline(std::cin, name);

  FILE * f = fopen(argv[1], "ab"); // open file, append mode!
  for0(i, argc){
    if(i > 1){
      FILE * g = fopen(argv[i], "rb");
      char s;
      while(1){
        s = fgetc(g);
        if(s == EOF){
          break;
        }
        fwrite(&s, sizeof(char), 1, f);
      }
      fclose(g);
    }
  }
  fclose(f);

  return 0;
}
