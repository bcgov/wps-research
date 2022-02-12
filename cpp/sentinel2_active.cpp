#include"misc.h"
int main(int argc, char ** argv){
	size_t nrow, ncol, nband;
	vector<string> s;
	hread(str("swir.hdr"), nrow, ncol, nband, s);
	cout << s << endl;
	return 0;
}


