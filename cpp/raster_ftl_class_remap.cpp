/*
 
   Mapping rules:

   And your mapping rules:

Rule	Target Class
Starts with "C"	--> "C"
Contains "D-1/2" -->	"D"
Contains "S" --> "S"
Contains "N" -->	"N"
Contains "O-1" -->	"O"
Contains "M-1/2" -->	"M"
Exactly "W"	--> "W"


Original Class	Matches Rule	Target Class
C-7	Starts with C	C
D-1/2	Contains D-1/2	D
C-5	Starts with C	C
C-3	Starts with C	C
B01_N	Contains N	N
O-1a/b	Contains O-1	O
M-1/2	Contains M-1/2	M
S-1	Starts with S	S
B03_N	Contains N	N
B26_O-1a/b	Contains O-1	O
W	Equals W	W
N	Contains N	N
B71_S-2	contains S S 
C-2	Starts with C	C
B21_D-1/2	Contains D-1/2	D
B71_O-1a/b	Contains O-1	O
B46_M-1/2	Contains M-1/2	M
B46_D-1/2	Contains D-1/2	D
B26_D-1/2	Contains D-1/2	D
S-3	Starts with S	S
S-2	Starts with S	S
C-4	Starts with C	C
C-6	Starts with C	C
*/


#include "misc.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>

using namespace std;

// Parse class names from ENVI header
vector<string> read_class_names(const string& hdr_file) {
    ifstream hf(hdr_file);
    if (!hf.is_open()) err("Failed to open header file");

    string line;
    vector<string> class_names;
    bool in_class_block = false;

    while (getline(hf, line)) {
        if (line.find("class names") != string::npos) {
            in_class_block = true;
            continue;
        }
        if (in_class_block) {
            if (line.find("}") != string::npos) break;
            // Remove commas and trim whitespace
            size_t start = line.find_first_not_of(" \t,");
            size_t end = line.find_last_not_of(" \t,\n\r");
            if (start != string::npos && end != string::npos)
                class_names.push_back(line.substr(start, end - start + 1));
        }
    }

    hf.close();
    return class_names;
}

// Mapping logic: fine class name -> coarse group
string map_class(const string& name) {
    if (name == "W") return "W";
    if (name.find("D-1/2") != string::npos) return "D";
    if (name.find("M-1/2") != string::npos) return "M";
    if (name.find("O-1") != string::npos) return "O";
    if (name.find("N") != string::npos) return "N";
    if (name.rfind("C", 0) == 0) return "C"; // starts with 'C'
    if (name.rfind("S", 0) == 0) return "S"; // starts with 'S'
    return "UNKNOWN";
}

int main(int argc, char** argv) {
    if (argc < 2) err("Usage: raster_ftl_class_remap [input classification raster]");

    string fn(argv[1]);
    string hfn = hdr_fn(fn);
    size_t nrow, ncol, nband;
    hread(hfn, nrow, ncol, nband);

    if (nband != 1) err("Input must be single-band classification image");

    // Load classification data
    float* in = bread(fn, nrow, ncol, nband);
    size_t np = nrow * ncol;

    // Parse class names
    vector<string> class_names = read_class_names(hfn);
    if (class_names.empty()) err("No class names found in header");

    // Build fine -> coarse class ID map
    unordered_map<string, int> coarse_class_ids;
    vector<string> coarse_class_names;
    unordered_map<int, int> remap; // fine class ID -> coarse class ID
    int next_coarse_id = 0;

    for (size_t i = 0; i < class_names.size(); ++i) {
        string coarse = map_class(class_names[i]);

        if (coarse_class_ids.find(coarse) == coarse_class_ids.end()) {
            coarse_class_ids[coarse] = next_coarse_id++;
            coarse_class_names.push_back(coarse);
        }

        remap[i] = coarse_class_ids[coarse];
    }

    // Remap pixels
    float* out = (float*)malloc(np * sizeof(float));
    if (!out) err("Memory allocation failed");

    for (size_t i = 0; i < np; ++i) {
        int val = static_cast<int>(in[i]);
        if (std::isnan(in[i]) || val < 0 || val >= (int)class_names.size()) {
            out[i] = NAN;
        } else {
            out[i] = static_cast<float>(remap[val]);
        }
    }

    // Write output raster
    string ofn = fn + "_remap.bin";
    string ohfn = fn + "_remap.hdr";
    hwrite(ohfn, nrow, ncol, 1); // single-band

    // Append coarse class names to header
    ofstream hf(ohfn, ios::app);
    hf << "class names = {\n";
    for (size_t i = 0; i < coarse_class_names.size(); ++i) {
        hf << "  " << coarse_class_names[i];
        if (i != coarse_class_names.size() - 1) hf << ",\n";
        else hf << "\n";
    }
    hf << "}" << endl;
    hf.close();

    // Write binary data
    FILE* f = fopen(ofn.c_str(), "wb");
    if (!f) err("Failed to open output file for writing");
    fwrite(out, sizeof(float), np, f);
    fclose(f);

    free(in);
    free(out);

    printf("Done: %s\n", ofn.c_str());
    return 0;
}

