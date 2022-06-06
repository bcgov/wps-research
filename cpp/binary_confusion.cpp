/* calculate confusion matrix for a binary class map with respect to truth

Assume that no relabeling is required

map tp, tn, fn, fp!

20220605: note: stats are calculated only where classification data raster, aren't nodata
*/
#include"misc.h"
int main(int argc, char ** argv){
  if(argc < 3){
    err("binary_confusion [input binary class file name] [input ground reference file name]");
  }

  float d, c_tn, c_tp, c_fn, c_fp, TN, TP, FN, FP, N, P; // confusion matrix params
  c_tn = c_tp = c_fn = c_fp = 0.;

  str fn(argv[1]); // input file name
  str hfn(hdr_fn(fn)); // auto-detect header file name
  size_t nrow, ncol, nband, np, i, j;
  hread(hfn, nrow, ncol, nband); // read header

  size_t nrow2, ncol2, nband2;
  str cfn(argv[2]);
  str chfn(hdr_fn(cfn));
  hread(chfn, nrow2, ncol2, nband2);

  // check the inputs are the same shape
  if(nrow != nrow2 || ncol != ncol2 || nband != nband2){
    err("please check dimensional consistency of data in headers");
  }

  np = nrow * ncol;
  if(nband != 1) err("this program defines results for 1-band images");

  // read data into float array
  float * dat = bread(fn, nrow, ncol, nband);
  float * cdat = bread(cfn, nrow, ncol, nband);

  str tnfn(fn + str("_tn.bin"));
  str tnhfn(fn+ str("_tn.hdr"));

  str tpfn(fn + str("_tp.bin"));
  str tphfn(fn+ str("_tp.hdr"));

  str fnfn(fn + str("_fn.bin"));
  str fnhfn(fn+ str("_fn.hdr"));

  str fpfn(fn + str("_fp.bin"));
  str fphfn(fn+ str("_fp.hdr"));

  
  TN = TP = FN = FP = 0.;
  // true negatives
  hwrite(tnhfn, nrow, ncol, 1);
  FILE * f = fopen(tnfn.c_str(), "wb");
  for0(i, np){
    d = dat[i];
    if(isnan(d)){
      d = NAN;
    }
    else{
      d = ((dat[i] == cdat[i] && dat[i] == 0.)? 1. : 0);
      if(cdat[i] == 1){
	      P += 1.;
      }
      if(cdat[i] == 0.){
	      N += 1.;
      }
    }
    fwrite(&d, sizeof(float), 1, f);
    if(d == 1.) TN += 1.;
  }
  fclose(f);

  // true positives
  hwrite(tphfn, nrow, ncol, 1);
  f = fopen(tpfn.c_str(), "wb");
  for0(i, np){
    d = dat[i];
    if(isnan(d)){
      d = NAN;
    }
    else{
      d = ((dat[i] == cdat[i] && dat[i] == 1.) ? 1. : 0);
    }
    fwrite(&d, sizeof(float), 1, f);
    if(d == 1.) TP += 1.;
  }
  fclose(f);

  // false negatives
  hwrite(fnhfn, nrow, ncol, 1);
  f = fopen(fnfn.c_str(), "wb");
  for0(i, np){
    d = dat[i];
    if(isnan(d)){
      d = NAN;
    }
    else{
      d = ((dat[i] != cdat[i] && dat[i] == 0.) ? 1. : 0);
    }
    fwrite(&d, sizeof(float), 1, f);
    if(d == 1.) FN += 1.;
  }
  fclose(f);

  //false positives
  hwrite(fphfn, nrow, ncol, 1);
  f = fopen(fpfn.c_str(), "wb");
  for0(i, np){
    d = dat[i];
    if(isnan(d)){
      d = NAN;
    }
    else{
      d = ((dat[i] != cdat[i] && dat[i] == 1.) ? 1. : 0);
    }
    fwrite(&d, sizeof(float), 1, f);
    if(d == 1.) FP += 1.;
  }
  fclose(f);

  printf("P,N,TP,TN,FP,FN\n");
  float fnp = 100. / ((float)np);
  printf("%f,%f,%f,%f,%f,%f\n", P*fnp, N*fnp, TP*fnp, TN*fnp, FP*fnp, FN*fnp);

  printf("TPR=%f\n", TP/P); // true positive rate
  printf("TNR=%f\n", TN/N); // true negative rate
  printf("PPV=%f\n", TP/(TP + FP)); // positive predictive value
  printf("NPV=%f\n", TN/(TN + FN)); // negative predictive value
  printf("FNR=%f\n", FN/P); // false negative rate
  printf("FPR=%f\n", FP/N); // false positive rate
  printf("FDR=%f\n", FP/(FP + TP)); // false discovery rate
  printf("FOR=%f\n", FN/(FN + TN)); // false omission rate
  printf("PLR=%f\n", (TP/P) / (FP/N)); // positive likelihood ratio
  printf("NLR=%f\n", (FN/P) / (TN/N)); // negative likelihood ratio
  printf("ACC=%f\n", (TP + TN) / (P + N)); // accuracy
  printf("BA =%f\n", ((TP/P) + (TN/N))/2.); // balanced accuracy 
  printf("F1 =%f\n", 2.* TP / ((2.* TP) + FP + FN)); // f1 score
  printf("BM =%f\n", (TP/P) + (TN/N) + 1.); // bookmaker informedness
  printf("MK =%f\n", ( TP/(TP + FP)) + (TN/(TN + FN)) -1.); // markedness

  return 0;
}
