#include "SDT.h"
#include "parameters.h"
#include "my_prj.h"
#include "hls_stream.h"

void my_prj(pt_arr_t x_pt, eta_arr_t x_eta, phi_arr_t x_phi, score_arr_t score){
  #pragma HLS array_partition variable=x_pt
  #pragma HLS array_partition variable=x_eta
  #pragma HLS array_partition variable=x_phi
  #pragma HLS array_partition variable=score
  #pragma HLS pipeline
  tree_0_0.decision_function(x_pt, x_eta, x_phi, score);
}

// void load(int N, accelerator_input_t* x, hls::stream<input_t>& x_stream){
//   for(int n = 0; n < N; n++){
//     for(int i = 0; i < n_features; i++){
//       #pragma HLS pipeline
//       input_t xi = x[n * n_features + i];
//       x_stream.write(xi);
//     }
//   }
// }

// void compute(int N, hls::stream<input_t>& x_stream, hls::stream<score_t>& score_stream){
//   for(int n = 0; n < N; n++){
//     input_arr_t x_int;
//     score_arr_t score_int;
//     for(int i = 0; i < n_features; i++){
//       #pragma HLS pipeline
//       x_int[i] = x_stream.read();
//     }
//     my_prj(x_int, score_int);
//     for(int i = 0; i < BDT::fn_classes(n_classes); i++){
//       #pragma HLS pipeline
//       score_stream.write(score_int[i]);
//     }
//   }
// }

// void store(int N, hls::stream<score_t>& score_stream, accelerator_output_t* score){
//   for(int n = 0; n < N; n++){
//     for(int i = 0; i < BDT::fn_classes(n_classes); i++){
//       #pragma HLS pipeline
//       score_t scorei = score_stream.read();
//       score[n * BDT::fn_classes(n_classes) + i] = scorei;
//     }
//   }
// }

// void my_prj_accelerator(int N, int& n_f, int& n_c, accelerator_input_t* x, accelerator_output_t* score){
//   #pragma HLS dataflow
//   n_f = n_features;
//   n_c = BDT::fn_classes(n_classes);

//   hls::stream<input_t> x_stream("x_stream");
//   hls::stream<score_t> score_stream("score_stream");
//   #pragma HLS STREAM variable=x_stream depth=1024
//   #pragma HLS STREAM variable=score_stream depth=1024

//   load(N, x, x_stream);
//   compute(N, x_stream, score_stream);
//   store(N, score_stream, score);
// }