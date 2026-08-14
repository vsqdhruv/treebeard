//SDT.h

#ifndef BDT_H__
#define BDT_H__

#include "ap_fixed.h"
#include "hls_math.h"
#include <cstring>
#include <functional>

namespace BDT{

/* ---
* Balanced tree reduce implementation.
* Reduces an array of inputs to a single value using the template binary operator 'Op',
* for example summing all elements with OpAdd, or finding the maximum with OpMax
* Use only when the input array is fully unrolled. Or, slice out a fully unrolled section
* before applying and accumulate the result over the rolled dimension.
* Required for emulation to guarantee equality of ordering.
* --- */

// compile-time helpers used to build balanced binary reduction tree in reduce<T,N,Op>
constexpr int floorlog2(int x) { return (x < 2) ? 0 : 1 + floorlog2(x / 2); }
constexpr int pow2(int x) { return x == 0 ? 1 : 2 * pow2(x - 1); }

/*takes array of N values and combines them into one using balanced binary tree of Op applications
* instead of a sequential left-to-right fold - guarantees deterministic ordering, as float addition isnt associative*/ 
template <class T, int N, class Op> T reduce(const T *x, Op op) {
  static constexpr int leftN = pow2(floorlog2(N - 1)) > 0 ? pow2(floorlog2(N - 1)) : 0;
  static constexpr int rightN = N - leftN > 0 ? N - leftN : 0;
  if (N == 1) { return x[0]; }
  if (N == 2) { return op(x[0], x[1]); }

  return op(reduce<T, leftN, Op>(x, op), reduce<T, rightN, Op>(x + leftN, op));
}

// passed into reduce as combining operation - what makes reduce a SUM and not a min or a max
template<class T> class OpAdd {
  public:
    T operator()(T a, T b) { return a + b; }
};

// sigmoid LUT - need to see whether this gets mapped to LUT or BRAM logic blocks on firmware after csynth report
template <class data_T, class table_t, int table_size, int range>
void init_sigmoid_table(table_t table_out[table_size]){
  for(int i = 0; i < table_size; i++){
    float in_val = 2.0 * range * (i - float(table_size) / 2.0) / table_size;
    table_out[i] = 1.0 / (1 + hls::exp(-in_val));
  }
} 

template<class data_T, class table_t, int table_size, int range>
table_t sigmoid_lut(data_T x){
  #ifdef __HLS_SYN__
    bool initialised = false;
    table_t sigmoid_table[table_size];
  #else 
    static bool initialised = false;
    static table_t sigmoid_table[table_size];
  #endif
  if(!initialised){
    init_sigmoid_table<data_T, table_t, table_size, range>(sigmoid_table);
    initialised = true;
  }

  int index = x * table_size / (2 * range) + table_size / 2;
  /*printf("sigmoid_lut: x=%f, raw_index=%d\n", (double)x, index);*/
  if(index < 0) index = 0;
  if(index > table_size - 1) index = table_size - 1;
  return sigmoid_table[index];
}
// Number of trees given number of classes
constexpr int fn_classes(int n_classes){
  return n_classes == 2 ? 1 : n_classes;
}

// modified Tree structure for SDT
template<int n_tree, int n_nodes, int n_leaves, int n_features, int n_classes, int n_objects, 
         class score_t, class weight_t, class logit_t, class gate_t,
         class pt_t, class eta_t, class phi_t>
struct Tree {
public:
  int feature[n_nodes];

  weight_t bias[n_nodes];
  weight_t weight_pt[n_nodes][n_objects];
  weight_t weight_eta[n_nodes][n_objects];
  weight_t weight_phi[n_nodes][n_objects];

  score_t leaf_probs[n_leaves][n_classes];  // NEW, replaces value[n_nodes]

  int children_left[n_nodes];
  int children_right[n_nodes];
  int parent[n_nodes];

  void decision_function(pt_t x_pt[n_objects], eta_t x_eta[n_objects], phi_t x_phi[n_objects], score_t score[n_classes]) const{ 
    #pragma HLS pipeline II = 1
    #pragma HLS ARRAY_PARTITION variable=feature
    #pragma HLS ARRAY_PARTITION variable=bias
    #pragma HLS ARRAY_PARTITION variable=weight_pt
    #pragma HLS ARRAY_PARTITION variable=weight_eta
    #pragma HLS ARRAY_PARTITION variable=weight_phi
    #pragma HLS ARRAY_PARTITION variable=leaf_probs
    #pragma HLS ARRAY_PARTITION variable=children_left
    #pragma HLS ARRAY_PARTITION variable=children_right
    #pragma HLS ARRAY_PARTITION variable=parent

    gate_t gate[n_nodes];
    gate_t path_prob[n_nodes];
    gate_t path_prob_leaf[n_nodes];

    #pragma HLS ARRAY_PARTITION variable=gate
    #pragma HLS ARRAY_PARTITION variable=path_prob
    #pragma HLS ARRAY_PARTITION variable=path_prob_leaf

    // DIAGNOSTIC ONLY — forces all inputs to have fanout so cosim's post-check
    // doesn't choke on dangling ports (e.g. x_phi_13 unused at depth=2).
    // Remove or narrow once confirmed / once testing a deeper tree.
    /*ap_fixed<32,16> dummy_keepalive = 0;
    for (int k = 0; k < n_objects; k++) {
        dummy_keepalive += x_pt[k] + x_eta[k] + x_phi[k];
    }
    // harmless: multiply by 0 so it can't affect the real result, but creates a genuine dependency
    score[0] += dummy_keepalive * score_t(0);*/

    // Gate: compute sigmoid(w.x), x[0]=1 so weight[i][0] acts as a bias, beta pre-folded
    Gate: for(int i=0; i < n_nodes; i++){
      #pragma HLS unroll
      // only occures with non-leaf nodes
      // negative values mean is leaf ?
      if(feature[i] >= 0){
        //logit_t accumulation = bias[i];
        logit_t accumulation = bias[i];
        for(int k = 0; k < n_objects; k++){
          accumulation += x_pt[k] * weight_pt[i][k]
                        +  x_eta[k] * weight_eta[i][k]
                        +  x_phi[k] * weight_phi[i][k];
        }
        /*if(i == 0){
          printf("C++ node 0 accumulation = %f, bias[0] = %f\n", (double)accumulation, (double)bias[i]);
        }*/
        gate[i] = sigmoid_lut<logit_t , gate_t, 1024, 8>(accumulation); // figure out what happens here
      }else{
        gate[i] = 0; // unused for leaves
      }
    }

    //printf("table[0] = %f, table[1023] = %f\n", (double)sigmoid_lut[0], (double)sigmoid_lut[1023]);

    /*printf("---- gate[] per DFS node-id ----\n");
    for(int i = 0; i < n_nodes; i++){
      printf("node %d: feature=%d gate=%f\n", i, feature[i], (double)gate[i]);
    }*/

    // Activate: propogate path probability root -> leaves
    int iLeaf = 0;
    Activate: for(int i = 0; i < n_nodes; i++){
      #pragma HLS unroll
      // root node always active
      if(i == 0){
        path_prob[i] = 1.0;
      }else{
        if(i == children_left[parent[i]]){
          path_prob[i] = gate[parent[i]] * path_prob[parent[i]];
        }else{
          path_prob[i] = (1 - gate[parent[i]]) * path_prob[parent[i]];
        }
      }
      if(children_left[i] == -1){  // dunno whats goin on here
        path_prob_leaf[iLeaf] = path_prob[i];
        iLeaf++;
      }
    }

    // path probability debug
    /*printf("---- path_prob[] per DFS node-id ----\n");
    for(int i = 0; i < n_nodes; i++){
      printf("node %d: path_prob=%f\n", i, (double)path_prob[i]);
    }*/

    // Select: weighted sum over all leaves and all classes
    Select: for(int c = 0; c < n_classes; c++){
      #pragma HLS unroll
      score_t acc = 0;
      for(int i = 0; i < n_leaves; i++){
        acc += path_prob_leaf[i] * leaf_probs[i][c];
      }
      score[c] = acc;
    }
  }
};


/*
template<int n_trees, int n_classes, int n_features, class input_t, class score_t, class weight_t, class threshold_t>
struct BDT{

public:
  score_t normalisation;
  score_t init_predict[fn_classes(n_classes)];
  OpAdd<score_t> op_add;

  void tree_scores(input_t x, score_t scores[fn_classes(n_classes)][n_trees]) const;

  void decision_function(input_t x, score_t score[fn_classes(n_classes)]) const{
    score_t scores[fn_classes(n_classes)][n_trees];
    #pragma HLS ARRAY_PARTITION variable=scores dim=0
    // Get predictions scores
    tree_scores(x, scores);
    // Reduce
    Reduce:
    for(int j = 0; j < fn_classes(n_classes); j++){
      // Init predictions
      score[j] = init_predict[j];
      // Sum predictions from trees via "reduce" method
      score[j] += reduce<score_t, n_trees, OpAdd<score_t>>(scores[j], op_add);
    }
    // Normalize predictions
    for(int j = 0; j < fn_classes(n_classes); j++){
      score[j] *= normalisation;
    }
  }

};
*/

}
#endif

