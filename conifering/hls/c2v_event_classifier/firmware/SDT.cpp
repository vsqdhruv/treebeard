//SDT.cpp

#include "SDT.h"
#include "parameters.h"

/*
bool (*split_fn)(const input_t*, const threshold_t*) = !strcmp(splitting_convention,"<=") ? [](const input_t *a, const threshold_t *b) { return *a <= *b; } : [](const input_t *a, const threshold_t *b) { return *a < *b;};

template<>
void BDT::BDT<n_trees, n_classes, n_features, input_arr_t, score_t, weight_t, threshold_t>::tree_scores(input_arr_t x, score_t scores[fn_classes(n_classes)][n_trees]) const {
  scores[0][0] = tree_0_0.decision_function(x, split_fn);
}
*/ 

void sdt_predict(pt_t x_pt[n_objects], 
                eta_t x_eta[n_objects],
                phi_t x_phi[n_objects],
                score_t score[n_classes]){
                  tree_0_0.decision_function(x_pt, x_eta, x_phi, score);
                }

