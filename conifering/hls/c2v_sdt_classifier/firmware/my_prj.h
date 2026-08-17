#ifndef MY_PRJ_H_
#define MY_PRJ_H_

#include "SDT.h"
#include "parameters.h"


// Prototype of top level function for C-synthesis
void my_prj(
	pt_arr_t x_pt,
	eta_arr_t x_eta,
	phi_arr_t x_phi,
	score_arr_t score);
	// score_t tree_scores[BDT::fn_classes(n_classes)]
#endif
