// testbench.cpp

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>
#include "SDT.h"
#include "parameters.h"

void sdt_predict(pt_t x_pt[n_objects], eta_t x_eta[n_objects], phi_t x_phi[n_objects], score_t score[n_classes]);

std::vector<std::vector<double>> read_csv(const char* path){
    std::vector<std::vector<double>> data;
    std::ifstream file(path);
    std::string line;
    while(std::getline(file, line)){
        std::vector<double> row;
        std::stringstream ss(line);
        std::string val;
        while(std::getline(ss, val, ',')){
            row.push_back(std::atof(val.c_str()));
        }
        data.push_back(row);
    }
    return data;
}

int main(){
    auto X = read_csv("/eos/home-i00/d/dhnaik/C2V_event_training_data/batch/X_batch.csv");   // n_events x 57
    auto Y = read_csv("/eos/home-i00/d/dhnaik/C2V_event_training_data/batch/y_batch.csv");   // n_events x 1

    int n_events = X.size();
    int correct = 0;
    int correct_flipped = 0;

    for(int e = 0; e < n_events; e++){
        pt_t x_pt[n_objects];
        eta_t x_eta[n_objects];
        phi_t x_phi[n_objects];
        score_t score[n_classes];

        for(int k = 0; k < n_objects; k++){
            x_pt[k]  = X[e][3*k];
            x_eta[k] = X[e][3*k + 1];
            x_phi[k] = X[e][3*k + 2];
            /*printf("obj %d: pt=%f eta=%f phi=%f\n", k, (double)x_pt[k], (double)x_eta[k], (double)x_phi[k]);*/
        }

        sdt_predict(x_pt, x_eta, x_phi, score);

        
        int pred         = (score[1] > score[0]) ? 1 : 0;
        int truth = (int)Y[e][0];

        if(pred == truth) correct++;

        /*printf("event %d: score=[%f, %f] pred=%d truth=%d\n",
               e, (double)score[0], (double)score[1], pred, truth);*/
    }

    printf("\nAccuracy (normal):  %d/%d = %f\n", correct, n_events, (double)correct / n_events);
    return 0;
}
