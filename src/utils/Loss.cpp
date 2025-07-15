#include "utils/Loss.h"
#include <stdexcept>
#include <math.h>

double mseLoss(const std::vector<double> &input, const std::vector<double> &target){
    if(input.size() != target.size()) throw std::invalid_argument("input size mismatch");
    if(input.size() == 0) return 0;

    double loss = 0;
    for(int i = 0; i < input.size(); i++){
        double diff = input[i] - target[i];
        loss += diff * diff;
    }

    return loss/input.size();
}

double maeLoss(const std::vector<double> &input, const std::vector<double> &target){
    if(input.size() != target.size()) throw std::invalid_argument("input size mismatch");
    if(input.size() == 0) return 0;

    double loss = 0;
    for(int i = 0; i < input.size(); i++){
        double diff = input[i] - target[i];
        loss += abs(diff);
    }

    return loss/input.size();
}

