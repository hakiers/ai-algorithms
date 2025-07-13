#include "Loss.h"
#include <stdexcept>
#include <math.h>

long double mseLoss(const std::vector<long double> &input, const std::vector<long double> &target){
    if(input.size() != target.size()) throw std::invalid_argument("input size mismatch");
    if(input.size() == 0) return 0;

    long double loss = 0;
    for(int i = 0; i < input.size(); i++){
        long double diff = input[i] - target[i];
        loss += diff * diff;
    }

    return loss/input.size();
}

long double maeLoss(const std::vector<long double> &input, const std::vector<long double> &target){
    if(input.size() != target.size()) throw std::invalid_argument("input size mismatch");
    if(input.size() == 0) return 0;

    long double loss = 0;
    for(int i = 0; i < input.size(); i++){
        long double diff = input[i] - target[i];
        loss += abs(diff);
    }

    return loss/input.size();
}

