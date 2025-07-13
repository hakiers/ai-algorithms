#ifndef LOSS_H
#define LOSS_H
#include <vector>

long double mseLoss(const std::vector<long double> &, const std::vector<long double> &);

long double maeLoss(const std::vector<long double> &, const std::vector<long double> &);

#endif //LOSS_H
