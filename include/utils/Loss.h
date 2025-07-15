#ifndef LOSS_H
#define LOSS_H
#include <vector>

double mseLoss(const std::vector<double> &, const std::vector<double> &);

double maeLoss(const std::vector<double> &, const std::vector<double> &);

#endif //LOSS_H
