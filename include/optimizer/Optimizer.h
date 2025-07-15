#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <vector>

class LinearRegression;

class Optimizer {
    public:
    virtual ~Optimizer() {}

    virtual void optimize(LinearRegression &,
                      const std::vector<std::vector<double>> &,
                      const std::vector<double> &,
                      int) = 0;
};

#endif //OPTIMIZER_H
