#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <vector>

class LinearRegression;

class Optimizer {
    public:
    virtual ~Optimizer() {}

    virtual void optimize(LinearRegression &model,
                      const std::vector<std::vector<long double>> &X,
                      const std::vector<long double> &Y,
                      int epochs) = 0;
};

#endif //OPTIMIZER_H
