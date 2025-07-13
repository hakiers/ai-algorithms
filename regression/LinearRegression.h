#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <vector>

class Optimizer;

class LinearRegression {
    friend class Optimizer;
    private:
        std::vector<double> weights;
        double bias;

        bool is_trained;
    public:
        LinearRegression() : bias(0), is_trained(false) {}
        ~LinearRegression() = default;

        double predict(const std::vector<double> &) const;
        std::vector<double> predict(const std::vector<std::vector<double>> &) const;

        void fit(const std::vector<std::vector<double>> &, const std::vector<double> &,
                 Optimizer &, int);

        std::vector<double> getWeights() const;
        LinearRegression& setWeights(const std::vector<double> &);

        double getBias() const;
        LinearRegression& setBias(double);

        LinearRegression& reset();
};



#endif //LINEARREGRESSION_H
