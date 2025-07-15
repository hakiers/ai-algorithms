#ifndef CSVREADER_H
#define CSVREADER_H

#include <string>
#include <vector>

class CSVReader {
private:
    std::string filename;
    char delimiter;

public:
    CSVReader(const std::string &filename, char delimiter = ',') : filename(filename), delimiter(delimiter) {};
    void readData(std::vector<std::vector<double>> &, std::vector<double> &);
};

#endif
