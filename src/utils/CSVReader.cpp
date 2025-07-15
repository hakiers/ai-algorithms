#include "utils/CSVReader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

void CSVReader::readData(std::vector<std::vector<double>> &features, std::vector<double> &outputs) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;

	//# skip headers
	std::getline(file, line);



    while (std::getline(file, line)) {
        std::vector<double> rowFeatures;
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cells;

        while (std::getline(ss, cell, delimiter)) {
            cells.push_back(cell);
        }

        if (cells.size() < 2) {
            throw std::runtime_error("Invalid CSV row format — expected at least 2 columns.");
        }

        for (size_t i = 0; i < cells.size() - 1; ++i) {
            rowFeatures.push_back(std::stod(cells[i]));
        }

        outputs.push_back(std::stod(cells.back()));

        features.push_back(rowFeatures);
    }

    file.close();
}
