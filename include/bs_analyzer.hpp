
#include "problem.hpp"

#include <random>

using Rgen = std::mt19937;

class BsAnalyzer {
  public:
    BsAnalyzer(Problem &pb, RoadefParams params);

    void run();
    void runSearch();
    void runStep();

  private:
    Problem &pb;

    std::vector<int> targetStartTimes;

    Rgen rgen;
    RoadefParams params;
};

