
#include "problem.hpp"

#include <random>

using Rgen = std::mt19937;

class BsOptimizer {
  public:
    BsOptimizer(Problem &pb, RoadefParams params);

    void run();
    void runAttempt();

    std::vector<int> getInterventionOrder();
    void initBeam();
    void expandBeam(int intervention);
    void recordSolution();

    void analyze();

  private:
    Problem &pb;

    std::vector<int> bestStartTimes;
    Problem::Objective bestObj;

    std::vector<std::vector<int> > beam;

    std::vector<std::vector<int> > allSolutions;

    Rgen rgen;
    RoadefParams params;
    int beamWidth;
};

