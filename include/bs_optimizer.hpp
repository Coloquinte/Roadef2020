
#include "problem.hpp"

#include <random>

using Rgen = std::mt19937;

class BsOptimizer {
  public:
    BsOptimizer(Problem &pb, RoadefParams params);

    void run();

    // Beam initialization
    void resetBeam();
    void resetBeamPartial(int backtrackSize);

    // Beam exploration
    void runBeam();
    std::vector<int> getInterventionOrder();
    void expandBeam(int intervention);
    void recordSolution();

    bool solutionFound() const;

    void logBeamStart() const;
    void logBeamEnd() const;

  private:
    Problem &pb;

    std::vector<int> bestStartTimes;
    Problem::Objective bestObj;

    std::vector<std::vector<int> > beam;

    Rgen rgen;
    RoadefParams params;
    int beamWidth;
};

