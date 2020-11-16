
#include "problem.hpp"

#include <random>

using Rgen = std::mt19937;

class BsOptimizer {
  public:
    BsOptimizer(Problem &pb, RoadefParams params);

    void initSolution();
    void run();

    // Beam initialization
    void resetBeamPartial(int backtrackDepth);

    // Random parameters
    int getBeamWidth();
    int getBeamWidthFixed();
    int getBeamWidthRandomUniform();
    int getBeamWidthRandomGeom();
    int getBacktrackDepth();
    int getBacktrackDepthFixed();
    int getBacktrackDepthRandomUniform();
    int getBacktrackDepthRandomGeom();
    std::vector<int> getInterventionOrder();
    std::vector<int> getInterventionOrderRanking();
    std::vector<int> getInterventionOrderRandom();

    // Beam exploration
    void runBeam(int beamWidth);
    void expandBeam(int intervention, int beamWidth);
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
};

