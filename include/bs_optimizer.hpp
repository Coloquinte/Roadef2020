
#include "problem.hpp"

#include <random>

using Rgen = std::mt19937;

class BsOptimizer {
  public:
    BsOptimizer(Problem &pb, RoadefParams params);

    void initSolution();
    void run();

    // Beam initialization
    void restartBeam(int restartDepth);

    // Random parameters
    int getBeamWidth();
    int getBeamWidthFixed();
    int getBeamWidthRandomUniform();
    int getBeamWidthRandomGeom();
    int getBacktrackDepth();
    int getBacktrackDepthFixed();
    int getBacktrackDepthRandomUniform();
    int getBacktrackDepthRandomGeom();
    int getRestartDepth();
    int getRestartDepthFixed();
    int getRestartDepthRandomUniform();
    int getRestartDepthRandomGeom();
    std::vector<int> getSearchPriority();
    std::vector<int> getSearchPriorityDemandRanking();
    std::vector<int> getSearchPriorityRiskRanking();
    std::vector<int> getSearchPriorityRandom();
    std::vector<int> getRestartPriority();
    std::vector<int> getRestartPriorityConflicts();
    std::vector<int> getRestartPriorityTimesteps();
    std::vector<int> getRestartPriorityRandom();
    std::vector<int> getInterventionOrderRandom();
    std::vector<int> getTimestepOrderRandom();

    // Beam exploration
    void runBeam(int beamWidth, int backtrackDepth);
    void backtrackBeam(int beamWidth, int backtrackDepth);
    void expandBeam(int intervention, int beamWidth);
    void recordSolution();

    bool solutionFound() const;
    bool alreadyAssigned(int intervention) const;

    void logBeamStart() const;
    void logBeamEnd() const;
    void logSearchEnd() const;

  private:
    Problem &pb;

    std::vector<int> bestStartTimes;
    Problem::Objective bestObj;

    std::vector<std::vector<int> > beam;

    std::vector<std::vector<size_t> > assignmentCounts;

    Rgen rgen;
    RoadefParams params;
};

