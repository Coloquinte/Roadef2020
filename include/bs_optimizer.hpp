
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
    int getRestartDepthMixed();
    int getRestartDepthRandomGeom();

    std::vector<int> getSearchPriority();
    std::vector<int> getRestartPriority();

    std::vector<int> getPriorityDemandRanking();
    std::vector<int> getPriorityRiskRanking();
    std::vector<int> getPriorityConflicts();
    std::vector<int> getPriorityTimesteps();
    std::vector<int> getPriorityOverflowCost();

    std::vector<int> getInterventionOrderRandom();
    std::vector<int> getTimestepOrderRandom();
    std::vector<int> getTimestepOrderOverflowCost();
    std::vector<int> getInterventionOrderFromTimestepOrder(const std::vector<int> &timestepOrder);

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

    // Setup for the current search
    int choiceSearchPriority;
    int choiceRestartPriority;
    int choiceBeamWidth;
    int choiceRestartDepth;
    int choiceBacktrackDepth;
};

