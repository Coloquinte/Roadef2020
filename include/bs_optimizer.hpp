
#include "move.hpp"

class BsOptimizer {
  public:
    BsOptimizer(Problem &pb, RoadefParams params);

    void run();
    void runAttempt();
    void makeDecision(int intervention);
    std::vector<int> getInterventionOrder();

  private:
    Problem &pb;

    std::vector<int> bestStartTimes;
    Problem::Objective bestObj;

    Rgen rgen;
    RoadefParams params;
};

