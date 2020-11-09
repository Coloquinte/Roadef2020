
#include "move.hpp"

class LsOptimizer {
  public:
    LsOptimizer(Problem &pb, const std::string &solutionFilename);

    void run();
    void runReoptRestart(int start);

    void doMove();
    void doSimpleMove();
    void doPerturbation();
    void doCrossover();

    void initMoves();
    void initSimpleMoves();
    void initPerturbations();
    void initSolutions();

    void save(int start);
    void restore(int start);

  private:
    const int nbStarts = 8;
    const int maxMovesPerRestart = 20000;
    const int maxMovesPerReopt = 20000;

    Problem &pb;
    std::string solutionFilename;
    int nbMoves;
    std::vector<std::vector<int> > solutionStartTimes;
    std::vector<Problem::Objective> solutionObjs;
    std::vector<int> bestStartTimes;
    Problem::Objective bestObj;

    std::vector<Move> moves;
    std::vector<Move> simpleMoves;
    std::vector<Move> perturbations;
    Rgen rgen;
};

