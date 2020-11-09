
#include "ls_optimizer.hpp"

#include <iostream>
#include <algorithm>

LsOptimizer::LsOptimizer(Problem &pb, const std::string &solutionFilename) : pb(pb), solutionFilename(solutionFilename) {
    nbMoves = 0;
    rgen = Rgen(std::random_device()());
    initMoves();
    initSimpleMoves();
    initPerturbations();
    initSolutions();
}

void LsOptimizer::initSolutions() {
    for (int i = 0; i < nbStarts; ++i) {
        // TODO: keep initial solution if present
        Move::full().force(pb, rgen);
        solutionObjs.push_back(pb.objective());
        solutionStartTimes.push_back(pb.startTimes());
    }
    bestStartTimes = solutionStartTimes[0];
    bestObj = solutionObjs[0];
}

void LsOptimizer::run() {
    int nbIters = 0;
    int itersToCrossover = 1;
    while (true) {
        ++nbIters;
        for (int i = 0; i < nbStarts; ++i) {
            runReoptRestart(i);
        }
        if (nbIters >= itersToCrossover) {
            doCrossover();
            ++itersToCrossover;
            nbIters = 0;
        }
    }
}

void LsOptimizer::runReoptRestart(int start) {
    restore(start);
    for (int i = 0; i < maxMovesPerReopt; ++i) {
        doMove();
    }
    save(start);
    doPerturbation();
    for (int i = 0; i < maxMovesPerRestart; ++i) {
        doSimpleMove();
    }
    save(start);
}

void LsOptimizer::save(int i) {
    if (pb.objective() <= solutionObjs[i]) {
        solutionStartTimes[i] = pb.startTimes();
        solutionObjs[i] = pb.objective();
    }
    if (pb.objective() < bestObj) {
        bestStartTimes = pb.startTimes();
        bestObj = pb.objective();
        std::cout << "Improved by start #" << i << ": "
              << pb.exclusionValue() << " exclusions, "
              << pb.resourceValue() << " overflow, "
              << pb.riskValue() << " risk (" << pb.meanRiskValue() << " + " << pb.quantileRiskValue() << ")" << std::endl;
        pb.writeSolutionFile(solutionFilename);
    }
}

void LsOptimizer::restore(int i) {
    pb.reset(solutionStartTimes[i]);
}

void LsOptimizer::doMove() {
    std::uniform_int_distribution<int> dist(0, moves.size()-1);
    const Move &move = moves[dist(rgen)];
    ++nbMoves;
    move.apply(pb, rgen);
}

void LsOptimizer::doSimpleMove() {
    std::uniform_int_distribution<int> dist(0, simpleMoves.size()-1);
    const Move &move = simpleMoves[dist(rgen)];
    ++nbMoves;
    move.apply(pb, rgen);
}

void LsOptimizer::doPerturbation() {
    std::uniform_int_distribution<int> dist(0, perturbations.size()-1);
    const Move &move = perturbations[dist(rgen)];
    move.force(pb, rgen);
}

void LsOptimizer::doCrossover() {
    std::uniform_int_distribution<int> dist(0, nbStarts-1);
    std::bernoulli_distribution selDist;
    int start1 = dist(rgen);
    int start2 = dist(rgen);
    std::vector<int> crossover;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        bool sel = selDist(rgen);
        crossover.push_back(sel ? solutionStartTimes[start1][i] : solutionStartTimes[start2][i]);
    }
    int bestStart = std::min_element(solutionObjs.begin(), solutionObjs.end()) - solutionObjs.begin();
    std::uniform_int_distribution<int> replaceDist(0, nbStarts-2);
    int replaced = replaceDist(rgen);
    if (replaced >= bestStart) ++replaced;
    pb.reset(crossover);
    solutionStartTimes[replaced] = crossover;
    solutionObjs[replaced] = pb.objective();
}

void LsOptimizer::initMoves() {
    moves.clear();
    moves.push_back(Move::random(1));
    moves.push_back(Move::random(2));
    moves.push_back(Move::random(3));
    moves.push_back(Move::randomPerturbation(1, 50));
    moves.push_back(Move::randomPerturbation(2, 20));
    moves.push_back(Move::randomPerturbation(3, 10));
    moves.push_back(Move::cycle(2));
    moves.push_back(Move::cycle(3));
    moves.push_back(Move::cycle(4));
    moves.push_back(Move::cyclePerturbation(2, 20));
    moves.push_back(Move::cyclePerturbation(3, 10));
    moves.push_back(Move::path(2));
    moves.push_back(Move::path(3));
    moves.push_back(Move::path(4));
    moves.push_back(Move::same(2, 10));
    moves.push_back(Move::same(3, 10));
    moves.push_back(Move::same(4, 10));
    moves.push_back(Move::closeSame(2, 4, 50));
    moves.push_back(Move::closeSame(3, 4, 20));
    moves.push_back(Move::closeSame(4, 4, 20));
    moves.push_back(Move::closeSame(5, 4, 20));
    moves.push_back(Move::closePerturbation(2, 4, 20));
    moves.push_back(Move::closePerturbation(3, 4, 10));
    moves.push_back(Move::closePerturbation(4, 4, 5));
    moves.push_back(Move::closePerturbation(5, 4, 5));
}

void LsOptimizer::initSimpleMoves() {
    simpleMoves.clear();
    simpleMoves.push_back(Move::random(1));
    simpleMoves.push_back(Move::cycle(2));
    simpleMoves.push_back(Move::path(2));
    simpleMoves.push_back(Move::closeSame(2, 4, 50));
}

void LsOptimizer::initPerturbations() {
    perturbations.clear();
    perturbations.push_back(Move::random(1));
    perturbations.push_back(Move::random(5));
    perturbations.push_back(Move::randomPerturbation(5, 10));
    perturbations.push_back(Move::randomPerturbation(20, 5));
    perturbations.push_back(Move::cycle(5));
    perturbations.push_back(Move::cycle(10));
    perturbations.push_back(Move::closePerturbation(5, 4, 5));
}

