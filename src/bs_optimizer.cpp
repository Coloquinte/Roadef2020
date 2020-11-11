
#include "bs_optimizer.hpp"

#include <iostream>
#include <algorithm>

BsOptimizer::BsOptimizer(Problem &pb, RoadefParams params) : pb(pb), params(params) {
    rgen = Rgen(params.seed);
}

void BsOptimizer::run() {
    for (int i = 0; i < 1000; ++i) {
        runAttempt();
    }
}

void BsOptimizer::runAttempt() {
    pb.reset();
    std::vector<int> interventions = getInterventionOrder();
    for (int i : interventions) {
        makeDecision(i);
    }
    if (pb.objective() < bestObj || bestStartTimes.empty()) {
        pb.writeSolutionFile(params.solution);
        bestObj = pb.objective();
        bestStartTimes = pb.startTimes();
    }
}

void BsOptimizer::makeDecision(int intervention) {
    int bestTime = 0;
    Problem::Objective bestDecisionObj;
    for (int t = 0; t < pb.maxStartTime(intervention); ++t) {
        pb.set(intervention, t);
        if (t == 0 || pb.objective() < bestDecisionObj) {
            bestTime = t;
            bestDecisionObj = pb.objective();
        }
        pb.unset(intervention);
    }
    pb.set(intervention, bestTime);
}

std::vector<int> BsOptimizer::getInterventionOrder() {
    std::vector<int> order;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        order.push_back(i);
    }
    std::shuffle(order.begin(), order.end(), rgen);
    return order;
}
