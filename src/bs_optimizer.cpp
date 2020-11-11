
#include "bs_optimizer.hpp"

#include <iostream>
#include <algorithm>
#include <unordered_set>

using namespace std;

BsOptimizer::BsOptimizer(Problem &pb, RoadefParams params) : pb(pb), params(params) {
    rgen = Rgen(params.seed);
}

void BsOptimizer::run() {
    for (int i = 0; i < 200; ++i) {
        runAttempt();
    }
    analyze();
}

void BsOptimizer::runAttempt() {
    pb.reset();
    vector<int> interventions = getInterventionOrder();
    for (int i : interventions) {
        makeDecision(i);
    }
    recordSolution();
}

void BsOptimizer::recordSolution() {
    allSolutions.push_back(pb.startTimes());
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

vector<int> BsOptimizer::getInterventionOrder() {
    vector<int> order;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        order.push_back(i);
    }
    shuffle(order.begin(), order.end(), rgen);
    return order;
}

void BsOptimizer::analyze() {
    vector<int> nbDifferent(pb.nbInterventions());
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        unordered_set<int> times;
        for (int s = 0; s < allSolutions.size(); ++s) {
            times.insert(allSolutions[s][i]);
        }
        nbDifferent[i] = times.size();
    }
    int maxDiff = *max_element(nbDifferent.begin(), nbDifferent.end());
    int minDiff = *min_element(nbDifferent.begin(), nbDifferent.end());
    double avg = 0;
    for (int n : nbDifferent) {
        avg += n / (double) pb.nbInterventions();
    }
    cout << "Out of " << pb.nbTimesteps() << " timesteps, "
         << "average " << avg << " different times "
         << "(" << minDiff << "-" << maxDiff << ")"
         << " after " << allSolutions.size() << " trials"
         << endl;
}
