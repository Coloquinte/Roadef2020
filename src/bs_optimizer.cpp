
#include "bs_optimizer.hpp"

#include <iostream>
#include <algorithm>
#include <unordered_set>

using namespace std;

BsOptimizer::BsOptimizer(Problem &pb, RoadefParams params) : pb(pb), params(params) {
    rgen = Rgen(params.seed);
    beamWidth = 100;
}

void BsOptimizer::run() {
    for (int i = 0; i < 200; ++i) {
        runAttempt();
    }
    pb.reset(bestStartTimes);
    //analyze();
}

void BsOptimizer::runAttempt() {
    pb.reset();
    vector<int> interventions = getInterventionOrder();
    initBeam();
    for (int i : interventions) {
        expandBeam(i);
    }
    recordSolution();
}

void BsOptimizer::recordSolution() {
    for (vector<int> startTimes : beam) {
        pb.reset(startTimes);
        allSolutions.push_back(startTimes);
        if (pb.objective() < bestObj || bestStartTimes.empty()) {
            pb.writeSolutionFile(params.solution);
            bestObj = pb.objective();
            bestStartTimes = startTimes;
            cout << "New solution with "
                 << pb.exclusionValue() << " exclusions, "
                 << pb.resourceValue() << " overflow, "
                 << pb.riskValue() << " risk "
                 << "(" << pb.meanRiskValue() << " + " << pb.quantileRiskValue() << ")"
                 << endl;
        }
    }
}

struct NextBeamElement {
    int node;
    int time;
    Problem::Objective obj;

    NextBeamElement(int n, int t, Problem::Objective o) : node(n), time(t), obj(o) {}

    bool operator<(const NextBeamElement &o) const { return obj < o.obj; }
};

void BsOptimizer::initBeam() {
    beam.clear();
    for (int i = 0; i < beamWidth; ++i) {
        beam.push_back(vector<int>(pb.nbInterventions(), -1));
    }
}

void BsOptimizer::expandBeam(int intervention) {
    vector<NextBeamElement> beamTrials;
    for (int i = 0; i < beamWidth; ++i) {
        pb.set(beam[i]);
        for (int t = 0; t < pb.maxStartTime(intervention); ++t) {
            pb.set(intervention, t);
            beamTrials.emplace_back(i, t, pb.objective());
            pb.unset(intervention);
        }
    }
    sort(beamTrials.begin(), beamTrials.end());
    vector<vector<int> > newBeam;
    for (int i = 0; i < beamWidth && i < beamTrials.size(); ++i) {
        NextBeamElement elt = beamTrials[i];
        vector<int> node = beam[elt.node];
        node[intervention] = elt.time;
        newBeam.push_back(node);
    }
    beam = newBeam;
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
