
#include "bs_optimizer.hpp"

#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <cassert>

using namespace std;

BsOptimizer::BsOptimizer(Problem &pb, RoadefParams params) : pb(pb), params(params) {
    rgen = Rgen(params.seed);
    beamWidth = 10;
}

void BsOptimizer::run() {
    resetBeam();
    runBeam();
    for (int i = 0; i < 1000000; ++i) {
        resetBeamPartial(50);
        runBeam();
    }
    assert (solutionFound());
    pb.reset(bestStartTimes);
}

void BsOptimizer::runBeam() {
    logBeamStart();
    vector<int> interventions = getInterventionOrder();
    for (int i : interventions) {
        expandBeam(i);
    }
    logBeamEnd();
    recordSolution();
}

void BsOptimizer::logBeamStart() const {
    assert (!beam.empty());
    if (params.verbosity < 3) return;
    int cnt = 0;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        if (beam[0][i] == -1) {
            ++cnt;
        }
    }
    cout << "Beam run started with " << cnt << " interventions to assign" << endl;
}

void BsOptimizer::logBeamEnd() const {
    assert (!beam.empty());
    if (params.verbosity < 3) return;
    cout << "Beam run done" << endl;
    vector<int> differences;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        vector<int> values;
        for (int b = 0; b < beam.size(); ++b) {
            values.push_back(beam[b][i]);
        }
        sort(values.begin(), values.end());
        values.erase(unique(values.begin(), values.end()), values.end());
        if (values.size() >= 2) {
            cout << "\tIntervention " << i << ": \t" << values.size() << " different times" << endl;
        }
    }
}

void BsOptimizer::recordSolution() {
    assert (!beam.empty());
    for (vector<int> startTimes : beam) {
        for (int t : startTimes) {
            assert (t >= 0 && t < pb.nbTimesteps());
        }
        pb.reset(startTimes);
        if (pb.objective().betterThan(bestObj) || bestStartTimes.empty()) {
            pb.writeSolutionFile(params.solution);
            bestObj = pb.objective();
            bestStartTimes = startTimes;
            if (params.verbosity >= 2) {
                cout << "New solution with "
                     << pb.exclusionValue() << " exclusions, "
                     << pb.resourceValue() << " overflow, "
                     << pb.riskValue() << " risk "
                     << "(" << pb.meanRiskValue() << " + " << pb.quantileRiskValue() << ")"
                     << endl;
            }
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

void BsOptimizer::resetBeam() {
    pb.reset();
    beam.clear();
    beam.push_back(vector<int>(pb.nbInterventions(), -1));
}

void BsOptimizer::resetBeamPartial(int backtrackSize) {
    assert (solutionFound());
    pb.reset();
    beam.clear();
    vector<int> startTimes = bestStartTimes;
    for (int i = 0; i < backtrackSize; ++i) {
        int intervention = uniform_int_distribution<int>(0, pb.nbInterventions()-1)(rgen);
        startTimes[intervention] = -1;
    }
    beam.push_back(startTimes);
}

bool BsOptimizer::solutionFound() const {
    return !bestStartTimes.empty();
}

void BsOptimizer::expandBeam(int intervention) {
    vector<NextBeamElement> beamTrials;
    for (int i = 0; i < beam.size(); ++i) {
        assert (beam[i][intervention] == -1);
        pb.set(beam[i]);
        for (int t = 0; t < pb.maxStartTime(intervention); ++t) {
            // Insert in the sorted vector
            Problem::Objective threshold = beamTrials.empty() ? Problem::Objective() : beamTrials.back().obj;
            NextBeamElement elt (i, t, pb.objectiveIf(intervention, t, threshold));
            auto it = std::upper_bound(beamTrials.begin(), beamTrials.end(), elt);
            beamTrials.insert(it, elt);
            if (beamTrials.size() > beamWidth) {
                beamTrials.pop_back();
            }
        }
    }
    vector<vector<int> > newBeam;
    for (NextBeamElement elt : beamTrials) {
        vector<int> node = beam[elt.node];
        node[intervention] = elt.time;
        newBeam.push_back(node);
    }
    beam = newBeam;
}

vector<int> BsOptimizer::getInterventionOrder() {
    assert (beam.size() == 1);
    // Only keep interventions that are not set yet
    vector<int> interventions;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        if (beam[0][i] == -1) {
            interventions.push_back(i);
        }
    }
    // Use a heuristic measure to decide the order
    vector<double> ranking = pb.measureSpanMeanRisk();
    vector<pair<double, int> > costs;
    for (int i : interventions) {
        costs.emplace_back(-ranking[i], i);
    }
    sort(costs.begin(), costs.end());
    vector<int> order;
    for (auto p : costs) {
        order.push_back(p.second);
    }
    return order;
}

