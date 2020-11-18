
#include "bs_optimizer.hpp"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <unordered_set>
#include <cassert>
#include <chrono>

using namespace std;

BsOptimizer::BsOptimizer(Problem &pb, RoadefParams params) : pb(pb), params(params) {
    rgen = Rgen(params.seed);
}

void BsOptimizer::run() {
    initSolution();
    while (true) {
        if (chrono::steady_clock::now() >= params.endTime) {
            break;
        }
        int restartDepth = getRestartDepth();
        resetBeam(restartDepth);
        int beamWidth = getBeamWidth();
        runBeam(beamWidth);
    }
    if (solutionFound()) {
        pb.reset(bestStartTimes);
    }
}

void BsOptimizer::runBeam(int beamWidth) {
    assert (!beam.empty());
    logBeamStart();
    vector<int> interventions = getInterventionOrder();
    for (int i : interventions) {
        backtrackBeam(beamWidth, getBacktrackDepth());
        expandBeam(i, beamWidth);
    }
    logBeamEnd();
    recordSolution();
}

void BsOptimizer::backtrackBeam(int beamWidth, int depth) {
    assert (!beam.empty());
    int intervention = uniform_int_distribution<int>(0, pb.nbInterventions()-1)(rgen);
    if (beam[0][intervention] == -1) return;
    for (vector<int> &elt : beam) {
        elt[intervention] = -1;
    }
    // Avoid creating duplicates
    sort(beam.begin(), beam.end());
    beam.erase(unique(beam.begin(), beam.end()), beam.end());
    expandBeam(intervention, beamWidth);
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

void BsOptimizer::initSolution() {
    if (pb.validSolution()) {
        bestObj = pb.objective();
        bestStartTimes = pb.startTimes();
        if (params.verbosity >= 2) {
            cout << "Initial solution with "
                 << pb.exclusionValue() << " exclusions, "
                 << fixed << setprecision(2) << pb.resourceValue() << " overflow, "
                 << fixed << setprecision(5) << pb.riskValue() << " risk "
                 << fixed << setprecision(2) << "(" << pb.meanRiskValue() << " + " << pb.quantileRiskValue() << ")"
                 << endl;
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
                chrono::duration<double> elapsed = chrono::steady_clock::now() - params.startTime;
                cout << "New solution with "
                     << pb.exclusionValue() << " exclusions, "
                     << fixed << setprecision(2) << pb.resourceValue() << " overflow, "
                     << fixed << setprecision(5) << pb.riskValue() << " risk "
                     << fixed << setprecision(2) << "(" << pb.meanRiskValue() << " + " << pb.quantileRiskValue() << ")"
                     << fixed << setprecision(1) << ", elapsed " << elapsed.count() << "s"
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

void BsOptimizer::resetBeam(int restartDepth) {
    pb.reset();
    beam.clear();
    if (solutionFound()) {
        vector<int> startTimes = bestStartTimes;
        for (int i = 0; i < restartDepth; ++i) {
            int intervention = uniform_int_distribution<int>(0, pb.nbInterventions()-1)(rgen);
            startTimes[intervention] = -1;
        }
        beam.push_back(startTimes);
    }
    else {
        beam.push_back(vector<int>(pb.nbInterventions(), -1));
    }
}

bool BsOptimizer::solutionFound() const {
    return !bestStartTimes.empty();
}

void BsOptimizer::expandBeam(int intervention, int beamWidth) {
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

int BsOptimizer::getBeamWidth() {
    return getBeamWidthRandomGeom();
}

int BsOptimizer::getBeamWidthFixed() {
    return params.beamWidth;
}

int BsOptimizer::getBeamWidthRandomUniform() {
    return uniform_int_distribution<int>(1, 2*params.beamWidth-1)(rgen);
}

int BsOptimizer::getBeamWidthRandomGeom() {
    double mean = params.beamWidth;
    return 1 + geometric_distribution<int>(1.0/(mean-1.0))(rgen);
}

int BsOptimizer::getBacktrackDepth() {
    return getBacktrackDepthRandomGeom();
}

int BsOptimizer::getBacktrackDepthFixed() {
    return params.backtrackDepth;
}

int BsOptimizer::getBacktrackDepthRandomUniform() {
    double mean = params.backtrackDepth;
    if (mean <= 0) return 0;
    return uniform_int_distribution<int>(0, 2*params.backtrackDepth)(rgen);
}

int BsOptimizer::getBacktrackDepthRandomGeom() {
    double mean = params.backtrackDepth;
    if (mean <= 0) return 0;
    return geometric_distribution<int>(1.0/mean)(rgen);
}

int BsOptimizer::getRestartDepth() {
    return getRestartDepthRandomGeom();
}

int BsOptimizer::getRestartDepthFixed() {
    return params.restartDepth;
}

int BsOptimizer::getRestartDepthRandomUniform() {
    return uniform_int_distribution<int>(1, 2*params.restartDepth-1)(rgen);
}

int BsOptimizer::getRestartDepthRandomGeom() {
    double mean = params.restartDepth;
    return 1 + geometric_distribution<int>(1.0/(mean-1.0))(rgen);
}

vector<int> BsOptimizer::getInterventionOrder() {
    return getInterventionOrderRanking();
}

vector<int> BsOptimizer::getInterventionOrderRanking() {
    assert (beam.size() >= 1);
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

vector<int> BsOptimizer::getInterventionOrderRandom() {
    assert (beam.size() >= 1);
    vector<int> interventions;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        if (beam[0][i] == -1) {
            interventions.push_back(i);
        }
    }
    shuffle(interventions.begin(), interventions.end(), rgen);
    return interventions;
}

