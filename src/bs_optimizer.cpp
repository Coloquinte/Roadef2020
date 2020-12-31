
#include "bs_optimizer.hpp"
#include "measures.hpp"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <unordered_set>
#include <cassert>
#include <chrono>

using namespace std;

BsOptimizer::BsOptimizer(Problem &pb, RoadefParams params) : pb(pb), params(params) {
    rgen = Rgen(params.seed);
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        assignmentCounts.push_back(vector<size_t>(pb.maxStartTime(i), 0));
    }
}

void BsOptimizer::run() {
    initSolution();
    while (true) {
        if (chrono::steady_clock::now() >= params.endTime) {
            break;
        }
        int restartDepth = getRestartDepth();
        restartBeam(restartDepth);
        int beamWidth = getBeamWidth();
        int backtrackDepth = getBacktrackDepth();
        runBeam(beamWidth, backtrackDepth);
        recordSolution();
        pb.reset(bestStartTimes);
    }
    logSearchEnd();
}

void BsOptimizer::runBeam(int beamWidth, int backtrackDepth) {
    assert (!beam.empty());
    logBeamStart();
    vector<int> interventions = getSearchPriority();
    for (int i : interventions) {
        if (!alreadyAssigned(i)) {
            backtrackBeam(beamWidth, backtrackDepth);
            expandBeam(i, beamWidth);
        }
    }
    logBeamEnd();
}

void BsOptimizer::backtrackBeam(int beamWidth, int backtrackDepth) {
    assert (!beam.empty());
    for (int i = 0; i < backtrackDepth; ++i) {
        int intervention = uniform_int_distribution<int>(0, pb.nbInterventions()-1)(rgen);
        if (beam[0][intervention] == -1) continue;
        for (vector<int> &elt : beam) {
            elt[intervention] = -1;
        }
        // Avoid creating duplicates
        sort(beam.begin(), beam.end());
        beam.erase(unique(beam.begin(), beam.end()), beam.end());
        expandBeam(intervention, beamWidth);
    }
}

void BsOptimizer::logBeamStart() const {
    assert (!beam.empty());
    if (params.verbosity < 4) return;
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
    if (params.verbosity < 4) return;
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

void BsOptimizer::logSearchEnd() const {
    if (params.verbosity < 4) return;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        size_t totCount = 0;
        for (size_t count : assignmentCounts[i]) {
            totCount += count;
        }
        vector<pair<double, int> > timeCounts;
        for (int t = 0; t < pb.maxStartTime(i); ++t) {
            if (assignmentCounts[i][t] > 0) {
                timeCounts.emplace_back(100.0 * assignmentCounts[i][t] / totCount, t);
            }
        }
        sort(timeCounts.begin(), timeCounts.end(), greater<>());
        cout << pb.interventionNames()[i] << ": " << timeCounts.size() << " times" << endl;
        for (auto p : timeCounts) {
            if (p.first > 0.1) {
                cout << "\t" << p.second << ":\t" << p.first << "%" << endl;
            }
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
            if (params.verbosity >= 3) {
                cout << "\tParameters were: "
                     << "restart depth: " << choiceRestartDepth << ", "
                     << "backtrack depth: " << choiceBacktrackDepth << ", "
                     << "beam width: " << choiceBeamWidth << ", "
                     << "restart priority: " << choiceRestartPriority << ", "
                     << "search priority: " << choiceSearchPriority
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

void BsOptimizer::restartBeam(int restartDepth) {
    pb.reset();
    beam.clear();
    if (solutionFound()) {
        vector<int> startTimes = bestStartTimes;
        vector<int> interventions = getRestartPriority();
        for (int i = 0; i < restartDepth && i < pb.nbInterventions(); ++i) {
            startTimes[interventions[i]] = -1;
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

bool BsOptimizer::alreadyAssigned(int intervention) const {
    assert (!beam.empty());
    return beam[0][intervention] != -1;
}

void BsOptimizer::expandBeam(int intervention, int beamWidth) {
    vector<NextBeamElement> beamTrials;
    for (int i = 0; i < beam.size(); ++i) {
        assert (beam[i][intervention] == -1);
        pb.set(beam[i]);
        for (int t = 0; t < pb.maxStartTime(intervention); ++t) {
            // Insert in the sorted vector
            Problem::Objective threshold = beamTrials.size() < beamWidth ? Problem::Objective() : beamTrials.back().obj;
            NextBeamElement elt (i, t, pb.objectiveIfSet(intervention, t, threshold));
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
        ++assignmentCounts[intervention][elt.time];
    }
    beam = newBeam;
}

int BsOptimizer::getBeamWidth() {
    int beamWidth = getBeamWidthRandomGeom();
    choiceBeamWidth = beamWidth;
    return beamWidth;
}

int BsOptimizer::getBeamWidthFixed() {
    return params.beamWidth;
}

int BsOptimizer::getBeamWidthRandomUniform() {
    return uniform_int_distribution<int>(1, 2*params.beamWidth-1)(rgen);
}

int BsOptimizer::getBeamWidthRandomGeom() {
    double mean = params.beamWidth;
    if (mean <= 1.0 + 1e-8) return 1;
    return 1 + geometric_distribution<int>(1.0/mean)(rgen);
}

int BsOptimizer::getBacktrackDepth() {
    int backtrackDepth = getBacktrackDepthRandomGeom();
    choiceBacktrackDepth = backtrackDepth;
    return backtrackDepth;
}

int BsOptimizer::getBacktrackDepthFixed() {
    return params.backtrackDepth;
}

int BsOptimizer::getBacktrackDepthRandomUniform() {
    double mean = params.backtrackDepth;
    if (mean < 0.5) return 0;
    return uniform_int_distribution<int>(0, 2*params.backtrackDepth)(rgen);
}

int BsOptimizer::getBacktrackDepthRandomGeom() {
    double mean = params.backtrackDepth;
    if (mean <= 0.00001) return 0;
    return geometric_distribution<int>(1.0/(1.0+mean))(rgen);
}

int BsOptimizer::getRestartDepth() {
    int restartDepth = getRestartDepthMixed();
    choiceRestartDepth = restartDepth;
    return restartDepth;
}

int BsOptimizer::getRestartDepthFixed() {
    return params.restartDepth;
}

int BsOptimizer::getRestartDepthRandomUniform() {
    return uniform_int_distribution<int>(1, 2*params.restartDepth-1)(rgen);
}

int BsOptimizer::getRestartDepthMixed() {
    // Mix small backtracks (size specified) and large restarts (restart about half)
    double smallMean = params.restartDepth;
    double largeMean = pb.nbInterventions() / 2.0;
    double ratio = 2.0; // Spend 2x more times on small
    double probaSmall = ratio * largeMean / (smallMean + ratio * largeMean);
    double mean = largeMean;
    if (uniform_real_distribution<double>()(rgen) < probaSmall) {
        mean = smallMean;
    }
    if (mean <= 1.0 + 1e-8) return 1;
    return 1 + geometric_distribution<int>(1.0/mean)(rgen);
}

int BsOptimizer::getRestartDepthRandomGeom() {
    double mean = params.restartDepth;
    if (mean <= 1.0 + 1e-8) return 1;
    return 1 + geometric_distribution<int>(1.0/mean)(rgen);
}

vector<int> BsOptimizer::getSearchPriority() {
    int choice = uniform_int_distribution<int>(0, 2)(rgen);
    choiceSearchPriority = choice;
    if (choice == 0) {
        return getSearchPriorityRandom();
    }
    else if (choice == 1) {
        return getSearchPriorityDemandRanking();
    }
    else {
        return getSearchPriorityRiskRanking();
    }
}

vector<int> rankingFromMeasure(const vector<double> &measure) {
    vector<pair<double, int> > costs;
    for (int i = 0; i < measure.size(); ++i) {
        costs.emplace_back(-measure[i], i);
    }
    sort(costs.begin(), costs.end());
    vector<int> order;
    for (auto p : costs) {
        order.push_back(p.second);
    }
    return order;
}

vector<int> BsOptimizer::getSearchPriorityRiskRanking() {
    return rankingFromMeasure(measureSpanMeanRisk(pb));
}

vector<int> BsOptimizer::getSearchPriorityDemandRanking() {
    return rankingFromMeasure(measureAverageDemand(pb));
}

vector<int> BsOptimizer::getSearchPriorityRandom() {
    return getInterventionOrderRandom();
}

vector<int> BsOptimizer::getRestartPriority() {
    int choice = uniform_int_distribution<int>(0, 2)(rgen);
    choiceRestartPriority = choice;
    if (choice == 0) {
        return getRestartPriorityRandom();
    }
    else if (choice == 1) {
        return getRestartPriorityConflicts();
    }
    else {
        return getRestartPriorityTimesteps();
    }
}

vector<int> BsOptimizer::getRestartPriorityConflicts() {
    // Pick the most used timesteps for an intervention
    // Restart all interventions that are in conflict with those
    if (!solutionFound()) return getInterventionOrderRandom();
    vector<int> interventions = getInterventionOrderRandom();
    vector<char> interventionSeen(pb.nbInterventions(), 0);
    vector<int> order;
    vector<int> toVisit = interventions;
    while (!toVisit.empty()) {
        int i = toVisit.back();
        toVisit.pop_back();
        if (interventionSeen[i]) {
            continue;
        }
        interventionSeen[i] = 1;
        order.push_back(i);
        vector<pair<size_t, int> > toSort;
        for (int t = 0; t < pb.maxStartTime(i); ++t) {
            toSort.emplace_back(assignmentCounts[i][t], t);
        }
        sort(toSort.begin(), toSort.end(), greater<>());
        for (int j = 0; j < toSort.size(); ++j) {
            int startTime = toSort[j].second;
            if (startTime == bestStartTimes[i]) {
                // Do not free the current best start time
                continue;
            }
            if (toSort[j].first == 0 || toSort[j].first < 0.05 * toSort[0].first || j >= 8) {
                // Only free "likely" start times
                break;
            }
            vector<int> interventionsPresent;
            for (int t = startTime; t < startTime + pb.duration(i, startTime); ++t) {
                for (int p : pb.presence(t)) {
                    interventionsPresent.push_back(p);
                }
            }
            shuffle(interventionsPresent.begin(), interventionsPresent.end(), rgen);
            for (int k : interventionsPresent) {
                if (!interventionSeen[k]) {
                    toVisit.push_back(k);
                }
            }
        }
    }
    vector<int> remaining = getInterventionOrderRandom();
    for (int i : remaining) {
        if (!interventionSeen[i]) {
            order.push_back(i);
            interventionSeen[i] = 1;
        }
    }
    return order;
}

vector<int> BsOptimizer::getRestartPriorityTimesteps() {
    // Pick the timesteps in random order
    // Restart all interventions that are present here
    if (!solutionFound()) return getInterventionOrderRandom();
    vector<int> timesteps = getTimestepOrderRandom();
    vector<char> interventionSeen(pb.nbInterventions(), 0);
    vector<int> order;
    for (int t : timesteps) {
        vector<int> interventionsPresent = pb.presence(t);
        shuffle(interventionsPresent.begin(), interventionsPresent.end(), rgen);
        for (int i : interventionsPresent) {
            if (!interventionSeen[i]) {
                order.push_back(i);
                interventionSeen[i] = 1;
            }
        }
    }
    vector<int> remaining = getInterventionOrderRandom();
    for (int i : remaining) {
        if (!interventionSeen[i]) {
            order.push_back(i);
            interventionSeen[i] = 1;
        }
    }
    return order;
}

vector<int> BsOptimizer::getRestartPriorityRandom() {
    return getInterventionOrderRandom();
}

vector<int> BsOptimizer::getInterventionOrderRandom() {
    vector<int> order;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        order.push_back(i);
    }
    shuffle(order.begin(), order.end(), rgen);
    return order;
}

vector<int> BsOptimizer::getTimestepOrderRandom() {
    vector<int> order;
    for (int i = 0; i < pb.nbTimesteps(); ++i) {
        order.push_back(i);
    }
    shuffle(order.begin(), order.end(), rgen);
    return order;
}
