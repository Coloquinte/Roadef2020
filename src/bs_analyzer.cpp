
#include "bs_analyzer.hpp"

#include <iostream>
#include <algorithm>
#include <cassert>

using namespace std;

BsAnalyzer::BsAnalyzer(Problem &pb, RoadefParams params) : pb(pb), params(params) {
    rgen = Rgen(params.seed);
    targetStartTimes = pb.startTimes();
}

void BsAnalyzer::run() {
    for (int i = 0; i < 10; ++i) {
        runSearch();
    }
    pb.reset(targetStartTimes);
    // Now analyze everything we can
    vector<double> averageRank(pb.nbInterventions(), 0.0);
    for (auto order : orders) {
        for (int r = 0; r < pb.nbInterventions(); ++r) {
            averageRank[order[r]] += ((double) r) / orders.size();
        }
    }
    vector<pair<double, int> > sortedInter;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        sortedInter.emplace_back(averageRank[i], i);
    }
    sort(sortedInter.begin(), sortedInter.end());
    vector<double> spanMeanRisk = pb.measureSpanMeanRisk();
    vector<double> averageDemand = pb.measureAverageDemand();
    for (auto p : sortedInter) {
        cout << "Intervention " << p.second
        << ": \taverage " << p.first
        << ", " << pb.maxStartTime(p.second) << " valid timesteps"
        << ", " << spanMeanRisk[p.second] << " mean risk span"
        << ", " << averageDemand[p.second] << " average demand"
        << endl;
    }
}

void BsAnalyzer::runSearch() {
    //cout << "Looking for a search matching the target times" << endl;
    pb.reset();
    orders.emplace_back();
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        runStep();
    }
}

void BsAnalyzer::runStep() {
    vector<int> ranks(pb.nbInterventions(), -1);
    vector<vector<pair<Problem::Objective, int> > > rankings(pb.nbInterventions());
    vector<int> bestInterventions;
    int bestRank = numeric_limits<int>::max();
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        if (pb.assigned(i)) continue;
        for (int t = 0; t < pb.maxStartTime(i); ++t) {
            pb.set(i, t);
            rankings[i].emplace_back(pb.objective(), t);
            pb.unset(i);
        }
        sort(rankings[i].begin(), rankings[i].end());
        for (int rank = 0; rank < rankings[i].size(); ++rank) {
            if (rankings[i][rank].second == targetStartTimes[i]) {
                ranks[i] = rank;
                if (rank < bestRank) {
                    bestInterventions.clear();
                    bestInterventions.push_back(i);
                    bestRank = rank;
                }
                else if (rank == bestRank) {
                    bestInterventions.push_back(i);
                }
            }
        }
    }
    assert (!bestInterventions.empty());
    shuffle(bestInterventions.begin(), bestInterventions.end(), rgen);
    int bestIntervention = bestInterventions.front();
    //cout << "\tIntervention is " << bestIntervention
    //     << ",\t rank " << bestRank
    //     << ",\t out of " << bestInterventions.size() << " candidates"
    //     << ",\t time " << targetStartTimes[bestIntervention]
    //     << endl;
    pb.set(bestIntervention, targetStartTimes[bestIntervention]);
    orders.back().push_back(bestIntervention);
}

