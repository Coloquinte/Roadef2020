
#include "bs_analyzer.hpp"

#include <iostream>
#include <algorithm>

using namespace std;

BsAnalyzer::BsAnalyzer(Problem &pb, RoadefParams params) : pb(pb), params(params) {
    rgen = Rgen(params.seed);
    targetStartTimes = pb.startTimes();
}

void BsAnalyzer::run() {
    for (int i = 0; i < 1; ++i) {
        runSearch();
    }
    pb.reset(targetStartTimes);
}

void BsAnalyzer::runSearch() {
    cout << "Looking for a search matching the target times" << endl;
    pb.reset();
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        runStep();
    }
}

void BsAnalyzer::runStep() {
    vector<int> ranks(pb.nbInterventions(), -1);
    vector<vector<pair<Problem::Objective, int> > > rankings(pb.nbInterventions());
    int bestIntervention = -1;
    int bestRank = -1;
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
                if (bestIntervention == -1 || rank < bestRank) {
                    bestIntervention = i;
                    bestRank = rank;
                }
            }
        }
    }
    cout << "\tIntervention is " << bestIntervention
         << ",\t rank " << bestRank
         << "\t (time " << targetStartTimes[bestIntervention] << ")"
         << endl;
    pb.set(bestIntervention, targetStartTimes[bestIntervention]);
}

