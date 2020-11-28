
#include "bs_analyzer.hpp"
#include "measures.hpp"

#include <iostream>
#include <algorithm>
#include <cassert>

using namespace std;

BsAnalyzer::BsAnalyzer(Problem &pb, RoadefParams params) : pb(pb), params(params) {
    rgen = Rgen(params.seed);
    targetStartTimes = pb.startTimes();
}

void BsAnalyzer::showStats() {
    cout << "Intervention    "
         << "\tMeanRiskSpan\tMeanRiskSpanObj"
         << "\tAverageDemand"
         << "\tAverageDuration"
         << "\tValidity\tValidityObj"
         << endl;
    vector<double> spanMeanRisk = measureSpanMeanRisk(pb);
    vector<double> averageDemand = measureAverageDemand(pb);
    vector<double> averageDuration = measureAverageDuration(pb);
    vector<double> validTimestepRatio = measureValidTimestepRatio(pb);
    vector<double> spanMeanRiskObj = measureSpanMeanRisk(pb, pb.riskValue());
    vector<double> validTimestepRatioObj = measureValidTimestepRatio(pb, pb.riskValue());
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        cout << pb.interventionNames()[i] << "  "
             << "\t" << spanMeanRisk[i]
             << "\t" << spanMeanRiskObj[i]
             << "\t" << averageDemand[i]
             << "\t" << averageDuration[i]
             << "\t" << validTimestepRatio[i]
             << "\t" << validTimestepRatioObj[i]
             << endl;
    }
}

void BsAnalyzer::run() {
    showStats();
    for (int i = 0; i < 100; ++i) {
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
    vector<double> spanMeanRisk = measureSpanMeanRisk(pb);
    vector<double> averageDemand = measureAverageDemand(pb);
    vector<double> averageDuration = measureAverageDuration(pb);
    vector<double> validTimestepRatio = measureValidTimestepRatio(pb);
    vector<double> spanMeanRiskObj = measureSpanMeanRisk(pb, pb.riskValue());
    vector<double> validTimestepRatioObj = measureValidTimestepRatio(pb, pb.riskValue());
    cout << endl << endl << endl;
    cout << "Intervention    \tAveragePos"
         << "\tMeanRiskSpan\tMeanRiskSpanObj"
         << "\tAverageDemand"
         << "\tAverageDuration"
         << "\tValidity\tValidityObj"
         << endl;
    for (auto p : sortedInter) {
        cout << pb.interventionNames()[p.second]
             << "\t" << p.first
             << "\t" << spanMeanRisk[p.second]
             << "\t" << spanMeanRiskObj[p.second]
             << "\t" << averageDemand[p.second]
             << "\t" << averageDuration[p.second]
             << "\t" << validTimestepRatio[p.second]
             << "\t" << validTimestepRatioObj[p.second]
        << endl;
    }
    cout << endl << endl << endl;
}

void BsAnalyzer::runSearch() {
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
    pb.set(bestIntervention, targetStartTimes[bestIntervention]);
    orders.back().push_back(bestIntervention);
}

