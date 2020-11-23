
#include "measures.hpp"

#include <algorithm>

using namespace std;

vector<double> measureSpanMeanRisk(const Problem &pb) {
    vector<double> ret;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        auto p = minmax_element(pb.meanRisk().contribs()[i].begin(), pb.meanRisk().contribs()[i].end());
        ret.push_back(*p.second - *p.first);
    }
    return ret;
}

vector<double> measureAverageDemand(const Problem &pb) {
    vector<double> averageCapacities(pb.nbResources(), 0.0);
    for (int i = 0; i < pb.nbResources(); ++i) {
        double cap = 0.0;
        for (double b : pb.resources().upperBound()[i]) {
            cap += b;
        }
        averageCapacities[i] = cap / pb.nbTimesteps();
    }
    vector<double> ret;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        vector<double> totDemand(pb.nbResources(), 0.0);
        for (const auto &demands : pb.resources().demands()[i]) {
            for (const auto &d : demands) {
                totDemand[d.resource] += d.amount;
            }
        }
        double normalizedDemand = 0.0;
        for (int j = 0; j < pb.nbResources(); ++j) {
            normalizedDemand += min(totDemand[j] / averageCapacities[j], 1.0);
        }
        ret.push_back(normalizedDemand / pb.maxStartTime(i));
    }
    return ret;
}

vector<double> measureAverageDuration(const Problem &pb) {
    vector<double> ret;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        double sumDurations = 0.0;
        for (int t = 0; t < pb.maxStartTime(i); ++t) {
            sumDurations += pb.duration(i, t);
        }
        ret.push_back(sumDurations / pb.maxStartTime(i));
    }
    return ret;
}

vector<double> measureValidTimestepRatio(const Problem &pb) {
    vector<vector<int> > timesteps = validTimesteps(pb);
    vector<double> ret;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        ret.push_back(timesteps[i].size() / (double) pb.maxStartTime(i));
    }
    return ret;
}

vector<vector<int> > validTimesteps(const Problem &pb) {
    vector<vector<int> > ret;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        vector<int> validTimesteps;
        for (int startTime = 0; startTime < pb.maxStartTime(i); ++startTime) {
            for (Resources::ResourceContribution demand : pb.resources().demands()[i][startTime]) {
                if (demand.amount > pb.resources().upperBound()[demand.resource][demand.time] + 1.0e-5) {
                    continue;
                }
            }
            validTimesteps.push_back(startTime);
        }
        ret.push_back(validTimesteps);
    }
    return ret;
}

