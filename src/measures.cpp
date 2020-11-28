
#include "measures.hpp"

#include <algorithm>
#include <limits>
#include <cmath>

using namespace std;

vector<double> normalizeVector(const vector<double> &vec) {
    if (vec.empty()) return vec;
    double squaredNorm = 0.0;
    for (double v : vec) {
        squaredNorm += v * v;
    }
    double factor = 1.0 / sqrt(squaredNorm / vec.size() + 1.0e-8);
    vector<double> ret;
    for (double v : vec) {
        ret.push_back(factor * v);
    }
    return ret;
}

vector<double> minPossibleRisk(const Problem &pb) {
    vector<double> minRisks;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        double minRisk = *min_element(pb.meanRisk().contribs()[i].begin(), pb.meanRisk().contribs()[i].end());
        minRisks.push_back(minRisk);
    }
    return minRisks;
}

vector<double> maxAllowedRisk(const Problem &pb, double objectiveBound) {
    vector<double> minRisks = minPossibleRisk(pb);
    double totalMinRisk = 0.0;
    for (double r : minRisks) {
        totalMinRisk += r;
    }
    vector<double> maxAllowed;
    for (double r : minRisks) {
        maxAllowed.push_back(objectiveBound + totalMinRisk - r);
    }
    return maxAllowed;
}

vector<double> measureSpanMeanRisk(const Problem &pb) {
    return measureSpanMeanRisk(pb, numeric_limits<double>::infinity());
}

vector<double> measureSpanMeanRisk(const Problem &pb, double objectiveBound) {
    vector<double> maxAllowed = maxAllowedRisk(pb, objectiveBound);
    vector<double> ret;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        auto p = minmax_element(pb.meanRisk().contribs()[i].begin(), pb.meanRisk().contribs()[i].end());
        double maxVal = min(*p.second, maxAllowed[i]);
        double minVal = *p.first;
        ret.push_back(maxVal - minVal);
    }
    return normalizeVector(ret);
}

vector<double> measureSpanMinRisk(const Problem &pb) {
    return measureSpanMinRisk(pb, numeric_limits<double>::infinity());
}

vector<double> measureSpanMinRisk(const Problem &pb, double objectiveBound) {
    vector<double> ret;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        vector<double> vals;
        for (const auto &contribs: pb.quantileRisk().contribs()[i]) {
            double val = 0.0;
            for (const auto &c : contribs) {
                val += *min_element(c.risks.begin(), c.risks.end());
            }
            vals.push_back(val);
        }
        auto p = minmax_element(vals.begin(), vals.end());
        ret.push_back(*p.second - *p.first);
    }
    return normalizeVector(ret);
}

vector<double> measureSpanMedianRisk(const Problem &pb) {
    return measureSpanMedianRisk(pb, numeric_limits<double>::infinity());
}

vector<double> measureSpanMedianRisk(const Problem &pb, double objectiveBound) {
    vector<double> ret;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        vector<double> vals;
        for (const auto &contribs: pb.quantileRisk().contribs()[i]) {
            double val = 0.0;
            for (const auto &c : contribs) {
                vector<double> risks = c.risks;
                int pos = ceil(0.5 * risks.size()) - 1;
                std::nth_element(risks.begin(), risks.begin() + pos, risks.end());
                val += risks[pos];
            }
            vals.push_back(val);
        }
        auto p = minmax_element(vals.begin(), vals.end());
        ret.push_back(*p.second - *p.first);
    }
    return normalizeVector(ret);
}

vector<double> measureSpanQuantileRisk(const Problem &pb) {
    return measureSpanQuantileRisk(pb, numeric_limits<double>::infinity());
}

vector<double> measureSpanQuantileRisk(const Problem &pb, double objectiveBound) {
    vector<double> ret;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        vector<double> vals;
        for (const auto &contribs: pb.quantileRisk().contribs()[i]) {
            double val = 0.0;
            for (const auto &c : contribs) {
                vector<double> risks = c.risks;
                int pos = ceil(pb.quantile() * risks.size()) - 1;
                std::nth_element(risks.begin(), risks.begin() + pos, risks.end());
                val += risks[pos];
            }
            vals.push_back(val);
        }
        auto p = minmax_element(vals.begin(), vals.end());
        ret.push_back(*p.second - *p.first);
    }
    return normalizeVector(ret);
}


vector<double> measureSpanMaxRisk(const Problem &pb) {
    return measureSpanMaxRisk(pb, numeric_limits<double>::infinity());
}

vector<double> measureSpanMaxRisk(const Problem &pb, double objectiveBound) {
    vector<double> ret;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        vector<double> vals;
        for (const auto &contribs: pb.quantileRisk().contribs()[i]) {
            double val = 0.0;
            for (const auto &c : contribs) {
                val += *max_element(c.risks.begin(), c.risks.end());
            }
            vals.push_back(val);
        }
        auto p = minmax_element(vals.begin(), vals.end());
        ret.push_back(*p.second - *p.first);
    }
    return normalizeVector(ret);
}

vector<double> measureAverageDemand(const Problem &pb) {
    return measureAverageDemand(pb, numeric_limits<double>::infinity());
}

vector<double> measureAverageDemand(const Problem &pb, double objectiveBound) {
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
    return normalizeVector(ret);
}

vector<double> measureAverageDuration(const Problem &pb) {
    return measureAverageDuration(pb, numeric_limits<double>::infinity());
}

vector<double> measureAverageDuration(const Problem &pb, double objectiveBound) {
    vector<double> ret;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        double sumDurations = 0.0;
        for (int t = 0; t < pb.maxStartTime(i); ++t) {
            sumDurations += pb.duration(i, t);
        }
        ret.push_back(sumDurations / pb.maxStartTime(i));
    }
    return normalizeVector(ret);
}

vector<double> measureValidTimestepRatio(const Problem &pb) {
    return measureValidTimestepRatio(pb, numeric_limits<double>::infinity());
}

vector<double> measureValidTimestepRatio(const Problem &pb, double objectiveBound) {
    vector<vector<int> > timesteps = validTimesteps(pb, objectiveBound);
    vector<double> ret;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        ret.push_back(timesteps[i].size() / (double) pb.maxStartTime(i));
    }
    return ret;
}

vector<vector<int> > validTimesteps(const Problem &pb) {
    return validTimesteps(pb, numeric_limits<double>::infinity());
}

vector<vector<int> > validTimesteps(const Problem &pb, double objectiveBound) {
    vector<double> maxAllowed = maxAllowedRisk(pb, objectiveBound);
    vector<vector<int> > ret;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        vector<int> validTimesteps;
        for (int startTime = 0; startTime < pb.maxStartTime(i); ++startTime) {
            for (Resources::ResourceContribution demand : pb.resources().demands()[i][startTime]) {
                if (demand.amount > pb.resources().upperBound()[demand.resource][demand.time] + 1.0e-5) {
                    continue;
                }
            }
            if (pb.meanRisk().contribs()[i][startTime] > maxAllowed[i]) {
                continue;
            }
            validTimesteps.push_back(startTime);
        }
        ret.push_back(validTimesteps);
    }
    return ret;
}

