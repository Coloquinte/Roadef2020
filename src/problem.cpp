
#include "problem.hpp"
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>

using namespace std;

const string RESOURCES_STR = "Resources";
const string SEASONS_STR = "Seasons";
const string INTERVENTIONS_STR = "Interventions";
const string EXCLUSIONS_STR = "Exclusions";
const string T_STR = "T";
const string SCENARIO_NUMBER = "Scenarios_number";
const string RESOURCE_CHARGE_STR = "workload";
const string TMAX_STR = "tmax";
const string DELTA_STR = "Delta";
const string MAX_STR = "max";
const string MIN_STR = "min";
const string RISK_STR = "risk";
const string START_STR = "start";
const string QUANTILE_STR = "Quantile";
const string ALPHA_STR = "Alpha";
const string MODEL_STR = "Model";


int readInt(nlohmann::json &j) {
    if (j.is_number()) {
        return j.get<int>();
    }
    else {
        return stoi(j.get<string>());
    }
}

double readDouble(nlohmann::json &j) {
    if (j.is_number()) {
        return j.get<double>();
    }
    else {
        return stod(j.get<string>());
    }
}

string readString(nlohmann::json &j) {
    if (j.is_string()) {
        return j.get<string>();
    }
    else {
        return to_string(j.get<int>());
    }
}


Problem Problem::read(istream &is) {
    Problem pb;
    // Read the JSON
    nlohmann::json js;
    is >> js;

    auto exclusionsJ = js[EXCLUSIONS_STR];
    auto resourcesJ = js[RESOURCES_STR];
    auto interventionsJ = js[INTERVENTIONS_STR];

    // Basic data about the problem
    for (const auto &it : interventionsJ.items()) {
        pb.interventionNames_.push_back(it.key());
    }
    for (const auto &it : resourcesJ.items()) {
        pb.resourceNames_.push_back(it.key());
    }
    pb.nbTimesteps_ = js[T_STR].get<int>();
    for (string intName : pb.interventionNames_) {
        pb.maxStartTimes_.push_back(readInt(interventionsJ[intName][TMAX_STR]));
    }
    double alpha = readDouble(js[ALPHA_STR]);
    double quantile = readDouble(js[QUANTILE_STR]);
    vector<int> scenarioNumbers = js[SCENARIO_NUMBER].get<vector<int> >();

    // Exclusions
    // TODO

    // Resources
    for (string resourceName : pb.resourceNames_) {
        pb.resources_.lowerBound_.push_back(resourcesJ[resourceName][MIN_STR].get<vector<double> >());
        pb.resources_.upperBound_.push_back(resourcesJ[resourceName][MAX_STR].get<vector<double> >());
    }
    for (int maxStartTime : pb.maxStartTimes_) {
        pb.resources_.demands_.push_back(vector<vector<Resources::ResourceContribution> >(maxStartTime));
    }
    for (size_t i = 0; i < pb.interventionNames_.size(); ++i) {
        string interventionName = pb.interventionNames_[i];
        auto interventionWorkload = interventionsJ[interventionName][RESOURCE_CHARGE_STR];
        for (size_t j = 0; j < pb.resourceNames_.size(); ++j) {
            string resourceName = pb.resourceNames_[j];
            if (interventionWorkload.count(resourceName)) {
                for (const auto &elt1 : interventionWorkload[resourceName].items()) {
                    int resourceTime = stoi(elt1.key())-1;
                    for (const auto &elt2 : elt1.value().items()) {
                        int interventionTime = stoi(elt2.key())-1;
                        double usage = elt2.value().get<double>();
                        assert (interventionTime < pb.resources_.demands_[i].size());
                        pb.resources_.demands_[i][interventionTime].emplace_back(usage, resourceTime, j);
                    }
                }
            }
        }
    }

    // Mean risk
    pb.meanRisk_.contribs_.resize(pb.nbInterventions(), vector<double>(pb.nbTimesteps(), 0.0));
    for (size_t i = 0; i < pb.interventionNames_.size(); ++i) {
        string interventionName = pb.interventionNames_[i];
        auto intervention = interventionsJ[interventionName];
        auto interventionRisk = intervention[RISK_STR];
        auto deltaArray = intervention[DELTA_STR];
        for (int startTime = 0; startTime < pb.maxStartTime(i); ++startTime) {
            double meanContrib = 0.0;
            int delta = readInt(deltaArray[startTime]);
            for (int t = startTime; t < startTime + delta; ++t) {
                double factor = alpha;
                factor /= scenarioNumbers[t];
                factor /= pb.nbTimesteps();
                vector<double> riskArray = interventionRisk[to_string(t+1)][to_string(startTime+1)];
                assert (riskArray.size() == scenarioNumbers[t]);
                for (double r : riskArray) {
                    meanContrib += factor * r;
                }
            }
            pb.meanRisk_.contribs_[i][startTime] = meanContrib;
        }
    }

    // Quantile risk
    pb.quantileRisk_.nbScenarios_ = scenarioNumbers;
    for (int num : scenarioNumbers) {
        pb.quantileRisk_.quantileScenarios_.push_back(ceil(num * quantile) - 1);
    }
    for (int maxStartTime : pb.maxStartTimes_) {
        pb.quantileRisk_.contribs_.push_back(vector<vector<QuantileRisk::RiskContribution> >(maxStartTime));
    }
    for (size_t i = 0; i < pb.interventionNames_.size(); ++i) {
        string interventionName = pb.interventionNames_[i];
        auto intervention = interventionsJ[interventionName];
        auto interventionRisk = intervention[RISK_STR];
        auto deltaArray = intervention[DELTA_STR];
        for (int startTime = 0; startTime < pb.maxStartTime(i); ++startTime) {
            int delta = readInt(deltaArray[startTime]);
            for (int t = startTime; t < startTime + delta; ++t) {
                double factor = (1.0 - alpha) / pb.nbTimesteps();
                vector<double> riskArray = interventionRisk[to_string(t+1)][to_string(startTime+1)];
                assert (riskArray.size() == scenarioNumbers[t]);
                for (double &r : riskArray) {
                    r *= factor;
                }
                pb.quantileRisk_.contribs_[i][startTime].emplace_back(t, riskArray);
            }
        }
    }

    return pb;
}

Problem Problem::readFile(const string &fname) {
    ifstream f(fname);
    return read(f);
}

