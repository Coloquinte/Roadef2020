
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


int readInt(const nlohmann::json &j) {
    if (j.is_number()) {
        return j.get<int>();
    }
    else {
        return stoi(j.get<string>());
    }
}

double readDouble(const nlohmann::json &j) {
    if (j.is_number()) {
        return j.get<double>();
    }
    else {
        return stod(j.get<string>());
    }
}

string readString(const nlohmann::json &j) {
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
    pb.nbSeasons_ = js[SEASONS_STR].size();
    pb.nbTimesteps_ = readInt(js[T_STR]);
    for (const auto &it : interventionsJ.items()) {
        pb.interventionMappings_.emplace(it.key(), pb.interventionNames_.size());
        pb.interventionNames_.push_back(it.key());
    }
    for (const auto &it : resourcesJ.items()) {
        pb.resourceMappings_.emplace(it.key(), pb.resourceNames_.size());
        pb.resourceNames_.push_back(it.key());
    }
    for (string intName : pb.interventionNames_) {
        pb.maxStartTimes_.push_back(readInt(interventionsJ[intName][TMAX_STR]));
    }
    double alpha = readDouble(js[ALPHA_STR]);
    pb.alpha_ = alpha;
    double quantile = readDouble(js[QUANTILE_STR]);
    pb.quantile_ = quantile;
    vector<int> scenarioNumbers = js[SCENARIO_NUMBER].get<vector<int> >();

    // Exclusions
    pb.exclusions_.seasonInterdictions_.resize(pb.nbSeasons()+1, std::vector<std::vector<int> >(pb.nbInterventions()));
    for (size_t i = 0; i < pb.interventionNames_.size(); ++i) {
        string interventionName = pb.interventionNames_[i];
        auto intervention = interventionsJ[interventionName];
        auto deltaArray = intervention[DELTA_STR];
        assert (deltaArray.size() >= pb.maxStartTime(i));
        vector<int> durations;
        for (int startTime = 0; startTime < pb.maxStartTime(i); ++startTime) {
            durations.push_back(readInt(deltaArray[startTime]));
        }
        pb.exclusions_.durations_.push_back(durations);
    }
    auto exclusions = js[EXCLUSIONS_STR];
    pb.exclusions_.seasons_.resize(pb.nbTimesteps(), -1);
    int seasonCnt = 0;
    for (const auto &elt : js[SEASONS_STR].items()) {
        string seasonName = elt.key();
        for (const auto &exclusion : exclusions.items()) {
            assert (exclusion.value().size() == 3);
            string seasonExcl = exclusion.value()[2].get<string>();
            if (seasonExcl != seasonName) {
                continue;
            }
            int i1 = pb.interventionMappings_.at(exclusion.value()[0].get<string>());
            int i2 = pb.interventionMappings_.at(exclusion.value()[1].get<string>());
            pb.exclusions_.seasonInterdictions_[seasonCnt][i1].push_back(i2);
            pb.exclusions_.seasonInterdictions_[seasonCnt][i2].push_back(i1);
        }
        for (vector<int> &conflicts : pb.exclusions_.seasonInterdictions_[seasonCnt]) {
            sort(conflicts.begin(), conflicts.end());
        }
        for (const auto &timeStr : elt.value()) {
            int t = readInt(timeStr) - 1;
            assert (t >= 0 && t < pb.nbTimesteps());
            assert (pb.exclusions_.seasons_[t] == -1);
            pb.exclusions_.seasons_[t] = seasonCnt;
        }
        ++seasonCnt;
    }
    for (int &season : pb.exclusions_.seasons_) {
        // Create an additional phantom season
        if (season == -1) {
            season = seasonCnt;
        }
    }

    // Resources
    for (string resourceName : pb.resourceNames_) {
        pb.resources_.lowerBound_.push_back(resourcesJ[resourceName][MIN_STR].get<vector<double> >());
        pb.resources_.upperBound_.push_back(resourcesJ[resourceName][MAX_STR].get<vector<double> >());
    }
    // Add some tolerance; slightly less than used by the checker
    const double tolerance = 9.9e-6;
    for (vector<double> &lbs : pb.resources_.lowerBound_) {
        for (double &lb : lbs) {
            lb -= tolerance;
        }
    }
    for (vector<double> &ubs : pb.resources_.upperBound_) {
        for (double &ub : ubs) {
            ub += tolerance;
        }
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

    // Initial solution
    vector<int> startTimes(pb.nbInterventions(), -1);
    pb.reset(startTimes);

    return pb;
}

Problem Problem::readFile(const string &fname) {
    ifstream f(fname);
    return read(f);
}

void Problem::readSolution(istream &is) {
    vector<int> sol(nbInterventions(), -1);
    while (is.good()) {
        string name;
        int startTime;
        is >> name >> startTime;
        if (!is.fail()) {
            sol[interventionMappings_[name]] = startTime - 1;
        }
    }
    reset(sol);
}

void Problem::readSolutionFile(const string &fname) {
    ifstream f(fname);
    return readSolution(f);
}

void Problem::writeSolution(ostream &os) {
    for (int i = 0; i < nbInterventions(); ++i) {
        os << interventionNames_[i] << " " << startTimes_[i] + 1 << endl;
    }
}

void Problem::writeSolutionFile(const string &fname) {
    ofstream f(fname);
    return writeSolution(f);
}

void Exclusions::reset(const std::vector<int> &startTimes) {
    assert (startTimes.size() == nbInterventions());
    currentValue_ = 0;
    currentPresence_.clear();
    currentPresence_.resize(nbTimesteps());
    for (int i = 0; i < startTimes.size(); ++i) {
        if (startTimes[i] != -1) {
            set(i, startTimes[i]);
        }
    }
}

int Exclusions::objectiveIfSet(int intervention, int startTime) {
    set(intervention, startTime);
    int obj = value();
    unset(intervention, startTime);
    return obj;
}

void Exclusions::set(int intervention, int startTime) {
    if (startTime < 0) return;
    assert (intervention >= 0 && intervention < durations_.size());
    assert (startTime >= 0 && startTime < durations_[intervention].size());
    for (int t = startTime; t < startTime + durations_[intervention][startTime]; ++t) {
        assert (t < nbTimesteps());
        const std::vector<int> &interdictions = seasonInterdictions_[seasons_[t]][intervention];
        std::vector<int> &present = currentPresence_[t];
        for (int other : present) {
            bool forbidden = std::find(interdictions.begin(), interdictions.end(), other) != interdictions.end();
            if (forbidden)
                currentValue_ += 1;
        }
        present.push_back(intervention);
    }
}

void Exclusions::unset(int intervention, int startTime) {
    if (startTime < 0) return;
    assert (intervention >= 0 && intervention < durations_.size());
    assert (startTime >= 0 && startTime < durations_[intervention].size());
    for (int t = startTime; t < startTime + durations_[intervention][startTime]; ++t) {
        assert (t < nbTimesteps());
        const std::vector<int> &interdictions = seasonInterdictions_[seasons_[t]][intervention];
        std::vector<int> &present = currentPresence_[t];
        auto it = std::find(present.begin(), present.end(), intervention);
        assert(it != present.end());
        present.erase(it);
        for (int other : present) {
            bool forbidden = std::find(interdictions.begin(), interdictions.end(), other) != interdictions.end();
            if (forbidden)
                currentValue_ -= 1;
        }
    }
}

void Resources::reset(const std::vector<int> &startTimes) {
    assert (startTimes.size() == nbInterventions());
    currentValue_ = 0.0;
    currentUsage_.clear();
    currentUsage_.resize(nbResources(), std::vector<double>(nbTimesteps(), 0.0));
    for (int i = 0; i < startTimes.size(); ++i) {
        if (startTimes[i] != -1) {
            set(i, startTimes[i]);
        }
    }
}

double Resources::objectiveIfSet(int intervention, int startTime) const {
    assert (intervention >= 0 && intervention < demands_.size());
    assert (startTime >= 0 && startTime < demands_[intervention].size());
    double currentValue = currentValue_;
    for (ResourceContribution c : demands_[intervention][startTime]) {
        double lb = lowerBound_[c.resource][c.time];
        double ub = upperBound_[c.resource][c.time];
        double prevUsage = currentUsage_[c.resource][c.time];
        double nextUsage = prevUsage + c.amount;
        double prevCost = std::max(prevUsage - ub, 0.0) + std::max(lb - prevUsage, 0.0);
        double nextCost = std::max(nextUsage - ub, 0.0) + std::max(lb - nextUsage, 0.0);
        currentValue += (nextCost - prevCost);
    }
    return currentValue;
}

void Resources::set(int intervention, int startTime) {
    if (startTime < 0) return;
    assert (intervention >= 0 && intervention < demands_.size());
    assert (startTime >= 0 && startTime < demands_[intervention].size());
    for (ResourceContribution c : demands_[intervention][startTime]) {
        double lb = lowerBound_[c.resource][c.time];
        double ub = upperBound_[c.resource][c.time];
        double prevUsage = currentUsage_[c.resource][c.time];
        double nextUsage = prevUsage + c.amount;
        double prevCost = std::max(prevUsage - ub, 0.0) + std::max(lb - prevUsage, 0.0);
        double nextCost = std::max(nextUsage - ub, 0.0) + std::max(lb - nextUsage, 0.0);
        currentValue_ += (nextCost - prevCost);
        currentUsage_[c.resource][c.time] = nextUsage;
    }
}

void Resources::unset(int intervention, int startTime) {
    if (startTime < 0) return;
    assert (intervention >= 0 && intervention < demands_.size());
    assert (startTime >= 0 && startTime < demands_[intervention].size());
    for (ResourceContribution c : demands_[intervention][startTime]) {
        double lb = lowerBound_[c.resource][c.time];
        double ub = upperBound_[c.resource][c.time];
        double prevUsage = currentUsage_[c.resource][c.time];
        double nextUsage = prevUsage - c.amount;
        double prevCost = std::max(prevUsage - ub, 0.0) + std::max(lb - prevUsage, 0.0);
        double nextCost = std::max(nextUsage - ub, 0.0) + std::max(lb - nextUsage, 0.0);
        currentValue_ += (nextCost - prevCost);
        currentUsage_[c.resource][c.time] = nextUsage;
    }
}

void MeanRisk::reset(const std::vector<int> &startTimes) {
    assert (startTimes.size() == nbInterventions());
    currentValue_ = 0.0;
    for (int i = 0; i < startTimes.size(); ++i) {
        if (startTimes[i] != -1) {
            set(i, startTimes[i]);
        }
    }
}

double MeanRisk::objectiveIfSet(int intervention, int startTime) const {
    assert (intervention >= 0 && intervention < contribs_.size());
    assert (startTime >= 0 && startTime < contribs_[intervention].size());
    double currentValue = currentValue_;
    return currentValue + contribs_[intervention][startTime];
}

void MeanRisk::set(int intervention, int startTime) {
    if (startTime < 0) return;
    assert (intervention >= 0 && intervention < contribs_.size());
    assert (startTime >= 0 && startTime < contribs_[intervention].size());
    currentValue_ += contribs_[intervention][startTime];
}

void MeanRisk::unset(int intervention, int startTime) {
    if (startTime < 0) return;
    assert (intervention >= 0 && intervention < contribs_.size());
    assert (startTime >= 0 && startTime < contribs_[intervention].size());
    currentValue_ -= contribs_[intervention][startTime];
}

void QuantileRisk::reset(const std::vector<int> &startTimes) {
    assert (startTimes.size() == nbInterventions());
    currentValue_ = 0.0;
    currentExcesses_.clear();
    currentExcesses_.resize(nbTimesteps(), 0.0);
    currentRisks_.clear();
    for (int i = 0; i < nbTimesteps(); ++i) {
        currentRisks_.push_back(std::vector<double>(nbScenarios(i), 0.0));
    }
    for (int i = 0; i < startTimes.size(); ++i) {
        if (startTimes[i] != -1) {
            set(i, startTimes[i]);
        }
    }
}

double QuantileRisk::objectiveIfSet(int intervention, int startTime) {
    set(intervention, startTime);
    double obj = value();
    unset(intervention, startTime);
    return obj;
}


void QuantileRisk::set(int intervention, int startTime) {
    if (startTime < 0) return;
    assert (intervention >= 0 && intervention < contribs_.size());
    assert (startTime >= 0 && startTime < contribs_[intervention].size());
    for (const RiskContribution &c : contribs_[intervention][startTime]) {
        assert (c.risks.size() == nbScenarios_[c.time]);
        for (int i = 0; i < c.risks.size(); ++i) {
            currentRisks_[c.time][i] += c.risks[i];
        }
    }
    updateExcesses(intervention, startTime);
}

void QuantileRisk::unset(int intervention, int startTime) {
    if (startTime < 0) return;
    assert (intervention >= 0 && intervention < contribs_.size());
    assert (startTime >= 0 && startTime < contribs_[intervention].size());
    for (const RiskContribution &c : contribs_[intervention][startTime]) {
        assert (c.risks.size() == nbScenarios_[c.time]);
        for (int i = 0; i < c.risks.size(); ++i) {
            currentRisks_[c.time][i] -= c.risks[i];
        }
    }
    updateExcesses(intervention, startTime);
}

void QuantileRisk::updateExcess(int time) {
    double oldExcess = currentExcesses_[time];
    std::vector<double> workingSet = currentRisks_[time];
    double mean = std::accumulate(workingSet.begin(), workingSet.end(), 0.0) / workingSet.size();
    int pos = quantileScenarios_[time];
    std::nth_element(workingSet.begin(), workingSet.begin() + pos, workingSet.end());
    double quantile = workingSet[pos];
    double newExcess = std::max(quantile - mean, 0.0);
    currentValue_ += newExcess - oldExcess;
    currentExcesses_[time] = newExcess;
}

void QuantileRisk::updateExcesses(int intervention, int startTime) {
    std::vector<int> times;
    for (const RiskContribution &c : contribs_[intervention][startTime]) {
        times.push_back(c.time);
    }
    std::sort(times.begin(), times.end());
    times.erase(std::unique(times.begin(), times.end()), times.end());
    for (int t : times) {
        updateExcess(t);
    }
}

void Problem::reset(const std::vector<int> &startTimes) {
    assert (startTimes.size() == nbInterventions());
    startTimes_ = startTimes;
    exclusions_.reset(startTimes);
    resources_.reset(startTimes);
    meanRisk_.reset(startTimes);
    quantileRisk_.reset(startTimes);
}

void Problem::reset() {
    std::vector<int> startTimes(nbInterventions(), -1);
    reset(startTimes);
}

void Problem::set(int intervention, int startTime) {
    if (startTime < 0) return; // Nothing to set
    assert (startTimes_[intervention] == -1);
    exclusions_.set(intervention, startTime);
    resources_.set(intervention, startTime);
    meanRisk_.set(intervention, startTime);
    quantileRisk_.set(intervention, startTime);
    startTimes_[intervention] = startTime;
}

void Problem::unset(int intervention) {
    int startTime = startTimes_[intervention];
    if (startTime < 0) return; // Already unset
    exclusions_.unset(intervention, startTime);
    resources_.unset(intervention, startTime);
    meanRisk_.unset(intervention, startTime);
    quantileRisk_.unset(intervention, startTime);
    startTimes_[intervention] = -1;
}

void Problem::set(const std::vector<int> &startTimes) {
    // Incremental version of reset
    assert (startTimes.size() == nbInterventions());
    for (int i = 0; i < nbInterventions(); ++i) {
        if (startTimes[i] != startTime(i)) {
            unset(i);
            set(i, startTimes[i]);
        }
    }
}

Problem::Objective Problem::objectiveIfSet(int intervention, int startTime, Objective threshold) {
    assert (startTimes_[intervention] == -1);
    meanRisk_.set(intervention, startTime);
    if (objective() >= threshold) {
        meanRisk_.unset(intervention, startTime);
        return Objective();
    }
    resources_.set(intervention, startTime);
    if (objective() >= threshold) {
        meanRisk_.unset(intervention, startTime);
        resources_.unset(intervention, startTime);
        return Objective();
    }
    exclusions_.set(intervention, startTime);
    if (objective() >= threshold) {
        meanRisk_.unset(intervention, startTime);
        resources_.unset(intervention, startTime);
        exclusions_.unset(intervention, startTime);
        return Objective();
    }
    quantileRisk_.set(intervention, startTime);

    Objective ret = objective();

    meanRisk_.unset(intervention, startTime);
    resources_.unset(intervention, startTime);
    exclusions_.unset(intervention, startTime);
    quantileRisk_.unset(intervention, startTime);

    return ret;
}

bool Problem::validSolution() const {
    for (int i = 0; i < nbInterventions(); ++i) {
        if (!assigned(i)) return false;
    }
    return true;
}

