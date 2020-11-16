
#pragma once

#include <iosfwd>
#include <vector>
#include <unordered_map>
#include <string>
#include <limits>
#include <chrono>

struct RoadefParams {
    std::string instance;
    std::string solution;

    int verbosity;
    size_t seed;
    bool restart;

    double timeLimit;
    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point endTime;

    int beamWidth;
    int backtrackDepth;
};

class Exclusions {
  protected:
    // By intervention by start time
    std::vector<std::vector<int> > durations_;
    std::vector<int> seasons_;
    // By season by intervention
    std::vector<std::vector<std::vector<int> > > seasonInterdictions_;

    // Incremental data
    int currentValue_;
    std::vector<std::vector<int> > currentPresence_;

  public:
    int nbInterventions() const { return durations_.size(); }
    int nbTimesteps() const { return seasons_.size(); }
    int nbSeasons() const { return seasonInterdictions_.size(); }

    int value() const { return currentValue_; }
    const std::vector<int> &presence(int timestep) const { return currentPresence_[timestep]; }

    void set(int intervention, int startTime);
    void unset(int intervention, int startTime);
    void reset(const std::vector<int> &startTimes);

    friend class Problem;
};

class Resources {
  protected:
    struct ResourceContribution {
        double amount;
        int time;
        int resource;
        ResourceContribution(double a, int t, int r) : amount(a), time(t), resource(r) {}
    };
    // By resource by time
    std::vector<std::vector<double> > lowerBound_;
    std::vector<std::vector<double> > upperBound_;
    // By intervention by start time
    std::vector<std::vector<std::vector<ResourceContribution> > > demands_;

    // Incremental data
    double currentValue_;
    std::vector<std::vector<double> > currentUsage_;

  public:
    int nbInterventions() const { return demands_.size(); }
    int nbResources() const { return lowerBound_.size(); }
    int nbTimesteps() const { return lowerBound_.empty() ? 0 : lowerBound_.front().size(); }
    int maxStartTime(int intervention) const { return demands_[intervention].size(); }

    double value() const { return currentValue_; }

    void set(int intervention, int startTime);
    void unset(int intervention, int startTime);
    void reset(const std::vector<int> &startTimes);

    friend class Problem;
};

class MeanRisk {
  protected:
    // By intervention by start time
    std::vector<std::vector<double> > contribs_;

    // Incremental data
    double currentValue_;

  public:
    int nbInterventions() const { return contribs_.size(); }
    int maxStartTime(int intervention) const { return contribs_[intervention].size(); }

    double value() const { return currentValue_; }

    void set(int intervention, int startTime);
    void unset(int intervention, int startTime);
    void reset(const std::vector<int> &startTimes);

    friend class Problem;
};

class QuantileRisk {
  protected:
    struct RiskContribution {
        int time;
        std::vector<double> risks;
        RiskContribution(int t, const std::vector<double> &r) : time(t), risks(r) {}
    };
    // By time
    std::vector<int> nbScenarios_;
    std::vector<int> quantileScenarios_;
    // By intervention by start time
    std::vector<std::vector<std::vector<RiskContribution> > > contribs_;

    // Incremental data
    double currentValue_;
    std::vector<double> currentExcesses_;
    std::vector<std::vector<double> > currentRisks_;

  public:
    int nbInterventions() const { return contribs_.size(); }
    int nbTimesteps() const { return nbScenarios_.size(); }
    int nbScenarios(int timestep) const { return nbScenarios_[timestep]; }
    int maxStartTime(int intervention) const { return contribs_[intervention].size(); }

    double value() const { return currentValue_; }

    void set(int intervention, int startTime);
    void unset(int intervention, int startTime);
    void reset(const std::vector<int> &startTimes);

    friend class Problem;

  private:
    void updateExcess(int time);
    void updateExcesses(int intervention, int startTime);
};


class Problem {
  public:
    struct Objective;

    // Resource tolerance; slightly less than used by the checker
    static constexpr double resourceTol = 9.9e-6;
    // Risk tolerance; used only to compare solutions
    static constexpr double riskTol = 1.0e-8;

    static Problem read(std::istream &);
    static Problem readFile(const std::string&);
    void readSolution(std::istream &);
    void readSolutionFile(const std::string&);
    void writeSolution(std::ostream &);
    void writeSolutionFile(const std::string&);

    int nbResources() const { return resourceNames_.size(); }
    int nbInterventions() const { return interventionNames_.size(); }
    int nbTimesteps() const { return nbTimesteps_; }
    int nbSeasons() const { return nbSeasons_; }
    int maxStartTime(int intervention) const { return maxStartTimes_[intervention]; }

    int exclusionValue() const { return exclusions_.value(); }
    double resourceValue() const { return resources_.value() > resourceTol ? resources_.value() : 0.0; }
    double riskValue() const { return meanRisk_.value() + quantileRisk_.value(); }
    double meanRiskValue() const { return meanRisk_.value(); }
    double quantileRiskValue() const { return quantileRisk_.value(); }
    Objective objective() const;
    bool validSolution() const;

    bool assigned(int intervention) const { return startTimes_[intervention] != -1; }
    int startTime(int intervention) const { return startTimes_[intervention]; }
    const std::vector<int> &presence(int timestep) const { return exclusions_.presence(timestep); }
    const std::vector<int> &startTimes() const { return startTimes_; }

    void reset(const std::vector<int> &startTimes);
    void reset();
    void set(int intervention, int startTime);
    void unset(int intervention);
    void set(const std::vector<int> &startTimes);
    Objective objectiveIf(int intervention, int startTime, Objective threshold);

    // Heuristic measures to take decisions
    std::vector<double> measureSpanMeanRisk() const;
    std::vector<double> measureAverageDemand() const;

    // Access to internal data
    const Exclusions &exclusions() const { return exclusions_; }
    const Resources &resources() const { return resources_; }
    const MeanRisk &meanRisk() const { return meanRisk_; }
    const QuantileRisk &quantileRisk() const { return quantileRisk_; }

  private:
    // Names
    std::vector<std::string> interventionNames_;
    std::vector<std::string> resourceNames_;
    std::unordered_map<std::string, int> interventionMappings_;
    std::unordered_map<std::string, int> resourceMappings_;

    // Problem data
    int nbTimesteps_;
    int nbSeasons_;
    std::vector<int> maxStartTimes_;
    Exclusions exclusions_;
    Resources resources_;
    MeanRisk meanRisk_;
    QuantileRisk quantileRisk_;

    // Current solution
    std::vector<int> startTimes_;
};

struct Problem::Objective {
    Objective()
        : exclusion(std::numeric_limits<int>::max())
        , resource(std::numeric_limits<double>::infinity())
        , risk(std::numeric_limits<double>::infinity()) {}

    Objective(int exclusion, double resource, double risk)
        : exclusion(exclusion)
        , resource(resource)
        , risk(risk) {}

    int compare(const Objective &o) const {
        if (exclusion != o.exclusion)
            return exclusion < o.exclusion ? -1 : 1;
        if (resource != o.resource)
            return resource < o.resource ? -1 : 1;
        if (risk != o.risk)
            return risk < o.risk ? -1 : 1;
        return 0;
    }

    bool betterThan(const Objective &o) const {
        if (exclusion != o.exclusion)
            return exclusion < o.exclusion;
        if (resource != o.resource)
            return resource < o.resource;
        return risk + riskTol < o.risk;
    }

    bool operator<(const Objective &o) const {
        return compare(o) == -1;
    }

    bool operator>(const Objective &o) const {
        return compare(o) == 1;
    }

    bool operator==(const Objective &o) const {
        return compare(o) == 0;
    }

    bool operator>=(const Objective &o) const {
        return compare(o) != -1;
    }

    bool operator<=(const Objective &o) const {
        return compare(o) != 1;
    }

    bool operator!=(const Objective &o) const {
        return compare(o) != 0;
    }


    int exclusion;
    double resource;
    double risk;
};

inline Problem::Objective Problem::objective() const {
    return Objective(exclusionValue(), resourceValue(), riskValue());
}

