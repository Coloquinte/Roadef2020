
#pragma once

#include <iosfwd>
#include <vector>
#include <unordered_map>
#include <string>

enum class MoveStatus {
    Degraded, Same, Improved, NotFound
};

struct Assignment {
    int intervention;
    int startTime;
    Assignment(int i, int t) : intervention(i), startTime(t) {}

    bool operator<(const Assignment &o) const {
        if (intervention == o.intervention) {
            return startTime < o.startTime;
        }
        return intervention < o.intervention;
    }
};

struct Change {
    int intervention;
    int oldStartTime;
    int newStartTime;
    Change(int i, int ot, int nt)
        : intervention(i)
        , oldStartTime(ot)
        , newStartTime(nt)
        {}
};

struct RoadefParams {
    std::string instance;
    std::string solution;
    int verbosity;
    size_t seed;
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
    bool modificationDone_;

  public:
    int nbInterventions() const { return durations_.size(); }
    int nbTimesteps() const { return seasons_.size(); }
    int nbSeasons() const { return seasonInterdictions_.size(); }

    int value() const { return currentValue_; }
    const std::vector<int> &presence(int timestep) const { return currentPresence_[timestep]; }

    void set(int intervention, int startTime);
    void unset(int intervention, int startTime);

    void reset(const std::vector<int> &startTimes);
    void apply(const std::vector<Change> &changes);
    void commit(const std::vector<Change> &changes);
    void rollback(const std::vector<Change> &changes);

    friend class Problem;

  private:
    void move(int intervention, int oldStartTime, int newStartTime);
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
    bool modificationDone_;

  public:
    int nbInterventions() const { return demands_.size(); }
    int nbResources() const { return lowerBound_.size(); }
    int nbTimesteps() const { return lowerBound_.empty() ? 0 : lowerBound_.front().size(); }
    int maxStartTime(int intervention) const { return demands_[intervention].size(); }

    double value() const { return currentValue_; }

    void set(int intervention, int startTime);
    void unset(int intervention, int startTime);

    void reset(const std::vector<int> &startTimes);
    void apply(const std::vector<Change> &changes);
    void commit(const std::vector<Change> &changes);
    void rollback(const std::vector<Change> &changes);

    friend class Problem;

  private:
    void move(int intervention, int oldStartTime, int newStartTime);
};

class MeanRisk {
  protected:
    // By intervention by start time
    std::vector<std::vector<double> > contribs_;

    // Incremental data
    double currentValue_;
    bool modificationDone_;

  public:
    int nbInterventions() const { return contribs_.size(); }
    int maxStartTime(int intervention) const { return contribs_[intervention].size(); }

    double value() const { return currentValue_; }

    void set(int intervention, int startTime);
    void unset(int intervention, int startTime);

    void reset(const std::vector<int> &startTimes);
    void apply(const std::vector<Change> &changes);
    void commit(const std::vector<Change> &changes);
    void rollback(const std::vector<Change> &changes);

    friend class Problem;

  private:
    void move(int intervention, int oldStartTime, int newStartTime);
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
    bool modificationDone_;

  public:
    int nbInterventions() const { return contribs_.size(); }
    int nbTimesteps() const { return nbScenarios_.size(); }
    int nbScenarios(int timestep) const { return nbScenarios_[timestep]; }
    int maxStartTime(int intervention) const { return contribs_[intervention].size(); }

    double value() const { return currentValue_; }

    void set(int intervention, int startTime);
    void unset(int intervention, int startTime);

    void reset(const std::vector<int> &startTimes);
    void apply(const std::vector<Change> &changes);
    void commit(const std::vector<Change> &changes);
    void rollback(const std::vector<Change> &changes);

    friend class Problem;

  private:
    void updateExcess(int time);
    void updateExcesses(int intervention, int startTime);
    void move(int intervention, int oldStartTime, int newStartTime);
};


class Problem {
  public:
    struct Objective;

    static constexpr double resourceTol = 1.0e-6;
    static constexpr double riskTol = 1.0e-8;

    static Problem read(std::istream &);
    static Problem readFile(const std::string&);
    void writeSolution(std::ostream &);
    void writeSolutionFile(const std::string&);

    int nbResources() const { return resourceNames_.size(); }
    int nbInterventions() const { return interventionNames_.size(); }
    int nbTimesteps() const { return nbTimesteps_; }
    int nbSeasons() const { return nbSeasons_; }
    int maxStartTime(int intervention) const { return maxStartTimes_[intervention]; }

    int exclusionValue() const { return exclusions_.value(); }
    double resourceValue() const { return std::abs(resources_.value()) > resourceTol ? resources_.value() : 0.0; }
    double riskValue() const { return meanRisk_.value() + quantileRisk_.value(); }
    double meanRiskValue() const { return meanRisk_.value(); }
    double quantileRiskValue() const { return quantileRisk_.value(); }
    Objective objective() const;

    bool assigned(int intervention) const { return startTimes_[intervention] != -1; }
    int startTime(int intervention) const { return startTimes_[intervention]; }
    const std::vector<int> &presence(int timestep) const { return exclusions_.presence(timestep); }
    const std::vector<int> &startTimes() const { return startTimes_; }

    void reset(const std::vector<int> &startTimes);
    void reset();
    MoveStatus move(const std::vector<Assignment> &moves);
    void forceMove(const std::vector<Assignment> &moves);
    void set(int intervention, int startTime);
    void unset(int intervention);

  private:
    void commit(const std::vector<Change> &changes);
    void rollback(const std::vector<Change> &changes);
    std::vector<Change> movesToChanges(const std::vector<Assignment> &moves) const;

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
        : exclusion(0)
        , resource(0.0)
        , risk(0.0) {}

    Objective(int exclusion, double resource, double risk)
        : exclusion(exclusion)
        , resource(resource)
        , risk(risk) {}

    int compare(const Objective &o) const {
        if (exclusion != o.exclusion)
            return exclusion < o.exclusion ? -1 : 1;
        if (resource < (1.0 - resourceTol) * o.resource - resourceTol)
            return -1;
        if (resource > (1.0 + resourceTol) * o.resource + resourceTol)
            return 1;
        if (risk < (1.0 - riskTol) * o.risk - riskTol)
            return -1;
        if (risk > (1.0 + riskTol) * o.risk + riskTol)
            return 1;
        return 0;
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

