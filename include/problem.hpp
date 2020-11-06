
#pragma once

#include <iosfwd>
#include <vector>

class Exclusions {
  protected:
    // By intervention by start time
    std::vector<std::vector<int> > durations_;
    std::vector<int> seasons_;
    // By season by intervention
    std::vector<std::vector<std::vector<int> > > seasonInterdictions_;

  public:
    int nbInterventions() const { return durations_.size(); }
    int nbTimesteps() const { return seasons_.size(); }
    int nbSeasons() const { return seasonInterdictions_.size(); }

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

  public:
    int nbInterventions() const { return demands_.size(); }
    int nbResources() const { return lowerBound_.size(); }
    int nbTimesteps() const { return lowerBound_.empty() ? 0 : lowerBound_.front().size(); }
    int maxStartTime(int intervention) const { return demands_[intervention].size(); }

    friend class Problem;
};

class MeanRisk {
  protected:
    // By intervention by start time
    std::vector<std::vector<double> > contribs_;

  public:
    int nbInterventions() const { return contribs_.size(); }
    int maxStartTime(int intervention) const { return contribs_[intervention].size(); }

    friend class Problem;
};

class QuantileRisk {
  protected:
    struct RiskContribution {
        std::vector<double> risks;
        int time;
    };
    // By time
    std::vector<int> nbScenarios_;
    std::vector<int> quantileScenarios_;
    // By intervention by start time
    std::vector<std::vector<std::vector<RiskContribution> > > contribs_;

  public:
    int nbInterventions() const { return contribs_.size(); }
    int nbTimesteps() const { return nbScenarios_.size(); }
    int nbScenarios(int timestep) const { return nbScenarios_[timestep]; }
    int maxStartTime(int intervention) const { return contribs_[intervention].size(); }

    friend class Problem;
};


class Problem {
  public:
    static Problem read(std::istream &);
    static Problem readFile(const std::string&);
    void write(std::ostream &);
    void writeFile(const std::string&);

    int nbResources() const { return resourceNames_.size(); }
    int nbInterventions() const { return interventionNames_.size(); }
    int nbTimesteps() const { return nbTimesteps_; }
    int maxStartTime(int intervention) const { return maxStartTimes_[intervention]; }

  private:
    std::vector<std::string> interventionNames_;
    std::vector<std::string> resourceNames_;
    int nbTimesteps_;
    Exclusions exclusions_;
    Resources resources_;
    MeanRisk meanRisk_;
    QuantileRisk quantileRisk_;
    std::vector<int> maxStartTimes_;
};

