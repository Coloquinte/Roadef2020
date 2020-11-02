
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
};

class Resources {
  protected:
    struct ResourceContribution {
        double amount;
        int time;
        int resource;
    };
    // By resource by time
    std::vector<std::vector<double> > lowerBound_;
    std::vector<std::vector<double> > upperBound_;
    // By intervention by start time
    std::vector<std::vector<std::vector<ResourceContribution> > > resourceContributions_;

  public:
    int nbInterventions() const { return resourceContributions_.size(); }
    int nbResources() const { return lowerBound_.size(); }
    int nbTimesteps() const { return lowerBound_.empty() ? 0 : lowerBound_.front().size(); }
    int maxStartTime(int intervention) const { return resourceContributions_[intervention].size(); }
};

class MeanRisk {
  protected:
    // By intervention by start time
    std::vector<std::vector<double> > meanRiskContributions_;

  public:
    int nbInterventions() const { return meanRiskContributions_.size(); }
    int maxStartTime(int intervention) const { return meanRiskContributions_[intervention].size(); }
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
    std::vector<std::vector<std::vector<RiskContribution> > > riskContributions_;

  public:
    int nbInterventions() const { return riskContributions_.size(); }
    int nbTimesteps() const { return nbScenarios_.size(); }
    int nbScenarios(int timestep) const { return nbScenarios_[timestep]; }
};


class Problem {
  public:
    static Problem read(std::istream &);
    static Problem readFile(const std::string&);
    void write(std::ostream &);
    void writeFile(const std::string&);

  private:
    Exclusions exclusions_;
    Resources resources_;
    MeanRisk meanRisk_;
    QuantileRisk quantileRisk_;
};

