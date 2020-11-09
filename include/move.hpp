
#pragma once

#include "problem.hpp"

#include <memory>
#include <string>
#include <random>

typedef std::mt19937 Rgen;

class InterventionSelector {
  public:
    virtual std::vector<int> run(const Problem &, Rgen &) const =0;
};

class ChangeSelector {
  public:
    virtual std::vector<Assignment> run(const Problem &, const std::vector<int> &interventions, Rgen &) const =0;
};

class RandomInterventionSelector : public InterventionSelector {
  public:
    RandomInterventionSelector(int nbElements) : nbElements_(nbElements) {}
    std::vector<int> run(const Problem &, Rgen &) const;

  private:
    int nbElements_;
};

class CloseInterventionSelector : public InterventionSelector {
  public:
    CloseInterventionSelector(int nbElements, int nbTimesteps) : nbElements_(nbElements), nbTimesteps_(nbTimesteps) {}
    std::vector<int> run(const Problem &, Rgen &) const;

  private:
    int nbElements_;
    int nbTimesteps_;
};

class AllInterventionSelector : public InterventionSelector {
  public:
    AllInterventionSelector() {}
    std::vector<int> run(const Problem &, Rgen &) const;
};

class RandomChangeSelector : public ChangeSelector {
  public:
    RandomChangeSelector() {}
    std::vector<Assignment> run(const Problem &, const std::vector<int> &interventions, Rgen &) const;
};

class PerturbationChangeSelector : public ChangeSelector {
  public:
    PerturbationChangeSelector(int maxDist) : maxDist_(maxDist) {}
    std::vector<Assignment> run(const Problem &, const std::vector<int> &interventions, Rgen &) const;

  private:
    int maxDist_;
};

class SameChangeSelector : public ChangeSelector {
  public:
    SameChangeSelector(int maxDist) : maxDist_(maxDist) {}
    std::vector<Assignment> run(const Problem &, const std::vector<int> &interventions, Rgen &) const;

  private:
    int maxDist_;
};

class CycleChangeSelector : public ChangeSelector {
  public:
    CycleChangeSelector() {}
    std::vector<Assignment> run(const Problem &, const std::vector<int> &interventions, Rgen &) const;
};

class CyclePerturbationChangeSelector : public ChangeSelector {
  public:
    CyclePerturbationChangeSelector(int maxDist) : maxDist_(maxDist) {}
    std::vector<Assignment> run(const Problem &, const std::vector<int> &interventions, Rgen &) const;

  private:
    int maxDist_;
};

class PathChangeSelector : public ChangeSelector {
  public:
    PathChangeSelector() {}
    std::vector<Assignment> run(const Problem &, const std::vector<int> &interventions, Rgen &) const;
};

class Move {
  public:
    MoveStatus apply(Problem &, Rgen &) const;
    void force(Problem &, Rgen &) const;
    const std::string &name() const { return name_; }

    // Random selection random destination
    static Move random(int nbElements);
    // Random selection but small perturbation
    static Move randomPerturbation(int nbElements, int maxDist);
    // Interchange interventions
    static Move cycle(int nbElements);
    // Interchange interventions with small perturbation
    static Move cyclePerturbation(int nbElements, int maxDist);
    // Interchange interventions until the last element
    static Move path(int nbElements);
    // Random selection same perturbation
    static Move same(int nbElements, int maxDist);
    // Close selection same perturbation
    static Move closeSame(int nbElements, int nbTimesteps, int maxDist);
    // Close selection with small perturbation
    static Move closePerturbation(int nbElements, int nbTimesteps, int maxDist);
    // Randomize everything
    static Move full();

  private:
    Move(std::unique_ptr<InterventionSelector> isel, std::unique_ptr<ChangeSelector> csel)
      : isel_(std::move(isel))
      , csel_(std::move(csel)) {}
    std::unique_ptr<InterventionSelector> isel_;
    std::unique_ptr<ChangeSelector> csel_;
    std::string name_;
};
