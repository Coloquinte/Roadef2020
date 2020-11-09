
#include "move.hpp"

#include <cassert>
#include <algorithm>
#include <random>

Move Move::random(int nbElements) {
    return Move(
        std::make_unique<RandomInterventionSelector>(nbElements),
        std::make_unique<RandomChangeSelector>());
}

Move Move::randomPerturbation(int nbElements, int maxDist) {
    return Move(
        std::make_unique<RandomInterventionSelector>(nbElements),
        std::make_unique<PerturbationChangeSelector>(maxDist));
}

Move Move::cycle(int nbElements) {
    return Move(
        std::make_unique<RandomInterventionSelector>(nbElements),
        std::make_unique<CycleChangeSelector>());
}

Move Move::path(int nbElements) {
    return Move(
        std::make_unique<RandomInterventionSelector>(nbElements),
        std::make_unique<PathChangeSelector>());
}

Move Move::cyclePerturbation(int nbElements, int maxDist) {
    return Move(
        std::make_unique<RandomInterventionSelector>(nbElements),
        std::make_unique<CyclePerturbationChangeSelector>(maxDist));
}

Move Move::same(int nbElements, int maxDist) {
    return Move(
        std::make_unique<RandomInterventionSelector>(nbElements),
        std::make_unique<SameChangeSelector>(maxDist));
}

Move Move::closeSame(int nbElements, int nbTimesteps, int maxDist) {
    return Move(
        std::make_unique<CloseInterventionSelector>(nbElements, nbTimesteps),
        std::make_unique<SameChangeSelector>(maxDist));
}

Move Move::closePerturbation(int nbElements, int nbTimesteps, int maxDist) {
    return Move(
        std::make_unique<CloseInterventionSelector>(nbElements, nbTimesteps),
        std::make_unique<PerturbationChangeSelector>(maxDist));
}

Move Move::full() {
    return Move(
        std::make_unique<AllInterventionSelector>(),
        std::make_unique<RandomChangeSelector>());
}

MoveStatus Move::apply(Problem &pb, Rgen &rgen) const {
    std::vector<int> interventions = isel_->run(pb, rgen);
    std::vector<Assignment> moves = csel_->run(pb, interventions, rgen);
    return pb.move(moves);
}

void Move::force(Problem &pb, Rgen &rgen) const {
    std::vector<int> interventions = isel_->run(pb, rgen);
    std::vector<Assignment> moves = csel_->run(pb, interventions, rgen);
    pb.forceMove(moves);
}

std::vector<int> RandomInterventionSelector::run(const Problem &pb, Rgen &rgen) const {
    std::vector<int> ret;
    std::uniform_int_distribution<int> dist(0, pb.nbInterventions()-1);
    for (int i = 0; i < nbElements_; ++i) {
        int selected = 0;
        for (int t = 0; t < 10; ++t) {
            selected = dist(rgen);
            bool unique = true;
            for (int o : ret) {
                if (o == selected) unique = false;
            }
            if (unique) break;
        }
        ret.push_back(selected);
    }
    return ret;
}

std::vector<int> CloseInterventionSelector::run(const Problem &pb, Rgen &rgen) const {
    std::uniform_int_distribution<int> dist(0, pb.nbTimesteps()-1);
    int timestep = dist(rgen);
    std::vector<int> interventions;
    for (int t = timestep; t < timestep + nbTimesteps_ && t < pb.nbTimesteps(); ++t) {
        for (int i : pb.presence(t)) {
            interventions.push_back(i);
        }
    }
    std::sort(interventions.begin(), interventions.end());
    interventions.erase(std::unique(interventions.begin(), interventions.end()), interventions.end());
    std::shuffle(interventions.begin(), interventions.end(), rgen);
    std::vector<int> ret;
    for (int i = 0; i < nbElements_ && i < interventions.size(); ++i) {
        ret.push_back(interventions[i]);
    }
    return ret;
}

std::vector<int> AllInterventionSelector::run(const Problem &pb, Rgen &rgen) const {
    std::vector<int> ret;
    for (int i = 0; i < pb.nbInterventions(); ++i) {
        ret.push_back(i);
    }
    std::shuffle(ret.begin(), ret.end(), rgen);
    return ret;
}

std::vector<Assignment> RandomChangeSelector::run(const Problem &pb, const std::vector<int> &interventions, Rgen &rgen) const {
    std::vector<Assignment> ret;
    for (int intervention : interventions) {
        std::uniform_int_distribution<int> dist(0, pb.maxStartTime(intervention)-1);
        ret.emplace_back(intervention, dist(rgen));
    }
    return ret;
}

std::vector<Assignment> PerturbationChangeSelector::run(const Problem &pb, const std::vector<int> &interventions, Rgen &rgen) const {
    std::vector<Assignment> ret;
    for (int intervention : interventions) {
        int origPos = pb.startTime(intervention);
        int minPos = std::max(0, origPos - maxDist_);
        int maxPos = std::min(pb.maxStartTime(intervention)-1, origPos + maxDist_);
        assert (minPos <= maxPos);
        std::uniform_int_distribution<int> dist(minPos, maxPos);
        ret.emplace_back(intervention, dist(rgen));
    }
    return ret;
}

std::vector<Assignment> SameChangeSelector::run(const Problem &pb, const std::vector<int> &interventions, Rgen &rgen) const {
    std::uniform_int_distribution<int> dist(-maxDist_, maxDist_);
    int delta = dist(rgen);
    std::vector<Assignment> ret;
    for (int intervention : interventions) {
        int pos = pb.startTime(intervention) + delta;
        pos = std::max(pos, 0);
        pos = std::min(pos, pb.maxStartTime(intervention)-1);
        ret.emplace_back(intervention, pos);
    }
    return ret;
}

std::vector<Assignment> CycleChangeSelector::run(const Problem &pb, const std::vector<int> &interventions, Rgen &rgen) const {
    std::vector<Assignment> ret;
    for (int i = 0; i < interventions.size(); ++i) {
        int intervention = interventions[i];
        int nextIntervention = interventions[(i+1) % interventions.size()];
        int pos = pb.startTime(nextIntervention);
        pos = std::min(pos, pb.maxStartTime(intervention)-1);
        ret.emplace_back(intervention, pos);
    }
    return ret;
}

std::vector<Assignment> CyclePerturbationChangeSelector::run(const Problem &pb, const std::vector<int> &interventions, Rgen &rgen) const {
    std::vector<Assignment> ret;
    for (int i = 0; i < interventions.size(); ++i) {
        int intervention = interventions[i];
        int nextIntervention = interventions[(i+1) % interventions.size()];
        int origPos = pb.startTime(nextIntervention);
        int minPos = std::min(pb.maxStartTime(intervention)-1, std::max(0, origPos - maxDist_));
        int maxPos = std::min(pb.maxStartTime(intervention)-1, std::max(0, origPos + maxDist_));
        assert (minPos <= maxPos);
        std::uniform_int_distribution<int> dist(minPos, maxPos);
        ret.emplace_back(intervention, dist(rgen));
    }
    return ret;
}

std::vector<Assignment> PathChangeSelector::run(const Problem &pb, const std::vector<int> &interventions, Rgen &rgen) const {
    std::vector<Assignment> ret;
    for (int i = 0; i+1 < interventions.size(); ++i) {
        int intervention = interventions[i];
        int nextIntervention = interventions[i+1];
        int pos = pb.startTime(nextIntervention);
        pos = std::min(pos, pb.maxStartTime(intervention)-1);
        ret.emplace_back(intervention, pos);
    }
    int lastIntervention = interventions.back();
    std::uniform_int_distribution<int> dist(0, pb.maxStartTime(lastIntervention)-1);
    ret.emplace_back(lastIntervention, dist(rgen));
    return ret;
}


