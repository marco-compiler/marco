#include "marco/Modeling/Matching.h"

namespace marco::modeling::internal::matching {
Matchable::Matchable(std::shared_ptr<const IndexSet> matchableIndices)
    : matchableIndices(std::move(matchableIndices)) {
  this->unmatched = *this->matchableIndices;
}

const IndexSet &Matchable::getMatchableIndices() const {
  assert(matchableIndices && "Matchable indices not set");
  return *matchableIndices;
}

const IndexSet &Matchable::getMatched() const { return matched; }

const IndexSet &Matchable::getUnmatched() const { return unmatched; }

bool Matchable::allComponentsMatched() const { return unmatched.empty(); }

void Matchable::setMatch(IndexSet indices) {
  assert(getMatchableIndices().contains(indices) &&
         "The matched indices set is not a subset of the matchable indices");

  unmatched = getMatchableIndices() - indices;
  matched = std::move(indices);
  assert(matched + unmatched == getMatchableIndices());
}

void Matchable::addMatch(const IndexSet &newMatch) {
  assert(getMatchableIndices().contains(newMatch) &&
         "The matched indices set is not a subset of the matchable indices");

  matched += newMatch;
  unmatched -= newMatch;
  assert(matched + unmatched == getMatchableIndices());
}

void Matchable::removeMatch(const IndexSet &removedMatch) {
  matched -= removedMatch;
  unmatched += removedMatch;
  assert(matched + unmatched == getMatchableIndices());
}
} // namespace marco::modeling::internal::matching
