#include "marco/Modeling/Matching.h"

namespace marco::modeling::internal::matching {
Matchable::Matchable(IndexSet matchableIndices)
    : matchableIndices(std::move(matchableIndices)) {
  this->unmatched = this->matchableIndices;
}

const IndexSet &Matchable::getMatched() const { return matched; }

const IndexSet &Matchable::getUnmatched() const { return unmatched; }

bool Matchable::allComponentsMatched() const { return unmatched.empty(); }

void Matchable::addMatch(const IndexSet &newMatch) {
  assert(matchableIndices.contains(newMatch) &&
         "The matched indexes set is not a subset of the matchable indexes");

  matched += newMatch;
  unmatched -= newMatch;
  assert(matched + unmatched == matchableIndices);
}

void Matchable::removeMatch(const IndexSet &removedMatch) {
  matched -= removedMatch;
  unmatched += removedMatch;
  assert(matched + unmatched == matchableIndices);
}
} // namespace marco::modeling::internal::matching
