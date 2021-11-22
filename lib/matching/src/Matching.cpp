#include <marco/matching/Matching.h>

using namespace marco::matching::detail;

Matchable::Matchable(IncidenceMatrix initialMatch) : match(std::move(initialMatch))
{
  assert(initialMatch.getEquationRanges().flatSize() == 1 || initialMatch.getVariableRanges().flatSize() == 1);
}

const IncidenceMatrix& Matchable::getMatched() const
{
  return match;
}

IncidenceMatrix Matchable::getUnmatched() const
{
  return !match;
}

bool Matchable::allComponentsMatched() const
{
  return llvm::all_of(match.getIndexes(), [&](const auto& indexes) {
      return match.get(indexes);
  });
}

void Matchable::addMatch(IncidenceMatrix newMatch)
{
  match += newMatch;
}

void Matchable::removeMatch(IncidenceMatrix removedMatch)
{
  match -= removedMatch;
}
