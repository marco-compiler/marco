#include "marco/VariableFilter/VariableFilter.h"
#include "marco/VariableFilter/Parser.h"
#include "marco/VariableFilter/Token.h"
#include "marco/Utils/LogMessage.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Regex.h"
#include <map>

using namespace ::marco;
using namespace ::marco::vf;

namespace marco
{
  void VariableFilter::dump() const
  {
    dump(llvm::outs());
  }

  void VariableFilter::dump(llvm::raw_ostream& os) const
  {
    // TODO: improve output

    /*
	for (int s = 0; s < 12; ++s)
		os << "#";

	os << "\n *** TRACKED VARIABLES *** : " << "\n";

	for (const auto& tracker : _variables)
		tracker.getValue().dump(os);

	for (int s = 0; s < 12; ++s)
		os << "#";

	os << "\n *** TRACKED DERIVATIVES *** : " << "\n";

	for (const auto& tracker : _derivatives)
		tracker.getValue().dump(os);

	for (int s = 0; s < 12; ++s)
		os << "#";

	os << "\n *** TRACKED REGEX(s) *** : " << "\n";

	for (const auto &regex : _regex) {
		os << "Regex: /" << regex.c_str() << "/";
	}
	 */
  }

  bool VariableFilter::isEnabled() const
  {
    return enabled;
  }

  void VariableFilter::setEnabled(bool isEnabled)
  {
    enabled = isEnabled;
  }

  void VariableFilter::addVariable(Tracker var)
  {
    setEnabled(true);
    variables[var.getName()] = var;
  }

  void VariableFilter::addDerivative(Tracker var)
  {
    setEnabled(true);
    derivatives[var.getName()] = var;
  }

  void VariableFilter::addRegexString(llvm::StringRef newRegex)
  {
    setEnabled(true);
    regex.push_back(newRegex.str());
  }

  VariableFilter::Filter VariableFilter::getVariableInfo(llvm::StringRef name, unsigned int expectedRank) const
  {
    bool visibility = !isEnabled();
    llvm::SmallVector<Range, 3> ranges;

    if (matchesRegex(name)) {
      visibility = true;
      Range unboundedRange(Range::kUnbounded, Range::kUnbounded);
      ranges.insert(ranges.begin(), expectedRank, unboundedRange);
    }

    if (variables.count(name) != 0) {
      visibility = true;
      auto tracker = variables.lookup(name);
      ranges.clear();

      // If the requested rank is lower than the one known by the variable filter,
      // then only keep an amount of ranges equal to the rank.

      auto trackerRanges = tracker.getRanges();
      unsigned int amount = expectedRank < trackerRanges.size() ? expectedRank : trackerRanges.size();
      auto it = trackerRanges.begin();
      ranges.insert(ranges.begin(), it, it + amount);
    }

    // If the requested rank is higher than the one known by the variable filter,
    // then set the remaining ranges as unbounded.

    for (size_t i = ranges.size(); i < expectedRank; ++i)
      ranges.emplace_back(Range::kUnbounded, Range::kUnbounded);

    return VariableFilter::Filter(visibility, ranges);
  }

  Filter VariableFilter::getVariableDerInfo(llvm::StringRef name, unsigned int expectedRank) const
  {
    bool visibility = false;
    llvm::SmallVector<Range, 3> ranges;

    if (derivatives.count(name) != 0) {
      visibility = true;
      auto tracker = derivatives.lookup(name);
    }

    // For now, derivatives are always fully printed

    for (size_t i = ranges.size(); i < expectedRank; ++i) {
      ranges.emplace_back(Range::kUnbounded, Range::kUnbounded);
    }

    return VariableFilter::Filter(visibility, ranges);
  }

  bool VariableFilter::matchesRegex(llvm::StringRef identifier) const
  {
    return llvm::any_of(regex, [&identifier](const auto& expression) {
      llvm::Regex llvmRegex(expression);
      return llvmRegex.match(identifier);
    });
  }

  llvm::Expected<VariableFilter> VariableFilter::fromString(llvm::StringRef str)
  {
    VariableFilter vf;
    Parser parser(vf, str);

    if (auto error = parser.run()) {
      return std::move(error);
    }

    return vf;
  }
}
