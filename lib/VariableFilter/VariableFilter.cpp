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
    variables[var.getName()].push_back(var);
  }

  void VariableFilter::addDerivative(Tracker var)
  {
    setEnabled(true);
    derivatives[var.getName()].push_back(var);
  }

  void VariableFilter::addRegexString(llvm::StringRef newRegex)
  {
    setEnabled(true);
    regex.push_back(newRegex.str());
  }

  std::vector<Filter> VariableFilter::getVariableInfo(llvm::StringRef name, unsigned int expectedRank) const
  {
    std::vector<Filter> filters;

    bool visibility = !isEnabled();

    if (matchesRegex(name)) {
      std::vector<Range> ranges;
      visibility = true;
    }

    if (auto trackersIt = variables.find(name); trackersIt != variables.end()) {
      for (const auto& tracker : trackersIt->second) {
        visibility = true;

        std::vector<Range> ranges;

        // If the requested rank is lower than the one known by the variable filter,
        // then only keep an amount of ranges equal to the rank.

        auto trackerRanges = tracker.getRanges();
        unsigned int amount = expectedRank < trackerRanges.size() ? expectedRank : trackerRanges.size();
        auto it = trackerRanges.begin();
        ranges.insert(ranges.begin(), it, it + amount);

        // If the requested rank is higher than the one known by the variable filter,
        // then set the remaining ranges as unbounded.

        for (size_t i = ranges.size(); i < expectedRank; ++i) {
          ranges.emplace_back(Range::kUnbounded, Range::kUnbounded);
        }

        filters.push_back(VariableFilter::Filter(visibility, ranges));
      }
    }

    if (filters.empty()) {
      std::vector<Range> ranges;

      for (size_t i = ranges.size(); i < expectedRank; ++i) {
        ranges.emplace_back(Range::kUnbounded, Range::kUnbounded);
      }

      filters.push_back(VariableFilter::Filter(visibility, ranges));
    }

    return filters;
  }

  std::vector<Filter> VariableFilter::getVariableDerInfo(llvm::StringRef name, unsigned int expectedRank) const
  {
    std::vector<Filter> result;

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

    result.push_back(VariableFilter::Filter(visibility, ranges));

    return result;
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
    auto sourceFile = std::make_shared<SourceFile>("-", llvm::MemoryBuffer::getMemBuffer(str));
    Parser parser(vf, sourceFile);

    if (auto error = parser.run()) {
      return std::move(error);
    }

    return vf;
  }
}
