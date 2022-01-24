#pragma once

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

/**
 * Keeps track of variables, arrays, derivatives (and regex for matching)
 * that has to be printed during the simulation.
 */
namespace marco
{
	namespace vf
	{
		/**
		 * Represents an array range, $ special character is '-1'.
		 */
		class ArrayRange
		{
			public:
			static constexpr long unbounded = -1;

			ArrayRange(long lowerBound, long upperBound);

			bool hasLowerBound() const;
			long getLowerBound() const;

			bool hasUpperBound() const;
			long getUpperBound() const;

			private:
			long lowerBound, upperBound;
		};

		/**
		 * Keeps tracks of a single variable, array or derivative that has been
		 * specified by command line argument.
		 */
		class Tracker
		{
			public:
			Tracker();
			Tracker(llvm::StringRef name);
			Tracker(llvm::StringRef name, llvm::ArrayRef<ArrayRange> ranges);

			void setRanges(llvm::ArrayRef<ArrayRange> ranges);

			llvm::StringRef getName() const;

			llvm::ArrayRef<ArrayRange> getRanges() const;

			ArrayRange getRangeOfDimension(unsigned int dimensionIndex) const;

			private:
			std::string name;
			llvm::SmallVector<ArrayRange, 3> ranges;
		};

		class Filter
		{
			public:
			Filter(bool visibility, llvm::ArrayRef<ArrayRange> ranges);

			bool isVisible() const;
			llvm::ArrayRef<ArrayRange> getRanges() const;

			static Filter visibleScalar();
			static Filter visibleArray(llvm::ArrayRef<long> shape);

			private:
			bool visibility;
			llvm::SmallVector<ArrayRange, 3> ranges;
		};
	}

	class VariableFilter
	{
		public:
		using Tracker = vf::Tracker;
		using Filter = vf::Filter;

		void dump() const;

		void dump(llvm::raw_ostream& os) const;

		[[nodiscard]] bool isEnabled() const;

		void setEnabled(bool enabled);

		void addVariable(Tracker var);

		void addDerivative(Tracker var);

		void addRegexString(llvm::StringRef regex);

		Filter getVariableInfo(llvm::StringRef name, unsigned int expectedRank = 0) const;

		Filter getVariableDerInfo(llvm::StringRef name, unsigned int expectedRank = 0) const;

		static llvm::Expected<VariableFilter> fromString(llvm::StringRef str);

		private:
		/**
		 * Check whether a variable identifier matches any of the regular
		 * expressions stored within the variable filter.
		 *
		 * @param identifier	variable identifier
		 * @return true if a regular expression has been matched
		 */
		bool matchesRegex(llvm::StringRef identifier) const;

		private:
		llvm::StringMap<Tracker> _variables;
		llvm::StringMap<Tracker> _derivatives;
		llvm::SmallVector<std::string> _regex;
		bool _enabled = false;
	};
}
