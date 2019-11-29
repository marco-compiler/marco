#include "modelica/utils/Interval.hpp"

namespace modelica
{
	template<typename T>
	class IntervalSet
	{
		private:
		llvm::SmallVector<Interval<T>, 2> intervals;

		public:
		IntervalSet(Interval<T> interval): intervals({ interval }) {}
		IntervalSet(): intervals() {}

		[[nodiscard]] bool contains(T element) const
		{
			const auto inElement = [element](const Interval<T>& interval) {
				return interval.contains(element);
			};

			return std::find_if(intervals.begin(), intervals.end(), inElement) !=
						 intervals.end();
		}

		IntervalSet& unite(const IntervalSet& other)
		{
			for (const auto& el : other.intervals)
				intervals.push_back(el);
			return *this;
		}

		IntervalSet& unite(Interval<T> other)
		{
			intervals.push_back(other);
			return *this;
		}

		IntervalSet& minus(Interval<T> other) { return minus(other); }

		IntervalSet& minus(const Interval<T>& other)
		{
			for (const auto& inter : intervals)
				inter.remove(other);

			return *this;
		}
	};
}	 // namespace modelica
