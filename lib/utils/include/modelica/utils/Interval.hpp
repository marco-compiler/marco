#include <functional>
#include <iterator>
#include <type_traits>

#include "llvm/ADT/SmallVector.h"

namespace modelica
{
	template<typename T>
	class Interval
	{
		private:
		T beginValue;
		T endValue;
		T stepValue;

		class IntervalIterator
		{
			public:
			using iterator_category = std::input_iterator_tag;
			using value_type = T;
			using difference_type = void;
			using pointer = T*;
			using reference = T&;

			IntervalIterator(const Interval<T>& interval)
					: original(interval), current(original.front())
			{
			}
			IntervalIterator(const Interval<T>& interval, T value)
					: original(interval), current(value)
			{
			}
			[[nodiscard]] T operator*() const { return current; }
			[[nodiscard]] T operator->() const { return current; }
			IntervalIterator operator++()
			{
				IntervalIterator old = *this;

				current += original.step();
				if (!original.contains(current))
					current = original.back();

				return old;
			}
			[[nodiscard]] bool operator==(const IntervalIterator& other)
			{
				return &original == &other.original && current == other.current;
			}
			[[nodiscard]] bool operator!=(const IntervalIterator& other)
			{
				return !(*this == other);
			}

			private:
			const Interval<T>& original;
			T current;
		};

		public:
		constexpr Interval(T begin, T end, T step = 1)
				: beginValue(begin), endValue(end), stepValue(step)
		{
			static_assert(
					std::is_fundamental<T>::value,
					"using interval of non primitive types is forbidden");
			assert(begin <= end);
		}

		constexpr Interval(const Interval& begin, T end)
				: beginValue(begin.beginValue), endValue(end), stepValue(begin.step)
		{
			assert(begin <= end);
		}
		constexpr Interval(T begin, const Interval& end)
				: beginValue(begin), endValue(end.endValue), stepValue(begin.step)
		{
			assert(begin <= end);
		}

		[[nodiscard]] constexpr bool operator==(const Interval& other) const
		{
			return front() == other.front() && back() == other.back() &&
						 step() == other.step();
		}

		[[nodiscard]] constexpr bool operator!=(const Interval& other) const
		{
			return !(*this == other);
		}

		[[nodiscard]] constexpr T front() const { return beginValue; }
		[[nodiscard]] constexpr T back() const { return endValue; }
		[[nodiscard]] constexpr T step() const { return stepValue; }
		[[nodiscard]] constexpr size_t size() const
		{
			return (endValue - beginValue) / stepValue;
		}
		[[nodiscard]] IntervalIterator begin() const
		{
			return IntervalIterator(*this);
		}
		[[nodiscard]] IntervalIterator end() const
		{
			return IntervalIterator(*this, endValue);
		}
		[[nodiscard]] T nthElement(int64_t elm) const
		{
			T el = front() + (elm * step());
			assert(el < endValue);
			return el;
		}
		[[nodiscard]] constexpr bool contains(T val) const
		{
			if ((val - beginValue) % std::abs(stepValue) != 0)
				return false;

			return val >= beginValue && val < endValue;
		}
	};

	using IntInterval = Interval<int>;
	using LongInterval = Interval<long>;

	template<typename T>
	[[nodiscard]] bool multipleOf(T left, T right)
	{
		assert(left != 0 && "cannot be zero");
		assert(right != 0 && "cannot be zero");
		return left % right == 0 || right % left == 0;
	}

	template<typename T>
	[[nodiscard]] std::pair<T, bool> minCommon(
			const Interval<T>& left, const Interval<T>& right)
	{
		auto el = std::find_if(left.begin(), left.end(), [&right](T el) {
			return right.contains(el);
		});

		if (el != left.end())
			return std::make_pair(*el, true);

		return std::make_pair(0, false);
	}

	template<typename T>
	[[nodiscard]] bool disjoint(const Interval<T>& left, const Interval<T>& right)
	{
		auto [elm, foundEl] = minCommon(left, right);
		return !foundEl;
	}

	template<typename T>
	[[nodiscard]] llvm::SmallVector<Interval<T>, 2> intervalFilter(
			const Interval<T>& vector, std::function<bool(T)> predicate)
	{
		llvm::SmallVector<Interval<T>, 2> toReturn;

		T currentIntervalStart = vector.front();
		bool inInterval = false;

		auto step = vector.step();

		for (T current : vector)
		{
			bool keepEl = predicate(current);
			if (inInterval && !keepEl)
			{
				toReturn.emplace_back(currentIntervalStart, current, step);
				inInterval = false;
			}

			if (!inInterval && keepEl)
			{
				inInterval = true;
				currentIntervalStart = current;
			}
		}
		if (inInterval)
			toReturn.emplace_back(currentIntervalStart, vector.back(), step);

		return toReturn;
	}

	template<typename T>
	[[nodiscard]] llvm::SmallVector<Interval<T>, 2> simpleIntersection(
			const Interval<T>& left, const Interval<T>& right)
	{
		using Type = const Interval<T>&;
		auto [smaller, larger] = [&]() {
			if (left.size() >= right.size())
				return std::make_pair<Type, Type>(right, left);
			return std::make_pair<Type, Type>(left, right);
		}();

		auto larg = larger;
		return intervalFilter<T>(
				smaller, [larg](T el) { return larg.contains(el); });
	}

	template<typename T>
	[[nodiscard]] llvm::SmallVector<Interval<T>, 2> intersection(
			const Interval<T>& left, const Interval<T>& right)
	{
		auto [firstElm, foundEl] = minCommon(left, right);
		if (!foundEl)
			return { Interval<T>(0, 0, 1) };

		// if the steps are unrelated
		// there is nothing we can do if not check everything
		if (!multipleOf(left.step(), right.step()))
			return simpleIntersection(left, right);

		T maxStep = std::max(left.step(), right.step());
		T lastElm = std::min(left.back(), right.back());
		return { Interval<T>(firstElm, lastElm, maxStep) };
	}

	template<typename T>
	[[nodiscard]] llvm::SmallVector<Interval<T>, 2> intervalUnion(
			const Interval<T>& left, const Interval<T>& right)
	{
		if (left.step() != right.step())
			return { left, right };
		if (left.front() == right.back())
			return { Interval<T>(left.front(), right.back(), right.step()) };
		if (left.front() == right.back())
			return { Interval<T>(right.front(), left.back(), right.step()) };

		auto [el, overlapping] = minCommon(left, right);
		if (!overlapping)
			return { left, right };

		return { Interval<T>(
				std::min(left.front(), right.front()),
				std::max(left.back(), right.back()),
				left.step()) };
	}

	template<typename T>
	[[nodiscard]] llvm::SmallVector<Interval<T>, 2> intervalDifference(
			const Interval<T>& original, const Interval<T>& toRemove)
	{
		auto [position, notDisjoint] = minCommon(original, toRemove);
		if (!notDisjoint)
			return { original };

		llvm::SmallVector<Interval<T>, 2> toReturn;
		if (original.step() == toRemove.step())
		{
			if (original.front() < toRemove.front())
				toReturn.emplace_back(original.front(), position, original.step());
			if (original.back() > toRemove.back())
				toReturn.emplace_back(
						toRemove.back(), original.back(), original.step());
			return toReturn;
		}

		return intervalFilter<T>(
				original, [&](T el) { return !toRemove.contains(el); });
	}
}	 // namespace modelica
