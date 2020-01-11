#pragma once
#include "llvm/Support/raw_ostream.h"
#include "modelica/utils/Interval.hpp"

namespace modelica
{
	class IndexSet
	{
		public:
		IndexSet() = default;
		IndexSet(MultiDimInterval interval): values({ std::move(interval) }) {}
		template<typename... T>
		[[nodiscard]] bool contains(T... arg) const
		{
			return contains({ arg... });
		}
		[[nodiscard]] auto begin() const { return values.begin(); }
		[[nodiscard]] auto begin() { return values.begin(); }
		[[nodiscard]] auto end() { return values.end(); }
		[[nodiscard]] auto end() const { return values.end(); }
		[[nodiscard]] bool contains(llvm::ArrayRef<size_t> point) const;
		void unite(IndexSet other)
		{
			for (auto& range : other)
				unite(std::move(range));
		}
		void unite(std::initializer_list<Interval> list)
		{
			unite(MultiDimInterval(std::move(list)));
		}
		void unite(MultiDimInterval other);
		void intersecate(const IndexSet& other);
		void intersecate(const MultiDimInterval& other);
		void remove(const IndexSet& other);
		void remove(const MultiDimInterval& other);
		[[nodiscard]] bool disjoint(const IndexSet& other) const
		{
			for (auto& el : other)
				if (!disjoint(el))
					return false;
			return true;
		}
		[[nodiscard]] bool disjoint(const MultiDimInterval& other) const;
		[[nodiscard]] bool empty() const { return values.empty(); }
		[[nodiscard]] size_t partitionsCount() const { return values.size(); }

		[[nodiscard]] bool operator==(const IndexSet& other) const
		{
			return values == other.values;
		}
		[[nodiscard]] bool operator!=(const IndexSet& other) const
		{
			return !(values == other.values);
		}
		[[nodiscard]] size_t size() const
		{
			size_t toReturn = 0;
			for (const auto& el : values)
				toReturn += el.size();
			return toReturn;
		}

		void dump(llvm::raw_ostream& OS) const;
		[[nodiscard]] std::string toString() const;

		private:
		void compact();
		llvm::SmallVector<MultiDimInterval, 2> values;
	};

	[[nodiscard]] IndexSet remove(
			const MultiDimInterval& left, const MultiDimInterval& right);
}	 // namespace modelica
