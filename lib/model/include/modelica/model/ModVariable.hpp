#pragma once
#include <limits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include "modelica/model/ModExp.hpp"
#include "modelica/utils/IRange.hpp"
#include "modelica/utils/IndexSet.hpp"
#include "modelica/utils/Interval.hpp"

namespace modelica
{
	class ModVariable
	{
		public:
		ModVariable(
				std::string name,
				ModExp exp,
				bool isState = false,
				bool isConst = false)
				: name(std::move(name)),
					init(std::move(exp)),
					isConst(isConst),
					contribuitesToState(isState)
		{
		}

		[[nodiscard]] const std::string& getName() const { return name; }
		[[nodiscard]] const ModExp& getInit() const { return init; }
		ModExp& getInit() { return init; }
		[[nodiscard]] IndexSet toIndexSet() const;
		[[nodiscard]] MultiDimInterval toMultiDimInterval() const;
		void setIsState(bool isState) { contribuitesToState = isState; }
		[[nodiscard]] bool isState() const { return contribuitesToState; }
		[[nodiscard]] bool isConstant() const { return isConst; }
		void dump(llvm::raw_ostream& OS) const;
		[[nodiscard]] size_t size() const { return init.getModType().flatSize(); }
		[[nodiscard]] size_t indexOfElement(llvm::ArrayRef<size_t> access) const
		{
			const auto& type = getInit().getModType();
			assert(access.size() == type.getDimensionsCount());
			size_t index = 0;
			size_t maxIndex = 1;
			for (size_t i = access.size() - 1;
					 i != std::numeric_limits<size_t>::max();
					 i--)
			{
				index += access[i] * maxIndex;
				maxIndex *= type.getDimension(i);
			}

			return index;
		}

		private:
		std::string name;
		ModExp init;
		bool isConst;
		bool contribuitesToState;
	};
}	 // namespace modelica
