#pragma once
#include "llvm/Support/raw_ostream.h"
#include "modelica/model/ModExp.hpp"
#include "modelica/utils/IndexSet.hpp"

namespace modelica
{
	class ModVariable
	{
		public:
		ModVariable(std::string name, ModExp exp, bool isState = true)
				: name(std::move(name)),
					init(std::move(exp)),
					contribuitesToState(isState)
		{
		}

		[[nodiscard]] const std::string& getName() const { return name; }
		[[nodiscard]] const ModExp& getInit() const { return init; }
		ModExp& getInit() { return init; }
		[[nodiscard]] IndexSet toIndexSet() const;
		[[nodiscard]] bool isState() const { return contribuitesToState; }
		void dump(llvm::raw_ostream& OS) const;

		private:
		std::string name;
		ModExp init;
		bool contribuitesToState;
	};
}	 // namespace modelica
