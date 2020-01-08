#pragma once
#include "modelica/model/ModExp.hpp"
#include "modelica/utils/IndexSet.hpp"

namespace modelica
{
	class ModVariable
	{
		public:
		ModVariable(std::string name, ModExp exp)
				: name(std::move(name)), init(std::move(exp))
		{
		}

		[[nodiscard]] const std::string& getName() const { return name; }
		[[nodiscard]] const ModExp& getInit() const { return init; }
		ModExp& getInit() { return init; }
		[[nodiscard]] IndexSet toIndexSet() const;

		private:
		std::string name;
		ModExp init;
	};
}	 // namespace modelica
