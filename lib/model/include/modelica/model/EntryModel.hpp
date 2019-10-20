#pragma once

#include "modelica/model/Model.hpp"

namespace modelica
{
	class EntryModel: public Model
	{
		public:
		EntryModel(
				std::vector<ModEquation> equations, llvm::StringMap<ModVariable> vars)
				: Model(std::move(equations), std::move(vars))
		{
		}
		EntryModel() = default;

		void dump(llvm::raw_ostream& OS = llvm::outs()) const
		{
			OS << "vars\n";
			for (auto i = varbegin(); i != varend(); i++)
			{
				OS << i->first() << " = ";
				i->second.getInit().dump(OS);

				OS << "\n";
			}

			OS << "equations\n";
			for (const auto& update : *this)
				update.dump(OS);
		}
	};
}	 // namespace modelica
