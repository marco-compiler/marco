#pragma once

#include "llvm/ADT/SmallVector.h"
#include "modelica/model/Model.hpp"

namespace modelica
{
	class EntryModel: public Model
	{
		public:
		EntryModel(
				llvm::SmallVector<ModEquation, 2> equations,
				llvm::StringMap<ModVariable> vars)
				: Model(std::move(equations), std::move(vars))
		{
		}
		EntryModel() = default;

		void dump(llvm::raw_ostream& OS = llvm::outs()) const
		{
			OS << "init\n";
			for (auto i = varbegin(); i != varend(); i++)
			{
				OS << i->first() << " = ";
				i->second.getInit().dump(OS);

				OS << "\n";
			}

			OS << "update\n";
			for (const auto& update : *this)
				update.dump(OS);
		}
	};
}	 // namespace modelica
