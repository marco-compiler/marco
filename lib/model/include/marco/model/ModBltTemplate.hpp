#pragma once

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "marco/model/ModEquation.hpp"

namespace marco
{
	class ModBltTemplate
	{
		public:
		ModBltTemplate(
				llvm::SmallVector<ModEquation, 3> equs, std::string templateName)
				: equations(equs), templateName(templateName)
		{
		}

		void dump(bool dumpName = true, llvm::raw_ostream& OS = llvm::outs()) const
		{
			if (dumpName)
				OS << "blt-block-" << templateName << "\n";

			OS << "\ttemplate\n";
			for (const ModEquation& eq : equations)
			{
				OS << "\t";
				eq.getTemplate()->dump(true, OS);
				OS << "\n";
			}

			OS << "\tupdate\n";
			for (const ModEquation& eq : equations)
			{
				OS << "\t";
				eq.dump(OS);
			}
		}

		[[nodiscard]] const std::string& getName() const { return templateName; }
		void setName(std::string name) { templateName = name; }

		private:
		llvm::SmallVector<ModEquation, 3> equations;
		std::string templateName;
	};
}	 // namespace marco
