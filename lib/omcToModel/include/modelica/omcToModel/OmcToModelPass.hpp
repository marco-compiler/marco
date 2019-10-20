#pragma once

#include "modelica/AST/Visitor.hpp"
#include "modelica/model/EntryModel.hpp"

namespace modelica
{
	class OmcToModelPass
	{
		public:
		OmcToModelPass(EntryModel& toPopulate): model(toPopulate) {}

		std::unique_ptr<ComponentClause> visit(
				std::unique_ptr<ComponentClause> decl);

		std::unique_ptr<ForEquation> visit(std::unique_ptr<ForEquation> decl);
		std::unique_ptr<SimpleEquation> visit(std::unique_ptr<SimpleEquation> decl);

		template<typename T>
		std::unique_ptr<T> visit(std::unique_ptr<T> decl)
		{
			return decl;
		}

		void afterChildrenVisit(Equation* eq)
		{
			if (llvm::isa<ForEquation>(eq))
				forEqNestingLevel--;
		}

		template<typename T>
		void afterChildrenVisit(T*)
		{
		}

		private:
		bool handleSimpleMod(llvm::StringRef name, const SimpleModification& mod);
		EntryModel& model;
		size_t forEqNestingLevel{ 0 };
	};
}	 // namespace modelica
