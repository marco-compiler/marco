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

		template<typename T>
		std::unique_ptr<T> visit(std::unique_ptr<T> decl)
		{
			return decl;
		}

		template<typename T>
		void afterChildrenVisit(T*)
		{
		}

		private:
		bool handleSimpleMod(llvm::StringRef name, const SimpleModification& mod);
		EntryModel& model;
	};
}	 // namespace modelica
