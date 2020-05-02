#pragma once

#include "llvm/Support/Error.h"
#include "modelica/frontend/Class.hpp"
#include "modelica/frontend/Equation.hpp"
#include "modelica/frontend/Expression.hpp"
#include "modelica/frontend/ForEquation.hpp"
#include "modelica/frontend/SymbolTable.hpp"
#include "modelica/model/EntryModel.hpp"
#include "modelica/model/ModCall.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModType.hpp"

namespace modelica
{
	class OmcToModelPass
	{
		public:
		OmcToModelPass(EntryModel& toPopulate): model(toPopulate) {}

		[[nodiscard]] llvm::Error lower(Class& cl, const SymbolTable& table);
		[[nodiscard]] llvm::Expected<ModEquation> lower(
				Equation& eq, const SymbolTable& table);
		[[nodiscard]] llvm::Expected<ModEquation> lower(
				ForEquation& eq, const SymbolTable& table);
		[[nodiscard]] llvm::Expected<ModExp> lower(
				Expression& exp, const SymbolTable& table);
		[[nodiscard]] llvm::Expected<ModCall> lowerCall(
				Expression& call, const SymbolTable& table);

		[[nodiscard]] llvm::Expected<ModExp> lowerOperation(
				Expression& op, const SymbolTable& table);
		[[nodiscard]] llvm::Error lower(Member& member, const SymbolTable& table);
		[[nodiscard]] llvm::Expected<ModType> lower(
				const Type& tp, const SymbolTable& table);

		[[nodiscard]] llvm::Expected<ModExp> lowerReference(
				Expression& ref, const SymbolTable& table);

		[[nodiscard]] llvm::Expected<ModExp> initializer(
				Member& member, const SymbolTable& table);

		[[nodiscard]] llvm::Expected<ModExp> lowerStart(
				Member& member, const SymbolTable& table);

		[[nodiscard]] llvm::Expected<ModExp> defaultInitializer(
				const Member& mem, const SymbolTable& table);

		private:
		EntryModel& model;
	};
}	 // namespace modelica
