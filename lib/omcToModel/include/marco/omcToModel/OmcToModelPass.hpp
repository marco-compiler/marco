#pragma once

#include "llvm/Support/Error.h"
#include "marco/frontend/AST.h"
#include "marco/frontend/SymbolTable.hpp"
#include "marco/model/ModCall.hpp"
#include "marco/model/ModExp.hpp"
#include "marco/model/ModType.hpp"
#include "marco/model/Model.hpp"

namespace marco
{
	class OmcToModelPass
	{
		public:
		OmcToModelPass(Model& toPopulate): model(toPopulate) {}

		template<typename T>
		[[nodiscard]] llvm::Error lower(
				frontend::Class& cls, const frontend::SymbolTable& table);

		[[nodiscard]] llvm::Expected<ModEquation> lower(
				frontend::Equation& eq, const frontend::SymbolTable& table, int nestingLevel);

		[[nodiscard]] llvm::Expected<ModEquation> lower(
				frontend::ForEquation& eq, const frontend::SymbolTable& table);

		[[nodiscard]] llvm::Expected<ModExp> lower(
				frontend::Expression& exp, const frontend::SymbolTable& table);

		[[nodiscard]] llvm::Expected<ModCall> lowerCall(
				frontend::Expression& call, const frontend::SymbolTable& table);

		[[nodiscard]] llvm::Expected<ModExp> lowerOperation(
				frontend::Expression& op, const frontend::SymbolTable& table);

		[[nodiscard]] llvm::Error lower(
				frontend::Member& member, const frontend::SymbolTable& table);

		[[nodiscard]] llvm::Expected<ModType> lower(
				const frontend::Type& tp, const frontend::SymbolTable& table);

		[[nodiscard]] llvm::Expected<ModExp> lowerReference(
				frontend::Expression& ref, const frontend::SymbolTable& table);

		[[nodiscard]] llvm::Expected<ModExp> initializer(
				frontend::Member& member, const frontend::SymbolTable& table);

		[[nodiscard]] llvm::Expected<ModExp> lowerStart(
				frontend::Member& member, const frontend::SymbolTable& table);

		[[nodiscard]] llvm::Expected<ModExp> defaultInitializer(
				const frontend::Member& mem, const frontend::SymbolTable& table);

		private:
		Model& model;
	};

	template<>
	llvm::Error OmcToModelPass::lower<frontend::Class>(
			frontend::Class& cls, const frontend::SymbolTable& table);

	template<>
	llvm::Error OmcToModelPass::lower<frontend::PartialDerFunction>(
			frontend::Class& cls, const frontend::SymbolTable& table);

	template<>
	llvm::Error OmcToModelPass::lower<frontend::StandardFunction>(
			frontend::Class& cls, const frontend::SymbolTable& table);

	template<>
	llvm::Error OmcToModelPass::lower<frontend::Model>(
			frontend::Class& cls, const frontend::SymbolTable& table);

	template<>
	llvm::Error OmcToModelPass::lower<frontend::Package>(
			frontend::Class& cls, const frontend::SymbolTable& table);

	template<>
	llvm::Error OmcToModelPass::lower<frontend::Record>(
			frontend::Class& cls, const frontend::SymbolTable& table);
}	 // namespace marco
