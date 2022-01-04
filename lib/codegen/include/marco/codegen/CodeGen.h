#pragma once

#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <marco/ast/AST.h>
#include <marco/ast/SymbolTable.hpp>
#include <marco/codegen/dialects/modelica/ModelicaBuilder.h>
#include <marco/utils/SourcePosition.h>

#include "Passes.h"

namespace marco::codegen
{
	struct ModelicaOptions
	{
		double startTime = 0;
		double endTime = 10;
		double timeStep = 0.1;

		/**
		 * Get a statically allocated copy of the default options.
		 *
		 * @return default options
		 */
		static const ModelicaOptions& getDefaultOptions() {
			static ModelicaOptions options;
			return options;
		}
	};

	class Reference
	{
		public:
		Reference();

		[[nodiscard]] mlir::Value operator*();
		[[nodiscard]] mlir::Value getReference() const;

		void set(mlir::Value value);

		[[nodiscard]] static Reference ssa(mlir::OpBuilder* builder, mlir::Value value);
		[[nodiscard]] static Reference memory(mlir::OpBuilder* builder, mlir::Value value);
		[[nodiscard]] static Reference member(mlir::OpBuilder* builder, mlir::Value value);

		private:
		Reference(mlir::OpBuilder* builder,
							mlir::Value value,
							std::function<mlir::Value(mlir::OpBuilder*, mlir::Value)> reader,
							std::function<void(mlir::OpBuilder* builder, Reference& destination, mlir::Value)> writer);

		mlir::OpBuilder* builder;
		mlir::Value value;
		std::function<mlir::Value(mlir::OpBuilder*, mlir::Value)> reader;
		std::function<void(mlir::OpBuilder*, Reference&, mlir::Value)> writer;
	};

	class MLIRLowerer
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		explicit MLIRLowerer(mlir::MLIRContext& context, ModelicaOptions options = ModelicaOptions::getDefaultOptions());

		llvm::Optional<mlir::ModuleOp> run(llvm::ArrayRef<std::unique_ptr<ast::Class>> classes);

		private:
		mlir::Operation* lower(const ast::Class& cls);
		mlir::Operation* lower(const ast::PartialDerFunction& function);
		mlir::Operation* lower(const ast::StandardFunction& function);
		mlir::Operation* lower(const ast::Model& model);
		mlir::Operation* lower(const ast::Package& package);
		mlir::Operation* lower(const ast::Record& record);

		mlir::Type lower(const ast::Type& type, modelica::BufferAllocationScope allocationScope);
		mlir::Type lower(const ast::BuiltInType& type, modelica::BufferAllocationScope allocationScope);
		mlir::Type lower(const ast::PackedType& type, modelica::BufferAllocationScope allocationScope);
		mlir::Type lower(const ast::UserDefinedType& type, modelica::BufferAllocationScope allocationScope);

		template<typename Context>
		void lower(const ast::Member& member);

		void lower(const ast::Equation& equation);
		void lower(const ast::ForEquation& forEquation);

		void lower(const ast::Algorithm& algorithm);
		void lower(const ast::Statement& statement);
		void lower(const ast::AssignmentStatement& statement);
		void lower(const ast::IfStatement& statement);
		void lower(const ast::ForStatement& statement);
		void lower(const ast::WhileStatement& statement);
		void lower(const ast::WhenStatement& statement);
		void lower(const ast::BreakStatement& statement);
		void lower(const ast::ReturnStatement& statement);

		template<typename T>
		Container<Reference> lower(const ast::Expression& expression);

		/**
		 * The builder is a helper class to create IR inside a function. The
		 * builder is stateful, in particular it keeps an "insertion point":
		 * this is where the next operations will be introduced.
		 */
		modelica::ModelicaBuilder builder;

		/**
		 * The symbol table maps a variable name to a value in the current scope.
		 * Entering a function creates a new scope, and the function arguments
		 * are added to the mapping. When the processing of a function is
		 * terminated, the scope is destroyed and the mappings created in this
		 * scope are dropped.
		 */
		llvm::ScopedHashTable<llvm::StringRef, Reference> symbolTable;

		/**
		 * The stack represent the list of the nested scope names in which the
		 * lowerer currently is.
		 */
		std::deque<llvm::StringRef> scopes;

		 /**
		  * Apply a binary operation to a list of values.
		  *
		  * @param args      arguments
		  * @param callback  callback that should process the current args and return a result
		  * @return folded value
		  */
		 mlir::Value foldBinaryOperation(llvm::ArrayRef<mlir::Value> args, std::function<mlir::Value(mlir::Value, mlir::Value)> callback);

		/**
		 * Lower the arguments of an operation.
		 *
		 * @param operation operation whose arguments have to be lowered
		 * @return lowered args
 		 */
		Container<mlir::Value> lowerOperationArgs(const ast::Operation& operation);

		/**
		 * Helper to convert an AST location to a MLIR location.
		 *
		 * @param location frontend location
		 * @return MLIR location
		 */
		mlir::Location loc(SourcePosition location);

		/**
		 * Helper to convert an AST location to a MLIR location.
		 *
		 * @param location frontend location
		 * @return MLIR location
		 */
		mlir::Location loc(SourceRange location);

		ModelicaOptions options;
	};

	template<>
	void MLIRLowerer::lower<ast::Model>(
			const ast::Member& member);

	template<>
	void MLIRLowerer::lower<ast::Function>(
			const ast::Member& member);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<ast::Expression>(
			const ast::Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<ast::Operation>(
			const ast::Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<ast::Constant>(
			const ast::Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<ast::ReferenceAccess>(
			const ast::Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<ast::Call>(
			const ast::Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<ast::Tuple>(
			const ast::Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<ast::Array>(
			const ast::Expression& expression);
}
