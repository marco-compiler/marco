#pragma once

#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <modelica/frontend/AST.h>
#include <modelica/utils/SourceRange.hpp>

#include "ModelicaBuilder.h"
#include "Passes.h"

namespace modelica
{
	struct ModelicaOptions {

		bool x64 = true;

		unsigned int getBitWidth()
		{
			if (x64)
				return 64;

			return 32;
		}

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

	struct ModelicaConversionOptions : public ModelicaToLLVMConversionOptions
	{
		bool inlining = true;
		bool cse = true;
		bool openmp = false;
		bool debug = true;

		/**
		 * Get a statically allocated copy of the default options.
		 *
		 * @return default options
		 */
		static const ModelicaConversionOptions& getDefaultOptions() {
			static ModelicaConversionOptions options;
			return options;
		}
	};

	class Reference
	{
		public:
		Reference();

		[[nodiscard]] mlir::Value operator*();
		[[nodiscard]] mlir::Value getReference() const;
		[[nodiscard]] bool isInitialized() const;

		[[nodiscard]] static Reference ssa(ModelicaBuilder* builder, mlir::Value value);
		[[nodiscard]] static Reference memory(ModelicaBuilder* builder, mlir::Value value, bool initialized);

		private:
		Reference(ModelicaBuilder* builder,
							mlir::Value value,
							bool initialized,
							std::function<mlir::Value(ModelicaBuilder*, mlir::Value)> reader);

		ModelicaBuilder* builder;
		mlir::Value value;
		bool initialized;
		std::function<mlir::Value(ModelicaBuilder* builder, mlir::Value ref)> reader;
	};

	class MLIRLowerer
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		explicit MLIRLowerer(mlir::MLIRContext& context, ModelicaOptions options = ModelicaOptions::getDefaultOptions());

		mlir::LogicalResult convertToLLVMDialect(mlir::ModuleOp& module, ModelicaConversionOptions options = ModelicaConversionOptions::getDefaultOptions());

		llvm::Optional<mlir::ModuleOp> lower(llvm::ArrayRef<modelica::ClassContainer> classes);

		private:
		mlir::Operation* lower(const Class& cls);
		mlir::FuncOp lower(const Function& function);
		mlir::Type lower(const Type& type, BufferAllocationScope allocationScope);
		mlir::Type lower(const BuiltInType& type, BufferAllocationScope allocationScope);
		mlir::Type lower(const UserDefinedType& type, BufferAllocationScope allocationScope);
		void lower(const Member& member);
		void lower(const Algorithm& algorithm);
		void lower(const Statement& statement);
		void lower(const AssignmentStatement& statement);
		void lower(const IfStatement& statement);
		void lower(const ForStatement& statement);
		void lower(const WhileStatement& statement);
		void lower(const WhenStatement& statement);
		void lower(const BreakStatement& statement);
		void lower(const ReturnStatement& statement);

		template<typename T>
		Container<Reference> lower(const Expression& expression);

		/**
		 * The builder is a helper class to create IR inside a function. The
		 * builder is stateful, in particular it keeps an "insertion point":
		 * this is where the next operations will be introduced.
		 */
		ModelicaBuilder builder;

		/**
		 * The symbol table maps a variable name to a value in the current scope.
		 * Entering a function creates a new scope, and the function arguments
		 * are added to the mapping. When the processing of a function is
		 * terminated, the scope is destroyed and the mappings created in this
		 * scope are dropped.
		 */
		llvm::ScopedHashTable<llvm::StringRef, Reference> symbolTable;

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
		Container<mlir::Value> lowerOperationArgs(const Operation& operation);

		/**
		 * Helper to convert an AST location to a MLIR location.
		 *
		 * @param location frontend location
		 * @return MLIR location
		 */
		mlir::Location loc(SourcePosition location);
	};

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<Expression>(const Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<Operation>(const Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<Constant>(const Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<ReferenceAccess>(const Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<Call>(const Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<Tuple>(const Expression& expression);
}
