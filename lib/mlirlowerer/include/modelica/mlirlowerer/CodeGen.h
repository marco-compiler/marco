#pragma once

#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <modelica/frontend/AST.h>
#include <modelica/frontend/SymbolTable.hpp>
#include <modelica/matching/Matching.hpp>
#include <modelica/matching/SccCollapsing.hpp>
#include <modelica/matching/Schedule.hpp>
#include <modelica/model/ModType.hpp>
#include <modelica/model/Model.hpp>
#include <modelica/omcToModel/OmcToModelPass.hpp>
#include <modelica/passes/ConstantFold.hpp>
#include <modelica/passes/SolveModel.hpp>
#include <modelica/utils/SourcePosition.h>

#include "ModelicaBuilder.h"
#include "Passes.h"

namespace modelica::codegen
{
	struct ModelicaOptions {

		bool x64 = true;
		double startTime = 0;
		double endTime = 10;
		double timeStep = 0.1;

		[[nodiscard]] unsigned int getBitWidth() const
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
		bool resultBuffersToArgs = true;
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

		llvm::Optional<mlir::ModuleOp> lower(llvm::ArrayRef<frontend::ClassContainer> classes);

		private:
		mlir::Operation* lower(frontend::Class& cls);
		mlir::Operation* lower(frontend::Function& function);
		mlir::Operation* lower(frontend::Package& package);
		mlir::Operation* lower(frontend::Record& record);

		mlir::Type lower(frontend::Type& type, BufferAllocationScope allocationScope);
		mlir::Type lower(frontend::BuiltInType& type, BufferAllocationScope allocationScope);
		mlir::Type lower(frontend::PackedType& type, BufferAllocationScope allocationScope);
		mlir::Type lower(frontend::UserDefinedType& type, BufferAllocationScope allocationScope);

		template<typename Context>
		void lower(frontend::Member& member);

		void lower(frontend::Equation& equation);
		void lower(frontend::ForEquation& forEquation);

		void lower(frontend::Algorithm& algorithm);
		void lower(frontend::Statement& statement);
		void lower(frontend::AssignmentStatement& statement);
		void lower(frontend::IfStatement& statement);
		void lower(frontend::ForStatement& statement);
		void lower(frontend::WhileStatement& statement);
		void lower(frontend::WhenStatement& statement);
		void lower(frontend::BreakStatement& statement);
		void lower(frontend::ReturnStatement& statement);

		void assign(mlir::Location location, Reference memory, mlir::Value value);

		template<typename T>
		Container<Reference> lower(frontend::Expression& expression);

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
		Container<mlir::Value> lowerOperationArgs(frontend::Operation& operation);

		/**
		 * Helper to convert an AST location to a MLIR location.
		 *
		 * @param location frontend location
		 * @return MLIR location
		 */
		mlir::Location loc(SourcePosition location);

		ModelicaOptions options;
	};

	template<>
	void MLIRLowerer::lower<frontend::Class>(frontend::Member& member);

	template<>
	void MLIRLowerer::lower<frontend::Function>(frontend::Member& member);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<frontend::Expression>(frontend::Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<frontend::Operation>(frontend::Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<frontend::Constant>(frontend::Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<frontend::ReferenceAccess>(frontend::Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<frontend::Call>(frontend::Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<frontend::Tuple>(frontend::Expression& expression);

	template<>
	MLIRLowerer::Container<Reference> MLIRLowerer::lower<frontend::Array>(frontend::Expression& expression);
}
