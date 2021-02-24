#pragma once

#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <modelica/frontend/ClassContainer.hpp>
#include <modelica/mlirlowerer/ModelicaBuilder.h>
#include <modelica/utils/SourceRange.hpp>

namespace modelica
{
	struct ModelicaOptions {

		bool x64 = false;

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

	/**
	 * Convert an MLIR module to the LLVM dialect.
	 *
	 * @param context MLIR context
	 * @param module  module
	 * @param options Modelica lowering options
	 * @return success if the conversion was successful
	 */
	[[nodiscard]] mlir::LogicalResult convertToLLVMDialect(mlir::MLIRContext* context, mlir::ModuleOp module, ModelicaOptions options = ModelicaOptions::getDefaultOptions());

	class Reference
	{
		public:
		Reference();

		[[nodiscard]] mlir::Value operator*();

		[[nodiscard]] mlir::Value getReference() const;

		[[nodiscard]] static Reference ssa(ModelicaBuilder* builder, mlir::Value value);
		[[nodiscard]] static Reference memref(ModelicaBuilder* builder, mlir::Value value);
		[[nodiscard]] static Reference placeholder(ModelicaBuilder* builder);

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

	class MlirLowerer
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		explicit MlirLowerer(mlir::MLIRContext& context, ModelicaOptions options = ModelicaOptions::getDefaultOptions());

		mlir::ModuleOp lower(llvm::ArrayRef<const modelica::ClassContainer> classes);
		mlir::Operation* lower(const modelica::Class& cls);
		mlir::FuncOp lower(const modelica::Function& function);
		mlir::Type lower(const modelica::Type& type);
		mlir::Type lower(const modelica::BuiltInType& type);
		mlir::Type lower(const modelica::UserDefinedType& type);
		void lower(const modelica::Member& member);
		void lower(const modelica::Algorithm& algorithm);
		void lower(const modelica::Statement& statement);
		void lower(const modelica::AssignmentStatement& statement);
		void lower(const modelica::IfStatement& statement);
		void lower(const modelica::ForStatement& statement);
		void lower(const modelica::WhileStatement& statement);
		void lower(const modelica::WhenStatement& statement);
		void lower(const modelica::BreakStatement& statement);
		void lower(const modelica::ReturnStatement& statement);

		template<typename T>
		Container<Reference> lower(const modelica::Expression& expression);

		private:
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
		 * Options for the lowerer.
		 */
		 ModelicaOptions options;

		/**
		 * Lower the arguments of an operation.
		 *
		 * @param operation operation whose arguments have to be lowered
		 * @return lowered args
 		 */
		Container<mlir::Value> lowerOperationArgs(const modelica::Operation& operation);

		/**
		 * Helper to convert an AST location to a MLIR location.
		 *
		 * @param location frontend location
		 * @return MLIR location
		 */
		mlir::Location loc(SourcePosition location);

		mlir::Type constantToType(const Constant& constant)
		{
			if (constant.isA<BuiltInType::Boolean>())
				return builder.getBooleanType();

			if (constant.isA<BuiltInType::Integer>())
				return builder.getIntegerType();

			if (constant.isA<BuiltInType::Float>())
				return builder.getRealType();

			assert(false && "Unreachable");
			return builder.getNoneType();
		}

		template<typename T>
		constexpr mlir::Attribute getAttribute(const T& value)
		{
			constexpr BuiltInType type = typeToFrontendType<T>();

			if constexpr (type == BuiltInType::Boolean)
				return builder.getBoolAttr(value);
			else if constexpr (type == BuiltInType::Integer)
				return builder.getIntegerAttribute(value);
			else if constexpr (type == BuiltInType::Float)
				return builder.getRealAttribute(value);

			assert(false && "Unknown type");
			return builder.getZeroAttr(builder.getNoneType());
		}

		mlir::Value cast(mlir::Value value, mlir::Type destination);
	};

	template<>
	MlirLowerer::Container<Reference> MlirLowerer::lower<modelica::Expression>(const modelica::Expression& expression);

	template<>
	MlirLowerer::Container<Reference> MlirLowerer::lower<modelica::Operation>(const modelica::Expression& expression);

	template<>
	MlirLowerer::Container<Reference> MlirLowerer::lower<modelica::Constant>(const modelica::Expression& expression);

	template<>
	MlirLowerer::Container<Reference> MlirLowerer::lower<modelica::ReferenceAccess>(const modelica::Expression& expression);

	template<>
	MlirLowerer::Container<Reference> MlirLowerer::lower<modelica::Call>(const modelica::Expression& expression);

	template<>
	MlirLowerer::Container<Reference> MlirLowerer::lower<modelica::Tuple>(const modelica::Expression& expression);
}
