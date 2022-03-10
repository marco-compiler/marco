#ifndef MARCO_CODEGEN_TRANSFORMS_MODEL_MODELCONVERTER_H
#define MARCO_CODEGEN_TRANSFORMS_MODEL_MODELCONVERTER_H

#include "llvm/ADT/StringRef.h"
#include "marco/Codegen/Conversion/Modelica/TypeConverter.h"
#include "marco/Codegen/dialects/modelica/ModelicaDialect.h"
#include "marco/Codegen/Transforms/Model/Scheduling.h"
#include "marco/Codegen/Transforms/ModelSolving.h"
#include <map>
#include <set>

namespace marco::codegen
{
  class ModelConverter
  {
    private:
      // The extra data within the simulation data structure that is placed
      // before the simulation variables.

      static constexpr size_t timeVariablePosition = 0;

      // The derivatives map keeps track of whether a variable is the derivative
      // of another one. Each variable is identified by its position within the
      // list of the "body" region arguments.

      using DerivativesPositionsMap = std::map<size_t, size_t>;

      // Name for the functions of the simulation
      static constexpr llvm::StringLiteral mainFunctionName = "main";
      static constexpr llvm::StringLiteral initFunctionName = "init";
      static constexpr llvm::StringLiteral updateNonStateVariablesFunctionName = "updateNonStateVariables";
      static constexpr llvm::StringLiteral updateStateVariablesFunctionName = "updateStateVariables";
      static constexpr llvm::StringLiteral incrementTimeFunctionName = "incrementTime";
      static constexpr llvm::StringLiteral printHeaderFunctionName = "printHeader";
      static constexpr llvm::StringLiteral printFunctionName = "print";
      static constexpr llvm::StringLiteral deinitFunctionName = "deinit";
      static constexpr llvm::StringLiteral runFunctionName = "runSimulation";

      struct ConversionInfo
      {
        std::set<std::unique_ptr<Equation>> explicitEquations;
        std::map<ScheduledEquation*, Equation*> explicitEquationsMap;
        std::set<ScheduledEquation*> implicitEquations;
        std::set<ScheduledEquation*> cyclicEquations;

        std::set<unsigned int> IDAVariables;
        std::set<ScheduledEquation*> IDAEquations;
      };

    public:
      ModelConverter(SolveModelOptions options, TypeConverter& typeConverter)
          : options(std::move(options)),
            typeConverter(&typeConverter)
      {
      }

      /// Convert a scheduled model into the algorithmic functions that compose the simulation.
      /// The usage of such functions is delegated to the runtime library, which is statically
      /// linked with the code generated by the compiler. This decoupling allows to relieve the
      /// code generation phase from the generation of functions that are independent from the
      /// model being processed.
      mlir::LogicalResult convert(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model,
          const mlir::BlockAndValueMapping& derivatives) const;

    private:
      /// Get the MLIR type corresponding to void*.
      mlir::Type getVoidPtrType() const;

      mlir::LLVM::LLVMFuncOp lookupOrCreateHeapAllocFn(mlir::OpBuilder& builder, mlir::ModuleOp module) const;

      mlir::LLVM::LLVMFuncOp lookupOrCreateHeapFreeFn(mlir::OpBuilder& builder, mlir::ModuleOp module) const;

      /// Create the main function, which is called when the executable of the simulation is run.
      /// In order to keep the code generation simpler, the real implementation of the function
      /// managing the simulation lives within the runtime library and the main just consists in
      /// a call to such function.
      mlir::LogicalResult createMainFunction(
          mlir::OpBuilder& builder, const Model<ScheduledEquationsBlock>& model) const;

      /// Load the data structure from the opaque pointer that is passed around the
      /// simulation functions.
      ///
      /// @param builder	 operation builder
      /// @param ptr 	     opaque pointer
      /// @param varTypes  types of the variables
      /// @return data structure containing the variables
      mlir::Value loadDataFromOpaquePtr(mlir::OpBuilder& builder, mlir::Value ptr, mlir::TypeRange varTypes) const;

      /// Extract a value from the data structure shared between the various
      /// simulation main functions.
      ///
      /// @param builder 			  operation builder
      /// @param typeConverter  type converter
      /// @param structValue 	  data structure
      /// @param type 				  value type
      /// @param position 		  value position
      /// @return extracted value
      mlir::Value extractValue(
          mlir::OpBuilder& builder, mlir::Value structValue, mlir::Type type, unsigned int position) const;

      /// Bufferize the variables and convert the subsequent load/store operations to operate on the
      /// allocated memory buffer.
      mlir::Value convertMember(mlir::OpBuilder& builder, modelica::MemberCreateOp op) const;

      /// Create the initialization function that allocates the variables and
      /// stores them into an appropriate data structure to be passed to the other
      /// simulation functions.
      mlir::LogicalResult createInitFunction(
          mlir::OpBuilder& builder,
          modelica::ModelOp modelOp,
          const ConversionInfo& conversionInfo,
          const mlir::BlockAndValueMapping& derivatives) const;

      /// Create a function to be called when the simulation has finished and the
      /// variables together with its data structure are not required anymore and
      /// thus can be deallocated.
      mlir::LogicalResult createDeinitFunction(mlir::OpBuilder& builder, modelica::ModelOp modelOp) const;

      mlir::FuncOp createEquationFunction(
          mlir::OpBuilder& builder,
          const ScheduledEquation& equation,
          llvm::StringRef equationFunctionName,
          mlir::FuncOp templateFunction,
          std::multimap<mlir::FuncOp, mlir::CallOp>& equationTemplateCalls,
          mlir::TypeRange varsTypes) const;

      mlir::LogicalResult createUpdateNonStateVariablesFunction(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model,
          const ConversionInfo& conversionInfo) const;

      /// Create the functions that calculates the values that the state variables will have
      /// in the next iteration.
      mlir::LogicalResult createUpdateStateVariablesFunction(
          mlir::OpBuilder& builder, modelica::ModelOp modelOp, const DerivativesPositionsMap& derivatives) const;

      mlir::LogicalResult createIncrementTimeFunction(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model) const;

      void printSeparator(mlir::OpBuilder& builder, mlir::Value separator) const;

      mlir::Value getOrCreateGlobalString(
          mlir::Location loc,
          mlir::OpBuilder& builder,
          mlir::StringRef name,
          mlir::StringRef value,
          mlir::ModuleOp module) const;

      mlir::Value getSeparatorString(mlir::Location loc, mlir::OpBuilder& builder, mlir::ModuleOp module) const;

      mlir::Value getNewlineString(mlir::Location loc, mlir::OpBuilder& builder, mlir::ModuleOp module) const;

      mlir::LLVM::LLVMFuncOp getOrInsertPrintf(mlir::OpBuilder& builder, mlir::ModuleOp module) const;

      void printVariableName(
          mlir::OpBuilder& builder,
          mlir::Value name,
          mlir::Type type,
          VariableFilter::Filter filter,
          std::function<mlir::Value()> structValue,
          unsigned int position,
          mlir::ModuleOp module,
          mlir::Value separator,
          bool shouldPreprendSeparator = true) const;

      void printScalarVariableName(
          mlir::OpBuilder& builder,
          mlir::Value name,
          mlir::ModuleOp module,
          mlir::Value separator,
          bool shouldPrependSeparator) const;

      void printArrayVariableName(
          mlir::OpBuilder& builder,
          mlir::Value name,
          mlir::Type type,
          VariableFilter::Filter filter,
          std::function<mlir::Value()> structValue,
          unsigned int position,
          mlir::ModuleOp module,
          mlir::Value separator,
          bool shouldPrependSeparator) const;

      mlir::LogicalResult createPrintHeaderFunction(
          mlir::OpBuilder& builder,
          modelica::ModelOp op,
          DerivativesPositionsMap& derivativesPositions) const;

      void printVariable(
          mlir::OpBuilder& builder,
          mlir::Value var,
          VariableFilter::Filter filter,
          mlir::Value separator,
          bool shouldPreprendSeparator = true) const;

      void printScalarVariable(
          mlir::OpBuilder& builder, mlir::Value var, mlir::Value separator, bool shouldPreprendSeparator = true) const;

      void printArrayVariable(
          mlir::OpBuilder& builder,
          mlir::Value var,
          VariableFilter::Filter filter,
          mlir::Value separator,
          bool shouldPreprendSeparator = true) const;

      void printElement(mlir::OpBuilder& builder, mlir::Value value) const;

      mlir::LogicalResult createPrintFunction(
          mlir::OpBuilder& builder,
          modelica::ModelOp op,
          DerivativesPositionsMap& derivativesPositions) const;

      mlir::LogicalResult createPrintFunctionBody(
          mlir::OpBuilder& builder,
          modelica::ModelOp op,
          mlir::TypeRange varTypes,
          DerivativesPositionsMap& derivativesPositions,
          llvm::StringRef functionName,
          std::function<mlir::LogicalResult(std::function<mlir::Value()>, llvm::StringRef, unsigned int, VariableFilter::Filter, mlir::Value)> elementCallback) const;

    private:
      SolveModelOptions options;
      TypeConverter* typeConverter;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODEL_MODELCONVERTER_H