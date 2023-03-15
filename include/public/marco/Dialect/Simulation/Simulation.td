#ifndef MARCO_DIALECTS_SIMULATION_SIMULATION_TD
#define MARCO_DIALECTS_SIMULATION_SIMULATION_TD

include "marco/Dialect/Simulation/SimulationAttributes.td"
include "marco/Dialect/Simulation/SimulationInterfaces.td"
include "marco/Dialect/Simulation/SimulationTypes.td"
include "marco/Dialect/Simulation/SimulationDialect.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

//===---------------------------------------------------------------------===//
// Simulation operation definitions
//===---------------------------------------------------------------------===//

class Simulation_Op<string mnemonic, list<Trait> traits = []> :
    Op<Simulation_Dialect, mnemonic, traits>;

def Simulation_ModuleOp : Simulation_Op<"module",
    [IsolatedFromAbove,
    NoRegionArguments,
    SingleBlock,
    NoTerminator,
    HasParent<"mlir::ModuleOp">]>
{
    let summary = "The module containing all the simulation functions.";

    let description = [{
        The operation is intended to contain all the operations that are
        tightly related to the functions managing the simulation.

        It expects three attributes:
          1. the name of the model
          2. the list of variables
          3. the map of the derivatives (in form of list)

        Example:

        ```mlir
        #var0 = #simulation.variable<...>
        #var1 = #simulation.variable<...>
        #der0 = #simulation.derivative<...>

        simulation.module attributes {modelName = "test", variables = [#var0, #var1], derivatives = [#der0]} {

        }
        ```
    }];

    let arguments = (ins
        DefaultValuedStrAttr<StrAttr, "<unknown>">:$modelName,
        DefaultValuedAttr<VariablesListAttr, "llvm::None">:$variables,
        DefaultValuedAttr<DerivativesListAttr, "llvm::None">:$derivatives);

    let regions = (region AnyRegion:$bodyRegion);

    let assemblyFormat = "attr-dict-with-keyword $bodyRegion";
    let skipDefaultBuilders = 1;

    let builders = [
        OpBuilder<(ins
            "::llvm::StringRef":$modelName,
            "::llvm::ArrayRef<::mlir::Attribute>":$variables,
            "::llvm::ArrayRef<::mlir::Attribute>":$derivatives)>
    ];

    let extraClassDeclaration = [{
        /// Get the types of the variables.
        ::llvm::SmallVector<::mlir::Type> getVariablesTypes();

        /// Get a map from each derivative name to the its derived variable
        /// name.
        ::llvm::StringMap<::llvm::StringRef> getDerivativesMap();
    }];
}

def Simulation_InitFunctionOp : Simulation_Op<"init_function",
    [IsolatedFromAbove,
    AutomaticAllocationScope,
    HasParent<"ModuleOp">,
    Simulation_FunctionOpInterface]>
{
    let summary = [{
        Function in charge of creating and initializing the simulation data.
    }];

    let description = [{
        The operation represents the function in charge of creating an
        initializing the simulation data.
        The body should contain the operations used to allocate (and, if
        necessary, initialize) the data structure representing each variable.
        The yielded values are the ones that will be stored (through copies) in
        the runtime data structure as representative of the variables (and,
        thus, must have value semantics).

        Example:

        ```mlir
        #var0 = simulation.variable<...> : !llvm.ptr<f64>
        #var1 = simulation.variable<...> : !llvm.ptr<f64>

        simulation.init_function () -> (!llvm.ptr<f64>, !llvm.ptr<f64>) {
            %0 = ... : !llvm.ptr<f64>
            %1 = ... : !llvm.ptr<f64>
            simulation.yield %0, %1 : !llvm.ptr<f64>, !llvm.ptr<f64>
        }
        ```
    }];

    let arguments = (ins TypeAttrOf<FunctionType>:$function_type);
    let regions = (region AnyRegion:$bodyRegion);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "::mlir::TypeRange":$variableTypes), [{
            auto functionType =
                $_builder.getFunctionType(llvm::None, variableTypes);

            build($_builder, $_state, functionType);
        }]>
    ];

    let extraClassDeclaration = [{
        /// @name FunctionOpInterface
        /// {

        /// Returns the argument types of this function.
        ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        ::llvm::ArrayRef<::mlir::Type> getResultTypes() {
            return getFunctionType().getResults();
        }

        /// }
    }];
}

def Simulation_DeinitFunctionOp : Simulation_Op<"deinit_function",
    [IsolatedFromAbove,
    AutomaticAllocationScope,
    HasParent<"ModuleOp">,
    Simulation_FunctionOpInterface]>
{
    let summary = "Function in charge of deleting the simulation data.";

    let description = [{
        The operation represents the function in charge of deleting the
        simulation data.
        The arguments represent the variables of the simulation.
        The body should contain the operations used to deallocate the data
        structure representing each variable.

        Example:

        ```mlir
        #var0 = #simulation.variable<...> : memref<3xf64>
        #var0 = #simulation.variable<...> : memref<5xf64>

        simulation.deinit_function (%arg0 : memref<3xf64>, %arg1: memref<5xf64>) {
            memref.dealloc %arg0 : memref<3xf64>
            memref.dealloc %arg1 : memref<5xf64>
            simulation.yield
        }
        ```
    }];

    let arguments = (ins TypeAttrOf<FunctionType>:$function_type);
    let regions = (region AnyRegion:$bodyRegion);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "::mlir::TypeRange":$variableTypes), [{
            auto functionType =
                $_builder.getFunctionType(variableTypes, llvm::None);

            build($_builder, $_state, functionType);
        }]>
    ];

    let extraClassDeclaration = [{
        /// Get the arguments of the function representing the variables.
        ::mlir::ValueRange getVariables();

        /// @name FunctionOpInterface
        /// {

        /// Returns the argument types of this function.
        ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        ::llvm::ArrayRef<::mlir::Type> getResultTypes() {
            return getFunctionType().getResults();
        }

        /// }
    }];
}

class Simulation_InitSolversFunctionOp<string mnemonic>
    : Simulation_Op<mnemonic,
        [IsolatedFromAbove,
        AutomaticAllocationScope,
        HasParent<"ModuleOp">,
        Simulation_FunctionOpInterface]>
{
    let arguments = (ins TypeAttrOf<FunctionType>:$function_type);
    let regions = (region AnyRegion:$bodyRegion);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "::mlir::TypeRange":$solverTypes, "::mlir::TypeRange":$variableTypes), [{
            ::mlir::OpBuilder::InsertionGuard guard($_builder);

            auto functionType =
                $_builder.getFunctionType(variableTypes, solverTypes);

            build($_builder, $_state, functionType);
        }]>
    ];

    let extraClassDeclaration = [{
        /// Get the types of the solvers.
        ::mlir::TypeRange getSolverTypes()
        {
            return getFunctionType().getResults();
        }

        /// Get the types of the variables.
        ::mlir::TypeRange getVariableTypes()
        {
            return getFunctionType().getInputs();
        }

        /// Get the arguments of the function representing the variables.
        ::llvm::ArrayRef<::mlir::BlockArgument> getVariables()
        {
            return getBodyRegion().getArguments();
        }

        /// @name FunctionOpInterface
        /// {

        /// Returns the argument types of this function.
        ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        ::llvm::ArrayRef<::mlir::Type> getResultTypes() {
            return getFunctionType().getResults();
        }

        /// }
    }];
}

def Simulation_InitICSolversFunctionOp
    : Simulation_InitSolversFunctionOp<"init_ic_solvers_function">
{
    let summary = [{
        Function in charge of creating and initializing the 'initial
        conditions' solvers.
    }];

    let description = [{
        The operation represents the function in charge of creating and
        initializing the solvers used to solve the 'initial conditions' model.
        The arguments of the function represent the variables of the
        simulation, which may need to be used for the initialization of the
        solvers.
        The yielded values are the ones that will be stored (through copies) in
        the runtime data structure as representative of the solvers (and, thus,
        must have value semantics).

        Example:

        ```mlir
        #var0 = #simulation.variable<...> : !llvm.ptr<f64>
        #var1 = #simulation.variable<...> : !llvm.ptr<f64>

        simulation.init_ic_solvers_function (%arg0: !llvm.ptr<f64>, %arg1: !llvm.ptr<f64>) {
            %0 = ... : !solver.type1
            %1 = ... : !solver.type2
            simulation.yield %0, %1 : !solver.type1, !solver.type2
        }
        ```
    }];
}

def Simulation_InitMainSolversFunctionOp
    : Simulation_InitSolversFunctionOp<"init_main_solvers_function">
{
    let summary = [{
        Function in charge of creating and initializing the 'main' solvers.
    }];

    let description = [{
        The operation represents the function in charge of creating and
        initializing the solvers used to solve the 'main' model.
        The arguments of the function represent the variables of the
        simulation, which may need to be used for the initialization of the
        solvers.
        The yielded values are the ones that will be stored (through copies) in
        the runtime data structure as representative of the solvers (and, thus,
        must have value semantics).

        Example:

        ```mlir
        #var0 = #simulation.variable<...> : !llvm.ptr<f64>
        #var1 = #simulation.variable<...> : !llvm.ptr<f64>

        simulation.init_main_solvers_function (%arg0: !llvm.ptr<f64>, %arg1: !llvm.ptr<f64>) {
            %0 = ... : !solver.type1
            %1 = ... : !solver.type2
            simulation.yield %0, %1 : !solver.type1, !solver.type2
        }
        ```
    }];
}

class Simulation_DeinitSolversFunctionOp<string mnemonic>
    : Simulation_Op<mnemonic,
        [IsolatedFromAbove,
        AutomaticAllocationScope,
        HasParent<"ModuleOp">,
        Simulation_FunctionOpInterface]>
{
    let arguments = (ins TypeAttrOf<FunctionType>:$function_type);
    let regions = (region AnyRegion:$bodyRegion);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "::mlir::TypeRange":$solverTypes), [{
            ::mlir::OpBuilder::InsertionGuard guard($_builder);

            auto functionType =
                $_builder.getFunctionType(solverTypes, llvm::None);

            build($_builder, $_state, functionType);
        }]>
    ];

    let extraClassDeclaration = [{
        /// Get the types of the solvers.
        ::mlir::TypeRange getSolverTypes()
        {
            return getFunctionType().getInputs();
        }

        /// Get the arguments of the function representing the solvers.
        ::llvm::ArrayRef<::mlir::BlockArgument> getSolvers()
        {
            return getBodyRegion().getArguments();
        }

        /// @name FunctionOpInterface
        /// {

        /// Returns the argument types of this function.
        ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        ::llvm::ArrayRef<::mlir::Type> getResultTypes() {
            return getFunctionType().getResults();
        }

        /// }
    }];
}

def Simulation_DeinitICSolversFunctionOp
    : Simulation_DeinitSolversFunctionOp<"deinit_ic_solvers_function">
{
    let summary = [{
        Function in charge of deleting the 'initial conditions' solvers.
    }];

    let description = [{
        The operation represents the function in charge of deleting the solvers
        used to solve the 'initial conditions' model.
        The arguments of the function represent the solvers of the simulation.

        Example:

        ```mlir
        simulation.deinit_ic_solvers_function (%arg0: !solver.type1, %arg1: !solver.type2) {
            simulation.yield
        }
        ```
    }];
}

def Simulation_DeinitMainSolversFunctionOp
    : Simulation_DeinitSolversFunctionOp<"deinit_main_solvers_function">
{
    let summary = "Function in charge of deleting the 'main' solvers.";

    let description = [{
        The operation represents the function in charge of deleting the solvers
        used to solve the 'main' model.
        The arguments of the function represent the solvers of the simulation.

        Example:

        ```mlir
        simulation.deinit_main_solvers_function (%arg0: !solver.type1, %arg1: !solver.type2) {
            simulation.yield
        }
        ```
    }];
}

def Simulation_VariableGetterOp : Simulation_Op<"variable_getter",
    [IsolatedFromAbove,
    AutomaticAllocationScope,
    HasParent<"ModuleOp">,
    Simulation_FunctionOpInterface]>
{
    let summary = "Getter interface for a simulation variable.";

    let description = [{
        The operation represents a function to be called to retrieve the value
        of a specific scalar variable.
        It takes as argument the (possibly) array variable and an amount of
        indices equal to its rank (zero, if the variable is a scalar one).
        A getter can be attached to multiple variables, as long they all have
        the same type, in order to avoid code duplication.

        Example:

        ```mlir
        #var0 = #simulation.variable<...> : !llvm.ptr<f64>
        #var1 = #simulation.variable<...> : !llvm.ptr<f64>

        simulation.variable_getter [#var0, #var1](%arg0: !llvm.ptr<f64>, %arg1: index) -> f64 {
            %result = ... f64
            simulation.yield %result : f64
        }
        ```
    }];

    let arguments = (ins
        TypeAttrOf<FunctionType>:$function_type,
        TypedArrayAttrBase<
            VariableAttr,
            "The variables to which the function can be applied">:$variables);

    let regions = (region AnyRegion:$bodyRegion);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins
            "::llvm::ArrayRef<::mlir::Attribute>":$variables,
            "::mlir::Type":$resultType,
            "::mlir::Type":$variableType,
            "int64_t":$variableRank), [{
                llvm::SmallVector<mlir::Type, 4> argTypes;
                argTypes.push_back(variableType);
                argTypes.append(variableRank, $_builder.getIndexType());

                build($_builder, $_state,
                      $_builder.getFunctionType(argTypes, resultType),
                      $_builder.getArrayAttr(variables));
            }]>
    ];

    let extraClassDeclaration = [{
        /// Get the argument representing the array variable.
        ::mlir::BlockArgument getVariable();

        /// Get the rank of the variable.
        int64_t getVariableRank();

        /// Get the arguments representing the indices of the scalar variable.
        ::llvm::ArrayRef<::mlir::BlockArgument> getIndices();

        /// Get the argument representing the index of a specific dimension.
        ::mlir::BlockArgument getIndex(int64_t dimension);

        /// @name FunctionOpInterface
        /// {

        /// Returns the argument types of this function.
        ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        ::llvm::ArrayRef<::mlir::Type> getResultTypes() {
            return getFunctionType().getResults();
        }

        /// }
    }];
}

def Simulation_YieldOp : Simulation_Op<"yield",
    [Terminator,
    ParentOneOf<["InitFunctionOp", "DeinitFunctionOp",
                 "InitICSolversFunctionOp", "InitMainSolversFunctionOp",
                 "DeinitICSolversFunctionOp", "DeinitMainSolversFunctionOp",
                 "VariableGetterOp"]>]>
{
    let summary = "Yield values to the parent operation.";

    let description = [{
        The operation yields zero or more SSA values from an op region and
        terminates the region. The semantics of how the values yielded are used
        is defined by the parent operation.
    }];

    let arguments = (ins Variadic<AnyType>:$values);
    let results = (outs);

    let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

def Simulation_FunctionOp : Simulation_Op<"function",
    [IsolatedFromAbove,
    AutomaticAllocationScope,
    Symbol,
    Simulation_FunctionOpInterface,
    CallableOpInterface,
    HasParent<"ModuleOp">]>
{
    let summary = "Generic simulation function.";

    let description = [{
        The operation represents a generic function that is requested by the
        runtime library in order to run the simulation.
        The arguments consist in the instances of the solvers, the time
        variable, the simulation variables and optional extra arguments.
        The function is allowed to return at most one result.

        Example:

        ```mlir
        simulation.func @foo(solvers: [], time: [%arg0 : f64], variables: [], extra_args: []) -> f64 {
            ...
            simulation.return %arg0 : f64
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        OptionalAttr<StrAttr>:$sym_visibility,
        TypeAttrOf<FunctionType>:$function_type,
        I64Attr:$solversAmount,
        I64Attr:$variablesAmount);

    let regions = (region AnyRegion:$bodyRegion);

    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;

    let builders = [
        OpBuilder<(ins
            "::llvm::StringRef":$name,
            "::mlir::TypeRange":$solverTypes,
            "::mlir::Type":$timeType,
            "::mlir::TypeRange":$variableTypes,
            "::mlir::TypeRange":$extraArgs,
            "::mlir::TypeRange":$resultTypes)>
    ];

    let extraClassDeclaration = [{
        /// Get the types of the solvers.
        ::llvm::SmallVector<::mlir::Type> getSolverTypes();

        /// Get the type of the time variable.
        ::mlir::Type getTimeType();

        /// Get the types of the variables.
        ::llvm::SmallVector<::mlir::Type> getVariableTypes();

        /// Get the types of the extra arguments.
        ::llvm::SmallVector<::mlir::Type> getExtraArgTypes();

        /// Get the arguments of the function representing the solvers.
        ::llvm::ArrayRef<::mlir::BlockArgument> getSolvers();

        /// Get the argument of the function representing the time variable.
        ::mlir::BlockArgument getTime();

        /// Get the arguments of the function representing the variables.
        ::llvm::ArrayRef<::mlir::BlockArgument> getVariables();

        /// Get the arguments of the function representing the extra arguments.
        ::llvm::ArrayRef<::mlir::BlockArgument> getExtraArgs();

        /// @name CallableOpInterface methods
        /// {

        /// Returns the region on the current operation that is callable. This
        /// may return null in the case of an external callable object, e.g. an
        /// external function.
        ::mlir::Region* getCallableRegion()
        {
            return &getBodyRegion();
        }

        /// Returns the results types that the callable region produces when
        /// executed.
        ::llvm::ArrayRef<::mlir::Type> getCallableResults()
        {
            return getFunctionType().getResults();
        }

        /// }
        /// @name FunctionOpInterface methods
        /// {

        /// Returns the argument types of this function.
        ::llvm::ArrayRef<::mlir::Type> getArgumentTypes()
        {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        ::llvm::ArrayRef<::mlir::Type> getResultTypes()
        {
            return getFunctionType().getResults();
        }

        /// }
        /// @name SymbolOpInterface methods
        /// {

        bool isDeclaration()
        {
            return false;
        }

        /// }
    }];
}

def Simulation_ReturnOp : Simulation_Op<"return",
    [Terminator,
    HasParent<"FunctionOp">]>
{
    let summary = "Function return operation.";

    let description = [{
        The `simulation.return` operation represents a return operation within
        a generic function. The operand number and types must match the
        signature of the function that contains the operation.

        Example:

        ```mlir
        simulation.func @foo(solvers: [], time: [%arg0 : f64], variables: []) -> f64 {
            ...
            simulation.return %arg0 : f64
        }
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$values);
    let results = (outs);

    let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

#endif // MARCO_DIALECTS_SIMULATION_SIMULATION_TD
