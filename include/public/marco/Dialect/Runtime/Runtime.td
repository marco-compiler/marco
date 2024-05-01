#ifndef MARCO_DIALECTS_RUNTIME_RUNTIME_TD
#define MARCO_DIALECTS_RUNTIME_RUNTIME_TD

include "marco/Dialect/Runtime/RuntimeAttributes.td"
include "marco/Dialect/Runtime/RuntimeInterfaces.td"
include "marco/Dialect/Runtime/RuntimeTypes.td"
include "marco/Dialect/Runtime/RuntimeDialect.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"

//===---------------------------------------------------------------------===//
// Type constraints
//===---------------------------------------------------------------------===//

// Scheduler instance type.
def SchedulerInstance : Type<CPred<"$_self.isa<::mlir::ida::SchedulerType>()">,
                             "scheduler", "::mlir::runtime::SchedulerType">;

//===---------------------------------------------------------------------===//
// Attribute constraints
//===---------------------------------------------------------------------===//

// Multidimensional range attribute.
def MultidimensionalRangeAttr
    : Attr<CPred<"$_self.isa<::mlir::runtime::MultidimensionalRangeAttr>()">,
                 "Multidimensional range attribute">
{
  let storageType = [{ ::mlir::runtime::MultidimensionalRangeAttr }];
  let returnType = [{ ::mlir::runtime::MultidimensionalRangeAttr }];
  let convertFromStorage = "$_self";
}

// Index set attribute.
def IndexSetAttr
    : Attr<CPred<"$_self.isa<::mlir::runtime::IndexSetAttr>()">,
                 "Index set attribute">
{
  let storageType = [{ ::mlir::runtime::IndexSetAttr }];
  let returnType = [{ ::mlir::runtime::IndexSetAttr }];
  let convertFromStorage = "$_self";
}

//===---------------------------------------------------------------------===//
// Runtime operation definitions
//===---------------------------------------------------------------------===//

class Runtime_Op<string mnemonic, list<Trait> traits = []> :
    Op<Runtime_Dialect, mnemonic, traits>;

def Runtime_SchedulerOp : Runtime_Op<"scheduler",
    [Symbol,
    HasParent<"mlir::ModuleOp">]>
{
    let summary = "Declare an instance of a scheduler.";

    let description = [{
        Declare an instance of a scheduler.
    }];

    let arguments = (ins SymbolNameAttr:$sym_name);
    let results = (outs);
    let assemblyFormat = "$sym_name attr-dict";
}

def Runtime_SchedulerCreateOp : Runtime_Op<"scheduler_create", []>
{
    let summary = "Create a scheduler instance.";

    let description = [{
        The operation create a new scheduler instance.

        Example:

        ```mlir
        runtime.scheduler_create @scheduler
        ```
    }];

    let arguments = (ins SymbolNameAttr:$scheduler);
    let results = (outs);
    let assemblyFormat = "$scheduler attr-dict";
}

def Runtime_SchedulerDestroyOp : Runtime_Op<"scheduler_destroy", []>
{
    let summary = "Destroy a scheduler instance.";

    let description = [{
        The operation destroys a scheduler instance.

        Example:

        ```mlir
        runtime.scheduler_destroy @scheduler
        ```
    }];

    let arguments = (ins SymbolNameAttr:$scheduler);
    let results = (outs);
    let assemblyFormat = "$scheduler attr-dict";
}

def Runtime_SchedulerAddEquationOp : Runtime_Op<"scheduler_add_equation", []>
{
    let summary = "Add an equation to a scheduler.";

    let description = [{
        The operation is intended to inform a scheduler about an equation it
        has to execute.

        Example:

        ```mlir
        runtime.scheduler_add_equation @scheduler {ranges : #modeling<multidim_range [0,7][0,13]>}
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$scheduler,
        FlatSymbolRefAttr:$function,
        OptionalAttr<MultidimensionalRangeAttr>:$ranges,
        DefaultValuedAttr<BoolAttr, "false">:$independentIndices);

    let results = (outs);
    let assemblyFormat = "$scheduler attr-dict";
}

def Runtime_SchedulerRunOp : Runtime_Op<"scheduler_run", []>
{
    let summary = "Run the equations owned by a scheduler.";

    let description = [{
        The operation runs the equations owned by a scheduler.

        Example:

        ```mlir
        runtime.run @scheduler
        ```
    }];

    let arguments = (ins SymbolNameAttr:$scheduler);
    let results = (outs);
    let assemblyFormat = "$scheduler attr-dict";
}

def Runtime_VariableOp : Runtime_Op<"variable",
    [HasParent<"mlir::ModuleOp">, Symbol]>
{
    let summary = "Variable of the simulation.";

    let description = [{
        The operation represents a variable of the simulation.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type,
        I64ArrayAttr:$dimensions,
        DefaultValuedAttr<BoolAttr, "false">:$printable,
        OptionalAttr<TypedArrayAttrBase<
            MultidimensionalRangeAttr, "Printable indices">>:$printable_indices);

    let results = (outs);

    let assemblyFormat = "$sym_name attr-dict `:` $type";
}

def Runtime_ModelNameOp : Runtime_Op<"model_name",
    [HasParent<"mlir::ModuleOp">]>
{
    let summary = [{
        Operation holding the information about the name of the model.
    }];

    let description = [{
        The operation holds the information about the name of the compiled
        model.

        Example:

        ```mlir
        runtime.model_name "model"
        ```
    }];

    let arguments = (ins StrAttr:$name);

    let assemblyFormat = "$name attr-dict";
}

def Runtime_NumberOfVariablesOp : Runtime_Op<"number_of_variables",
    [HasParent<"mlir::ModuleOp">]>
{
    let summary = [{
        Operation holding the information about the number of array variables.
    }];

    let description = [{
        The operation holds the information about the number of array variables
        of the simulation.

        Example:

        ```mlir
        runtime.number_of_variables 5
        ```
    }];

    let arguments = (ins I64Attr:$numberOfVariables);

    let assemblyFormat = "$numberOfVariables attr-dict";
}

def Runtime_VariableNamesOp : Runtime_Op<"variable_names",
    [HasParent<"mlir::ModuleOp">]>
{
    let summary = [{
        Operation holding the variable names.
    }];

    let description = [{
        The operation holds the the variable names.

        Example:

        ```mlir
        runtime.variable_names ["x", "y", "z"]
        ```
    }];

    let arguments = (ins StrArrayAttr:$names);

    let assemblyFormat = "$names attr-dict";
}

def Runtime_VariableRanksOp : Runtime_Op<"variable_ranks",
    [HasParent<"mlir::ModuleOp">]>
{
    let summary = [{
        Operation holding the variable ranks.
    }];

    let description = [{
        The operation holds the the variable ranks.

        Example:

        ```mlir
        runtime.variable_ranks [0, 1, 2]
        ```
    }];

    let arguments = (ins I64ArrayAttr:$ranks);

    let assemblyFormat = "$ranks attr-dict";
}

def Runtime_DerivativesMapOp : Runtime_Op<"derivatives_map",
    [HasParent<"mlir::ModuleOp">]>
{
    let summary = [{
        Operation holding the derivative index of each variable.
    }];

    let description = [{
        The operation holds the list of derivative mappings.
        The i-th element of the array contains the index of the derivative of
        variable i.
        The value -1 is used to indicate that the variable has no derivative.

        Example:

        ```mlir
        runtime.derivatives_map [-1, 2, -1]
        ```
    }];

    let arguments = (ins I64ArrayAttr:$derivatives);

    let assemblyFormat = "$derivatives attr-dict";
}

def Runtime_PrintableIndicesOp : Runtime_Op<"printable_indices",
    [HasParent<"mlir::ModuleOp">]>
{
    let summary = [{
        Operation holding the information about the printable indices of
        variables.
    }];

    let description = [{
        The operation holds the information about the printable indices of
        variables.
        The i-th element of the array contains the information about the
        printable indices of variable i.
        The information can be a boolean for scalar variables (i.e., the ones
        with rank zero), while it must be an index set for non-scalar ones.

        Example:

        ```mlir
        runtime.derivatives_map [false, true, #modeling<index_set {[0,1][2,3], [3,7][2,3]}>]
        ```
    }];

    let arguments = (ins ArrayAttr:$value);

    let assemblyFormat = "$value attr-dict";
}

def Runtime_InitFunctionOp : Runtime_Op<"init_function",
    [AutomaticAllocationScope,
    IsolatedFromAbove,
    HasParent<"mlir::ModuleOp">]>
{
    let summary = [{
        Function in charge of initializing the simulation data.
    }];

    let description = [{
        The operation represents the function in charge of creating an
        initializing the simulation data.

        Example:

        ```mlir
        runtime.init_function {
           ...
            runtime.yield
        }
        ```
    }];

    let arguments = (ins TypeAttrOf<FunctionType>:$function_type);
    let regions = (region AnyRegion:$bodyRegion);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins), [{
            auto functionType = $_builder.getFunctionType(
                std::nullopt, std::nullopt);

            build($_builder, $_state, functionType);
        }]>
    ];
}

def Runtime_DeinitFunctionOp : Runtime_Op<"deinit_function",
    [AutomaticAllocationScope,
    IsolatedFromAbove,
    HasParent<"mlir::ModuleOp">]>
{
    let summary = [{
        Function in charge of deinitializing the simulation data.
    }];

    let description = [{
        The operation represents the function in charge of deinitializing the
        simulation data.

        Example:

        ```mlir
        runtime.deinit_function {
           ...
            runtime.yield
        }
        ```
    }];

    let arguments = (ins TypeAttrOf<FunctionType>:$function_type);
    let regions = (region AnyRegion:$bodyRegion);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins), [{
            auto functionType = $_builder.getFunctionType(
                std::nullopt, std::nullopt);

            build($_builder, $_state, functionType);
        }]>
    ];
}

def Runtime_VariableGetterOp : Runtime_Op<"variable_getter",
    [AutomaticAllocationScope,
    IsolatedFromAbove,
    Symbol,
    CallableOpInterface,
    FunctionOpInterface,
    HasParent<"mlir::ModuleOp">]>
{
    let summary = "Getter interface for a simulation variable.";

    let description = [{
        The operation represents a function to be called to retrieve the value
        of a specific scalar variable.
        It takes as argument an amount of indices equal to its rank (zero, if
        the variable is a scalar one).

        Example:

        ```mlir
        runtime.variable_getter @name(%arg0: index, %arg1: index) -> f64 {
            %result = ... f64
            runtime.yield %result : f64
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs);

    let regions = (region AnyRegion:$bodyRegion);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "::llvm::StringRef":$name, "uint64_t":$variableRank)>
    ];

    let extraClassDeclaration = [{
        /// Get the argument representing the array variable.
        ::mlir::BlockArgument getVariable();

        /// Get the rank of the variable.
        uint64_t getVariableRank();

        /// Get the arguments representing the indices of the scalar variable.
        ::llvm::ArrayRef<::mlir::BlockArgument> getIndices();

        /// Get the argument representing the index of a specific dimension.
        ::mlir::BlockArgument getIndex(uint64_t dimension);

        /// @name CallableOpInterface methods
        /// {

        ::mlir::Region* getCallableRegion()
        {
            return isExternal() ? nullptr : &getBodyRegion();
        }

        llvm::ArrayRef<mlir::Type> getCallableResults()
        {
            return getFunctionType().getResults();
        }

        /// }
        /// @name FunctionOpInterface methods
        /// {

        llvm::ArrayRef<mlir::Type> getArgumentTypes()
        {
            return getFunctionType().getInputs();
        }

        llvm::ArrayRef<mlir::Type> getResultTypes()
        {
            return getFunctionType().getResults();
        }

        /// }
        /// @name SymbolOpInterface methods
        /// {

        bool isDeclaration() {
            return isExternal();
        }

        /// }
    }];
}

def Runtime_VariableGettersOp : Runtime_Op<"variable_getters",
    [HasParent<"mlir::ModuleOp">]>
{
    let summary = [{
        Operation holding the names of the function to be called to get the
        values of the variables.
    }];

    let description = [{
        The operation holds the the names of the function to be called to get the
        values of the variables.
        The i-th element of the array contains the name of the function to query
        the array variable at position i.

        Example:

        ```mlir
        runtime.variable_getters [@getter1, @getter2, @getter3]
        ```
    }];

    let arguments = (ins ArrayAttr:$names);

    let assemblyFormat = "$names attr-dict";
}

def Runtime_YieldOp : Runtime_Op<"yield",
    [Terminator,
    ParentOneOf<["InitFunctionOp", "DeinitFunctionOp", "VariableGetterOp"]>]>
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

def Runtime_ICModelBeginOp : Runtime_Op<"ic_model_begin",
    [HasParent<"mlir::ModuleOp">,
    SingleBlock,
    NoTerminator,
    IsolatedFromAbove]>
{
    let arguments = (ins);
    let regions = (region SizedRegion<1>:$bodyRegion);
    let assemblyFormat = "$bodyRegion attr-dict";
}

def Runtime_ICModelEndOp : Runtime_Op<"ic_model_end",
    [HasParent<"mlir::ModuleOp">,
    SingleBlock,
    NoTerminator,
    IsolatedFromAbove]>
{
    let arguments = (ins);
    let regions = (region SizedRegion<1>:$bodyRegion);
    let assemblyFormat = "$bodyRegion attr-dict";
}

def Runtime_DynamicModelBeginOp : Runtime_Op<"dynamic_model_begin",
    [HasParent<"mlir::ModuleOp">,
    SingleBlock,
    NoTerminator,
    IsolatedFromAbove]>
{
    let arguments = (ins);
    let regions = (region SizedRegion<1>:$bodyRegion);
    let assemblyFormat = "$bodyRegion attr-dict";
}

def Runtime_DynamicModelEndOp : Runtime_Op<"dynamic_model_end",
    [HasParent<"mlir::ModuleOp">,
    SingleBlock,
    NoTerminator,
    IsolatedFromAbove]>
{
    let arguments = (ins);
    let regions = (region SizedRegion<1>:$bodyRegion);
    let assemblyFormat = "$bodyRegion attr-dict";
}

def Runtime_EquationFunctionOp : Runtime_Op<"equation_function",
    [HasParent<"mlir::ModuleOp">,
    IsolatedFromAbove,
    AutomaticAllocationScope,
    Symbol,
    FunctionOpInterface,
    CallableOpInterface]>
{
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<StrAttr>:$sym_visibility,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs);

    let regions = (region AnyRegion:$body);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins
            "llvm::StringRef":$name,
            "uint64_t":$numOfInductions,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
            CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>
    ];

    let extraClassDeclaration = [{
        /// @name CallableOpInterface methods
        /// {

        /// Returns the region on the current operation that is callable. This may
        /// return null in the case of an external callable object, e.g. an external
        /// function.
        mlir::Region* getCallableRegion()
        {
            return isExternal() ? nullptr : &getBody();
        }

        /// Returns the results types that the callable region produces when
        /// executed.
        llvm::ArrayRef<mlir::Type> getCallableResults()
        {
            return getFunctionType().getResults();
        }

        /// }
        /// @name FunctionOpInterface methods
        /// {

        /// Returns the argument types of this function.
        llvm::ArrayRef<mlir::Type> getArgumentTypes()
        {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        llvm::ArrayRef<mlir::Type> getResultTypes()
        {
            return getFunctionType().getResults();
        }

        /// }
        /// @name SymbolOpInterface methods
        /// {

        bool isDeclaration()
        {
            return isExternal();
        }

        /// }
    }];
}

def Runtime_FunctionOp : Runtime_Op<"function",
    [AutomaticAllocationScope,
    IsolatedFromAbove,
    Symbol,
    CallableOpInterface,
    FunctionOpInterface,
    HasParent<"mlir::ModuleOp">]>
{
    let summary = "Generic simulation function.";

    let description = [{
        The operation represents a generic function that is requested by the
        runtime library in order to run the simulation.
        The function is allowed to return at most one result.

        Example:

        ```mlir
        runtime.func @foo(%arg0: i64, %arg1: f64) -> f64 {
            ...
            runtime.return %arg1 : f64
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        OptionalAttr<StrAttr>:$sym_visibility,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs);

    let regions = (region AnyRegion:$bodyRegion);

    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;

    let builders = [
        OpBuilder<(ins "::llvm::StringRef":$name, "mlir::FunctionType":$type)>
    ];

    let extraClassDeclaration = [{
        /// @name CallableOpInterface methods
        /// {

        /// Returns the region on the current operation that is callable. This may
        /// return null in the case of an external callable object, e.g. an external
        /// function.
        mlir::Region* getCallableRegion()
        {
            return isExternal() ? nullptr : &getBodyRegion();
        }

        /// Returns the results types that the callable region produces when
        /// executed.
        llvm::ArrayRef<mlir::Type> getCallableResults()
        {
            return getFunctionType().getResults();
        }

        /// }
        /// @name FunctionOpInterface methods
        /// {

        /// Returns the argument types of this function.
        llvm::ArrayRef<mlir::Type> getArgumentTypes()
        {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        llvm::ArrayRef<mlir::Type> getResultTypes()
        {
            return getFunctionType().getResults();
        }

        /// }
        /// @name SymbolOpInterface methods
        /// {

        bool isDeclaration()
        {
            return isExternal();
        }

        /// }
    }];
}

def Runtime_ReturnOp : Runtime_Op<"return",
    [Terminator,
    ParentOneOf<["FunctionOp", "VariableGetterOp", "EquationFunctionOp"]>]>
{
    let summary = "Function return operation.";

    let description = [{
        The `runtime.return` operation represents a return operation within
        a generic function. The operand number and types must match the
        signature of the function that contains the operation.

        Example:

        ```mlir
        runtime.func @foo(%arg0: f64) -> f64 {
            ...
            runtime.return %arg0 : f64
        }
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$values);
    let results = (outs);

    let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

#endif // MARCO_DIALECTS_RUNTIME_RUNTIME_TD
