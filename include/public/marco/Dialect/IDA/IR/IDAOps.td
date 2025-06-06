#ifndef MARCO_DIALECT_IDA_IR_IDAOPS_TD
#define MARCO_DIALECT_IDA_IR_IDAOPS_TD

include "marco/Dialect/IDA/IR/IDA.td"
include "marco/Dialect/IDA/IR/IDATypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===---------------------------------------------------------------------===//
// Type constraints
//===---------------------------------------------------------------------===//

// IDA instance type.
def IDAInstance : Type<CPred<"mlir::isa<::mlir::ida::InstanceType>($_self)">,
                       "instance", "::mlir::ida::InstanceType">;

// IDA equation type.
def IDAEquation : Type<CPred<"mlir::isa<::mlir::ida::EquationType>($_self)">,
                             "equation", "::mlir::ida::EquationType">;

// IDA variable type.
def IDAVariable : Type<CPred<"mlir::isa<::mlir::ida::VariableType>($_self)">,
                             "variable", "::mlir::ida::VariableType">;

//===---------------------------------------------------------------------===//
// Attribute constraints
//===---------------------------------------------------------------------===//

// Multidimensional range attribute.
def MultidimensionalRangeAttr
    : Attr<CPred<"mlir::isa<::mlir::ida::MultidimensionalRangeAttr>($_self)">,
                 "Multidimensional range attribute">
{
  let storageType = [{ ::mlir::ida::MultidimensionalRangeAttr }];
  let returnType = [{ ::mlir::ida::MultidimensionalRangeAttr }];
  let convertFromStorage = "$_self";
}

//===---------------------------------------------------------------------===//
// IDA operation definitions
//===---------------------------------------------------------------------===//

class IDA_Op<string mnemonic, list<Trait> traits = []> :
    Op<IDA_Dialect, mnemonic, traits>;

def IDA_InternalEquationsOp : IDA_Op<"internal_equations",
    [SingleBlock, NoTerminator, IsolatedFromAbove]>
{
    let arguments = (ins);
    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let assemblyFormat = "$bodyRegion attr-dict";
}

def IDA_InstanceOp : IDA_Op<"instance",
    [Symbol,
    HasParent<"mlir::ModuleOp">]>
{
    let summary = "Declare an instance of the IDA solver.";

    let description = [{
        Declare an instance of the IDA solver.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name
    );

    let results = (outs);

    let assemblyFormat = [{
        $sym_name attr-dict
    }];
}

def IDA_InitOp : IDA_Op<"create", [MemoryEffects<[MemWrite]>]>
{
    let summary = "Create an IDA instance.";

    let description = [{
        The operation create a new IDA instance.

        Example:

        ```mlir
        ida.create @ida
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$instance);

    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict
    }];
}

def IDA_SetStartTimeOp
    : IDA_Op<"set_start_time", [MemoryEffects<[MemWrite]>]>
{
    let summary = "Set the start time of the simulation.";

    let description = [{
        The operation sets the start time to be used by IDA during the
        simulation.

        Example:

        ```mlir
        ida.set_start_time @instance { time = 0.0 }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$instance,
        F64Attr:$time);

    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict
    }];
}

def IDA_SetEndTimeOp
    : IDA_Op<"set_end_time", [MemoryEffects<[MemWrite]>]>
{
    let summary = "Set the end time of the simulation.";

    let description = [{
        The operation sets the end time to be used by IDA during the
        simulation.

        Example:

        ```mlir
        ida.set_end_time @instance { time = 0.0 }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$instance,
        F64Attr:$time);

    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict
    }];
}

def IDA_GetCurrentTimeOp
    : IDA_Op<"get_current_time", [MemoryEffects<[MemRead]>]>
{
    let summary = "Get the time reached by IDA.";

    let description = [{
        The operation returns the time reached by the simulation.

        Example:

        ```mlir
        %0 = ida.get_current_time @instance : f64
        ```
    }];

    let arguments = (ins SymbolNameAttr:$instance);
    let results = (outs F64:$time);

    let builders = [
        OpBuilder<(ins "llvm::StringRef":$idaInstance), [{
            auto resultType = $_builder.getF64Type();
            build($_builder, $_state, resultType, idaInstance);
        }]>
    ];

    let assemblyFormat = [{
        $instance attr-dict `:` type($time)
    }];
}

def IDA_AddEquationOp
    : IDA_Op<"add_equation", [MemoryEffects<[MemWrite]>]>
{
    let summary = "Add the information regarding an equation iteration range.";

    let description = [{
        The operation is intended to inform IDA about the iteration ranges of
        an equation.

        The ranges are expressed as integer pairs of values, where each pair
        consists in the beginning and ending iteration value, with the former
        being strictly less than the latter.

        For example, an equation iterating on two indices `i`, `j` and `k`,
        with `i` belonging to `[4,7)`, `j` belonging to `[9, 13)` and `k`
        belonging to `[1, 3)`, would be represented by the following 2-D 2x3
        array: `[[4, 7], [9, 13], [1, 3]]`.

        The operation returns a reference to the equation added to the IDA
        instance.

        Example:

        ```mlir
        %0 = ... : !ida.variable
        ida.add_equation @instance { equationRanges : #modeling<multidim_range [4,7][9,13][1,3]> } : !ida.equation
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$instance,
        MultidimensionalRangeAttr:$equationRanges,
        OptionalAttr<StrAttr>:$stringRepresentation);

    let results = (outs IDAEquation:$equation);

    let builders = [
        OpBuilder<(ins "llvm::StringRef":$idaInstance, "MultidimensionalRangeAttr":$equationRanges), [{
            auto resultType = ida::EquationType::get($_builder.getContext());
            build($_builder, $_state, resultType, idaInstance, equationRanges, nullptr);
        }]>
    ];

    let assemblyFormat = [{
        $instance attr-dict `:` type($equation)
    }];
}

def IDA_AddAlgebraicVariableOp
    : IDA_Op<"add_algebraic_variable", [MemoryEffects<[MemWrite]>]>
{
    let summary = "Add the information regarding the dimensions of an algebraic variable.";

    let description = [{
        The operation is intended to inform IDA about the dimensions of
        an algebraic variable.

        The operation expects three attributes:
          1. 'arrayDimensions': contains the dimensions of the array variable;
             scalar variables must be represented by a 1-D array of size 1.
          2. 'getter': the name of the function to be used by IDA to get the
             value of a scalar variable inside the array.
          3. 'setter': the name of the function to be used by IDA to set the
             value of a scalar variable inside the array.

        The operation returns a reference to the variable added to the IDA
        instance.

        Example:

        ```mlir
        ida.add_algebraic_variable @instance { dimensions = [3, 5], getter = @x_getter, setter = @x_setter } : !ida.variable
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$instance,
        I64ArrayAttr:$dimensions,
        SymbolRefAttr:$getter,
        SymbolRefAttr:$setter,
        OptionalAttr<StrAttr>:$name);

    let results = (outs IDAVariable:$idaVariable);

    let builders = [
        OpBuilder<(ins "llvm::StringRef":$idaInstance, "ArrayAttr":$dimensions, "SymbolRefAttr":$getter, "SymbolRefAttr":$setter), [{
            auto resultType = ida::VariableType::get($_builder.getContext());
            build($_builder, $_state, resultType, idaInstance, dimensions, getter, setter, nullptr);
        }]>,
        OpBuilder<(ins "llvm::StringRef":$idaInstance, "ArrayAttr":$dimensions, "StringRef":$getter, "StringRef":$setter), [{
            auto resultType = ida::VariableType::get($_builder.getContext());
            auto getterAttr = SymbolRefAttr::get($_builder.getContext(), getter);
            auto setterAttr = SymbolRefAttr::get($_builder.getContext(), setter);
            build($_builder, $_state, resultType, idaInstance, dimensions, getterAttr, setterAttr, nullptr);
        }]>
    ];

    let assemblyFormat = [{
        $instance attr-dict `:` type($idaVariable)
    }];
}

def IDA_AddStateVariableOp
    : IDA_Op<"add_state_variable", [MemoryEffects<[MemWrite]>]>
{
    let summary = "Add the information regarding a state variable and its derivative.";

    let description = [{
        The operation is intended to inform IDA about the dimensions of
        a state variable.

        The operation expects five attributes:
          1. 'arrayDimensions': contains the dimensions of the array variable;
             scalar variables must be represented by a 1-D array of size 1.
          2. 'stateGetter': the name of the function to be used by IDA to get
             the value of a scalar variable inside the array of the state
             variable.
          3. 'stateSetter': the name of the function to be used by IDA to set
             the value of a scalar variable inside the array of the state
             variable.
          4. 'derivativeGetter': the name of the function to be used by IDA to
             get the value of a scalar variable inside the array of the
             derivative variable.
          5. 'derivativeSetter': the name of the function to be used by IDA to
             set the value of a scalar variable inside the array of the
             derivative variable.

        The operation returns a reference to the variable added to the IDA
        instance.

        Example:

        ```mlir
        ida.add_state_variable @instance { dimensions = [3, 5], getter = @x_getter, setter = @x_setter } : !ida.variable
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$instance,
        I64ArrayAttr:$dimensions,
        SymbolRefAttr:$stateGetter,
        SymbolRefAttr:$stateSetter,
        SymbolRefAttr:$derivativeGetter,
        SymbolRefAttr:$derivativeSetter,
        OptionalAttr<StrAttr>:$name);

    let results = (outs IDAVariable:$idaVariable);

    let builders = [
        OpBuilder<(ins "llvm::StringRef":$idaInstance, "ArrayAttr":$dimensions, "SymbolRefAttr":$stateGetter, "SymbolRefAttr":$stateSetter, "SymbolRefAttr":$derivativeGetter, "SymbolRefAttr":$derivativeSetter), [{
            auto resultType = ida::VariableType::get($_builder.getContext());
            build($_builder, $_state, resultType, idaInstance, dimensions, stateGetter, stateSetter, derivativeGetter, derivativeSetter, nullptr);
        }]>,
        OpBuilder<(ins "llvm::StringRef":$idaInstance, "ArrayAttr":$dimensions, "StringRef":$stateGetter, "StringRef":$stateSetter, "StringRef":$derivativeGetter, "StringRef":$derivativeSetter), [{
            auto resultType = ida::VariableType::get($_builder.getContext());
            auto stateGetterAttr = SymbolRefAttr::get($_builder.getContext(), stateGetter);
            auto stateSetterAttr = SymbolRefAttr::get($_builder.getContext(), stateSetter);
            auto derivativeGetterAttr = SymbolRefAttr::get($_builder.getContext(), derivativeGetter);
            auto derivativeSetterAttr = SymbolRefAttr::get($_builder.getContext(), derivativeSetter);
            build($_builder, $_state, resultType, idaInstance, dimensions, stateGetterAttr, stateSetterAttr, derivativeGetterAttr, derivativeSetterAttr, nullptr);
        }]>
    ];

    let assemblyFormat = [{
        $instance attr-dict `:` type($idaVariable)
    }];
}

def IDA_AddVariableAccessOp
    : IDA_Op<"add_variable_access", [MemoryEffects<[MemWrite]>]>
{
    let summary = "Add the information regarding the access to an array variable inside an equation.";

    let description = [{
        The operation is used to inform IDA about how an equation accesses an
        array variable.

        ```mlir
        %0 = ... : !ida.equation
        %1 = ... : !ida.variable
        ida.add_variable_access @instance, %0, %1 { access = affine_map<(d0, d1)[s0] -> (d1, d0 + s0)> } : !ida.equation, !ida.variable
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$instance,
        IDAEquation:$equation,
        IDAVariable:$variable,
        SymbolNameAttr:$accessFunction);

    let results = (outs);

    let assemblyFormat = [{
        $instance `,` $equation `,` $variable attr-dict `:` type($equation) `,` type($variable)
    }];
}

def IDA_ResidualFunctionOp : IDA_Op<"residual_function",
    [IsolatedFromAbove,
    AutomaticAllocationScope,
    Symbol,
    CallableOpInterface,
    FunctionOpInterface]>
{
    let summary = "Residual function of a loop equation.";

    let description = [{
        The residual function of an equation is a function that computes the
        residual error of that equation.

        The residual error of an equation consists in the difference between
        its right-hand side and left-hand side values.

        The function takes a variadic list of arguments:
          1. Current time of the simulation (which will be provided by IDA when
             calling the residual function).
          2. The indices of the scalar equation to be examined.

        Example:

        ```mlir
        ida.residual_function @foo(%time : f64, %i : index, %j : index, %k : index) -> f64 {
            %diff = ... : f64
            ida.return %diff : f64
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs);

    let results = (outs);

    let regions = (region
        AnyRegion:$body
    );

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "StringRef":$sym_name, "uint64_t":$equationRank), [{
            auto nameAttr = $_builder.getStringAttr(sym_name);

            SmallVector<Type, 3> args;
            args.push_back($_builder.getF64Type());

            for (size_t i = 0; i < equationRank; ++i) {
                args.push_back($_builder.getIndexType());
            }

            auto functionType = $_builder.getFunctionType(args, $_builder.getF64Type());
            auto typeAttr = TypeAttr::get(functionType);
            build($_builder, $_state, nameAttr, typeAttr, nullptr, nullptr);
        }]>
    ];

    let extraClassDeclaration = [{
        BlockArgument getTime();
        ArrayRef<BlockArgument> getEquationIndices();

        //===-------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===-------------------------------------------------------------===//

        bool isDeclaration()
        {
            return isExternal();
        }

        //===-------------------------------------------------------------===//
        // CallableOpInterface Methods
        //===-------------------------------------------------------------===//

        /// Returns the region on the current operation that is callable. This
        /// may return null in the case of an external callable object, e.g. an
        /// external function.
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

        //===-------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===-------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes()
        {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes()
        {
            return getFunctionType().getResults();
        }
    }];
}

def IDA_JacobianFunctionOp : IDA_Op<"jacobian_function",
    [IsolatedFromAbove,
    AutomaticAllocationScope,
    Symbol,
    CallableOpInterface,
    FunctionOpInterface]>
{
    let summary = "Jacobian function of a vectorized equation.";

    let description = [{
        The Jacobian function of an equation is a function that computes an
        element of the Jacobian matrix of that equation.

        The function expects an attribute named `equationRank` indicating the
        number of iteration variables of the equation and an attribute named
        `variableRank` indicating the rank of the variable with respect to the
        equation is derived.

        The function takes a variadic list of arguments:
          1. Current time of the simulation (which will be provided by IDA when
             calling the residual function).
          2. The indices of the scalar equation to be examined.
          3. The indices of the scalar variable with respect to the equation is
             derived.
          4. The IDA `alpha` parameter.
          5. The identifier of the memory pool used for allocating the AD seeds.
          6. The identifiers of the AD buffers that have been allocated in the
             memory pool with the given identifier.

        Example:

        ```mlir
        ida.jacobian_function @foo(%time : f64, %i : index, %j : index, %k : index, %l : index, %m : index, %alpha: f64, %pool: index, %seed1: index, %seed2: index) -> f64 attributes { equationRank = 3, variableRank = 2 } {
            %der = ... : f64
            ida.return %der : f64
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs,
        IndexAttr:$equationRank,
        IndexAttr:$variableRank);

    let results = (outs);

    let regions = (region
        AnyRegion:$body
    );

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "StringRef":$sym_name, "size_t":$equationRank, "size_t":$variableRank, "size_t":$numOfADSeeds), [{
            auto nameAttr = $_builder.getStringAttr(sym_name);

            SmallVector<Type, 3> args;
            args.push_back($_builder.getF64Type());

            for (size_t i = 0; i < equationRank; ++i) {
                args.push_back($_builder.getIndexType());
            }

            for (size_t i = 0; i < variableRank; ++i) {
                args.push_back($_builder.getIndexType());
            }

            args.push_back($_builder.getF64Type());
            args.push_back($_builder.getIndexType());

            for (size_t i = 0; i < numOfADSeeds; ++i) {
                args.push_back($_builder.getIndexType());
            }

            auto functionType = $_builder.getFunctionType(args, $_builder.getF64Type());
            auto typeAttr = TypeAttr::get(functionType);
            build($_builder, $_state, nameAttr, typeAttr, nullptr, nullptr,
                  $_builder.getIndexAttr(equationRank),
                  $_builder.getIndexAttr(variableRank));
        }]>
    ];

    let extraClassDeclaration = [{
        BlockArgument getTime();
        ArrayRef<BlockArgument> getEquationIndices();
        ArrayRef<BlockArgument> getVariableIndices();
        BlockArgument getAlpha();
        BlockArgument getMemoryPool();
        ArrayRef<BlockArgument> getADSeeds();

        //===-------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===-------------------------------------------------------------===//

        bool isDeclaration()
        {
            return isExternal();
        }

        //===-------------------------------------------------------------===//
        // CallableOpInterface Methods
        //===-------------------------------------------------------------===//

        /// Returns the region on the current operation that is callable. This
        /// may return null in the case of an external callable object, e.g. an
        /// external function.
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

        //===-------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===-------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes()
        {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes()
        {
            return getFunctionType().getResults();
        }
    }];
}

def IDA_ReturnOp : IDA_Op<"return",
    [NoMemoryEffect,
     ParentOneOf<["ResidualFunctionOp", "JacobianFunctionOp"]>,
     ReturnLike, Terminator]>
{
    let summary = "Return operation.";

    let description = [{
        The operation represents a return operation within a function.
        The operation takes variable number of operands and produces no
        results. The operand number and types must match the signature of the
        function that contains the operation.
    }];

    let arguments = (ins Variadic<AnyType>:$operands);

    let builders = [
        OpBuilder<(ins), [{
            build($_builder, $_state, std::nullopt);
        }]>];

    let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def IDA_SetResidualOp
    : IDA_Op<"set_residual", [MemoryEffects<[MemWrite]>]>
{
    let summary = "Add the residual function of an equation.";

    let description = [{
        The operation is intended to inform IDA about the residual function to
        be used for an equation already declared inside IDA.

        ```mlir
        ida.residual_function @foo {
            ...
        }

        %0 = ... : !ida.equation
        ida.set_residual @instance, %0 { function : @foo } : !ida.equation
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$instance,
        IDAEquation:$equation,
        SymbolNameAttr:$function);

    let results = (outs);

    let assemblyFormat = [{
        $instance `,` $equation attr-dict `:` type($equation)
    }];
}

def IDA_AddJacobianOp
    : IDA_Op<"add_jacobian", [MemoryEffects<[MemWrite]>]>
{
    let summary = "Add the Jacobian function of an equation.";

    let description = [{
        The operation is intended to inform IDA about the Jacobian function to
        be used for an equation already declared inside IDA.

        ```mlir
        ida.jacobian_function @foo {
            ...
        }

        %0 = ... : !ida.equation
        %1 = ... : !ida.variable
        ida.add_jacobian @instance, %0, %1 { function : @foo, seedSizes = [3, 6, 2] } : !ida.equation, !ida.variable
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$instance,
        IDAEquation:$equation,
        IDAVariable:$variable,
        SymbolNameAttr:$function,
        I64ArrayAttr:$seedSizes);

    let results = (outs);

    let assemblyFormat = [{
        $instance `,` $equation `,` $variable attr-dict `:` type($equation) `,` type($variable)
    }];
}

def IDA_CalcICOp : IDA_Op<"calc_ic", [MemoryEffects<[MemWrite]>]>
{
    let summary = "Compute the initial values of the variables.";

    let description = [{
        The operation instructs IDA to compute the initial variables of the
        variables.

        Example:
        ```mlir
        ida.calc_ic @instance
        ```
    }];

    let arguments = (ins SymbolNameAttr:$instance);
    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict
    }];
}

def IDA_StepOp : IDA_Op<"step", [MemoryEffects<[MemWrite]>]>
{
    let summary = "Perform a step within the simulation.";

    let description = [{
        The operation instructs IDA to perform a simulation step.

        Example:
        ```mlir
        ida.step @instance
        ```
    }];

    let arguments = (ins SymbolNameAttr:$instance);
    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict
    }];
}

def IDA_FreeOp : IDA_Op<"free", [MemoryEffects<[MemFree]>]>
{
    let summary = "Deallocates the IDA data structures and the array variables.";

    let description = [{
        The operation deallocates the IDA data structures and the memory
        reserved for the array variables that have been added to IDA.

        Example:
        ```mlir
        ida.free @instance
        ```
    }];

    let arguments = (ins SymbolNameAttr:$instance);
    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict
    }];
}

def IDA_PrintStatisticsOp
    : IDA_Op<"print_statistics", [MemoryEffects<[MemWrite]>]>
{
    let summary = "Print statistics regarding IDA.";

    let description = [{
        Print to stderr some statistics regarding IDA.

        Example:
        ```mlir
        %0 = ... : !ida.instance
        ida.print_statistics %0 : !ida.instance
        ```
    }];

    let arguments = (ins SymbolNameAttr:$instance);
    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict
    }];
}

#endif // MARCO_DIALECT_IDA_IR_IDAOPS_TD
