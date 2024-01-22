#ifndef MARCO_DIALECTS_KINSOL_KINSOL_TD
#define MARCO_DIALECTS_KINSOL_KINSOL_TD

include "marco/Dialect/KINSOL/KINSOLAttributes.td"
include "marco/Dialect/KINSOL/KINSOLDialect.td"
include "marco/Dialect/KINSOL/KINSOLTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===---------------------------------------------------------------------===//
// Type constraints
//===---------------------------------------------------------------------===//

// KINSOL instance type.
def KINSOLInstance : Type<CPred<"$_self.isa<::mlir::kinsol::InstanceType>()">,
                                "instance", "::mlir::kinsol::InstanceType">;

// KINSOL equation type.
def KINSOLEquation : Type<CPred<"$_self.isa<::mlir::kinsol::EquationType>()">,
                                "equation", "::mlir::kinsol::EquationType">;

// KINSOL variable type.
def KINSOLVariable : Type<CPred<"$_self.isa<::mlir::kinsol::VariableType>()">,
                                "variable", "::mlir::kinsol::VariableType">;

//===---------------------------------------------------------------------===//
// Attribute constraints
//===---------------------------------------------------------------------===//

// Multidimensional range attribute.
def MultidimensionalRangeAttr
    : Attr<CPred<"$_self.isa<::mlir::kinsol::MultidimensionalRangeAttr>()">,
                 "Multidimensional range attribute">
{
  let storageType = [{ ::mlir::kinsol::MultidimensionalRangeAttr }];
  let returnType = [{ ::mlir::kinsol::MultidimensionalRangeAttr }];
  let convertFromStorage = "$_self";
}

//===---------------------------------------------------------------------===//
// KINSOL operation definitions
//===---------------------------------------------------------------------===//

class KINSOL_Op<string mnemonic, list<Trait> traits = []> :
    Op<KINSOL_Dialect, mnemonic, traits>;

def KINSOL_InstanceOp : KINSOL_Op<"instance",
    [Symbol,
    HasParent<"mlir::ModuleOp">]>
{
    let summary = "Declare an instance of the KINSOL solver.";

    let description = [{
        Declare an instance of the KINSOL solver.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name
    );

    let results = (outs);

    let assemblyFormat = [{
        $sym_name attr-dict
    }];
}

def KINSOL_InitOp : KINSOL_Op<"create", []>
{
    let summary = "Create an KINSOL instance.";

    let description = [{
        The operation create a new KINSOL instance.

        Example:

        ```mlir
        kinsol.create @kinsol
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$instance);

    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict
    }];
}

def KINSOL_AddEquationOp : KINSOL_Op<"add_equation", []>
{
    let summary = "Add the information regarding an equation iteration range.";

    let description = [{
        The operation is intended to inform KINSOL about the iteration ranges
        of an equation.

        The ranges are expressed as integer pairs of values, where each pair
        consists in the beginning and ending iteration value, with the former
        being strictly less than the latter.

        For example, an equation iterating on two indices `i`, `j` and `k`,
        with `i` belonging to `[4,7)`, `j` belonging to `[9, 13)` and `k`
        belonging to `[1, 3)`, would be represented by the following 2-D 2x3
        array: `[[4, 7], [9, 13], [1, 3]]`.

        The operation returns a reference to the equation added to the KINSOL
        instance.

        Example:

        ```mlir
        %0 = ... : !kinsol.variable
        kinsol.add_equation @instance, %0 { equationRanges : #modeling<multidim_range [4,7][9,13][1,3]>, writeAccess : affine_map<(d0, d1, d2)[s0] -> (d1, d0 + s0, d2)> } : !kinsol.variable -> !kinsol.equation
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$instance,
        MultidimensionalRangeAttr:$equationRanges,
        KINSOLVariable:$writtenVariable,
        SymbolNameAttr:$writeAccessFunction,
        OptionalAttr<StrAttr>:$stringRepresentation);

    let results = (outs KINSOLEquation:$equation);

    let builders = [
        OpBuilder<(ins "llvm::StringRef":$kinsolInstance, "MultidimensionalRangeAttr":$equationRanges, "Value":$writtenVariable, "llvm::StringRef":$writeAccessFunction), [{
            auto resultType = kinsol::EquationType::get($_builder.getContext());
            build($_builder, $_state, resultType, kinsolInstance, equationRanges, writtenVariable, writeAccessFunction, nullptr);
        }]>
    ];

    let assemblyFormat = [{
        $instance `,` $writtenVariable attr-dict `:` type($writtenVariable) `->` type($equation)
    }];
}

def KINSOL_AddVariableOp : KINSOL_Op<"add_variable", []>
{
    let summary = "Add the information regarding the dimensions of a variable.";

    let description = [{
        The operation is intended to inform KINSOL about the dimensions of
        a variable.

        The operation expects three attributes:
          1. 'arrayDimensions': contains the dimensions of the array variable;
             scalar variables must be represented by a 1-D array of size 1.
          2. 'getter': the name of the function to be used by KINSOL to get the
             value of a scalar variable inside the array.
          3. 'setter': the name of the function to be used by KINSOL to set the
             value of a scalar variable inside the array.

        The operation returns a reference to the variable added to the KINSOL
        instance.

        Example:

        ```mlir
        kinsol.add_variable @instance { dimensions = [3, 5], getter = @x_getter, setter = @x_setter } : !kinsol.variable
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$instance,
        I64ArrayAttr:$dimensions,
        SymbolRefAttr:$getter,
        SymbolRefAttr:$setter,
        OptionalAttr<StrAttr>:$name);

    let results = (outs KINSOLVariable:$kinsolVariable);

    let builders = [
        OpBuilder<(ins "llvm::StringRef":$kinsolInstance, "ArrayAttr":$dimensions, "SymbolRefAttr":$getter, "SymbolRefAttr":$setter), [{
            auto resultType = kinsol::VariableType::get($_builder.getContext());
            build($_builder, $_state, resultType, kinsolInstance, dimensions, getter, setter, nullptr);
        }]>,
        OpBuilder<(ins "llvm::StringRef":$kinsolInstance, "ArrayAttr":$dimensions, "StringRef":$getter, "StringRef":$setter), [{
            auto resultType = kinsol::VariableType::get($_builder.getContext());
            auto getterAttr = SymbolRefAttr::get($_builder.getContext(), getter);
            auto setterAttr = SymbolRefAttr::get($_builder.getContext(), setter);
            build($_builder, $_state, resultType, kinsolInstance, dimensions, getterAttr, setterAttr, nullptr);
        }]>
    ];

    let assemblyFormat = [{
        $instance attr-dict `:` type($kinsolVariable)
    }];
}

def KINSOL_AddVariableAccessOp : KINSOL_Op<"add_variable_access", []>
{
    let summary = "Add the information regarding the access to an array variable inside an equation.";

    let description = [{
        The operation is used to inform KINSOL about how an equation accesses an
        array variable.

        ```mlir
        %0 = ... : !kinsol.equation
        %1 = ... : !kinsol.variable
        kinsol.add_variable_access @instance, %0, %1 { access = affine_map<(d0, d1)[s0] -> (d1, d0 + s0)> } : !kinsol.equation, !kinsol.variable
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$instance,
        KINSOLEquation:$equation,
        KINSOLVariable:$variable,
        SymbolNameAttr:$accessFunction);

    let results = (outs);

    let assemblyFormat = [{
        $instance `,` $equation `,` $variable attr-dict `:` type($equation) `,` type($variable)
    }];
}

def KINSOL_ResidualFunctionOp : KINSOL_Op<"residual_function",
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
          1. The indices of the scalar equation to be examined.

        Example:

        ```mlir
        kinsol.residual_function @foo(%i : index, %j : index, %k : index) -> f64 {
            %diff = ... : f64
            kinsol.return %diff : f64
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

            for (size_t i = 0; i < equationRank; ++i) {
                args.push_back($_builder.getIndexType());
            }

            auto functionType = $_builder.getFunctionType(args, $_builder.getF64Type());
            auto typeAttr = TypeAttr::get(functionType);
            build($_builder, $_state, nameAttr, typeAttr, nullptr, nullptr);
        }]>
    ];

    let extraClassDeclaration = [{
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

def KINSOL_JacobianFunctionOp : KINSOL_Op<"jacobian_function",
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
          1. The indices of the scalar equation to be examined.
          2. The indices of the scalar variable with respect to the equation is
             derived.

        Example:

        ```mlir
        kinsol.jacobian_function @foo(%i : index, %j : index, %k : index, %l : index, %m : index) -> f64 attributes { equationRank = 3, variableRank = 2 } {
            %der = ... : f64
            kinsol.return %der : f64
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
        OpBuilder<(ins "StringRef":$sym_name, "size_t":$equationRank, "size_t":$variableRank), [{
            auto nameAttr = $_builder.getStringAttr(sym_name);
            SmallVector<Type, 3> args;

            for (size_t i = 0; i < equationRank; ++i) {
                args.push_back($_builder.getIndexType());
            }

            for (size_t i = 0; i < variableRank; ++i) {
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
        ArrayRef<BlockArgument> getEquationIndices();
        ArrayRef<BlockArgument> getVariableIndices();

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

def KINSOL_ReturnOp : KINSOL_Op<"return",
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

def KINSOL_SetResidualOp : KINSOL_Op<"set_residual", []>
{
    let summary = "Add the residual function of an equation.";

    let description = [{
        The operation is intended to inform KINSOL about the residual function to
        be used for an equation already declared inside KINSOL.

        ```mlir
        kinsol.residual_function @foo {
            ...
        }

        %0 = ... : !kinsol.equation
        kinsol.set_residual @instance, %0 { function : @foo } : !kinsol.equation
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$instance,
        KINSOLEquation:$equation,
        SymbolNameAttr:$function);

    let results = (outs);

    let assemblyFormat = [{
        $instance `,` $equation attr-dict `:` type($equation)
    }];
}

def KINSOL_AddJacobianOp : KINSOL_Op<"add_jacobian", []>
{
    let summary = "Add the Jacobian function of an equation.";

    let description = [{
        The operation is intended to inform KINSOL about the Jacobian function to
        be used for an equation already declared inside KINSOL.

        ```mlir
        kinsol.jacobian_function @foo {
            ...
        }

        %0 = ... : !kinsol.equation
        %1 = ... : !kinsol.variable
        kinsol.add_jacobian @instance, %0, %1 { function : @foo } : !kinsol.equation, !kinsol.variable
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$instance,
        KINSOLEquation:$equation,
        KINSOLVariable:$variable,
        SymbolNameAttr:$function);

    let results = (outs);

    let assemblyFormat = [{
        $instance `,` $equation `,` $variable attr-dict `:` type($equation) `,` type($variable)
    }];
}

def KINSOL_SolveOp : KINSOL_Op<"solve", []>
{
    let summary = "Perform a step within the simulation.";

    let description = [{
        The operation instructs KINSOL to perform a simulation step.

        Example:
        ```mlir
        kinsol.step @instance
        ```
    }];

    let arguments = (ins SymbolNameAttr:$instance);
    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict
    }];
}

def KINSOL_FreeOp : KINSOL_Op<"free", []>
{
    let summary = "Deallocates the KINSOL data structures and the array variables.";

    let description = [{
        The operation deallocates the KINSOL data structures and the memory
        reserved for the array variables that have been added to KINSOL.

        Example:
        ```mlir
        kinsol.free @instance
        ```
    }];

    let arguments = (ins SymbolNameAttr:$instance);
    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict
    }];
}

#endif // MARCO_DIALECTS_KINSOL_KINSOL_TD
