#ifndef MARCO_DIALECTS_KINSOL_KINSOL_TD
#define MARCO_DIALECTS_KINSOL_KINSOL_TD

include "marco/Dialect/KINSOL/KINSOLAttributes.td"
include "marco/Dialect/KINSOL/KINSOLDialect.td"
include "marco/Dialect/KINSOL/KINSOLTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

// KINSOL instance type
def KINSOLInstance : Type<CPred<"$_self.isa<::mlir::kinsol::InstanceType>()">,
                       "instance", "::mlir::kinsol::InstanceType">;

// KINSOL equation type
def KINSOLEquation : Type<CPred<"$_self.isa<::mlir::kinsol::EquationType>()">,
                             "equation", "::mlir::kinsol::EquationType">;

// KINSOL variable type
def KINSOLVariable : Type<CPred<"$_self.isa<::mlir::kinsol::VariableType>()">,
                             "variable", "::mlir::kinsol::VariableType">;

//===----------------------------------------------------------------------===//
// KINSOL operation definitions
//===----------------------------------------------------------------------===//

class KINSOL_Op<string mnemonic, list<Trait> traits = []> :
    Op<KINSOL_Dialect, mnemonic, traits>;

def KINSOL_CreateOp : KINSOL_Op<"create",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Create an KINSOL instance.";

    let description = [{
        The operation create a new KINSOL instance.

        The operations expects an attribute called 'scalarEquations'
        indicating the total amount of scalar equations of the model.
        The number of scalar variables is not needed as it matches
        by construction the number of scalar equation.
        The underlying data structures are allocated but not
        initialized; initialization is delegated to the InitOp
        operation, which has to be called after all the data
        regarding equations and variables have been set by means
        of the appropriate operations (i.e. AddEquationOp, etc.).

        Example:

        ```mlir
        %0 = kinsol.create { scalarEquations = 23 } : !kinsol.instance
        ```
    }];

    let arguments = (ins I64Attr:$scalarEquations);

    let results = (outs KINSOLInstance:$instance);

    let builders = [
        OpBuilder<(ins "IntegerAttr":$scalarEquations), [{
            auto resultType = kinsol::InstanceType::get($_builder.getContext());
            build($_builder, $_state, resultType, scalarEquations);
        }]>
    ];

    let assemblyFormat = [{
        attr-dict `:` type($instance)
    }];
}


def KINSOL_AddEquationOp : KINSOL_Op<"add_equation",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Add the information regarding an equation iteration range.";

    let description = [{
        The operation is intended to inform KINSOL about the iteration
        ranges of an equation.

        The ranges are expressed as integer pairs of values, where
        each pair consists in the beginning and ending iteration value,
        with the former being strictly less than the latter.

        For example, an equation iterating on two indices `i`, `j` and `k`,
        with `i` belonging to `[4,7)`, `j` belonging to `[9, 13)` and `k`
        belonging to `[1, 3)`, would be represented by the following 2-D
        2x3 array: `[[4, 7], [9, 13], [1, 3]]`.

        The operation returns a reference to the equation added to the
        KINSOL instance.

        Example:

        ```mlir
        %0 = ... : !kinsol.instance
        kinsol.add_equation %0 { equationRanges : [[4,7], [9,13], [1,3]] } : kinsol.instance -> kinsol.equation
        ```
    }];

    let arguments = (ins
        KINSOLInstance:$instance,
        ArrayAttr:$equationRanges);

    let results = (outs KINSOLEquation:$equation);

    let builders = [
        OpBuilder<(ins "Value":$kinsolInstance, "ArrayAttr":$equationRanges), [{
            auto resultType = kinsol::EquationType::get($_builder.getContext());
            build($_builder, $_state, resultType, kinsolInstance, equationRanges);
        }]>
    ];

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance) `->` type($equation)
    }];
}

def KINSOL_AddAlgebraicVariableOp : KINSOL_Op<"add_algebraic_variable",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Add the information regarding the dimensions of an algebraic variable.";

    let description = [{
        The operation is intended to inform KINSOL about the dimensions of
        an algebraic variable.

        The operation expects three attributes:
          1. 'arrayDimensions': contains the dimensions of the array
             variable; scalar variables must be represented by a 1-D
             array of size 1.
          2. 'getter': the name of the function to be used by KINSOL to
             get the value of a scalar variable inside the array.
          3. 'setter': the name of the function to be used by KINSOL to
             set the value of a scalar variable inside the array.

        The operation returns a reference to the variable added to the
        KINSOL instance.

        Example:

        ```mlir
        %0 = ... : !kinsol.instance
        kinsol.add_algebraic_variable %0, %1 { dimensions = [3, 5], getter = @x_getter, setter = @x_setter } : (!kinsol.instance, !mydialect.type) -> !kinsol.variable
        ```
    }];

    let arguments = (ins
        KINSOLInstance:$instance,
        AnyType:$variable,
        I64ArrayAttr:$dimensions,
        SymbolRefAttr:$getter,
        SymbolRefAttr:$setter);

    let results = (outs KINSOLVariable:$kinsolVariable);

    let builders = [
        OpBuilder<(ins "Value":$kinsolInstance, "Value":$variable, "ArrayAttr":$dimensions, "SymbolRefAttr":$getter, "SymbolRefAttr":$setter), [{
            auto resultType = kinsol::VariableType::get($_builder.getContext());
            build($_builder, $_state, resultType, kinsolInstance, variable, dimensions, getter, setter);
        }]>,
        OpBuilder<(ins "Value":$kinsolInstance, "Value":$variable, "ArrayAttr":$dimensions, "StringRef":$getter, "StringRef":$setter), [{
            auto resultType = kinsol::VariableType::get($_builder.getContext());
            auto getterAttr = SymbolRefAttr::get($_builder.getContext(), getter);
            auto setterAttr = SymbolRefAttr::get($_builder.getContext(), setter);
            build($_builder, $_state, resultType, kinsolInstance, variable, dimensions, getterAttr, setterAttr);
        }]>
    ];

    let assemblyFormat = [{
        $instance `,` $variable attr-dict `:` `(` type($instance) `,` type($variable) `)` `->` type($kinsolVariable)
    }];
}

def KINSOL_VariableGetterOp : KINSOL_Op<"variable_getter",
    [IsolatedFromAbove, FunctionOpInterface, AutomaticAllocationScope, Symbol]>
{
    let summary = "Getter interface for an external variable.";

    let description = [{
        The `kinsol.variable_getter` is a function-like operation that
        acts as a bridge between KINSOL and the custom dialect types.
        It allows to define how an array variable can be accessed
        and one of its scalar values retrieved.

        The function signature consists in a variadic list of arguments
        and one result.
        The first argument is the array variable (with its original
        type); the remaining ones consist in the indices of the
        scalar variable to be accessed.
        The return value is indeed the accessed scalar variable.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type);

    let results = (outs);

    let regions = (region
        AnyRegion:$bodyRegion
    );

    let builders = [
        OpBuilder<(ins "StringRef":$sym_name, "Type":$resultType, "Type":$variableType, "unsigned int":$variableRank), [{
            auto nameAttr = $_builder.getStringAttr(sym_name);

            SmallVector<Type, 2> args;

            args.push_back(variableType);
            args.append(variableRank, $_builder.getIndexType());

            auto functionType = $_builder.getFunctionType(args, resultType);
            auto typeAttr = TypeAttr::get(functionType);
            build($_builder, $_state, nameAttr, typeAttr);
        }]>
    ];

    let extraClassDeclaration = [{
        BlockArgument getVariable();
        ArrayRef<BlockArgument> getVariableIndices();

        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() {
            return getFunctionType().getResults();
        }

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() {
            return isExternal();
        }
    }];
}

def KINSOL_VariableSetterOp : KINSOL_Op<"variable_setter",
    [IsolatedFromAbove, FunctionOpInterface, AutomaticAllocationScope, Symbol]>
{
    let summary = "Setter interface for an external variable.";

    let description = [{
        The `kinsol.variable_setter` is a function-like operation that
        acts as a bridge between KINSOL and the custom dialect types.
        It allows to define how an array variable can be accessed
        and one of its scalar values updated.

        The function signature consists in a variadic list of arguments.
        The first argument is the array variable (with its original
        type); the second argument is the value to be set; the
        remaining ones consist in the indices of the scalar variable
        to be accessed.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type);

    let results = (outs);

    let regions = (region
        AnyRegion:$bodyRegion
    );

    let builders = [
        OpBuilder<(ins "StringRef":$sym_name, "Type":$variableType, "Type":$valueType, "unsigned int":$variableRank), [{
            auto nameAttr = $_builder.getStringAttr(sym_name);

            SmallVector<Type, 3> args;

            args.push_back(variableType);
            args.push_back(valueType);
            args.append(variableRank, $_builder.getIndexType());

            auto functionType = $_builder.getFunctionType(args, llvm::None);
            auto typeAttr = TypeAttr::get(functionType);
            build($_builder, $_state, nameAttr, typeAttr);
        }]>
    ];

    let extraClassDeclaration = [{
        BlockArgument getVariable();
        BlockArgument getValue();
        ArrayRef<BlockArgument> getVariableIndices();

        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() {
            return getFunctionType().getResults();
        }

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() {
            return isExternal();
        }
    }];
}

def KINSOL_AddVariableAccessOp : KINSOL_Op<"add_variable_access",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Add the information regarding the access to an array variable inside an equation.";

    let description = [{
        The operation is used to inform KINSOL about how an equation accesses
        an array variable.

        ```mlir
        %0 = ... : !kinsol.instance
        %1 = ... : !kinsol.equation
        %2 = ... : !kinsol.variable
        kinsol.add_variable_access %0, %1, %2 { access = affine_map<(d0, d1)[s0] -> (d1, d0 + s0)> } : !kinsol.instance, !kinsol.equation, !kinsol.variable
        ```
    }];

    let arguments = (ins
        KINSOLInstance:$instance,
        KINSOLEquation:$equation,
        KINSOLVariable:$variable,
        AffineMapAttr:$access);

    let results = (outs);

    let assemblyFormat = [{
        $instance `,` $equation `,` $variable attr-dict `:` type($instance) `,` type($equation) `,` type($variable)
    }];
}

def KINSOL_ResidualFunctionOp : KINSOL_Op<"residual_function",
    [IsolatedFromAbove, FunctionOpInterface, AutomaticAllocationScope, Symbol]>
{
    let summary = "Residual function of a loop equation.";

    let description = [{
        The residual function of an equation is a function that computes
        the residual error of that equation.

        The residual error of an equation consists in the difference
        between its right-hand side and left-hand side values.

        The function expects an attribute name `equationRank` indicating
        the number of iteration variables of the equation.

        The function takes a variadic list of arguments:
          1. Current time of the simulation (which will be provided by KINSOL when calling the residual function).
          2. The list of the original variables.
          3. The indices of the scalar equation to be examined.

        Example:

        ```mlir
        kinsol.residual_function @foo attributes { equationRank = 3 } {
            ^bbo(%time : f64, %var0 : !mydialect.type1, %var1 : !mydialect.type2, %i : index, %j : index, %k : index) {
                ...
                %diff = ... : f64
                kinsol.return %diff : f64
            }
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        IndexAttr:$equationRank);

    let results = (outs);

    let regions = (region
        AnyRegion:$bodyRegion
    );

    let builders = [
        OpBuilder<(ins "StringRef":$sym_name, "Type":$timeType, "TypeRange":$variablesTypes, "size_t":$equationRank, "Type":$differenceType), [{
            auto nameAttr = $_builder.getStringAttr(sym_name);

            SmallVector<Type, 3> args;

            args.push_back(timeType);

            for (const auto& variableType : variablesTypes) {
                args.push_back(variableType);
            }

            for (size_t i = 0; i < equationRank; ++i) {
                args.push_back($_builder.getIndexType());
            }

            auto functionType = $_builder.getFunctionType(args, differenceType);
            auto typeAttr = TypeAttr::get(functionType);
            build($_builder, $_state, nameAttr, typeAttr, $_builder.getIndexAttr(equationRank));
        }]>
    ];

    let extraClassDeclaration = [{
        BlockArgument getTime();
        ArrayRef<BlockArgument> getVariables();
        ArrayRef<BlockArgument> getEquationIndices();

        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() {
            return getFunctionType().getResults();
        }

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() {
            return isExternal();
        }
    }];
}

def KINSOL_JacobianFunctionOp : KINSOL_Op<"jacobian_function",
    [IsolatedFromAbove, FunctionOpInterface, AutomaticAllocationScope, Symbol]>
{
    let summary = "Jacobian function of a loop equation.";

    let description = [{
        The Jacobian function of an equation is a function that computes
        an element of the Jacobian matrix of that equation.

        The function expects an attribute named `equationRank` indicating
        the number of iteration variables of the equation and an attribute
        named `variableRank` indicating the rank of the variable with
        respect to the equation is derived.

        The function takes a variadic list of arguments:
          1. Current time of the simulation (which will be provided by KINSOL when calling the residual function).
          2. The list of the original variables.
          3. The indices of the scalar equation to be examined.
          4. The indices of the scalar variable with respect to the equation is derived.
          5. The KINSOL `alpha` parameter .

        Example:

        ```mlir
        kinsol.jacobian_function @foo attributes { equationRank = 3, variableRank = 2 } {
            ^bbo(%time : f64, %var0 : !mydialect.type1, %var1 : !mydialect.type2, %i : index, %j : index, %k : index, %l : index, %m : index, %alpha: f64) {
                ...
                %der = ... : f64
                kinsol.return %der : f64
            }
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        IndexAttr:$equationRank,
        IndexAttr:$variableRank);

    let results = (outs);

    let regions = (region
        AnyRegion:$bodyRegion
    );

    let builders = [
        OpBuilder<(ins "StringRef":$sym_name, "Type":$timeType, "TypeRange":$variablesTypes, "size_t":$equationRank, "size_t":$variableRank, "Type":$alphaType, "Type":$differenceType), [{
            auto nameAttr = $_builder.getStringAttr(sym_name);

            SmallVector<Type, 3> args;

            args.push_back(timeType);

            for (const auto& variableType : variablesTypes) {
                args.push_back(variableType);
            }

            for (size_t i = 0; i < equationRank; ++i) {
                args.push_back($_builder.getIndexType());
            }

            for (size_t i = 0; i < variableRank; ++i) {
                args.push_back($_builder.getIndexType());
            }

            args.push_back(alphaType);

            auto functionType = $_builder.getFunctionType(args, differenceType);
            auto typeAttr = TypeAttr::get(functionType);
            build($_builder, $_state, nameAttr, typeAttr, $_builder.getIndexAttr(equationRank), $_builder.getIndexAttr(variableRank));
        }]>
    ];

    let extraClassDeclaration = [{
        BlockArgument getTime();
        ArrayRef<BlockArgument> getVariables();
        ArrayRef<BlockArgument> getEquationIndices();
        ArrayRef<BlockArgument> getVariableIndices();
        BlockArgument getAlpha();

        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() {
            return getFunctionType().getResults();
        }

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration() {
            return isExternal();
        }
    }];
}

def KINSOL_ReturnOp : KINSOL_Op<"return",
    [NoSideEffect,
     ParentOneOf<["VariableGetterOp", "VariableSetterOp", "ResidualFunctionOp", "JacobianFunctionOp"]>,
     ReturnLike, Terminator]>
{
    let summary = "Return operation.";

    let description = [{
        The operation represents a return operation within a function.
        The operation takes variable number of operands and produces no results.
        The operand number and types must match the signature of the function
        that contains the operation.
    }];

    let arguments = (ins Variadic<AnyType>:$operands);

    let builders = [
        OpBuilder<(ins), [{
            build($_builder, $_state, llvm::None);
        }]>];

    let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def KINSOL_AddResidualOp : KINSOL_Op<"add_residual",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Add the residual function of an equation.";

    let description = [{
        The operation is intended to inform KINSOL about the residual
        function to be used for an equation already declared inside
        KINSOL.

        ```mlir
        kinsol.residual_function @foo {
            ...
        }

        %0 = ... : !kinsol.instance
        %1 = ... : !kinsol.equation
        kinsol.add_residual %0, %1 { function : @foo } : !kinsol.instance, !kinsol.equation
        ```
    }];

    let arguments = (ins
        KINSOLInstance:$instance,
        KINSOLEquation:$equation,
        SymbolNameAttr:$function);

    let results = (outs);

    let assemblyFormat = [{
        $instance `,` $equation attr-dict `:` type($instance) `,` type($equation)
    }];
}

def KINSOL_AddJacobianOp : KINSOL_Op<"add_jacobian",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Add the Jacobian function of an equation.";

    let description = [{
        The operation is intended to inform KINSOL about the Jacobian
        function to be used for an equation already declared inside
        KINSOL.

        ```mlir
        kinsol.jacobian_function @foo {
            ...
        }

        %0 = ... : !kinsol.instance
        %1 = ... : !kinsol.equation
        %2 = ... : !kinsol.variable
        kinsol.add_jacobian %0, %1, %2 { function : @foo } : !kinsol.instance, !kinsol.equation, !kinsol.variable
        ```
    }];

    let arguments = (ins
        KINSOLInstance:$instance,
        KINSOLEquation:$equation,
        KINSOLVariable:$variable,
        SymbolNameAttr:$function);

    let results = (outs);

    let assemblyFormat = [{
        $instance `,` $equation `,` $variable attr-dict `:` type($instance) `,` type($equation) `,` type($variable)
    }];
}

def KINSOL_InitOp : KINSOL_Op<"init",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Initialize the KINSOL environment and solve the system.";

    let description = [{
        The operation initializes the KINSOL environment.
        Note that all the equations and variables must have
        already been added to the KINSOL instance.
        The non linear system of equations is then solved.

        Example:

        ```mlir
        %0 = ... : !kinsol.instance
        %1 = kinsol.init %0 : !kinsol.instance
        ```
    }];

    let arguments = (ins KINSOLInstance:$instance);
    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

def KINSOL_FreeOp : KINSOL_Op<"free",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Deallocates the KINSOL data structures and the array variables.";

    let description = [{
        The operation deallocates the KINSOL data structures and
        the memory reserved for the array variables that have
        been added to KINSOL.

        Example:
        ```mlir
        %0 = ... : !kinsol.instance
        %1 = kinsol.free %0 : !kinsol.instance
        ```
    }];

    let arguments = (ins KINSOLInstance:$instance);
    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

def KINSOL_PrintStatisticsOp : KINSOL_Op<"print_statistics",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Print statistics regarding KINSOL.";

    let description = [{
        Print to stderr some statistics regarding KINSOL.

        Example:
        ```mlir
        %0 = ... : !kinsol.instance
        kinsol.print_statistics %0 : !kinsol.instance
        ```
    }];

    let arguments = (ins KINSOLInstance:$instance);
    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

#endif // MARCO_DIALECTS_KINSOL_KINSOL_TD
