#ifndef MARCO_DIALECTS_IDA_IDA_TD
#define MARCO_DIALECTS_IDA_IDA_TD

include "marco/Dialect/IDA/IDAAttributes.td"
include "marco/Dialect/IDA/IDADialect.td"
include "marco/Dialect/IDA/IDATypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// Type constraints
//===----------------------------------------------------------------------===//

// IDA instance type
def IDAInstance : Type<CPred<"$_self.isa<::mlir::ida::InstanceType>()">,
                       "instance", "::mlir::ida::InstanceType">;

// IDA equation type
def IDAEquation : Type<CPred<"$_self.isa<::mlir::ida::EquationType>()">,
                             "equation", "::mlir::ida::EquationType">;

// IDA variable type
def IDAVariable : Type<CPred<"$_self.isa<::mlir::ida::VariableType>()">,
                             "variable", "::mlir::ida::VariableType">;

//===----------------------------------------------------------------------===//
// IDA operation definitions
//===----------------------------------------------------------------------===//

class IDA_Op<string mnemonic, list<Trait> traits = []> :
    Op<IDA_Dialect, mnemonic, traits>;

def IDA_CreateOp : IDA_Op<"create",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Create an IDA instance.";

    let description = [{
        The operation create a new IDA instance.

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
        %0 = ida.create { scalarEquations = 23 } : !ida.instance
        ```
    }];

    let arguments = (ins I64Attr:$scalarEquations);

    let results = (outs IDAInstance:$instance);

    let builders = [
        OpBuilder<(ins "IntegerAttr":$scalarEquations), [{
            auto resultType = ida::InstanceType::get($_builder.getContext());
            build($_builder, $_state, resultType, scalarEquations);
        }]>
    ];

    let assemblyFormat = [{
        attr-dict `:` type($instance)
    }];
}

def IDA_SetStartTimeOp : IDA_Op<"set_start_time",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Set the start time of the simulation.";

    let description = [{
        The operation sets the start time to be used by
        IDA during the simulation.

        Example:

        ```mlir
        %0 = ... : !ida.instance
        ida.set_start_time %0 { time = 0.0 } : !ida.instance
        ```
    }];

    let arguments = (ins
        IDAInstance:$instance,
        F64Attr:$time);

    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

def IDA_SetEndTimeOp : IDA_Op<"set_end_time",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Set the end time of the simulation.";

    let description = [{
        The operation sets the end time to be used by
        IDA during the simulation.

        Example:

        ```mlir
        %0 = ... : !ida.instance
        ida.set_end_time %0 { time = 0.0 } : !ida.instance
        ```
    }];

    let arguments = (ins
        IDAInstance:$instance,
        F64Attr:$time);

    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

def IDA_SetTimeStepOp : IDA_Op<"set_time_step",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Set a fixed time step for the simulation.";

    let description = [{
        The operation enables the equidistant time grid inside
        IDA and sets the time step to the one provided through
        the 'timeStep' attribute.

        Example:

        ```mlir
        %0 = ... : !ida.instance
        ida.set_time_step %0 { timeStep = 0.1 } : !ida.instance
        ```
    }];

    let arguments = (ins
        IDAInstance:$instance,
        F64Attr:$timeStep);

    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

def IDA_GetCurrentTimeOp : IDA_Op<"get_current_time",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Get the time reached by IDA.";

    let description = [{
        The operation returns the time reached by the simulation.

        Example:

        ```mlir
        %0 = ... : !ida.instance
        %1 = ida.get_current_time %0 : !ida.instance -> f64
        ```
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs AnyType:$time);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance) `->` type($time)
    }];
}

def IDA_AddEquationOp : IDA_Op<"add_equation",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Add the information regarding an equation iteration range.";

    let description = [{
        The operation is intended to inform IDA about the iteration
        ranges of an equation.

        The ranges are expressed as integer pairs of values, where
        each pair consists in the beginning and ending iteration value,
        with the former being strictly less than the latter.

        For example, an equation iterating on two indices `i`, `j` and `k`,
        with `i` belonging to `[4,7)`, `j` belonging to `[9, 13)` and `k`
        belonging to `[1, 3)`, would be represented by the following 2-D
        2x3 array: `[[4, 7], [9, 13], [1, 3]]`.

        The operation returns a reference to the equation added to the
        IDA instance.

        Example:

        ```mlir
        %0 = ... : !ida.instance
        ida.add_equation %0 { equationRanges : [[4,7], [9,13], [1,3]] } : ida.instance -> ida.equation
        ```
    }];

    let arguments = (ins
        IDAInstance:$instance,
        ArrayAttr:$equationRanges);

    let results = (outs IDAEquation:$equation);

    let builders = [
        OpBuilder<(ins "Value":$idaInstance, "ArrayAttr":$equationRanges), [{
            auto resultType = ida::EquationType::get($_builder.getContext());
            build($_builder, $_state, resultType, idaInstance, equationRanges);
        }]>
    ];

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance) `->` type($equation)
    }];
}

def IDA_AddAlgebraicVariableOp : IDA_Op<"add_algebraic_variable",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Add the information regarding the dimensions of an algebraic variable.";

    let description = [{
        The operation is intended to inform IDA about the dimensions of
        an algebraic variable.

        The operation expects three attributes:
          1. 'arrayDimensions': contains the dimensions of the array
             variable; scalar variables must be represented by a 1-D
             array of size 1.
          2. 'getter': the name of the function to be used by IDA to
             get the value of a scalar variable inside the array.
          3. 'setter': the name of the function to be used by IDA to
             set the value of a scalar variable inside the array.

        The operation returns a reference to the variable added to the
        IDA instance.

        Example:

        ```mlir
        %0 = ... : !ida.instance
        ida.add_algebraic_variable %0, %1 { dimensions = [3, 5], getter = @x_getter, setter = @x_setter } : (!ida.instance, !mydialect.type) -> !ida.variable
        ```
    }];

    let arguments = (ins
        IDAInstance:$instance,
        AnyType:$variable,
        I64ArrayAttr:$dimensions,
        SymbolRefAttr:$getter,
        SymbolRefAttr:$setter);

    let results = (outs IDAVariable:$idaVariable);

    let builders = [
        OpBuilder<(ins "Value":$idaInstance, "Value":$variable, "ArrayAttr":$dimensions, "SymbolRefAttr":$getter, "SymbolRefAttr":$setter), [{
            auto resultType = ida::VariableType::get($_builder.getContext());
            build($_builder, $_state, resultType, idaInstance, variable, dimensions, getter, setter);
        }]>,
        OpBuilder<(ins "Value":$idaInstance, "Value":$variable, "ArrayAttr":$dimensions, "StringRef":$getter, "StringRef":$setter), [{
            auto resultType = ida::VariableType::get($_builder.getContext());
            auto getterAttr = SymbolRefAttr::get($_builder.getContext(), getter);
            auto setterAttr = SymbolRefAttr::get($_builder.getContext(), setter);
            build($_builder, $_state, resultType, idaInstance, variable, dimensions, getterAttr, setterAttr);
        }]>
    ];

    let assemblyFormat = [{
        $instance `,` $variable attr-dict `:` `(` type($instance) `,` type($variable) `)` `->` type($idaVariable)
    }];
}

def IDA_AddStateVariableOp : IDA_Op<"add_state_variable",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Add the information regarding the dimensions of a state variable.";

    let description = [{
        The operation is intended to inform IDA about the dimensions of
        a state variable.

        The operation expects three attributes:
          1. 'arrayDimensions': contains the dimensions of the array
             variable; scalar variables must be represented by a 1-D
             array of size 1.
          2. 'getter': the name of the function to be used by IDA to
             get the value of a scalar variable inside the array.
          3. 'setter': the name of the function to be used by IDA to
             set the value of a scalar variable inside the array.

        The operation returns a reference to the variable added to the
        IDA instance.

        Example:

        ```mlir
        %0 = ... : !ida.instance
        ida.add_state_variable %0, %1 { dimensions = [3, 5], getter = @x_getter, setter = @x_setter } : (!ida.instance, !mydialect.type) -> !ida.variable
        ```
    }];

    let arguments = (ins
        IDAInstance:$instance,
        AnyType:$variable,
        I64ArrayAttr:$dimensions,
        SymbolRefAttr:$getter,
        SymbolRefAttr:$setter);

    let results = (outs IDAVariable:$idaVariable);

    let builders = [
        OpBuilder<(ins "Value":$idaInstance, "Value":$variable, "ArrayAttr":$dimensions, "SymbolRefAttr":$getter, "SymbolRefAttr":$setter), [{
            auto resultType = ida::VariableType::get($_builder.getContext());
            build($_builder, $_state, resultType, idaInstance, variable, dimensions, getter, setter);
        }]>,
        OpBuilder<(ins "Value":$idaInstance, "Value":$variable, "ArrayAttr":$dimensions, "StringRef":$getter, "StringRef":$setter), [{
            auto resultType = ida::VariableType::get($_builder.getContext());
            auto getterAttr = SymbolRefAttr::get($_builder.getContext(), getter);
            auto setterAttr = SymbolRefAttr::get($_builder.getContext(), setter);
            build($_builder, $_state, resultType, idaInstance, variable, dimensions, getterAttr, setterAttr);
        }]>
    ];

    let assemblyFormat = [{
        $instance `,` $variable attr-dict `:` `(` type($instance) `,` type($variable) `)` `->` type($idaVariable)
    }];
}

def IDA_SetDerivativeOp : IDA_Op<"set_derivative",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Set the derivative of a state variable.";

    let description = [{
        The operation is used to set the derivative of a state variable
        that has been previously declared inside IDA.

        The operation expects the 'getter' and 'setter' attributes as
        the `ida.add_state_variable`, to be used by IDA to operate on
        the derivative array.

        Example:

        ```mlir
        %0 = ... : !ida.instance
        %1 = ... : !ida.variable
        ida.set_derivative %0, %1, %2 { getter = @der_x_getter, setter = @der_x_setter } : (!ida.instance, !ida.variable, !mydialect.type) -> !ida.variable
        ```
    }];

    let arguments = (ins
        IDAInstance:$instance,
        IDAVariable:$stateVariable,
        AnyType:$derivative,
        SymbolRefAttr:$getter,
        SymbolRefAttr:$setter);

    let results = (outs);

    let builders = [
        OpBuilder<(ins "Value":$idaInstance, "Value":$stateVariable, "Value":$derivative, "StringRef":$getter, "StringRef":$setter), [{
            auto getterAttr = SymbolRefAttr::get($_builder.getContext(), getter);
            auto setterAttr = SymbolRefAttr::get($_builder.getContext(), setter);
            build($_builder, $_state, idaInstance, stateVariable, derivative, getterAttr, setterAttr);
        }]>
    ];

    let assemblyFormat = [{
        $instance `,` $stateVariable `,` $derivative attr-dict `:` type($instance) `,` type($stateVariable) `,` type($derivative)
    }];
}

def IDA_VariableGetterOp : IDA_Op<"variable_getter",
    [IsolatedFromAbove, FunctionOpInterface, AutomaticAllocationScope, Symbol]>
{
    let summary = "Getter interface for an external variable.";

    let description = [{
        The `ida.variable_getter` is a function-like operation that
        acts as a bridge between IDA and the custom dialect types.
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

def IDA_VariableSetterOp : IDA_Op<"variable_setter",
    [IsolatedFromAbove, FunctionOpInterface, AutomaticAllocationScope, Symbol]>
{
    let summary = "Setter interface for an external variable.";

    let description = [{
        The `ida.variable_setter` is a function-like operation that
        acts as a bridge between IDA and the custom dialect types.
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

def IDA_AddVariableAccessOp : IDA_Op<"add_variable_access",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Add the information regarding the access to an array variable inside an equation.";

    let description = [{
        The operation is used to inform IDA about how an equation accesses
        an array variable.

        ```mlir
        %0 = ... : !ida.instance
        %1 = ... : !ida.equation
        %2 = ... : !ida.variable
        ida.add_variable_access %0, %1, %2 { access = affine_map<(d0, d1)[s0] -> (d1, d0 + s0)> } : !ida.instance, !ida.equation, !ida.variable
        ```
    }];

    let arguments = (ins
        IDAInstance:$instance,
        IDAEquation:$equation,
        IDAVariable:$variable,
        AffineMapAttr:$access);

    let results = (outs);

    let assemblyFormat = [{
        $instance `,` $equation `,` $variable attr-dict `:` type($instance) `,` type($equation) `,` type($variable)
    }];
}

def IDA_ResidualFunctionOp : IDA_Op<"residual_function",
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
          1. Current time of the simulation (which will be provided by IDA when calling the residual function).
          2. The list of the original variables.
          3. The indices of the scalar equation to be examined.

        Example:

        ```mlir
        ida.residual_function @foo attributes { equationRank = 3 } {
            ^bbo(%time : f64, %var0 : !mydialect.type1, %var1 : !mydialect.type2, %i : index, %j : index, %k : index) {
                ...
                %diff = ... : f64
                ida.return %diff : f64
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

def IDA_JacobianFunctionOp : IDA_Op<"jacobian_function",
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
          1. Current time of the simulation (which will be provided by IDA when calling the residual function).
          2. The list of the original variables.
          3. The indices of the scalar equation to be examined.
          4. The indices of the scalar variable with respect to the equation is derived.
          5. The IDA `alpha` parameter .

        Example:

        ```mlir
        ida.jacobian_function @foo attributes { equationRank = 3, variableRank = 2 } {
            ^bbo(%time : f64, %var0 : !mydialect.type1, %var1 : !mydialect.type2, %i : index, %j : index, %k : index, %l : index, %m : index, %alpha: f64) {
                ...
                %der = ... : f64
                ida.return %der : f64
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

def IDA_ReturnOp : IDA_Op<"return",
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

def IDA_AddResidualOp : IDA_Op<"add_residual",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Add the residual function of an equation.";

    let description = [{
        The operation is intended to inform IDA about the residual
        function to be used for an equation already declared inside
        IDA.

        ```mlir
        ida.residual_function @foo {
            ...
        }

        %0 = ... : !ida.instance
        %1 = ... : !ida.equation
        ida.add_residual %0, %1 { function : @foo } : !ida.instance, !ida.equation
        ```
    }];

    let arguments = (ins
        IDAInstance:$instance,
        IDAEquation:$equation,
        SymbolNameAttr:$function);

    let results = (outs);

    let assemblyFormat = [{
        $instance `,` $equation attr-dict `:` type($instance) `,` type($equation)
    }];
}

def IDA_AddJacobianOp : IDA_Op<"add_jacobian",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Add the Jacobian function of an equation.";

    let description = [{
        The operation is intended to inform IDA about the Jacobian
        function to be used for an equation already declared inside
        IDA.

        ```mlir
        ida.jacobian_function @foo {
            ...
        }

        %0 = ... : !ida.instance
        %1 = ... : !ida.equation
        %2 = ... : !ida.variable
        ida.add_jacobian %0, %1, %2 { function : @foo } : !ida.instance, !ida.equation, !ida.variable
        ```
    }];

    let arguments = (ins
        IDAInstance:$instance,
        IDAEquation:$equation,
        IDAVariable:$variable,
        SymbolNameAttr:$function);

    let results = (outs);

    let assemblyFormat = [{
        $instance `,` $equation `,` $variable attr-dict `:` type($instance) `,` type($equation) `,` type($variable)
    }];
}

def IDA_InitOp : IDA_Op<"init",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Initialize the IDA environment.";

    let description = [{
        The operation initializes the IDA environment.
        Note that all the equations and variables must have
        already been added to the IDA instance.

        Example:

        ```mlir
        %0 = ... : !ida.instance
        %1 = ida.init %0 : !ida.instance
        ```
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

def IDA_StepOp : IDA_Op<"step",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Perform a step within the simulation.";

    let description = [{
        The operation instructs IDA to perform a simulation step.

        Example:
        ```mlir
        %0 = ... : !ida.instance
        %1 = ida.step %0 : !ida.instance
        ```
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

def IDA_FreeOp : IDA_Op<"free",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Deallocates the IDA data structures and the array variables.";

    let description = [{
        The operation deallocates the IDA data structures and
        the memory reserved for the array variables that have
        been added to IDA.

        Example:
        ```mlir
        %0 = ... : !ida.instance
        %1 = ida.free %0 : !ida.instance
        ```
    }];

    let arguments = (ins IDAInstance:$instance);
    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

def IDA_PrintStatisticsOp : IDA_Op<"print_statistics",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
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

    let arguments = (ins IDAInstance:$instance);
    let results = (outs);

    let assemblyFormat = [{
        $instance attr-dict `:` type($instance)
    }];
}

#endif // MARCO_DIALECTS_IDA_IDA_TD
