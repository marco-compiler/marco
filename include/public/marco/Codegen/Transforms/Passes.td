#ifndef MARCO_CODEGEN_TRANSFORMS_PASSES_TD
#define MARCO_CODEGEN_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def AccessReplacementTestPass
    : Pass<"test-access-replacement", "ModelOp">
{
    let summary = "Test the access replacement.";

    let description = [{
        Test the replacement of accesses within equations.
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
    ];

    let constructor = "mlir::modelica::createAccessReplacementTestPass()";
}

def ArrayDeallocationPass : Pass<"array-deallocation", "mlir::ModuleOp">
{
    let summary = "Automatic array deallocation";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
    ];

    let constructor = "mlir::modelica::createArrayDeallocationPass()";
}

def AutomaticDifferentiationPass : Pass<"auto-diff", "mlir::ModuleOp">
{
    let summary = "Automatic differentiation of functions";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let constructor = "mlir::modelica::createAutomaticDifferentiationPass()";
}

def BindingEquationConversionPass
    : Pass<"convert-binding-equations", "ModelOp">
{
    let summary = "Convert binding equations.";

    let description = [{
        Convert the binding equations into equations or start operations.
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let constructor = "mlir::modelica::createBindingEquationConversionPass()";
}

def DerivativesAllocationPass
    : Pass<"allocate-derivatives", "ModelOp">
{
    let summary = "Create the variables for the derivatives.";

    let description = [{
        Create the derivative variable and remove the derivative operations.
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
    ];

    let constructor = "mlir::modelica::createDerivativesAllocationPass()";
}

def EquationAccessSplitPass
    : Pass<"split-overlapping-accesses", "ModelOp">
{
    let summary = "Split the overlapping read / write accesses.";

    let description = [{
        Split the indices of the equations so that no access can be a read and
        a write at the same time.
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
    ];

    let constructor = "mlir::modelica::createEquationAccessSplitPass()";
}

def EquationSidesSplitPass
    : Pass<"split-equation-sides", "ModelOp">
{
    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
    ];

    let constructor = "mlir::modelica::createEquationSidesSplitPass()";
}

def EquationInductionsExplicitationPass
    : Pass<"explicitate-equation-inductions", "ModelOp">
{
    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
    ];

    let constructor =
        "mlir::modelica::createEquationInductionsExplicitationPass()";
}

def EquationTemplatesCreationPass
    : Pass<"create-equation-templates", "ModelOp">
{
    let summary = "Create the equation templates and their instances.";

    let description = [{
        Create the equation templates and their instances from initial and
        non-initial equations.
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
    ];

    let constructor = "mlir::modelica::createEquationTemplatesCreationPass()";
}

def ExplicitCastInsertionPass
    : Pass<"explicit-cast-insertion", "mlir::ModuleOp">
{
    let summary = "Explicit cast insertion";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
        "mlir::scf::SCFDialect"
    ];

    let constructor = "mlir::modelica::createExplicitCastInsertionPass()";
}

def ExplicitInitialEquationsInsertionPass
    : Pass<"insert-explicit-initial-equations", "ModelOp">
{
    let summary = "Explicitate all the initial equations.";

    let description = [{
        Create the initial equations from the fixed start values and from the
        already existing non-initial equations.
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
    ];

    let constructor =
        "mlir::modelica::createExplicitInitialEquationsInsertionPass()";
}

def ExplicitStartValueInsertionPass
    : Pass<"insert-missing-start-values", "ModelOp">
{
    let summary = "Add the missing start values.";

    let description = [{
        Create the start values for variables that do not have one.
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
    ];

    let constructor =
        "mlir::modelica::createExplicitStartValueInsertionPass()";
}

def FunctionScalarizationPass : Pass<"scalarize", "mlir::ModuleOp">
{
    let summary = "Scalarize";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
        "mlir::arith::ArithDialect",
        "mlir::cf::ControlFlowDialect",
        "mlir::scf::SCFDialect"
    ];

    let options = [
        Option<"assertions", "assertions", "bool", "true",
               "Enable the runtime assertions">,
    ];

  let constructor = "mlir::modelica::createFunctionScalarizationPass()";
}

def FunctionInliningPass : Pass<"inline-functions", "mlir::ModuleOp">
{
    let summary = "Inline Modelica functions.";

    let description = [{
        Inline modelica functions.
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
    ];

    let constructor = "mlir::modelica::createFunctionInliningPass()";
}

def RangeBoundariesInferencePass : Pass<"infer-range-boundaries", "mlir::ModuleOp">
{
    let summary = "Infer the boundaries of unbounded ranges.";

    let description = [{
        Infer the boundaries of unbounded ranges.
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
    ];

    let constructor = "mlir::modelica::createRangeBoundariesInferencePass()";
}

def RecordInliningPass : Pass<"inline-records", "mlir::ModuleOp">
{
    let summary = "Instantiate a model.";

    let description = [{
        Instantiate a model by removing all the object-oriented features.
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let constructor = "mlir::modelica::createRecordInliningPass()";
}

def ModelAlgorithmConversionPass
    : Pass<"convert-model-algorithms", "mlir::ModuleOp">
{
    let summary = "Convert algorithms inside models.";

    let description = [{
        Convert the algorithms inside models.
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
    ];

    let constructor = "mlir::modelica::createModelAlgorithmConversionPass()";
}

def MatchingPass : Pass<"match", "ModelOp">
{
    let summary = "Perform the matching process on a model";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let options = [
        Option<"enableSimplificationAlgorithm",
                "enable-simplification-algorithm", "bool", "true",
               "Enable the usage of the simplification algorithm">
    ];

    let constructor = "mlir::modelica::createMatchingPass()";
}

def VariablesPromotionPass : Pass<"promote-variables-to-parameters", "ModelOp">
{
    let summary = "Promote to variables to parameters.";

    let description = [{
        Promote to parameters the variables that depend only on other parameters
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let constructor = "mlir::modelica::createVariablesPromotionPass()";
}

def ReadOnlyVariablesPropagationPass
    : Pass<"propagate-read-only-variables", "mlir::ModuleOp">
{
    let summary = "Propagate parameters and constants.";

    let description = [{
        Propagate the expression used to compute read-only variables (that is,
        parameters and constants).
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let options = [
        Option<"modelName", "model-name", "std::string", "",
               "Set the model onto which perform the transformation">,
        Option<"ignoredVariables", "ignored-variables", "std::string", "",
               "List of comma-separated variables that must not be propagated">
    ];

    let constructor = "mlir::modelica::createReadOnlyVariablesPropagationPass()";
}

def SCCDetectionPass : Pass<"detect-scc", "ModelOp">
{
    let summary = "Compute the SCCs of a model";

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let constructor = "mlir::modelica::createSCCDetectionPass()";
}

def SCCSolvingPass : Pass<"solve-scc", "ModelOp">
{
    let summary = "Solve the cycles of matched model by substitution";

    let description = [{
        Solve the cycles of matched model by substitution
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let options = [
        Option<"maxIterations", "max-iterations", "int64_t", "1000",
               "Set the maximum number of iterations">
    ];

    let constructor = "mlir::modelica::createSCCSolvingPass()";
}

def SchedulingPass : Pass<"schedule", "ScheduleOp">
{
    let summary = "Perform the scheduling process";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let constructor = "mlir::modelica::createSchedulingPass()";
}

def EquationExplicitationPass
    : Pass<"equation-explicitation", "mlir::ModuleOp">
{
    let summary = "Convert the explicitable equations into functions";

    let description = [{
        Convert the explicitable equations into functions.
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
        "mlir::arith::ArithDialect",
        "mlir::scf::SCFDialect"
    ];

    let constructor = "mlir::modelica::createEquationExplicitationPass()";
}

def EquationFunctionLoopHoistingPass
    : Pass<"equation-function-loop-hoist", "EquationFunctionOp">
{
    let summary = "Hoist operation out of loops inside equation functions";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let constructor = "mlir::modelica::createEquationFunctionLoopHoistingPass()";
}

def EquationExplicitationTestPass
    : Pass<"test-equation-explicitation", "ModelOp">
{
    let summary = "Test the explicitation of equations";

    let description = [{
        The pass is used to test the explicitation process of equations.
        It is not designed to be used within a compilation pipeline.
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let constructor = "mlir::modelica::createEquationExplicitationTestPass()";
}

def SimulationVariablesInsertionPass
    : Pass<"insert-simulation-variables", "mlir::ModuleOp">
{
    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
    ];

    let constructor = "mlir::modelica::createSimulationVariablesInsertionPass()";
}

def EulerForwardPass : Pass<"euler-forward", "mlir::ModuleOp">
{
    let summary = "TODO";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
        "mlir::simulation::SimulationDialect",
        "mlir::arith::ArithDialect",
        "mlir::cf::ControlFlowDialect",
        "mlir::func::FuncDialect",
        "mlir::LLVM::LLVMDialect",
        "mlir::scf::SCFDialect"
    ];

    let constructor = "mlir::modelica::createEulerForwardPass()";
}

def IDAPass : Pass<"ida", "mlir::ModuleOp">
{
    let summary = "Solve the model with IDA";

    let description = [{
        Apply the BDF integration algorithm by using the IDA external solver.
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect",
        "mlir::ida::IDADialect",
        "mlir::simulation::SimulationDialect",
        "mlir::arith::ArithDialect",
        "mlir::cf::ControlFlowDialect",
        "mlir::func::FuncDialect",
        "mlir::LLVM::LLVMDialect",
        "mlir::scf::SCFDialect"
    ];

    let options = [
        Option<"reducedSystem", "reduced-system", "bool", "true",
               "Compute a reduced system for IDA">,
        Option<"reducedDerivatives", "reduced-derivatives", "bool", "true",
               "Determine the derivatives that are always equal to zero">,
        Option<"jacobianOneSweep", "jacobian-one-sweep", "bool", "true",
               "Enable one-sweep computation of derivatives with respect to state and derivative variables">,
        Option<"debugInformation", "debug-information", "bool", "false",
               "Insert the information for runtime debugging">
    ];

    let constructor = "mlir::modelica::createIDAPass()";
}

def NegateOpDistributionPass : Pass<"distribute-neg", "mlir::ModuleOp">
{
    let summary = "Distribute the negation operations";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let constructor = "mlir::modelica::createNegateOpDistributionPass()";
}

def MulOpDistributionPass : Pass<"distribute-mul", "mlir::ModuleOp">
{
    let summary = "Distribute the multiplication operations";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let constructor = "mlir::modelica::createMulOpDistributionPass()";
}

def DivOpDistributionPass : Pass<"distribute-div", "mlir::ModuleOp">
{
    let summary = "Distribute the division operations";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let constructor = "mlir::modelica::createDivOpDistributionPass()";
}

def ModelDebugCanonicalizationPass : Pass<"canonicalize-model-for-debug", "ModelOp">
{
    let dependentDialects = [
        "mlir::modelica::ModelicaDialect"
    ];

    let constructor = "mlir::modelica::createModelDebugCanonicalizationPass()";
}

#endif // MARCO_CODEGEN_TRANSFORMS_PASSES_TD
