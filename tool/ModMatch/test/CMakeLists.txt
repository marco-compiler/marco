### This macros adds a test that checks if for a particular 
### modelica model named sourceModel exactly expectedMatchedEq 
### can be matched in maxIterations 
macro(addMatchingTest sourceModel expectedMatchedEqs maxIterations)

	### creates a output dir if it does not exists
	file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/simulations)

	### creates a command to generate the simulation file 
	### starting from the modelica file
	add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/simulations/${sourceModel}.sm
		COMMAND $<TARGET_FILE:omcc> -d -o=${CMAKE_CURRENT_BINARY_DIR}/simulations/${sourceModel}.sm ${CMAKE_CURRENT_SOURCE_DIR}/src/${sourceModel}.mo
		DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/${sourceModel}.mo marco::modc
		COMMENT "generating ${sourceModel} .sm file"
		VERBATIM
		)

	### promotes the simulation name to target so that it's possible to depend on it.
	add_custom_target(generated${sourceModel}
		ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/simulations/${sourceModel}.sm)

	### creates a test that tries to theck if the number is correct
	add_test(NAME matching${sourceModel}Test
			COMMAND $<TARGET_FILE:modmatch> --expectedMatches=${expectedMatchedEqs} --iterations=${maxIterations} ${CMAKE_CURRENT_BINARY_DIR}/simulations/${sourceModel}.sm
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/simulations/)
	
	### add the dependency
	set_property(TEST matching${sourceModel}Test APPEND PROPERTY DEPENDS generated${sourceModel})

endmacro(addMatchingTest)

addMatchingTest(SimpleVector 10 1)
addMatchingTest(TwoEquationSimpleVector 10 2)
addMatchingTest(OverlappedEquationSimpleVector 7 5)
addMatchingTest(AdvectionReaction 10 5)
addMatchingTest(HarmonicOscillator 8 5)
addMatchingTest(TransmissionLineEquations 7 5)
#addMatchingTest(SimpleAdvection 8 10)

