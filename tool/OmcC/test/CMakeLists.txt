find_program(CLANGC NAMES clang clang-9)


macro(addOmccTest INPUTFILE)
	### Copy test file
	configure_file(${INPUTFILE}.mo ${CMAKE_CURRENT_BINARY_DIR}/${INPUTFILE}.mo COPYONLY)

	### Creates the command that will run the simulator generator and will produce the bc file
	add_custom_command(OUTPUT ${INPUTFILE}.bc 
		COMMAND $<TARGET_FILE:omcc> --simTime=10000 -o=${INPUTFILE}.bc ${CMAKE_CURRENT_BINARY_DIR}/${INPUTFILE}.mo
		DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${INPUTFILE}.mo marco::omcc
		COMMENT "generates ${INPUTFILE} bc file"
		VERBATIM)


	### Creates the command that will run clang to generate the Obj file.
	add_custom_command(OUTPUT ${INPUTFILE}.o
		COMMAND ${CLANGC} -o ${INPUTFILE}.o -fPIC -c ${INPUTFILE}.bc
						COMMENT "generates ${INPUTFILE} simulation object file"
						DEPENDS ${INPUTFILE}.bc
						VERBATIM)

	### Creates a target so that other targets can depend on this one to make sure
	### the obj file is up to date
	add_custom_target(gen${INPUTFILE} ALL DEPENDS ${INPUTFILE}.o)

	add_executable(${INPUTFILE} dummy.cpp)
	target_link_libraries(${INPUTFILE} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/${INPUTFILE}.o marco::runtime)
	add_dependencies(${INPUTFILE} gen${INPUTFILE})
endmacro(addOmccTest)

# addOmccTest(FlattenedThermalChipODE)
# addOmccTest(FlattenedThermalChipCoolingOde)
# addOmccTest(ZeroingTest)
# addOmccTest(MUltiDimesionalZeroing)
# addOmccTest(DepthThreeCycle)
# addOmccTest(FlattenedThermalChipCoolingOOSimplified)
