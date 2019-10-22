find_program(CLANGC NAMES clang clang-9)

### Copy test file
configure_file(FlattenedThermalChipODE.mo ${CMAKE_CURRENT_BINARY_DIR}/FlattenedThermalChipODE.mo COPYONLY)

### Creates the command that will run the simulator generator and will produce the bc file
add_custom_command(OUTPUT FlattenedThermalChipODE.bc 
	COMMAND $<TARGET_FILE:omcc> --simTime=10000 -bc=FlattenedThermalChipODE.bc ${CMAKE_CURRENT_BINARY_DIR}/FlattenedThermalChipODE.mo
	DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/FlattenedThermalChipODE.mo modelica::omcc
	COMMENT "generates flattenedThermalchp bc file"
	VERBATIM)


### Creates the command that will run clang to generate the Obj file.
add_custom_command(OUTPUT FlattenedThermalChipODE.o
	COMMAND ${CLANGC} -o FlattenedThermalChipODE.o -fPIC -c FlattenedThermalChipODE.bc
					COMMENT "generates flattendThermalChip simulation object file"
					DEPENDS FlattenedThermalChipODE.bc
					VERBATIM)

### Creates a target so that other targets can depend on this one to make sure
### the obj file is up to date
add_custom_target(genFlattenedThermalChipODE ALL DEPENDS FlattenedThermalChipODE.o)

add_executable(FlattenedThermalChipODE dummy.cpp)
target_link_libraries(FlattenedThermalChipODE PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/FlattenedThermalChipODE.o modelica::runtime)
add_dependencies(FlattenedThermalChipODE genFlattenedThermalChipODE)
