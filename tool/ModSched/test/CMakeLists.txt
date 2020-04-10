### This macros adds a test that runs a the scheduling on a file
macro(addSchedTest sourceModel)


	### creates a test that tries to theck if the number is correct
	add_test(NAME matching${sourceModel}Test
		COMMAND $<TARGET_FILE:modsched> --dumpModel ${CMAKE_CURRENT_SOURCE_DIR}/src/${sourceModel}.sm
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/)
	
	### add the dependency
	set_property(TEST matching${sourceModel}Test APPEND PROPERTY DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/${sourceModel}.sm)

endmacro(addSchedTest)

addSchedTest(ArrayTest)
