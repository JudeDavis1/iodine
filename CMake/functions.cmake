function (include_local_dirs tgt)
	target_include_directories(${tgt} PUBLIC "../../Iodyn")
endfunction()