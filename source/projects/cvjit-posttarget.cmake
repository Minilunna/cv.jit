
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} PUBLIC ${EXTRA_LIBS} )

#set(C74_CXX_STANDARD 98)
include("${CMAKE_CURRENT_SOURCE_DIR}/../../max-sdk-base/script/max-posttarget.cmake")

# Copy the external to the externals directory
foreach (copy_destination ${COPY_DIR})
if (APPLE)
    add_custom_command(TARGET ${THIS_FOLDER_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    $<TARGET_BUNDLE_DIR:${THIS_FOLDER_NAME}>
    "${copy_destination}/$<TARGET_FILE_NAME:${THIS_FOLDER_NAME}>.mxo"
    )
else ()
    add_custom_command(TARGET ${THIS_FOLDER_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy 
    $<TARGET_FILE:${THIS_FOLDER_NAME}> 
    "${copy_destination}/$<TARGET_FILE_NAME:${THIS_FOLDER_NAME}>"
    )
endif()
    
endforeach()



